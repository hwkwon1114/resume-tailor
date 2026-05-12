"""CLI runner for the resume-tailoring harness.

Usage:
    # JD from a text file, resume defaults to resume_converted.json
    python tailor.py --jd path/to/jd.txt

    # JD piped from stdin
    pbpaste | python tailor.py

    # Custom resume JSON
    python tailor.py --jd jd.txt --resume other_resume.json

    # Custom output PDF path
    python tailor.py --jd jd.txt --out my_output.pdf

Output: tailored PDF saved to disk + metrics printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from harness.orchestrator import run
from harness.schema import Resume, autopopulate_bullet_ids

_ROOT = Path(__file__).parent
_DEFAULT_RESUME = _ROOT / "resume_converted.json"


def _load_resume(path: Path) -> Resume:
    raw = json.loads(path.read_text())
    return Resume.model_validate(autopopulate_bullet_ids(raw))


def _load_jd(path: Path | None) -> str:
    if path is not None:
        return path.read_text()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("ERROR: provide a JD via --jd <file> or pipe it via stdin.", file=sys.stderr)
    sys.exit(1)


def _print_metrics(metrics: dict, passed: bool) -> None:
    status = "PASSED" if passed else "FAILED (partial result — PDF still saved)"
    print(f"\n{'='*55}")
    print(f"  Result : {status}")
    print(f"  Pages  : {metrics.get('page_fit_pages', '?')}")
    print(f"  JD cov : {metrics.get('jd_coverage', 0):.2f}  (floor 0.50)")
    print(f"  Src att: {metrics.get('source_attribution_score', 0):.2f}  (floor 1.00)")
    print(f"  Schema : {'✓' if metrics.get('schema_valid') else '✗'}")
    print(f"  Field lock : {'✓' if metrics.get('field_lock_passed') else '✗'}")
    vd = metrics.get('voice_drift_max')
    print(f"  Voice drift (max): {vd:.2f}" if vd is not None else "  Voice drift: —")
    print(f"{'='*55}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tailor a resume to a job description.")
    parser.add_argument("--jd", type=Path, default=None, help="Path to JD text file (or pipe via stdin)")
    parser.add_argument("--resume", type=Path, default=_DEFAULT_RESUME, help="Resume JSON path")
    parser.add_argument("--out", type=Path, default=None, help="Output PDF path (default: <name>_tailored.pdf)")
    parser.add_argument("--retries", type=int, default=3, help="Max retries (default: 3)")
    args = parser.parse_args()

    jd = _load_jd(args.jd)
    resume = _load_resume(args.resume)

    out_path = args.out or _ROOT / f"{resume.contact.name.replace(' ', '_')}_tailored.pdf"

    from harness.models.gemini_subprocess import GeminiSubprocessClient
    model = GeminiSubprocessClient()

    attempt_log: list[str] = []

    def progress(stage: str, payload: dict) -> None:
        msg = f"  [{stage}] attempt {payload.get('attempt', '?')}"
        attempt_log.append(msg)
        print(msg, flush=True)

    print(f"\nTailoring resume for: {resume.contact.name}")
    print(f"Resume : {args.resume}")
    print(f"Output : {out_path}\n")

    result = run(
        jd=jd,
        input_resume=resume,
        model=model,
        judge_model=model,
        max_retries=args.retries,
        progress=progress,
    )

    _print_metrics(result.final_metrics, result.passed)

    from render.pdf import render_to_pdf
    pdf = render_to_pdf(result.final_resume)
    out_path.write_bytes(pdf)
    print(f"PDF saved → {out_path}")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(result.final_resume.model_dump_json(indent=2))
    print(f"JSON saved → {json_path}")


if __name__ == "__main__":
    main()
