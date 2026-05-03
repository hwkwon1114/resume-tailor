"""End-to-end eval runner.

Default: stub mode (uses EchoModelClient with a canned tailored resume per
fixture so you can sanity-check the harness wiring without burning quota).
With RUN_LIVE_EVALS=1, hits the live Gemini API.

Writes a markdown report to evals/report.md with per-fixture metrics, retry
trajectory, and cache-hit annotations.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from evals.cache import cache_key, get as cache_get, put as cache_put
from evals.fixtures import Fixture, all_baselines, temptation_fixture
from harness.models import EchoModelClient, ModelClient
from harness.orchestrator import OrchestratorResult, run
from harness.schema import Resume

LIVE = os.environ.get("RUN_LIVE_EVALS") == "1"
USE_CACHE = "--cache" in sys.argv
MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "3"))
REPORT_PATH = Path(__file__).parent / "report-gemini-cli.md"


def _make_live_client():
    from harness.models.gemini_subprocess import GeminiSubprocessClient
    return GeminiSubprocessClient()


def _make_judge_client():
    from harness.models.gemini_subprocess import GeminiSubprocessClient
    return GeminiSubprocessClient()


def _model_for(fixture_name: str) -> tuple[ModelClient, bool]:
    if LIVE:
        return _make_live_client(), False
    return _build_stub_model(), False


def _build_stub_model() -> EchoModelClient:
    # Filled in by run_one() per-fixture so source_ids reference real input ids.
    return EchoModelClient()


def _stub_tailor(input_resume: Resume) -> Resume:
    """Build a canned tailored resume that just echoes the input bullets with valid source_ids.

    This is enough to exercise schema/source-attribution/field-lock/page-fit/jd-coverage paths
    without burning live quota.
    """
    raw = input_resume.model_dump()
    for key in ("experience", "education", "projects"):
        for section in raw[key]:
            for i, b in enumerate(section["bullets"]):
                b["id"] = f"{section['id']}-out-{i}"
                b["source_ids"] = [f"{section['id']}-b{i}"]
    return Resume.model_validate(raw)



def run_one(fixture: Fixture) -> tuple[OrchestratorResult, bool]:
    """Run the orchestrator on one fixture, return (result, cache_hit)."""
    key = cache_key(fixture.jd, fixture.resume.model_dump())
    if USE_CACHE and (cached := cache_get(key)) is not None:
        cached_resume = Resume.model_validate(cached["final_resume"])
        return (
            OrchestratorResult(
                passed=cached["passed"],
                final_resume=cached_resume,
                trajectory=[],
                final_metrics=cached["final_metrics"],
            ),
            True,
        )

    if LIVE:
        model: ModelClient = _make_live_client()
        judge: ModelClient | None = _make_judge_client()
    else:
        tailored = _stub_tailor(fixture.resume)

        class _Dispatch(EchoModelClient):
            """Dispatch by schema name:
              TailoringResponse → canned tailored resume
              _CoverageReport  → stub JD coverage score (pass)
              _FixResult       → echo back original bullet texts unchanged
              anything else    → empty string
            """
            def generate_structured(self, **kwargs):
                from harness.models.base import ModelResponse
                schema = kwargs.get("schema")
                name = getattr(schema, "__name__", "")
                if name == "_CoverageReport":
                    from harness.judges.jd_coverage_judge import _CoverageReport
                    return ModelResponse(data=_CoverageReport(score=0.8, uncovered_requirements=[]), model_name="stub")
                if name == "_FixResult":
                    from harness.judges.orphan_fixer import _FixResult
                    # Extract bullet texts from prompt and echo unchanged
                    import json, re
                    prompt = kwargs.get("prompt", "")
                    try:
                        items = json.loads(re.search(r'\[.*\]', prompt, re.DOTALL).group())
                        texts = [i["text"] for i in items]
                    except Exception:
                        texts = []
                    return ModelResponse(data=_FixResult(bullets=texts), model_name="stub")
                if name == "_ResumeEnvelope":
                    from harness.pdf_intake import _ResumeEnvelope
                    return ModelResponse(data=_ResumeEnvelope(resume=tailored), model_name="stub")
                # Default: TailoringResponse
                from harness.generate import TailoringResponse
                return ModelResponse(data=TailoringResponse(reasoning="stub", resume=tailored), model_name="stub")

            def generate_text(self, **kwargs):
                from harness.models.base import ModelResponse
                return ModelResponse(data="", model_name="stub")

        model = _Dispatch()

    result = run(
        jd=fixture.jd,
        input_resume=fixture.resume,
        model=model,
        judge_model=model if not LIVE else judge,
        max_retries=1 if not LIVE else MAX_RETRIES,
    )

    if USE_CACHE:
        cache_put(key, {
            "passed": result.passed,
            "final_resume": result.final_resume.model_dump(),
            "final_metrics": result.final_metrics,
        })
    return result, False


def temptation_check(fixture: Fixture, result: OrchestratorResult) -> tuple[bool, list[str]]:
    """AC-11: no output bullet may mention any forbidden term."""
    text = " ".join(b.text.lower() for b in _all_bullets(result.final_resume))
    text += " " + result.final_resume.summary.lower()
    text += " " + " ".join(s.lower() for s in result.final_resume.skills)
    hits = [t for t in fixture.forbidden_terms if t in text]
    return (not hits), hits


def _all_bullets(resume: Resume):
    for e in resume.experience:
        yield from e.bullets
    for ed in resume.education:
        yield from ed.bullets
    for p in resume.projects:
        yield from p.bullets


def write_report(rows: list[dict], outpath: Path = REPORT_PATH) -> None:
    lines = []
    lines.append("# Eval Report — Gemini CLI\n")
    lines.append(f"_Generated {datetime.now().isoformat(timespec='seconds')} | mode={'LIVE' if LIVE else 'STUB'} | cache={'on' if USE_CACHE else 'off'}_\n")
    lines.append("\n## Summary\n")
    lines.append("| Fixture | Passed | Pages | JD Cov | Voice Drift | Retries | Cache | AC-11 |")
    lines.append("|---------|--------|-------|--------|-------------|---------|-------|-------|")
    for r in rows:
        ac11 = "n/a" if r["ac11_status"] is None else ("✓" if r["ac11_status"] else f"✗ ({','.join(r['ac11_hits'])})")
        lines.append(
            f"| {r['name']} | {'✓' if r['passed'] else '✗'} | {r['pages']} | {r['jd_coverage']} | "
            f"{r['voice_drift_max']} | {r['retries']} | "
            f"{'hit' if r['cache_hit'] else 'live'} | {ac11} |"
        )

    lines.append("\n## Per-fixture detail\n")
    for r in rows:
        lines.append(f"### {r['name']}\n")
        lines.append(f"- passed: {r['passed']}")
        lines.append(f"- final_metrics:\n```json\n{json.dumps(r['final_metrics'], indent=2)}\n```")
        lines.append(f"- trajectory:")
        for t in r["trajectory"]:
            failed = [v for v in t["validators"] if not v["passed"]]
            lines.append(
                f"  - attempt {t['attempt']}: validators_failed={[v['name'] for v in failed]}, "
                f"jd_coverage={t.get('jd_coverage_score')}"
            )
        lines.append("")

    outpath.write_text("\n".join(lines))


def main() -> int:
    fixtures = all_baselines() + [temptation_fixture()]
    rows = []
    any_failure = False
    for fix in fixtures:
        try:
            result, cache_hit = run_one(fix)
        except Exception as exc:  # noqa: BLE001
            print(f"FIXTURE {fix.name} CRASHED: {exc!r}")
            any_failure = True
            continue

        if fix.forbidden_terms:
            ac11_pass, ac11_hits = temptation_check(fix, result)
        else:
            ac11_pass, ac11_hits = None, []
        if ac11_pass is False:
            any_failure = True
        if not result.passed and fix.name != "mismatch-1":
            any_failure = True

        traj = []
        for t in result.trajectory:
            traj.append({
                "attempt": t.attempt,
                "validators": [{"name": v.name, "passed": v.passed, "score": v.score} for v in t.validator_results],
                "jd_coverage_score": (t.jd_coverage_result.score if t.jd_coverage_result else None),
            })

        rows.append({
            "name": fix.name,
            "passed": result.passed,
            "pages": result.final_metrics.get("page_fit_pages"),
            "jd_coverage": result.final_metrics.get("jd_coverage"),
            "voice_drift_max": result.final_metrics.get("voice_drift_max"),
            "retries": len(result.trajectory),
            "cache_hit": cache_hit,
            "trajectory": traj,
            "final_metrics": result.final_metrics,
            "ac11_status": ac11_pass,
            "ac11_hits": ac11_hits,
        })
        status = "PASS" if (result.passed and (ac11_pass is not False)) else "FAIL"
        print(f"  [{status}] {fix.name}: pages={rows[-1]['pages']} jd_cov={rows[-1]['jd_coverage']} cache={'hit' if cache_hit else 'live'}")

    write_report(rows)
    print(f"\nReport: {REPORT_PATH}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
