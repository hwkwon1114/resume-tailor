"""Streamlit UI for the resume-tailoring harness.

JD textarea + JSON resume uploader → tailored PDF + live quality scores.
Live values, not just labels (validates AC-7).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from harness.orchestrator import OrchestratorResult, run
from harness.pdf_intake import extract_text_from_pdf
from harness.schema import Resume, autopopulate_bullet_ids
from render.pdf import render_to_pdf

MAX_JD_CHARS = 20_000
MAX_RESUME_BYTES = 200 * 1024


def _load_resume(text: str) -> Resume:
    raw = json.loads(text)
    return Resume.model_validate(autopopulate_bullet_ids(raw))


def _gemini() -> "GeminiSubprocessClient":
    from harness.models.gemini_subprocess import GeminiSubprocessClient
    return GeminiSubprocessClient()


def _intake_parse(text: str) -> "Resume":
    from harness.pdf_intake import text_to_resume
    return text_to_resume(text, _gemini())


def _metric_color(value: float | None, floor: float, warn: float) -> str:
    if value is None:
        return "off"
    if value >= floor:
        return "normal"
    if value >= warn:
        return "off"
    return "inverse"


def _render_metrics(result: OrchestratorResult) -> None:
    m = result.final_metrics
    cols = st.columns(4)
    with cols[0]:
        pages = m.get("page_fit_pages")
        st.metric("Pages", pages if pages is not None else "—", delta="✓" if pages == 1 else "overflow", delta_color=_metric_color(1.0 if pages == 1 else 0.0, 1.0, 0.5))
    with cols[1]:
        jc = m.get("jd_coverage")
        st.metric("JD coverage", f"{jc:.2f}" if jc is not None else "—", delta="floor 0.50", delta_color=_metric_color(jc, 0.5, 0.4))
    with cols[2]:
        sv = m.get("schema_valid")
        st.metric("Schema valid", "✓" if sv else "✗", delta_color=_metric_color(1.0 if sv else 0.0, 1.0, 0.5))
    with cols[3]:
        sa = m.get("source_attribution_score")
        st.metric("Source attribution", f"{sa:.2f}" if sa is not None else "—", delta_color=_metric_color(sa, 1.0, 0.8))

    st.metric("Voice drift (max)", f"{m.get('voice_drift_max', 0):.2f}", delta="lower is better", delta_color="off")


def main() -> None:
    st.set_page_config(page_title="Resume Tailor", layout="wide")
    st.title("Resume Tailor")
    st.caption("Gemini CLI · schema · source-attribution · field-lock · page-fit · JD-coverage · orphan-fix")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Job description")
        jd = st.text_area("Paste the JD", height=240, max_chars=MAX_JD_CHARS, label_visibility="collapsed")
        st.subheader("Your resume")
        resume_obj: Resume | None = None
        tab_json, tab_pdf, tab_paste = st.tabs(["Upload JSON", "Upload PDF", "Paste text"])

        with tab_json:
            upload_json = st.file_uploader("Resume JSON (max 200 KB)", type=["json"], key="upload_json")
            if upload_json is not None:
                data = upload_json.read()
                if len(data) > MAX_RESUME_BYTES:
                    st.error(f"File too large ({len(data)} bytes; max {MAX_RESUME_BYTES}).")
                else:
                    try:
                        resume_obj = _load_resume(data.decode("utf-8"))
                        st.success(f"Loaded resume for {resume_obj.contact.name}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not parse JSON: {exc}")

        with tab_pdf:
            upload_pdf = st.file_uploader("Resume PDF (max 200 KB)", type=["pdf"], key="upload_pdf")
            if upload_pdf is not None:
                data = upload_pdf.read()
                if len(data) > MAX_RESUME_BYTES:
                    st.error(f"File too large ({len(data)} bytes; max {MAX_RESUME_BYTES}).")
                else:
                    with st.spinner("Extracting and parsing PDF…"):
                        try:
                            text = extract_text_from_pdf(data)
                            if not text.strip():
                                st.error("Could not extract any text from this PDF (may be image-only).")
                            else:
                                resume_obj = _intake_parse(text)
                                st.success(f"Parsed resume for {resume_obj.contact.name}")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"PDF parse failed: {exc}")

        with tab_paste:
            pasted = st.text_area("Paste your resume text here", height=200, key="paste_text")
            parse_btn = st.button("Parse pasted text", key="parse_paste", disabled=not pasted.strip())
            if parse_btn and pasted.strip():
                with st.spinner("Parsing resume text…"):
                    try:
                        resume_obj = _intake_parse(pasted)
                        st.success(f"Parsed resume for {resume_obj.contact.name}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Parse failed: {exc}")

        ready = bool(jd.strip()) and resume_obj is not None
        run_clicked = st.button("Tailor", type="primary", disabled=not ready)

    with right:
        st.subheader("Quality scores")
        metrics_slot = st.container()
        progress_slot = st.empty()
        result_slot = st.container()

    if run_clicked:
        progress_slot.info("Generating…")
        progress_log: list[str] = []

        def progress(stage: str, payload: dict) -> None:
            progress_log.append(f"{stage}: {payload}")
            progress_slot.info(f"{stage} (attempt {payload.get('attempt', '?')})")

        result = run(jd=jd, input_resume=resume_obj, model=_gemini(), judge_model=_gemini(), progress=progress)
        st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        result: OrchestratorResult = st.session_state["last_result"]
        with metrics_slot:
            _render_metrics(result)
        with result_slot:
            if result.passed:
                st.success(f"Tailored resume ready (page count: {result.final_metrics.get('page_fit_pages')})")
            else:
                st.warning("Result did not fully pass the harness; see debug for details. PDF is still downloadable.")
            try:
                pdf = render_to_pdf(result.final_resume)
                st.download_button("Download PDF", pdf, file_name=f"{result.final_resume.contact.name.replace(' ', '_')}_resume.pdf", mime="application/pdf")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Render failed: {exc}")

            with st.expander("Debug: trajectory + raw output"):
                st.json({
                    "metrics": result.final_metrics,
                    "trajectory": [
                        {
                            "attempt": t.attempt,
                            "validators": [{"name": v.name, "passed": v.passed, "score": v.score, "errors": v.errors[:3]} for v in t.validator_results],
                            "jd_coverage": {"score": t.jd_coverage_result.score, "passed": t.jd_coverage_result.passed} if t.jd_coverage_result else None,
                        }
                        for t in result.trajectory
                    ],
                })
                st.json(result.final_resume.model_dump())


if __name__ == "__main__":
    main()
