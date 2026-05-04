"""Orchestrator — generate -> validate -> jd-coverage-judge -> retry loop.

Each attempt:
  1. Generate (LLM)
  2. Drop empty sections + skill boost (mechanical)
  3. Mechanical validators: schema, source-attribution, field-lock, page-fit, jd-coverage-tokens
  4. PostPageFitTrim if overflow (mechanical)
  5. JDCoverageJudge (LLM) — semantic 0-1 score; uncovered reqs go into next retry feedback
  6. VoiceCheck (mechanical, informational)
  7. OrphanFixer on final candidate (LLM, only when needed)

Feedback replaces across attempts (no unbounded prompt growth).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from harness.generate import FeedbackMessage, generate
from harness.judges.jd_coverage_judge import JDCoverageJudge, JDCoverageResult
from harness.judges.orphan_fixer import detect_orphans, fix_orphans
from harness.judges.voice_check import VoiceCheck, VoiceCheckResult
from harness.models import ModelClient
from harness.ranking import rank_bullets_by_jd
from harness.schema import Resume, iter_bullets
from harness.validators import ValidationResult, run_all

log = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3


@dataclass(slots=True)
class RetryRecord:
    attempt: int
    validator_results: list[ValidationResult]
    jd_coverage_result: JDCoverageResult | None
    voice_result: VoiceCheckResult | None
    feedback: FeedbackMessage


@dataclass(slots=True)
class OrchestratorResult:
    passed: bool
    final_resume: Resume
    trajectory: list[RetryRecord]
    final_metrics: dict[str, Any]


def run(
    *,
    jd: str,
    input_resume: Resume,
    model: ModelClient,
    judge_model: ModelClient | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    progress: Callable[[str, dict], None] | None = None,
) -> OrchestratorResult:
    """Run the harness loop."""
    judge_client = judge_model or model
    trajectory: list[RetryRecord] = []
    feedback = FeedbackMessage()
    last_resume: Resume | None = None
    last_jd_cov: JDCoverageResult | None = None
    last_voice: VoiceCheckResult | None = None
    last_generate_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        if progress:
            progress("generating", {"attempt": attempt})

        try:
            gen_resp = generate(jd=jd, input_resume=input_resume, feedback=feedback, model=model)
        except Exception as exc:  # noqa: BLE001
            log.warning("generate() raised on attempt %d: %r", attempt, exc)
            last_generate_exc = exc
            feedback = FeedbackMessage(schema_errors=[f"previous attempt raised: {exc}"])
            trajectory.append(RetryRecord(attempt=attempt, validator_results=[], jd_coverage_result=None, voice_result=None, feedback=feedback))
            continue

        candidate = _drop_empty_sections(gen_resp.data)
        candidate = _boost_skills_from_jd(candidate, input_resume, jd)
        last_resume = candidate

        exp_summary = [(e.employer, len(e.bullets)) for e in candidate.experience]
        proj_summary = [(p.name[:30], len(p.bullets)) for p in candidate.projects]
        total_bullets = sum(b for _, b in exp_summary) + sum(b for _, b in proj_summary)
        log.info(
            "[attempt %d] generated — exp: %s | proj: %s | total bullets: %d",
            attempt, exp_summary, proj_summary, total_bullets,
        )

        if progress:
            progress("validating", {"attempt": attempt})
        validator_results = run_all(output=candidate, input_resume=input_resume, jd=jd)
        # page_fit overflow is non-blocking (post-processing trims it).
        # Underutilization IS retried so the LLM gets explicit "add more content" feedback,
        # but only on non-final attempts so we don't exhaust retries chasing an unfillable page.
        mechanical_passed = all(v.passed for v in validator_results if v.name != "page_fit")
        _pf_result = next((v for v in validator_results if v.name == "page_fit"), None)
        _util_pct = _pf_result.payload.get("page_utilization_pct", 100) if _pf_result else 100
        _pages = _pf_result.payload.get("pages", 1) if _pf_result else 1
        page_underutilized = _util_pct < 90 and attempt < max_retries
        page_overflowed = _pages > 1 and attempt < max_retries
        log.info(
            "[attempt %d] page_fit — pages=%s util=%d%% underutilized=%s mechanical_passed=%s",
            attempt,
            _pf_result.payload.get("pages", "?") if _pf_result else "?",
            _util_pct,
            page_underutilized,
            mechanical_passed,
        )

        # JD coverage judge runs every attempt — gives semantic score + uncovered reqs for feedback.
        if progress:
            progress("jd_coverage_judge", {"attempt": attempt})
        jd_cov_result: JDCoverageResult | None = None
        try:
            jd_cov_result = JDCoverageJudge().run(output_resume=candidate, jd=jd, model=judge_client)
        except Exception as exc:  # noqa: BLE001
            log.warning("jd_coverage judge failed on attempt %d: %r", attempt, exc)
        last_jd_cov = jd_cov_result

        voice_result = VoiceCheck().run(input_resume=input_resume, output_resume=candidate)
        last_voice = voice_result

        feedback = _build_feedback(validator_results, jd_cov_result)
        trajectory.append(RetryRecord(
            attempt=attempt,
            validator_results=validator_results,
            jd_coverage_result=jd_cov_result,
            voice_result=voice_result,
            feedback=feedback,
        ))

        jd_passed = jd_cov_result is None or jd_cov_result.passed
        _will_exit = mechanical_passed and jd_passed and not page_underutilized and not page_overflowed
        log.info(
            "[attempt %d] decision — jd_passed=%s page_underutilized=%s page_overflowed=%s → %s",
            attempt, jd_passed, page_underutilized, page_overflowed,
            "EXIT" if _will_exit else "RETRY",
        )
        if not _will_exit:
            log.info("[attempt %d] feedback:\n%s", attempt, feedback.render() if not feedback.is_empty() else "(empty)")
        if _will_exit:
            candidate = _fix_and_trim_orphans(candidate, judge_client, input_resume, jd)
            return OrchestratorResult(
                passed=True,
                final_resume=candidate,
                trajectory=trajectory,
                final_metrics=_metrics(validator_results, jd_cov_result, voice_result),
            )

    if last_resume is None:
        raise RuntimeError(
            f"All {max_retries + 1} generation attempt(s) failed without producing a candidate."
            + (f" Last error: {last_generate_exc}" if last_generate_exc else "")
        )
    last_resume = _fix_and_trim_orphans(last_resume, judge_client, input_resume, jd)
    return OrchestratorResult(
        passed=False,
        final_resume=last_resume,
        trajectory=trajectory,
        final_metrics=_metrics(trajectory[-1].validator_results, last_jd_cov, last_voice),
    )


def _build_feedback(validators: list[ValidationResult], jd_cov: JDCoverageResult | None) -> FeedbackMessage:
    fb = FeedbackMessage()
    by_name = {v.name: v for v in validators}

    if (sc := by_name.get("schema_check")) and not sc.passed:
        fb.schema_errors = list(sc.errors)
    if (sa := by_name.get("source_attribution")) and not sa.passed:
        fb.source_attribution_errors = list(sa.errors)
    if (fl := by_name.get("field_lock")) and not fl.passed:
        fb.field_lock_errors = list(fl.errors)
    if pf := by_name.get("page_fit"):
        if not pf.passed:
            fb.page_fit_overflow = list(pf.payload.get("overflow_bullet_ids", []))
        util = pf.payload.get("page_utilization_pct", 100)
        if util < 85:
            fb.page_fit_underutilized = (
                f"Page is only {util}% full — add a 3rd or 4th experience role, "
                "or add more bullets to existing roles, to reach ≥90% page utilization."
            )
    if jd_cov is not None and not jd_cov.passed:
        fb.jd_coverage_score = round(jd_cov.score, 3)
        fb.jd_coverage_missing = list(jd_cov.uncovered_requirements[:15])
    return fb


def _fix_and_trim_orphans(resume: Resume, model: ModelClient, input_resume: Resume, jd: str) -> Resume:
    """Post-processing: page trim → orphan fix → page trim again.

    1. Page-fit trim first (handles overflow from generation)
    2. LLM rewrite of danger-zone bullets (expand to 2 full lines or cut to 1)
    3. Mechanical trim fallback for any still in the danger zone
    4. Page-fit check again (orphan expansion can cause new overflow)
    """
    from harness.validators.page_fit import PageFitValidator

    # Step 1: trim any page overflow from generation
    pf = PageFitValidator().run(resume)
    if not pf.passed:
        overflow_ids = pf.payload.get("overflow_bullet_ids") or []
        if overflow_ids:
            resume = _trim_for_page_fit(resume, input_resume, jd, overflow_ids, max_iterations=5)

    # Step 2: fix orphan bullets
    orphans = detect_orphans(resume)
    if orphans:
        resume = fix_orphans(resume, orphans, model)
        still_bad = detect_orphans(resume)
        if still_bad:
            resume = _mechanical_trim_orphans(resume, still_bad)

    # Step 3: re-check page fit after orphan expansion
    pf2 = PageFitValidator().run(resume)
    if not pf2.passed:
        overflow_ids2 = pf2.payload.get("overflow_bullet_ids") or []
        if overflow_ids2:
            resume = _trim_for_page_fit(resume, input_resume, jd, overflow_ids2, max_iterations=5)

    exp_summary = [(e.employer, len(e.bullets)) for e in resume.experience]
    total_bullets = sum(b for _, b in exp_summary) + sum(len(p.bullets) for p in resume.projects)
    final_util = pf2.payload.get("page_utilization_pct", "?")
    log.info("post-processing final — exp: %s | total bullets: %d | util: %s%%", exp_summary, total_bullets, final_util)
    return resume


def _mechanical_trim_orphans(resume: Resume, orphan_ids: list[str]) -> Resume:
    """Last-resort: trim trailing words until bullet is ≤ 95 chars (clean single line)."""
    from harness.judges.orphan_fixer import SINGLE_LINE_MAX
    id_set = set(orphan_ids)
    data = resume.model_dump()
    for key in ("experience", "education", "projects"):
        for section in data[key]:
            for bullet in section["bullets"]:
                if bullet["id"] in id_set:
                    words = bullet["text"].split()
                    while len(" ".join(words)) > SINGLE_LINE_MAX and len(words) > 3:
                        words.pop()
                    bullet["text"] = " ".join(words)
    return Resume.model_validate(data)


def _prune_bullets(resume: Resume, drop_ids: set[str]) -> Resume:
    data = resume.model_dump()
    for key in ("experience", "education", "projects"):
        for section in data[key]:
            section["bullets"] = [b for b in section["bullets"] if b["id"] not in drop_ids]
    return _drop_empty_sections(Resume.model_validate(data))


def _boost_skills_from_jd(resume: Resume, input_resume: Resume, jd: str) -> Resume:
    """Mechanically add supported JD terms to the skills list.

    Only adds terms that are already in the INPUT resume's skills (no fabrication).
    Tokens are matched case-insensitively; the display form is preserved from input.
    """
    from harness.ranking import extract_jd_key_terms
    from harness.validators.jd_coverage import resume_keywords, tokenize

    jd_terms = set(extract_jd_key_terms(jd, top_n=50))
    output_kw = resume_keywords(resume)
    missing = jd_terms - output_kw
    if not missing:
        return resume

    # token → display string from input skills
    input_skill_map: dict[str, str] = {}
    for sk in input_resume.skills:
        for tok in tokenize(sk):
            input_skill_map.setdefault(tok, sk)

    seen = {s.lower() for s in resume.skills}
    to_add: list[str] = []
    for term in sorted(missing):
        if term not in input_skill_map:
            continue
        display = input_skill_map[term]
        if display.lower() in seen:
            continue
        to_add.append(display)
        seen.add(display.lower())

    if not to_add:
        return resume

    data = resume.model_dump()
    data["skills"] = data["skills"] + to_add
    return Resume.model_validate(data)


def _drop_empty_sections(resume: Resume) -> Resume:
    """Remove experience and project entries with zero bullets.

    Empty entries consume 2 header lines each and add no value. Education entries
    are kept even without bullets (degree + GPA lines are always worth showing).
    """
    data = resume.model_dump()
    for key in ("experience", "projects"):
        data[key] = [s for s in data[key] if s["bullets"]]
    return Resume.model_validate(data)


def _trim_for_page_fit(
    candidate: Resume,
    input_resume: Resume,
    jd: str,
    overflow_ids: list[str],
    *,
    max_iterations: int = 5,
) -> Resume:
    """Deterministic page-fit recovery.

    Drop the LOWEST-JD-ranked bullets among the overflowing ones, one at a time,
    re-rendering after each drop until pages == 1 or we run out of overflow
    candidates. The first input bullet of the highest-ranked role is never
    dropped (so we always preserve at least one piece of the strongest experience).
    """
    from harness.validators.page_fit import PageFitValidator

    # Rank OUTPUT bullets by their source bullets' JD-relevance (use the input
    # ranking as a proxy — output bullet inherits its source's relevance).
    input_scores = {bs.bullet_id: bs.score for bs in rank_bullets_by_jd(input_resume, jd)}

    def output_bullet_score(bullet_id: str, source_ids: list[str]) -> float:
        if not source_ids:
            return 0.0  # unsourced bullet → drop first
        return max(input_scores.get(sid, 0.0) for sid in source_ids)

    current = candidate
    for _ in range(max_iterations):
        # Re-validate page fit on the current candidate.
        pf_result = PageFitValidator().run(current)
        if pf_result.passed:
            return current
        live_overflow = pf_result.payload.get("overflow_bullet_ids") or []
        if not live_overflow:
            return current

        # Build (bullet_id, score) for currently-overflowing bullets only.
        scored: list[tuple[str, float]] = []
        for b in iter_bullets(current):
            if b.id in live_overflow:
                scored.append((b.id, output_bullet_score(b.id, b.source_ids)))
        if not scored:
            return current
        scored.sort(key=lambda kv: kv[1])  # ascending — drop lowest first
        drop_id = scored[0][0]
        current = _prune_bullets(current, {drop_id})
        current = _drop_empty_sections(current)  # clean up roles emptied by pruning
    return current


def _metrics(validators, jd_cov, voice) -> dict[str, Any]:
    by_name = {v.name: v for v in validators}
    jd_score = jd_cov.score if jd_cov is not None else (
        by_name["jd_coverage"].score if by_name.get("jd_coverage") else None
    )
    return {
        "schema_valid": by_name.get("schema_check") and by_name["schema_check"].passed,
        "source_attribution_score": by_name.get("source_attribution") and round(by_name["source_attribution"].score, 3),
        "field_lock_passed": by_name.get("field_lock") and by_name["field_lock"].passed,
        "page_fit_pages": by_name.get("page_fit") and by_name["page_fit"].payload.get("pages"),
        "page_fit_overflow_ids": by_name.get("page_fit") and by_name["page_fit"].payload.get("overflow_bullet_ids", []),
        "jd_coverage": round(jd_score, 3) if jd_score is not None else None,
        "voice_drift_mean": round(voice.mean_drift, 3) if voice is not None else None,
        "voice_drift_max": round(voice.max_drift, 3) if voice is not None else None,
    }
