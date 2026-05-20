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

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from harness.generate import FeedbackMessage, generate
from harness.judges.fabrication_audit import (
    THRESHOLD as FABRICATION_THRESHOLD,
    FabricationAudit,
    FabricationAuditResult,
)
from harness.judges.jd_coverage_judge import JDCoverageJudge, JDCoverageResult
from harness.judges.orphan_fixer import (
    SINGLE_LINE_MAX,
    OrphanCategory,
    detect_orphans,
    detect_orphans_geometry,
    fix_bullets,
    fix_orphans,
    llm_clean_truncated,
)
from harness.judges.voice_check import VoiceCheck, VoiceCheckResult
from harness.models import ModelClient
from harness.ranking import rank_bullets_by_jd
from harness.schema import Resume, iter_bullets
from harness.validators import ValidationResult, run_all

log = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3

_HANGING_CONNECTIVES = frozenset({
    "and", "or", "but", "with", "by", "for", "to", "in", "on", "at", "of",
    "a", "an", "the", "as", "while", "across", "through", "via",
    "including", "using", "from", "into", "than", "that", "which",
})


def _ensure_clean_ending(text: str) -> str:
    """Strip trailing dangling connectives and ensure the bullet ends with a period.

    A dangling connective (e.g. 'and team', 'across digital') signals a truncated
    phrase. We strip the connective AND the word that followed it together.
    """
    original = text
    text = text.rstrip(" .,;:-")
    words = text.split()
    # Repeatedly strip "<connective> <word>" pairs at the tail.
    while len(words) >= 2 and words[-2].lower().rstrip(".,;:") in _HANGING_CONNECTIVES:
        words.pop()  # drop the trailing word
        words.pop()  # drop the connective itself
    # Strip any remaining trailing standalone connective.
    while words and words[-1].lower().rstrip(".,;:") in _HANGING_CONNECTIVES:
        words.pop()
    text = " ".join(words).rstrip(" .,;:-")
    if not text:
        return original
    if text[-1] not in ".!?":
        text += "."
    return text


@dataclass(slots=True)
class RetryRecord:
    attempt: int
    validator_results: list[ValidationResult]
    jd_coverage_result: JDCoverageResult | None
    voice_result: VoiceCheckResult | None
    feedback: FeedbackMessage
    fabrication_result: FabricationAuditResult | None = None
    candidate: Resume | None = None


def _attempt_score(record: RetryRecord) -> tuple[int, int, int, int, int]:
    """Lexicographic attempt quality score; lower tuple = better.

    Used by the best-of-N fallback when no attempt achieved `_will_exit`.
    Priority order (each component is 0 = pass, 1 = fail except util):
      1. mechanical validators (schema, source attribution, field lock)
      2. fabrication audit (no flagged bullets)
      3. JD coverage at or above floor
      4. fits on one page
      5. utilization closer to 100% (capped) — prefers higher util
    """
    mech_pass = all(v.passed for v in record.validator_results if v.name != "page_fit")
    fab_pass = (
        record.fabrication_result is None
        or not record.fabrication_result.flagged_bullets
    )
    jd_pass = (record.jd_coverage_result is None or record.jd_coverage_result.passed)
    pf = next((v for v in record.validator_results if v.name == "page_fit"), None)
    pages = pf.payload.get("pages", 1) if pf else 1
    util = pf.payload.get("page_utilization_pct", 0) if pf else 0
    return (
        0 if mech_pass else 1,
        0 if fab_pass else 1,
        0 if jd_pass else 1,
        0 if pages == 1 else 1,
        100 - min(util, 100),
    )


@dataclass(slots=True)
class OrchestratorResult:
    passed: bool
    final_resume: Resume
    trajectory: list[RetryRecord]
    final_metrics: dict[str, Any]


_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "resume-tailor"


def _cache_key(jd: str, input_resume: Resume) -> str:
    """SHA-256 of jd + canonical resume JSON; stable across runs."""
    payload = jd + json.dumps(input_resume.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_load(key: str) -> OrchestratorResult | None:
    path = _CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("[cache] read failed for %s: %r — treating as miss", key[:12], exc)
        return None
    return OrchestratorResult(
        passed=data["passed"],
        final_resume=Resume.model_validate(data["final_resume"]),
        trajectory=[],
        final_metrics=data["final_metrics"],
    )


def _cache_store(key: str, result: OrchestratorResult) -> None:
    # Caching a failure would prevent retry-driven recovery on the next run.
    if not result.passed:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": result.passed,
        "final_resume": result.final_resume.model_dump(),
        "final_metrics": result.final_metrics,
    }
    (_CACHE_DIR / f"{key}.json").write_text(json.dumps(payload))


def _cache_enabled(use_cache: bool) -> bool:
    return use_cache and not os.environ.get("RESUME_TAILOR_NO_CACHE")


def run(
    *,
    jd: str,
    input_resume: Resume,
    model: ModelClient,
    judge_model: ModelClient | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    progress: Callable[[str, dict], None] | None = None,
    use_cache: bool = True,
) -> OrchestratorResult:
    """Run the harness loop."""
    cache_active = _cache_enabled(use_cache)
    cache_key = _cache_key(jd, input_resume) if cache_active else None
    if cache_key is not None:
        cached = _cache_load(cache_key)
        if cached is not None:
            log.info("[cache HIT] %s — skipping all LLM calls", cache_key[:12])
            return cached

    judge_client = judge_model or model
    trajectory: list[RetryRecord] = []
    feedback = FeedbackMessage()
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
        # page_fit overflow is non-blocking on non-final attempts (post-processing trims it).
        # Underutilization is a HARD gate on every attempt: a page below the 95% target is
        # never accepted via _will_exit. The fallthrough path picks the best attempt instead.
        mechanical_passed = all(v.passed for v in validator_results if v.name != "page_fit")
        _pf_result = next((v for v in validator_results if v.name == "page_fit"), None)
        _util_pct = _pf_result.payload.get("page_utilization_pct", 100) if _pf_result else 100
        _pages = _pf_result.payload.get("pages", 1) if _pf_result else 1
        page_underutilized = _util_pct < 95
        page_overflowed = _pages > 1 and attempt < max_retries
        log.info(
            "[attempt %d] page_fit — pages=%s util=%d%% underutilized=%s mechanical_passed=%s",
            attempt,
            _pf_result.payload.get("pages", "?") if _pf_result else "?",
            _util_pct,
            page_underutilized,
            mechanical_passed,
        )

        # JDCoverageJudge and FabricationAudit are independent LLM calls on the
        # same candidate — fan out in parallel via threads so wall-clock per
        # attempt drops from sum(jd, fab) to max(jd, fab). subprocess.run inside
        # GeminiSubprocessClient releases the GIL while waiting on the Gemini
        # CLI, so threading is sufficient — no asyncio refactor needed.
        # Strict policy: ANY bullet below FABRICATION_THRESHOLD triggers a retry.
        if progress:
            progress("jd_coverage_judge", {"attempt": attempt})
            progress("fabrication_audit", {"attempt": attempt})
        jd_cov_result: JDCoverageResult | None = None
        fab_result: FabricationAuditResult | None = None
        with ThreadPoolExecutor(max_workers=2) as judge_pool:
            jd_future = judge_pool.submit(
                JDCoverageJudge().run, output_resume=candidate, jd=jd, model=judge_client
            )
            fab_future = judge_pool.submit(
                FabricationAudit().run,
                input_resume=input_resume, output_resume=candidate, model=judge_client,
            )
            try:
                jd_cov_result = jd_future.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("jd_coverage judge failed on attempt %d: %r", attempt, exc)
            try:
                fab_result = fab_future.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("fabrication_audit failed on attempt %d: %r", attempt, exc)

        voice_result = VoiceCheck().run(input_resume=input_resume, output_resume=candidate)

        feedback = _build_feedback(validator_results, jd_cov_result, fab_result)
        trajectory.append(RetryRecord(
            attempt=attempt,
            validator_results=validator_results,
            jd_coverage_result=jd_cov_result,
            voice_result=voice_result,
            feedback=feedback,
            fabrication_result=fab_result,
            candidate=candidate,
        ))

        # A judge that ERRORED out (result is None — timeout, parse fail, etc.)
        # is NOT a pass. "Did not verify" must not be conflated with "verified
        # clean". Force a retry so the judge gets another chance; if every
        # attempt errors, the orchestrator falls through to best-of-N with
        # passed=False, an honest verdict.
        jd_passed = jd_cov_result is not None and jd_cov_result.passed
        fab_passed = fab_result is not None and not fab_result.flagged_bullets
        _will_exit = (
            mechanical_passed and jd_passed and fab_passed
            and not page_underutilized and not page_overflowed
        )
        log.info(
            "[attempt %d] decision — jd_passed=%s fab_passed=%s (flagged=%d) "
            "page_underutilized=%s page_overflowed=%s → %s",
            attempt, jd_passed, fab_passed,
            len(fab_result.flagged_bullets) if fab_result else 0,
            page_underutilized, page_overflowed,
            "EXIT" if _will_exit else "RETRY",
        )
        if not _will_exit:
            log.info("[attempt %d] feedback:\n%s", attempt, feedback.render() if not feedback.is_empty() else "(empty)")
        if _will_exit:
            candidate = _fix_and_trim_orphans(candidate, judge_client, input_resume, jd)
            metrics, pf_passed_after_postproc = _finalize_metrics(
                candidate, validator_results, jd_cov_result, voice_result, fab_result
            )
            # passed is honest about post-processing: if the orphan-fixer/trim
            # pipeline couldn't restore page_fit (rare, but possible when the
            # loop exited optimistically on the last attempt), passed=False.
            result = OrchestratorResult(
                passed=pf_passed_after_postproc,
                final_resume=candidate,
                trajectory=trajectory,
                final_metrics=metrics,
            )
            if cache_key is not None:
                _cache_store(cache_key, result)
            return result

    candidate_records = [r for r in trajectory if r.candidate is not None]
    if not candidate_records:
        raise RuntimeError(
            f"All {max_retries + 1} generation attempt(s) failed without producing a candidate."
            + (f" Last error: {last_generate_exc}" if last_generate_exc else "")
        )
    best = min(candidate_records, key=_attempt_score)
    log.info(
        "[fallthrough] best-of-N: attempt %d selected (score=%s, last=%d)",
        best.attempt, _attempt_score(best), candidate_records[-1].attempt,
    )
    best_resume = _fix_and_trim_orphans(best.candidate, judge_client, input_resume, jd)
    metrics, _ = _finalize_metrics(
        best_resume,
        best.validator_results,
        best.jd_coverage_result,
        best.voice_result,
        best.fabrication_result,
    )
    return OrchestratorResult(
        passed=False,
        final_resume=best_resume,
        trajectory=trajectory,
        final_metrics=metrics,
    )


def _finalize_metrics(
    resume: Resume,
    validator_results: list[ValidationResult],
    jd_cov: JDCoverageResult | None,
    voice: VoiceCheckResult | None,
    fab: FabricationAuditResult | None,
) -> tuple[dict[str, Any], bool]:
    """Re-validate page_fit on the post-processed resume so metrics + passed are honest.

    Returns (metrics_dict, targets_met). targets_met is True iff the final
    rendered PDF meets BOTH targets: pages == 1 AND page_utilization_pct ≥ 95.
    Callers use it as the OrchestratorResult.passed flag, so passed=True now
    means the resume genuinely satisfies the page count AND the utilization
    target — not just "fits on one page however empty."

    Only page_fit is re-run; post-processing cannot change schema, source
    attribution, or field-lock properties (it only rewrites bullet text and
    drops bullets — never edits employer/date fields or section ids).
    """
    from harness.validators.page_fit import PageFitValidator

    refreshed_pf = PageFitValidator().run(resume)
    util = refreshed_pf.payload.get("page_utilization_pct", 0)
    updated = [refreshed_pf if v.name == "page_fit" else v for v in validator_results]
    metrics = _metrics(updated, jd_cov, voice, fab)
    targets_met = refreshed_pf.passed and util >= 95
    return metrics, targets_met


def _build_feedback(
    validators: list[ValidationResult],
    jd_cov: JDCoverageResult | None,
    fab: FabricationAuditResult | None = None,
) -> FeedbackMessage:
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
        if util < 95:
            fb.page_fit_underutilized = (
                f"Page is only {util}% full — add a 3rd or 4th experience role, "
                "or add more bullets to existing roles, to reach ≥95% page utilization."
            )
    if jd_cov is not None and not jd_cov.passed:
        fb.jd_coverage_score = round(jd_cov.score, 3)
        fb.jd_coverage_missing = list(jd_cov.uncovered_requirements[:15])
    if fab is not None and fab.flagged_bullets:
        fb.fabrication_flagged = [
            {"bullet_id": a.bullet_id, "score": round(a.support_score, 2), "reason": a.reason}
            for a in fab.flagged_bullets
        ]
    return fb


def _fix_and_trim_orphans(resume: Resume, model: ModelClient, input_resume: Resume, jd: str) -> Resume:
    """Post-processing: LLM fixes orphans + trims/drops overflow candidates, then mechanical fallback.

    1. Identify orphan bullets (danger zone) and overflow candidates (if page > 1)
       without dropping anything yet.
    2. One LLM call handles both: fix orphans (expand or trim) and trim/drop
       overflow candidates — the LLM sees the full content and decides.
    3. Mechanical trim fallback for any orphans the LLM missed.
    4. Mechanical drop fallback if the resume is still > 1 page after the LLM pass.
    """
    from harness.validators.page_fit import PageFitValidator

    # Step 1: assess page fit and identify candidates — no drops yet.
    pf = PageFitValidator().run(resume)
    util = pf.payload.get("page_utilization_pct", 100)
    log.info(
        "[post-proc/0:incoming] pages=%d util=%d%% skills=%d",
        pf.payload.get("pages", 1), util, len(resume.skills),
    )

    overflow_candidate_ids: list[str] = []
    if not pf.passed:
        # Pass the bottom-ranked bullets as overflow candidates for the LLM to
        # trim or drop. Use enough candidates to plausibly fix the overflow.
        overflow_candidate_ids = _rank_worst_bullets(resume, input_resume, jd, n=5)

    # Step 2: combined LLM pass — fix orphans and trim/drop overflow candidates.
    orphan_cats = detect_orphans_geometry(resume)
    if orphan_cats:
        orphans = list(orphan_cats.keys())
    else:
        # Geometry unavailable (render failed / no weasyprint) — fall back to char-based.
        orphans = detect_orphans(resume)

    # Pre-flight skip: at target (pages=1, util>=95) the heavy fix_bullets LLM
    # rewrite is a net negative — empirically observed to drop util 100% → 88%
    # by over-trimming for orphan-free layout. Run only the cheap mechanical
    # shrink for any 3-line orphans (cosmetic cleanup) and return.
    if pf.passed and util >= 95:
        log.info(
            "[post-proc/0:at-target-skip] pages=1 util=%d%% orphans=%d — skipping fix_bullets LLM",
            util, len(orphans),
        )
        if orphan_cats:
            for bid, cat in orphan_cats.items():
                if cat in (OrphanCategory.THREE_LINE_ORPHAN, OrphanCategory.THREE_LINE_FULL):
                    resume = _iterative_shrink_to_fit(resume, bid, max_lines=2, model=model)
        return resume

    if orphans or overflow_candidate_ids:
        log.info(
            "[post-proc/1:before_llm_fix] orphans=%d overflow_cands=%d util_at_call=%d%%",
            len(orphans), len(overflow_candidate_ids), util,
        )
        resume = fix_bullets(
            resume, orphans, overflow_candidate_ids, model,
            page_utilization_pct=util,
            orphan_categories=orphan_cats or None,
        )
        _pf_after_llm = PageFitValidator().run(resume)
        log.info(
            "[post-proc/2:after_llm_fix] pages=%d util=%d%% skills=%d (Δpages from incoming: %+d)",
            _pf_after_llm.payload.get("pages", 1),
            _pf_after_llm.payload.get("page_utilization_pct", 100),
            len(resume.skills),
            _pf_after_llm.payload.get("pages", 1) - pf.payload.get("pages", 1),
        )
        # Re-measure after the LLM pass; any bullet still 3+ lines gets mechanically shrunk.
        post_cats = detect_orphans_geometry(resume)
        for bid, cat in post_cats.items():
            if cat in (OrphanCategory.THREE_LINE_ORPHAN, OrphanCategory.THREE_LINE_FULL):
                resume = _iterative_shrink_to_fit(resume, bid, max_lines=2, model=model)
        if not post_cats:  # geometry path unavailable — char-based fallback
            still_bad = detect_orphans(resume)
            if still_bad:
                resume = _mechanical_trim_orphans(resume, still_bad, model=model)
        # Below the 95% target, preserve 1-bullet roles — dropping them would make the page emptier.
        min_floor = 1 if util < 95 else 2
        resume = _drop_empty_sections(resume, min_exp_bullets=min_floor)

    # Step 3: mechanical drop fallback — only if the LLM pass was not enough.
    pf2 = PageFitValidator().run(resume)
    log.info(
        "[post-proc/3:after_shrink_loop] pages=%d util=%d%% passed=%s",
        pf2.payload.get("pages", 1),
        pf2.payload.get("page_utilization_pct", 100),
        pf2.passed,
    )
    if not pf2.passed:
        log.info("[post-proc/4:invoking_mechanical_trim] pages_before=%d", pf2.payload.get("pages", 1))
        # max_iterations must exceed (starting_skills - _SKILLS_FLOOR) so the loop
        # can exhaust skill-popping AND reach the bullet-drop branch. With up to
        # ~16 starting skills and floor=8, 8 iterations are needed before a single
        # bullet drop is even possible; 30 leaves comfortable headroom for the
        # subsequent bullet-drop phase. See trial_03 in .omc/debug/2page_repro/.
        resume = _trim_for_page_fit(resume, input_resume, jd, [], max_iterations=30)
        _pf_after_trim = PageFitValidator().run(resume)
        log.info(
            "[post-proc/5:after_mechanical_trim] pages=%d util=%d%% passed=%s",
            _pf_after_trim.payload.get("pages", 1),
            _pf_after_trim.payload.get("page_utilization_pct", 100),
            _pf_after_trim.passed,
        )

    # Iterative under-utilization rescue — try up to MAX_RESCUE_ITERATIONS expansion
    # passes to bring util to the 95% target. Each iteration targets the n shortest
    # 1-liner bullets (different ones each iter, since the previous iter's targets
    # are now 2-liners and no longer qualify). Stops on: target met, overflow risk,
    # no expandable bullets left, or zero progress between iterations.
    MAX_RESCUE_ITERATIONS = 3
    pf3 = PageFitValidator().run(resume)
    final_util = pf3.payload.get("page_utilization_pct", 100)
    for iteration in range(MAX_RESCUE_ITERATIONS):
        if not pf3.passed or final_util >= 95:
            break
        expand_ids = _shortest_expandable_bullets(resume, n=3)
        if not expand_ids:
            log.info(
                "[post-proc/rescue iter=%d] skipped — no expandable 1-liner bullets (util=%d%%)",
                iteration, final_util,
            )
            break
        log.info(
            "[post-proc/rescue iter=%d] firing — util=%d%% expand_ids=%s",
            iteration, final_util, expand_ids,
        )
        rescued = fix_bullets(
            resume, [], [], model,
            page_utilization_pct=100,
            force_expand_ids=expand_ids,
        )
        rescued_pf = PageFitValidator().run(rescued)
        rescued_util = rescued_pf.payload.get("page_utilization_pct", final_util)
        rescued_pages = rescued_pf.payload.get("pages", 1)
        if not rescued_pf.passed:
            log.info(
                "[post-proc/rescue iter=%d] rejected — expansion would overflow "
                "(rescued pages=%d util=%d%%); keeping pre-rescue state at util=%d%%",
                iteration, rescued_pages, rescued_util, final_util,
            )
            break
        if rescued_util <= final_util:
            log.info(
                "[post-proc/rescue iter=%d] no progress — util %d%% → %d%%; stopping",
                iteration, final_util, rescued_util,
            )
            break
        log.info(
            "[post-proc/rescue iter=%d] accepted — util %d%% → %d%% (pages=%d)",
            iteration, final_util, rescued_util, rescued_pages,
        )
        resume = rescued
        pf3 = rescued_pf
        final_util = rescued_util

    exp_summary = [(e.employer, len(e.bullets)) for e in resume.experience]
    total_bullets = sum(b for _, b in exp_summary) + sum(len(p.bullets) for p in resume.projects)
    final_pages = pf3.payload.get("pages", 1)
    log.info(
        "[post-proc/6:final] pages=%d util=%d%% passed=%s exp=%s total_bullets=%d skills=%d",
        final_pages, final_util, pf3.passed, exp_summary, total_bullets, len(resume.skills),
    )
    if final_pages > 1:
        log.warning(
            "[post-proc/FINAL_OVERFLOW] resume still %d pages after all fallbacks — "
            "this is the 2-page-leak bug",
            final_pages,
        )
    return resume


def _shortest_expandable_bullets(resume: Resume, n: int) -> list[str]:
    """Return ids of the n shortest TYPE A candidate bullets (≤120 chars).

    These are clean 1-liners that have headroom to grow into a full 2-liner.
    """
    candidates = [b for b in iter_bullets(resume) if len(b.text) <= SINGLE_LINE_MAX]
    candidates.sort(key=lambda b: len(b.text))
    return [b.id for b in candidates[:n]]


def _iterative_shrink_to_fit(
    resume: Resume,
    bullet_id: str,
    *,
    max_lines: int = 2,
    max_pops: int = 30,
    measure: Callable[[Resume], dict] | None = None,
    model: ModelClient | None = None,
) -> Resume:
    """Pop trailing words from bullet_id, re-measuring each time, until line_count <= max_lines.

    measure — injected for tests; defaults to measure_bullet_geometry.
    """
    if measure is None:
        from harness.validators.page_fit import measure_bullet_geometry
        measure = measure_bullet_geometry

    # Capture original before any popping so the LLM has full context.
    original_text: str | None = None
    for key in ("experience", "education", "projects"):
        for section in getattr(resume, key):
            for b in section.bullets:
                if b.id == bullet_id:
                    original_text = b.text
    if original_text is None:
        return resume

    current = resume
    for _ in range(max_pops):
        geom = measure(current)
        g = geom.get(bullet_id)
        if g is None or g.line_count <= max_lines:
            break
        data = current.model_dump()
        popped = False
        for key in ("experience", "education", "projects"):
            for section in data[key]:
                for bullet in section["bullets"]:
                    if bullet["id"] != bullet_id:
                        continue
                    words = bullet["text"].split()
                    if len(words) <= 3:
                        truncated = " ".join(words)
                        if model is not None:
                            clean = llm_clean_truncated(original_text, truncated, 240, model)
                        else:
                            clean = None
                        bullet["text"] = clean if clean is not None else _ensure_clean_ending(truncated)
                        return Resume.model_validate(data)
                    words.pop()
                    bullet["text"] = " ".join(words)
                    popped = True
                    break
                if popped:
                    break
            if popped:
                break
        if not popped:
            break
        current = Resume.model_validate(data)

    # Popping loop exited — apply LLM cleanup on the truncated result.
    truncated_text: str | None = None
    for key in ("experience", "education", "projects"):
        for section in getattr(current, key):
            for b in section.bullets:
                if b.id == bullet_id:
                    truncated_text = b.text
    if truncated_text is not None and model is not None:
        clean = llm_clean_truncated(original_text, truncated_text, 240, model)
        if clean is not None:
            data = current.model_dump()
            for key in ("experience", "education", "projects"):
                for section in data[key]:
                    for bullet in section["bullets"]:
                        if bullet["id"] == bullet_id:
                            bullet["text"] = clean
            return Resume.model_validate(data)

    return _apply_clean_ending(current, bullet_id)


def _apply_clean_ending(resume: Resume, bullet_id: str) -> Resume:
    data = resume.model_dump()
    for key in ("experience", "education", "projects"):
        for section in data[key]:
            for bullet in section["bullets"]:
                if bullet["id"] == bullet_id:
                    bullet["text"] = _ensure_clean_ending(bullet["text"])
                    return Resume.model_validate(data)
    return resume


def _mechanical_trim_orphans(
    resume: Resume, orphan_ids: list[str], model: ModelClient | None = None
) -> Resume:
    """Last-resort: trim trailing words until bullet is ≤ SINGLE_LINE_MAX chars (clean single line)."""
    from harness.judges.orphan_fixer import SINGLE_LINE_MAX
    id_set = set(orphan_ids)
    data = resume.model_dump()
    original_texts = {
        b["id"]: b["text"]
        for key in ("experience", "education", "projects")
        for section in data[key]
        for b in section["bullets"]
        if b["id"] in id_set
    }
    for key in ("experience", "education", "projects"):
        for section in data[key]:
            for bullet in section["bullets"]:
                if bullet["id"] not in id_set:
                    continue
                words = bullet["text"].split()
                while len(" ".join(words)) > SINGLE_LINE_MAX and len(words) > 3:
                    words.pop()
                truncated = " ".join(words)
                if model is not None:
                    clean = llm_clean_truncated(
                        original_texts[bullet["id"]], truncated, SINGLE_LINE_MAX, model
                    )
                else:
                    clean = None
                bullet["text"] = clean if clean is not None else _ensure_clean_ending(truncated)
    return Resume.model_validate(data)


def _rank_worst_bullets(resume: Resume, input_resume: Resume, jd: str, n: int) -> list[str]:
    """Return ids of the n lowest-JD-ranked bullets in the resume."""
    input_scores = {bs.bullet_id: bs.score for bs in rank_bullets_by_jd(input_resume, jd)}

    def score(b) -> float:
        if not b.source_ids:
            return 0.0
        return max(input_scores.get(sid, 0.0) for sid in b.source_ids)

    scored = sorted(iter_bullets(resume), key=score)
    return [b.id for b in scored[:n]]


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


def _drop_empty_sections(resume: Resume, *, min_exp_bullets: int = 2) -> Resume:
    """Remove low-bullet experience/project entries.

    - Experience entries with fewer than `min_exp_bullets` bullets are dropped.
      Default is 2: a 2-line header for 1 bullet wastes space and violates the
      generation prompt. Pass `min_exp_bullets=1` from post-processing call sites
      that have already detected underutilization — at low util, preserving a
      1-bullet role contributes more content than dropping a whole header+bullet.
    - Project entries with 0 bullets are dropped (1 bullet is acceptable for projects).
    - Education entries are kept even without bullets (degree + GPA lines always show).
    """
    data = resume.model_dump()
    data["experience"] = [s for s in data["experience"] if len(s["bullets"]) >= min_exp_bullets]
    data["projects"] = [s for s in data["projects"] if s["bullets"]]
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
    iter_count = 0
    skills_popped = 0
    bullets_dropped = 0
    for _ in range(max_iterations):
        iter_count += 1
        # Re-validate page fit on the current candidate.
        pf_result = PageFitValidator().run(current)
        if pf_result.passed:
            log.info(
                "[trim_for_page_fit] CONVERGED at iter=%d skills_popped=%d bullets_dropped=%d",
                iter_count, skills_popped, bullets_dropped,
            )
            return current
        live_overflow = pf_result.payload.get("overflow_bullet_ids") or []

        # Bug #2: Skills overflow leaves overflow_bullet_ids empty (the Skills
        # section has no data-bullet-id). Pop the trailing (least-relevant) skill
        # before sacrificing real achievement bullets. Floor at 8 per the prompt's
        # "8–14 skills total" guidance.
        if not live_overflow and len(current.skills) > _SKILLS_FLOOR:
            current = _pop_last_skill(current)
            skills_popped += 1
            continue

        # Score every bullet in the current resume.
        all_scored: list[tuple[str, float]] = [
            (b.id, output_bullet_score(b.id, b.source_ids)) for b in iter_bullets(current)
        ]
        if not all_scored:
            return current

        # If specific bullets were flagged as overflowing, restrict to those.
        # Otherwise (skills exhausted, still overflowing), drop the lowest-ranked
        # bullet overall to free up space.
        if live_overflow:
            overflow_set = set(live_overflow)
            scored = [(bid, score) for bid, score in all_scored if bid in overflow_set]
            if not scored:
                scored = all_scored
        else:
            scored = all_scored

        scored.sort(key=lambda kv: kv[1])  # ascending — drop lowest first
        drop_id = scored[0][0]
        current = _prune_bullets(current, {drop_id})
        current = _drop_empty_sections(current)  # clean up roles emptied by pruning
        bullets_dropped += 1
    log.warning(
        "[trim_for_page_fit] EXHAUSTED %d iterations — skills_popped=%d bullets_dropped=%d "
        "remaining skills=%d (likely cause of 2-page output)",
        max_iterations, skills_popped, bullets_dropped, len(current.skills),
    )
    return current


_SKILLS_FLOOR = 8


def _pop_last_skill(resume: Resume) -> Resume:
    """Return a copy of resume with the last skill removed.

    Skills are ordered by JD relevance (most important first), so popping the
    tail drops the least important skill.
    """
    if not resume.skills:
        return resume
    data = resume.model_dump()
    data["skills"] = data["skills"][:-1]
    return Resume.model_validate(data)


def _metrics(validators, jd_cov, voice, fab=None) -> dict[str, Any]:
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
        "fabrication_pass_rate": round(fab.pass_rate, 3) if fab is not None else None,
        "fabrication_flagged_count": len(fab.flagged_bullets) if fab is not None else None,
    }
