"""AC-8 (model pluggability) + AC-10 (retry budget cap)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from harness.generate import TailoringResponse
from harness.judges.fabrication_audit import _AuditReport
from harness.models import EchoModelClient, ModelResponse
from harness.orchestrator import (
    _ensure_clean_ending,
    _finalize_metrics,
    _fix_and_trim_orphans,
    _iterative_shrink_to_fit,
    _mechanical_trim_orphans,
    _pop_last_skill,
    _shortest_expandable_bullets,
    _trim_for_page_fit,
    run,
)
from harness.schema import Resume, autopopulate_bullet_ids
from harness.validators import ValidationResult
from harness.validators.page_fit import BulletGeometry

_RESUME_PATH = Path(__file__).resolve().parent.parent / "evals/fixtures/resume/me.json"


def _load() -> Resume:
    raw = autopopulate_bullet_ids(json.loads(_RESUME_PATH.read_text()))
    return Resume.model_validate(raw)


def _good_tailored(inp: Resume) -> Resume:
    raw = inp.model_dump()
    for key in ("experience", "education", "projects"):
        for section in raw[key]:
            for i, b in enumerate(section["bullets"]):
                b["id"] = f"{section['id']}-out-{i}"
                b["source_ids"] = [f"{section['id']}-b{i}"]
    return Resume.model_validate(raw)


def _good_audit(out: Resume) -> _AuditReport:
    audits = []
    for key in ("experience", "education", "projects"):
        for section in out.model_dump()[key]:
            for b in section["bullets"]:
                audits.append({"bullet_id": b["id"], "support_score": 1.0, "reason": "echo"})
    return _AuditReport.model_validate({"audits": audits})


class _SequenceModel:
    """Returns the next response of the matching schema type.

    Dispatches by the requested `schema` so generate (TailoringResponse) and
    judge (_AuditReport) calls don't cross-pollute each other's queues.
    Resume queue entries are auto-wrapped in a TailoringResponse envelope.
    """
    def __init__(self, resume_queue, audit_queue):
        self._envelopes = [TailoringResponse(reasoning="stub", resume=r) for r in resume_queue]
        self._audits = list(audit_queue)
        self.calls = 0

    def generate_structured(self, *, system, prompt, schema):
        self.calls += 1
        if schema is _AuditReport:
            return ModelResponse(data=self._audits.pop(0), model_name="seq")
        return ModelResponse(data=self._envelopes.pop(0), model_name="seq")

    def generate_text(self, **kw):
        self.calls += 1
        return ModelResponse(data="", model_name="seq")


def test_orchestrator_runs_against_stub_no_concrete_model_imports():
    """AC-8: orchestrator depends only on the protocol, not on Gemini."""
    inp = _load()
    good = _good_tailored(inp)
    audit = _good_audit(good)
    # AC-8 verifies orchestrator runs against a non-Gemini client. We don't assert .passed
    # because the input resume is multi-page and the test isn't about content quality —
    # we assert the orchestrator completed, used the stub for both generate and judge,
    # and emitted a structured trajectory.
    model = _SequenceModel(resume_queue=[good] * 4, audit_queue=[audit] * 4)
    jd = "python jax pytorch numpy scipy pytest docker simulation"
    result = run(jd=jd, input_resume=inp, model=model, max_retries=2)
    assert result.trajectory, "orchestrator must emit a trajectory"
    assert model.calls > 0, "orchestrator must have used the stub model"
    # AC-8 invariant: orchestrator module does not import any concrete model class.
    # Check actual import statements only — comments/docstrings may mention model names.
    import re

    import harness.orchestrator as orch

    src = open(orch.__file__).read()
    concrete_imports = re.findall(
        r"^\s*(?:from|import)\s+harness\.models\.(\w+)",
        src,
        flags=re.MULTILINE,
    )
    forbidden = {"gemini_3_flash", "gemma_4", "groq_llama", "deepseek_v4"}
    leaked = [m for m in concrete_imports if m in forbidden]
    assert not leaked, f"orchestrator leaked concrete model imports: {leaked}"


def _flagged_audit(out: Resume, flagged_bullet_id: str) -> _AuditReport:
    """Audit report where one specific bullet is flagged (below THRESHOLD), rest are clean."""
    from harness.judges.fabrication_audit import THRESHOLD
    audits = []
    for key in ("experience", "education", "projects"):
        for section in out.model_dump()[key]:
            for b in section["bullets"]:
                if b["id"] == flagged_bullet_id:
                    audits.append({
                        "bullet_id": b["id"],
                        "support_score": THRESHOLD - 0.1,
                        "reason": "introduces 'InDesign', not in cited inputs",
                    })
                else:
                    audits.append({"bullet_id": b["id"], "support_score": 1.0, "reason": "echo"})
    return _AuditReport.model_validate({"audits": audits})


def test_fabrication_audit_triggers_retry_and_records_flag():
    """A flagged bullet on attempt 0 triggers a retry — the flag is recorded into trajectory + feedback.

    Uses a small single-role fixture so the only retry trigger is fabrication
    (mechanical + page-fit + jd-coverage all pass on a minimal resume).
    """
    inp = _resume_with_skills(["Python"], bullets_per_role=2)
    good = _good_tailored(inp)
    flagged_id = good.experience[0].bullets[0].id

    # Attempt 0 flagged → must RETRY. Provide enough queue depth so the retry can run.
    model = _SequenceModel(
        resume_queue=[good, good, good, good],
        audit_queue=[_flagged_audit(good, flagged_id), _good_audit(good),
                     _good_audit(good), _good_audit(good)],
    )
    result = run(jd="python", input_resume=inp, model=model, max_retries=3)

    # Attempt 0 must have recorded the flag.
    assert len(result.trajectory) >= 1
    rec0 = result.trajectory[0]
    assert rec0.fabrication_result is not None, "attempt 0 must record a fabrication_result"
    assert len(rec0.fabrication_result.flagged_bullets) == 1, (
        f"attempt 0 should have exactly 1 flagged bullet, got "
        f"{len(rec0.fabrication_result.flagged_bullets)}"
    )
    assert rec0.fabrication_result.flagged_bullets[0].bullet_id == flagged_id

    # The flag must propagate into the retry feedback so the LLM sees it on attempt 1.
    assert rec0.feedback.fabrication_flagged, (
        "flagged bullets must be rendered into FeedbackMessage.fabrication_flagged"
    )
    assert rec0.feedback.fabrication_flagged[0]["bullet_id"] == flagged_id

    # The retry must have actually fired (attempt 1 exists).
    assert len(result.trajectory) >= 2, "flagged attempt 0 must trigger at least one retry"


def _stale_validator_results(pages: int = 2, util: int = 100) -> list[ValidationResult]:
    """Build a validator_results list with a STALE page_fit entry (simulates pre-post-proc state)."""
    return [
        ValidationResult(name="schema_check", passed=True, score=1.0, errors=[], payload={}),
        ValidationResult(name="source_attribution", passed=True, score=1.0, errors=[], payload={}),
        ValidationResult(name="field_lock", passed=True, score=1.0, errors=[], payload={}),
        ValidationResult(
            name="page_fit", passed=pages == 1, score=1.0 if pages == 1 else 0.5,
            errors=[] if pages == 1 else ["overflow"],
            payload={
                "pages": pages, "overflow_bullet_ids": [],
                "font_substituted": False, "page_utilization_pct": util,
            },
        ),
    ]


def test_finalize_metrics_substitutes_post_proc_page_fit_into_metrics():
    """Stale validator_results said pages=2, post-proc trimmed to 1 → metrics show pages=1.

    Uses util=97 in the post-proc stub so the targets-met flag is True (both
    pages==1 AND util≥95 must hold for passed=True under the post-95% contract).
    The test's load-bearing assertion is the metrics substitution itself.
    """
    resume = _resume_with_skills(["Python"], bullets_per_role=2)
    stale = _stale_validator_results(pages=2, util=100)  # what the loop saw on its last attempt

    def post_proc_pf(self, output):
        # Post-processing trimmed the resume to 1 page at 97% utilization.
        return ValidationResult(
            name="page_fit", passed=True, score=1.0, errors=[],
            payload={"pages": 1, "overflow_bullet_ids": [],
                     "font_substituted": False, "page_utilization_pct": 97},
        )

    with patch("harness.validators.page_fit.PageFitValidator.run", post_proc_pf):
        metrics, targets_met = _finalize_metrics(resume, stale, None, None, None)

    assert targets_met is True, "post-proc landed at pages=1 util=97 → both targets met"
    assert metrics["page_fit_pages"] == 1, (
        f"metrics must reflect POST-proc state (pages=1), not stale loop state (pages=2); "
        f"got {metrics['page_fit_pages']}"
    )


def test_finalize_metrics_reports_passed_false_when_post_proc_lands_below_95pct_util():
    """Even when pages==1, passed must be False if final util<95.

    Locks the post-95% contract: the passed flag honors the page-utilization
    target, not just page count. Without this, a run could ship a 1-page
    resume at e.g. 80% util while reporting Result: PASSED — exactly the
    silent-underutilization regression observed on the AWS content-developer
    JD run that motivated this gate.
    """
    resume = _resume_with_skills(["Python"], bullets_per_role=2)
    stale = _stale_validator_results(pages=1, util=98)  # loop saw a clean state

    def post_proc_pf(self, output):
        # Post-processing kept pages=1 but trimmed bullets down to util=91 —
        # exactly the state seen on the AWS run that exposed the gap.
        return ValidationResult(
            name="page_fit", passed=True, score=1.0, errors=[],
            payload={"pages": 1, "overflow_bullet_ids": [],
                     "font_substituted": False, "page_utilization_pct": 91},
        )

    with patch("harness.validators.page_fit.PageFitValidator.run", post_proc_pf):
        metrics, targets_met = _finalize_metrics(resume, stale, None, None, None)

    assert targets_met is False, (
        "pages=1 but util=91 (< 95% target) must yield targets_met=False — "
        "no silent shipping of under-utilized resumes"
    )
    assert metrics["page_fit_pages"] == 1, "page count is still 1"


def test_finalize_metrics_reports_passed_false_when_post_proc_fails_to_fix_overflow():
    """Even on the success path (`if _will_exit`), if post-proc can't rescue overflow, passed=False."""
    resume = _resume_with_skills(["Python"], bullets_per_role=2)
    stale = _stale_validator_results(pages=2, util=100)  # loop exited optimistically on last attempt

    def post_proc_pf(self, output):
        # Post-processing failed to fully rescue — still 2 pages.
        return ValidationResult(
            name="page_fit", passed=False, score=0.5, errors=["still overflowing"],
            payload={"pages": 2, "overflow_bullet_ids": ["exp-0-out-0"],
                     "font_substituted": False, "page_utilization_pct": 100},
        )

    with patch("harness.validators.page_fit.PageFitValidator.run", post_proc_pf):
        metrics, pf_passed = _finalize_metrics(resume, stale, None, None, None)

    assert pf_passed is False, (
        "post-proc could not rescue overflow → passed must be False (no silent 2-page shipping)"
    )
    assert metrics["page_fit_pages"] == 2


def test_fabrication_audit_clean_audit_does_not_trigger_retry():
    """Sanity check: when fabrication audit is clean, it does not BY ITSELF force a retry.

    Uses max_retries=0 so the loop budget allows exactly one attempt; the
    page_underutilized gate fires on the small fixture (can't fill 95% of a US
    Letter page) but cannot trigger a retry because the budget is exhausted.
    Trajectory length must therefore equal 1, and the clean fabrication audit
    must not have caused additional iterations.
    """
    inp = _resume_with_skills(["Python"], bullets_per_role=2)
    good = _good_tailored(inp)
    model = _SequenceModel(
        resume_queue=[good],
        audit_queue=[_good_audit(good)],
    )
    result = run(jd="python", input_resume=inp, model=model, max_retries=0)
    assert len(result.trajectory) == 1, (
        f"clean audit + max_retries=0 should exit on attempt 0; "
        f"got {len(result.trajectory)} attempts"
    )
    # The trajectory's only attempt must record a non-None, non-flagged audit result.
    rec0 = result.trajectory[0]
    assert rec0.fabrication_result is not None
    assert not rec0.fabrication_result.flagged_bullets, "no bullets should be flagged"


def test_retry_budget_cap_with_always_failing_model():
    """AC-10: orchestrator stops after max_retries+1 attempts and returns a partial result."""
    inp = _load()
    bad = _good_tailored(inp)
    raw = bad.model_dump()
    raw["experience"][0]["bullets"][0]["source_ids"] = []  # always fails source-attribution
    bad = Resume.model_validate(raw)
    audit = _good_audit(bad)

    # 4 generate calls (initial + 3 retries); judge runs only on the final attempt
    # because mechanical fails on every prior attempt (see orchestrator.run policy).
    model = _SequenceModel(resume_queue=[bad] * 4, audit_queue=[audit])
    result = run(jd="python", input_resume=inp, model=model, max_retries=3)
    assert not result.passed
    assert len(result.trajectory) == 4  # initial + 3 retries


def test_orchestrator_runs_both_judges_per_attempt():
    """JDCoverageJudge and FabricationAudit both fire per attempt and their
    results both land in the RetryRecord.

    The orchestrator runs the two judges in parallel via ThreadPoolExecutor.
    This test locks the *contract* (both judges always run, both results are
    surfaced) — not the threading implementation, which can be swapped without
    breaking the test. If parallelism were ever to silently drop one judge,
    this test would fail.
    """
    inp = _resume_with_skills(["Python"], bullets_per_role=2)
    good = _good_tailored(inp)

    # Sentinel objects whose identity we can assert downstream. .passed and
    # .flagged_bullets are set so the orchestrator's _will_exit gate evaluates
    # them as a clean attempt, but the test isn't checking that — only that
    # each sentinel arrives in its correct RetryRecord field.
    sentinel_jd = MagicMock(name="jd_cov_sentinel")
    sentinel_jd.passed = True
    sentinel_fab = MagicMock(name="fab_sentinel")
    sentinel_fab.flagged_bullets = []

    model = _SequenceModel(resume_queue=[good], audit_queue=[])

    with patch("harness.judges.jd_coverage_judge.JDCoverageJudge.run", return_value=sentinel_jd), \
         patch("harness.judges.fabrication_audit.FabricationAudit.run", return_value=sentinel_fab):
        result = run(jd="python", input_resume=inp, model=model, max_retries=0)

    assert len(result.trajectory) == 1, f"expected 1 attempt, got {len(result.trajectory)}"
    rec0 = result.trajectory[0]
    assert rec0.jd_coverage_result is sentinel_jd, (
        "JDCoverageJudge result must be wired into RetryRecord.jd_coverage_result"
    )
    assert rec0.fabrication_result is sentinel_fab, (
        "FabricationAudit result must be wired into RetryRecord.fabrication_result"
    )


# ── Underutilization gate (no escape hatch) + best-of-N selection ────────────

def _pf_validation(*, pages: int, util: int) -> ValidationResult:
    """Build a page_fit ValidationResult with the given pages/util payload."""
    return ValidationResult(
        name="page_fit",
        passed=pages == 1,
        score=1.0 if pages == 1 else 0.5,
        errors=[] if pages == 1 else ["overflow"],
        payload={
            "pages": pages,
            "overflow_bullet_ids": [],
            "font_substituted": False,
            "page_utilization_pct": util,
        },
    )


def test_underutilized_final_attempt_no_longer_triggers_exit():
    """Underutilization (util<95) blocks _will_exit on EVERY attempt, including the final one.

    Before the escape-hatch removal, `page_underutilized` was masked off on the
    final attempt, so the orchestrator would EXIT optimistically at any util
    on the last try. Now the gate fires on every attempt; the loop must reach
    the fallthrough (passed=False) instead of short-circuiting.

    JDCoverageJudge is patched to None so it doesn't pop from the same resume
    queue _SequenceModel uses for generate(); _fix_and_trim_orphans is a
    passthrough so the test doesn't depend on post-proc LLM calls.
    """
    inp = _resume_with_skills(["Python"], bullets_per_role=2)
    good = _good_tailored(inp)

    # Force PageFitValidator to always report util=82 (below 95% target), 1 page.
    def fake_pf_run(self, output):
        return _pf_validation(pages=1, util=82)

    def fake_fix(resume_in, model, input_resume, jd):
        return resume_in

    model = _SequenceModel(
        resume_queue=[good] * 2,
        audit_queue=[_good_audit(good)] * 2,
    )
    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run), \
         patch("harness.judges.jd_coverage_judge.JDCoverageJudge.run", return_value=None), \
         patch("harness.orchestrator._fix_and_trim_orphans", side_effect=fake_fix):
        result = run(jd="python", input_resume=inp, model=model, max_retries=1)

    # max_retries=1 → 2 attempts; both underutilized → no early EXIT, both attempts run.
    assert len(result.trajectory) == 2, (
        f"underutilized final attempt must not EXIT; expected 2 attempts, "
        f"got {len(result.trajectory)}"
    )
    assert result.passed is False, (
        "fallthrough path returns passed=False — never silently accept util<95"
    )


def test_best_of_n_selects_higher_util_attempt_when_final_is_worse():
    """When no attempt achieves _will_exit, the fallthrough picks the best-scoring
    earlier attempt, not the last attempt.

    Setup: 4 underutilized attempts at util 89%/85%/80%/75%. The final attempt
    (75%) is worst; the best-of-N scoring must select attempt 0 (89%).
    """
    inp = _resume_with_skills(["Python"], bullets_per_role=2)
    good = _good_tailored(inp)

    # Tag each candidate's summary so we can identify which one was selected.
    def _tag(base: Resume, marker: str) -> Resume:
        raw = base.model_dump()
        raw["summary"] = marker
        return Resume.model_validate(raw)

    candidates = [_tag(good, f"ATTEMPT_{i}") for i in range(4)]

    # Per-call util sequence for the 4 generate-time validations; later calls
    # (post-processing, _finalize_metrics) fall through to a steady 75%.
    util_seq = iter([89, 85, 80, 75])
    def fake_pf_run(self, output):
        try:
            util = next(util_seq)
        except StopIteration:
            util = 75
        return _pf_validation(pages=1, util=util)

    # Passthrough post-proc so the selected candidate survives identifiable into final_resume.
    def fake_fix(resume_in, model, input_resume, jd):
        return resume_in

    model = _SequenceModel(
        resume_queue=candidates,
        audit_queue=[_good_audit(c) for c in candidates],
    )
    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run), \
         patch("harness.judges.jd_coverage_judge.JDCoverageJudge.run", return_value=None), \
         patch("harness.orchestrator._fix_and_trim_orphans", side_effect=fake_fix):
        result = run(jd="python", input_resume=inp, model=model, max_retries=3)

    assert result.final_resume.summary == "ATTEMPT_0", (
        f"best-of-N must select the highest-util attempt (ATTEMPT_0 @89%); "
        f"got summary={result.final_resume.summary!r}"
    )
    assert len(result.trajectory) == 4, "all 4 attempts must have run"
    assert result.passed is False, "all attempts underutilized → passed=False"
    # The trajectory must carry the per-attempt candidate so best-of-N can replay.
    assert all(r.candidate is not None for r in result.trajectory), (
        "every successful attempt must record its candidate in trajectory"
    )


# ── Bug #1, #2, #3 regression tests ──────────────────────────────────────────

def _resume_with_skills(skills: list[str], bullets_per_role: int = 3) -> Resume:
    bullets = [
        {"id": f"exp-0-b{i}", "text": f"Bullet {i} text.", "source_ids": ["src"]}
        for i in range(bullets_per_role)
    ]
    return Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL", "bullets": bullets,
        }],
        "education": [], "projects": [], "skills": list(skills),
    })


class _NoopModel:
    def generate_structured(self, **kw):
        from harness.judges.orphan_fixer import _FixResult
        return ModelResponse(data=_FixResult(rewrites=[], drop_ids=[]), model_name="noop")


# Bug #1: _drop_empty_sections must run after fix_bullets so 1-bullet
# experience entries (created by LLM drops) are removed when the page is
# already well-utilized. (At low util the rule flips — see the test below.)
def test_fix_bullets_followed_by_drop_empty_sections(monkeypatch):
    """_fix_and_trim_orphans must call _drop_empty_sections after fix_bullets.

    Setup: a resume with one 2-bullet role; fix_bullets drops one bullet,
    leaving a 1-bullet role. Without the post-drop _drop_empty_sections call,
    the 1-bullet role would survive. Incoming state is pages=2 (overflow,
    so the at-target pre-flight skip is bypassed) but util=96 (≥95, so the
    strict 2-bullet floor applies — no underutilization-relaxation).
    """
    resume = Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": [
                {"id": "b0", "text": "a" * 130, "source_ids": ["src"]},  # orphan
                {"id": "b1", "text": "Other bullet.", "source_ids": ["src"]},
            ],
        }],
        "education": [], "projects": [], "skills": ["Python"],
    })

    # PageFitValidator stub: pages=2 (so pre-flight at-target skip is bypassed),
    # util=96 (≥95, so the strict 2-bullet floor applies in _drop_empty_sections).
    def fake_pf_run(self, output):
        return ValidationResult(
            name="page_fit", passed=False, score=0.0, errors=["overflow"],
            payload={"pages": 2, "overflow_bullet_ids": ["b0"],
                     "font_substituted": False, "page_utilization_pct": 96},
        )

    # fix_bullets stub: drop b1 (LLM "decides" to drop the non-orphan).
    def fake_fix_bullets(resume_in, orphans, overflow, model, **kw):
        data = resume_in.model_dump()
        for sec in data["experience"]:
            sec["bullets"] = [b for b in sec["bullets"] if b["id"] != "b1"]
        return Resume.model_validate(data)

    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run), \
         patch("harness.orchestrator.fix_bullets", side_effect=fake_fix_bullets):
        result = _fix_and_trim_orphans(resume, _NoopModel(), resume, "jd")

    # The role had 2 bullets, fix_bullets dropped one, leaving 1.
    # _drop_empty_sections must have removed the now-1-bullet role entirely.
    assert len(result.experience) == 0, (
        "Bug #1: _drop_empty_sections must run after fix_bullets to remove "
        f"experience entries with <2 bullets, got {len(result.experience)}"
    )


def test_fix_and_trim_orphans_skips_llm_when_already_at_target():
    """Pre-flight skip: when pages=1 and util>=95, the heavy fix_bullets LLM
    rewrite must NOT fire — even if cosmetic orphans are detected.

    Empirical motivation: in a real bench run the LLM rewrite on an
    already-passing page (pages=1 util=100% with 12 orphans) dropped util
    100% → 91% → 88%. The whole-resume rewrite optimizes for orphan-free
    layout, which trades fill for tidiness — wrong tradeoff at target.
    Lighter mechanical-only cleanup is preferred at target.
    """
    from harness.judges.orphan_fixer import OrphanCategory

    resume = Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": [
                {"id": "b0", "text": "a" * 200, "source_ids": ["src"]},
                {"id": "b1", "text": "b" * 200, "source_ids": ["src"]},
            ],
        }],
        "education": [], "projects": [], "skills": ["Python"],
    })

    # pages=1, util=97 — comfortably at target.
    def fake_pf_run(self, output):
        return ValidationResult(
            name="page_fit", passed=True, score=1.0, errors=[],
            payload={"pages": 1, "overflow_bullet_ids": [],
                     "font_substituted": False, "page_utilization_pct": 97},
        )

    # Geometry reports orphans — the trigger that would normally fire fix_bullets.
    fake_orphan_cats = {"b0": OrphanCategory.TWO_LINE_ORPHAN}
    fix_mock = MagicMock(side_effect=lambda r, *a, **kw: r)

    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run), \
         patch("harness.orchestrator.detect_orphans_geometry", return_value=fake_orphan_cats), \
         patch("harness.orchestrator.fix_bullets", fix_mock):
        _fix_and_trim_orphans(resume, _NoopModel(), resume, "jd")

    fix_mock.assert_not_called()


def test_fix_bullets_preserves_1bullet_role_when_underutilized():
    """At util<95, a role reduced to 1 bullet by fix_bullets is PRESERVED.

    The strict 2-bullet floor (test_fix_bullets_followed_by_drop_empty_sections)
    is intentional at-or-above target — a 2-line header for 1 bullet wastes
    space. Below the 95% target, that same rule destroys content the page
    actually needs: dropping a 1-bullet role nukes a real experience entry to
    enforce a visual-density preference, leaving the page even emptier. Under
    underutilization, _fix_and_trim_orphans must relax the floor to 1 so the
    surviving bullet keeps contributing content.
    """
    resume = Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": [
                {"id": "b0", "text": "a" * 130, "source_ids": ["src"]},  # orphan
                {"id": "b1", "text": "Other bullet.", "source_ids": ["src"]},
            ],
        }],
        "education": [], "projects": [], "skills": ["Python"],
    })

    # PageFitValidator stub: util=80 (well below 95% target), 1 page.
    def fake_pf_run(self, output):
        return ValidationResult(
            name="page_fit", passed=True, score=1.0, errors=[],
            payload={"pages": 1, "overflow_bullet_ids": [],
                     "font_substituted": False, "page_utilization_pct": 80},
        )

    # fix_bullets stub: drop b1 (same behavior as the strict-floor test).
    def fake_fix_bullets(resume_in, orphans, overflow, model, **kw):
        data = resume_in.model_dump()
        for sec in data["experience"]:
            sec["bullets"] = [b for b in sec["bullets"] if b["id"] != "b1"]
        return Resume.model_validate(data)

    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run), \
         patch("harness.orchestrator.fix_bullets", side_effect=fake_fix_bullets):
        result = _fix_and_trim_orphans(resume, _NoopModel(), resume, "jd")

    # The role had 2 bullets, fix_bullets dropped one, leaving 1.
    # Because util=80 < 95, the 1-bullet role must be preserved, not nuked.
    assert len(result.experience) == 1, (
        "at util<95, _drop_empty_sections must relax to min_exp_bullets=1; "
        f"the 1-bullet role should be preserved, got {len(result.experience)} roles"
    )
    assert len(result.experience[0].bullets) == 1, (
        f"the surviving bullet (b0) should still be there, "
        f"got {len(result.experience[0].bullets)} bullets"
    )


def test_fix_bullets_preserves_1bullet_role_at_util_94_boundary():
    """Boundary lock: util=94 is still <95, so the relaxed floor still applies.

    Pairs with the util=80 and util=95 cases to pin the strict `<` semantics of
    the `util < 95` gate at line 411 of harness/orchestrator.py — guards against
    an accidental flip to `<=` (which would only relax below 94).
    """
    resume = Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": [
                {"id": "b0", "text": "a" * 130, "source_ids": ["src"]},
                {"id": "b1", "text": "Other bullet.", "source_ids": ["src"]},
            ],
        }],
        "education": [], "projects": [], "skills": ["Python"],
    })

    def fake_pf_run(self, output):
        return ValidationResult(
            name="page_fit", passed=True, score=1.0, errors=[],
            payload={"pages": 1, "overflow_bullet_ids": [],
                     "font_substituted": False, "page_utilization_pct": 94},
        )

    def fake_fix_bullets(resume_in, orphans, overflow, model, **kw):
        data = resume_in.model_dump()
        for sec in data["experience"]:
            sec["bullets"] = [b for b in sec["bullets"] if b["id"] != "b1"]
        return Resume.model_validate(data)

    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run), \
         patch("harness.orchestrator.fix_bullets", side_effect=fake_fix_bullets):
        result = _fix_and_trim_orphans(resume, _NoopModel(), resume, "jd")

    assert len(result.experience) == 1, (
        f"at util=94 (still <95), the 1-bullet role must be preserved; "
        f"got {len(result.experience)} roles — did the orchestrator flip the "
        f"comparison to <=95 by mistake?"
    )


def test_iterative_rescue_makes_multiple_passes_when_util_climbing():
    """When each rescue pass brings util closer to the 95% target, the rescue
    iterates up to MAX_RESCUE_ITERATIONS (3) — not just one shot.

    Locks the iterative-rescue refactor. The AWS-content-developer live run
    landed at util=91% because the previous single-shot rescue couldn't fully
    recover. With iteration, util can climb across multiple passes
    (e.g. 80 → 85 → 92 → 96, stopping at target met).

    Setup: PageFitValidator stub returns a climbing util sequence; fix_bullets
    is spied (not actually run) so the climb is purely from the stub. The
    contract under test is "how many rescue iterations fire," not what
    fix_bullets does internally.
    """
    resume = _resume_with_skills(["Python"], bullets_per_role=3)

    # Util sequence consumed across PageFitValidator.run() calls inside
    # _fix_and_trim_orphans (when no orphans / no overflow):
    #   idx 0: initial pf at function top
    #   idx 1: pf2 after the (skipped) orphan-fix block
    #   idx 2: pf3 just before the rescue loop
    #   idx 3-5: rescued_pf after each of the 3 iterations
    util_sequence = iter([80, 80, 80, 85, 92, 96])

    def fake_pf_run(self, output):
        try:
            util = next(util_sequence)
        except StopIteration:
            util = 96
        return ValidationResult(
            name="page_fit", passed=True, score=1.0, errors=[],
            payload={"pages": 1, "overflow_bullet_ids": [],
                     "font_substituted": False, "page_utilization_pct": util},
        )

    fix_bullets_calls: list = []

    def spy_fix_bullets(resume_in, orphans, overflow, model, **kw):
        fix_bullets_calls.append(kw.get("force_expand_ids"))
        return resume_in  # unchanged; util climb comes from the pf stub

    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run), \
         patch("harness.orchestrator.fix_bullets", side_effect=spy_fix_bullets):
        _fix_and_trim_orphans(resume, _NoopModel(), resume, "jd")

    # Exactly 3 rescue iterations: util climbs 80→85 (iter 0), 85→92 (iter 1),
    # 92→96 (iter 2). Loop's top-of-iter break-check `final_util >= 95` would
    # exit iter 3 if it existed, but the for-loop bound is MAX_RESCUE_ITERATIONS=3.
    assert len(fix_bullets_calls) == 3, (
        f"iterative rescue should fire 3 times (util climb 80→85→92→96 across "
        f"3 iters); got {len(fix_bullets_calls)} fix_bullets calls"
    )
    for i, expand_ids in enumerate(fix_bullets_calls):
        assert expand_ids, f"rescue iter={i}: force_expand_ids should be non-empty"


def test_iterative_rescue_stops_on_no_progress():
    """The rescue stops iterating when an iteration produces zero util gain.

    Even with budget remaining (MAX_RESCUE_ITERATIONS=3), if rescued_util ≤
    final_util, further iterations would just waste Gemini calls. Locks the
    no-progress early-exit.
    """
    resume = _resume_with_skills(["Python"], bullets_per_role=3)

    # Util climbs once (80 → 85), then stalls at 85. iter 0 accepts; iter 1
    # sees rescued_util=85 ≤ final_util=85 → breaks. Only 2 fix_bullets calls.
    util_sequence = iter([80, 80, 80, 85, 85])

    def fake_pf_run(self, output):
        try:
            util = next(util_sequence)
        except StopIteration:
            util = 85
        return ValidationResult(
            name="page_fit", passed=True, score=1.0, errors=[],
            payload={"pages": 1, "overflow_bullet_ids": [],
                     "font_substituted": False, "page_utilization_pct": util},
        )

    fix_bullets_calls: list = []

    def spy_fix_bullets(resume_in, orphans, overflow, model, **kw):
        fix_bullets_calls.append(kw.get("force_expand_ids"))
        return resume_in

    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run), \
         patch("harness.orchestrator.fix_bullets", side_effect=spy_fix_bullets):
        _fix_and_trim_orphans(resume, _NoopModel(), resume, "jd")

    assert len(fix_bullets_calls) == 2, (
        f"rescue should bail after the first no-progress iteration; "
        f"expected 2 fix_bullets calls (iter 0 progress 80→85, iter 1 stalls 85→85), "
        f"got {len(fix_bullets_calls)}"
    )


# Bug #2: skills overflow → pop trailing skills before dropping bullets.
def test_pop_last_skill_drops_least_important():
    resume = _resume_with_skills(["Python", "JAX", "Rust"])
    popped = _pop_last_skill(resume)
    assert popped.skills == ["Python", "JAX"]


def test_trim_for_page_fit_pops_skills_before_bullets():
    """When overflow_bullet_ids is empty (skills overflow), drop trailing skills first."""
    skills = [f"skill-{i}" for i in range(15)]  # 15 > floor of 8
    resume = _resume_with_skills(skills, bullets_per_role=3)

    call_count = {"n": 0}

    def fake_pf_run(self, output):
        # First 4 calls: still overflowing with empty overflow_ids.
        # 5th call: passes — verifies we kept popping skills, not bullets.
        call_count["n"] += 1
        passed = call_count["n"] >= 5
        return ValidationResult(
            name="page_fit", passed=passed, score=1.0 if passed else 0.5,
            errors=[] if passed else ["overflow"],
            payload={"pages": 1 if passed else 2, "overflow_bullet_ids": [],
                     "font_substituted": False, "page_utilization_pct": 100},
        )

    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run):
        result = _trim_for_page_fit(resume, resume, "jd", [], max_iterations=10)

    # Bullets must be untouched; only skills should have been trimmed.
    assert len(result.experience[0].bullets) == 3, "no bullets should have been dropped"
    assert len(result.skills) < 15, "skills should have been trimmed"


def test_trim_for_page_fit_skills_floor_at_8():
    """Skills trimming stops at 8; further overflow falls through to bullet drops."""
    skills = [f"skill-{i}" for i in range(10)]  # 10 → can pop 2 before hitting floor
    resume = _resume_with_skills(skills, bullets_per_role=3)

    def fake_pf_run(self, output):
        # Always overflowing with empty overflow_ids — forces fall-through
        # to bullet dropping after skills hit the floor.
        return ValidationResult(
            name="page_fit", passed=False, score=0.5, errors=["overflow"],
            payload={"pages": 2, "overflow_bullet_ids": [],
                     "font_substituted": False, "page_utilization_pct": 100},
        )

    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run):
        result = _trim_for_page_fit(resume, resume, "jd", [], max_iterations=5)

    # Skills floor at 8, so at most 2 pops, then iterations spent dropping bullets.
    assert len(result.skills) == 8, f"expected skills floor at 8, got {len(result.skills)}"


def test_trim_for_page_fit_iteration_cap_must_cover_skill_floor_plus_bullet_drops():
    """Regression for the 2-page-leak observed in trial_03 of the 10-trial repro.

    When the overflowing content lives in a section without data-bullet-id (e.g.
    Skills), PageFitValidator returns overflow_bullet_ids=[] and the trim loop
    falls into the skill-pop branch — one skill per iteration. With realistic
    inputs (the project's me.json fixture seeds 14–16 skills) and floor=8, the
    loop needs ≥8 iterations just to exhaust skill-popping before it can ever
    reach the bullet-drop fallback.

    max_iterations=5 (the value the call site used to pass) is insufficient: the
    loop pops 5 skills, hits its cap, and returns a still-overflowing resume —
    the 2-page leak. max_iterations=30 (the post-fix value) gives the loop
    enough budget to pop 8 skills to the floor and then drop bullets.
    """
    def fixture():
        return _resume_with_skills([f"skill-{i}" for i in range(16)], bullets_per_role=3)

    def fake_pf_run(self, output):
        # Permanent overflow with empty overflow_ids — simulates skills (or
        # any non-bullet-tagged content) spilling onto page 2.
        return ValidationResult(
            name="page_fit", passed=False, score=0.5, errors=["overflow"],
            payload={"pages": 2, "overflow_bullet_ids": [],
                     "font_substituted": False, "page_utilization_pct": 100},
        )

    # --- Pre-fix repro: max_iterations=5 cannot reach the bullet-drop branch.
    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run):
        buggy = _trim_for_page_fit(fixture(), fixture(), "jd", [], max_iterations=5)
    assert len(buggy.skills) == 11, (
        f"trial_03 repro: max_iterations=5 should pop only 5 skills (16→11); "
        f"got {len(buggy.skills)}"
    )
    assert len(buggy.experience[0].bullets) == 3, (
        "trial_03 repro: with max_iterations=5 the loop must never reach "
        "the bullet-drop branch (this is the bug)"
    )

    # --- Post-fix: max_iterations=30 (the call-site value) reaches bullet drops.
    with patch("harness.validators.page_fit.PageFitValidator.run", fake_pf_run):
        fixed = _trim_for_page_fit(fixture(), fixture(), "jd", [], max_iterations=30)
    assert len(fixed.skills) == 8, (
        f"fix: max_iterations=30 should let skills hit floor=8; got {len(fixed.skills)}"
    )
    # Bullet drops + _drop_empty_sections: dropping 2 of 3 bullets leaves 1,
    # which is below the 2-bullet minimum, so the whole role is removed.
    assert len(fixed.experience) == 0, (
        "fix: max_iterations=30 must reach the bullet-drop branch — the "
        "1-role fixture loses its role once bullets fall below the 2-minimum"
    )


# Bug #3: shortest-bullet rescue when post-processing shrinks the page.
def test_shortest_expandable_bullets_returns_n_shortest():
    resume = Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": [
                {"id": "b0", "text": "a" * 50, "source_ids": ["src"]},
                {"id": "b1", "text": "a" * 80, "source_ids": ["src"]},
                {"id": "b2", "text": "a" * 30, "source_ids": ["src"]},
                {"id": "b3", "text": "a" * 200, "source_ids": ["src"]},  # excluded (>120)
            ],
        }],
        "education": [], "projects": [], "skills": ["Python"],
    })
    ids = _shortest_expandable_bullets(resume, n=2)
    assert ids == ["b2", "b0"], f"expected [b2, b0] (shortest two ≤120), got {ids}"


# ── _iterative_shrink_to_fit ─────────────────────────────────────────────────

def test_iterative_shrink_to_fit_pops_until_two_lines():
    """Shrink keeps popping trailing words until injected geometry says line_count <= max_lines."""
    resume = Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": [
                {"id": "b0",
                 "text": "one two three four five six seven eight nine ten eleven twelve",
                 "source_ids": ["src"]},
            ],
        }],
        "education": [], "projects": [], "skills": ["Python"],
    })

    # Stub: report 3 lines until the bullet has <= 8 words, then 2 lines.
    def measure(r):
        for sec in r.experience:
            for b in sec.bullets:
                if b.id == "b0":
                    n = len(b.text.split())
                    return {"b0": BulletGeometry(
                        line_count=3 if n > 8 else 2,
                        last_line_ratio=0.5, text_length=len(b.text),
                    )}
        return {}

    result = _iterative_shrink_to_fit(resume, "b0", max_lines=2, measure=measure)
    final_words = result.experience[0].bullets[0].text.split()
    assert len(final_words) == 8, f"expected 8 words after popping, got {len(final_words)}"


def test_iterative_shrink_to_fit_noop_when_already_fits():
    resume = Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": [{"id": "b0", "text": "short bullet text here", "source_ids": ["src"]}],
        }],
        "education": [], "projects": [], "skills": ["Python"],
    })

    def measure(r):
        return {"b0": BulletGeometry(line_count=1, last_line_ratio=0.5, text_length=22)}

    result = _iterative_shrink_to_fit(resume, "b0", max_lines=2, measure=measure)
    assert result.experience[0].bullets[0].text == "short bullet text here."


def test_iterative_shrink_to_fit_stops_at_three_word_floor():
    """Don't gut a bullet below 3 words even if geometry still says >max_lines."""
    resume = Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": [{"id": "b0", "text": "one two three four", "source_ids": ["src"]}],
        }],
        "education": [], "projects": [], "skills": ["Python"],
    })

    def measure(r):  # never satisfied
        return {"b0": BulletGeometry(line_count=5, last_line_ratio=0.1, text_length=20)}

    result = _iterative_shrink_to_fit(resume, "b0", max_lines=2, measure=measure)
    # 4 words → pop one to 3 → next iteration sees len(words)=3, returns early.
    assert result.experience[0].bullets[0].text == "one two three."


# ── _ensure_clean_ending ─────────────────────────────────────────────────────

def test_ensure_clean_ending_strips_dangling_connective_pair():
    assert _ensure_clean_ending("partners and team") == "partners."


def test_ensure_clean_ending_strips_across_digital():
    assert _ensure_clean_ending("engagement across digital") == "engagement."


def test_ensure_clean_ending_preserves_complete_sentence():
    assert _ensure_clean_ending("Built ML pipeline cutting latency 40%.") == "Built ML pipeline cutting latency 40%."


def test_ensure_clean_ending_adds_period_to_complete_phrase():
    assert _ensure_clean_ending("Reduced API latency 40 percent") == "Reduced API latency 40 percent."


def test_ensure_clean_ending_strips_trailing_comma():
    assert _ensure_clean_ending("Managed 60+ clients,") == "Managed 60+ clients."


def test_ensure_clean_ending_keeps_existing_terminator():
    assert _ensure_clean_ending("Question?") == "Question?"


def test_ensure_clean_ending_handles_solo_connective():
    assert _ensure_clean_ending("partners and") == "partners."


def test_ensure_clean_ending_empty_after_stripping_returns_original():
    original = "and or but"
    assert _ensure_clean_ending(original) == original


def test_iterative_shrink_to_fit_cleans_ending_after_pops():
    """After popping words, the final bullet text must pass through _ensure_clean_ending."""
    resume = Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": [
                {"id": "b0",
                 "text": "Managed deliverables for partners and team members ensuring communication",
                 "source_ids": ["src"]},
            ],
        }],
        "education": [], "projects": [], "skills": ["Python"],
    })

    # Stub: report 3 lines until <= 6 words (simulates trimming off "members ensuring communication",
    # leaving "Managed deliverables for partners and team" — a dangling connective pair).
    def measure(r):
        for sec in r.experience:
            for b in sec.bullets:
                if b.id == "b0":
                    n = len(b.text.split())
                    return {"b0": BulletGeometry(
                        line_count=3 if n > 6 else 2,
                        last_line_ratio=0.5, text_length=len(b.text),
                    )}
        return {}

    result = _iterative_shrink_to_fit(resume, "b0", max_lines=2, measure=measure)
    final_text = result.experience[0].bullets[0].text
    # "and team" is a dangling connective pair — must be stripped.
    assert not final_text.rstrip(".").endswith(" and"), f"dangling 'and' survived: {final_text!r}"
    assert not final_text.rstrip(".").endswith(" and team"), f"dangling 'and team' survived: {final_text!r}"
    assert final_text.endswith("."), f"bullet must end with period: {final_text!r}"


# ── LLM cleanup wiring ────────────────────────────────────────────────────────

def _make_shrink_resume(text: str) -> Resume:
    return Resume.model_validate({
        "contact": {"name": "T", "email": "t@t.com", "phone": "555-0000",
                    "location": "Chicago, IL", "links": []},
        "summary": "S.",
        "experience": [{
            "id": "exp-0", "title": "Engineer", "employer": "Acme",
            "start_date": "Jan 2020", "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": [{"id": "b0", "text": text, "source_ids": ["src"]}],
        }],
        "education": [], "projects": [], "skills": ["Python"],
    })


class _StubCleanModel:
    """Returns the given clean text from llm_clean_truncated (via _CleanRewrite schema)."""
    def __init__(self, clean_text: str):
        self._clean_text = clean_text

    def generate_structured(self, *, system, prompt, schema):
        return ModelResponse(data=schema.model_validate({"text": self._clean_text}), model_name="stub")


class _ErrorCleanModel:
    def generate_structured(self, **kw):
        raise RuntimeError("stub error")


def test_iterative_shrink_uses_llm_for_cleanup():
    """LLM cleanup is called after word-popping; result replaces truncated text."""
    long_text = "one two three four five six seven eight nine ten eleven twelve"
    clean_rewrite = "one two three four five six seven eight."
    resume = _make_shrink_resume(long_text)
    model = _StubCleanModel(clean_rewrite)

    def measure(r):
        for sec in r.experience:
            for b in sec.bullets:
                if b.id == "b0":
                    n = len(b.text.split())
                    return {"b0": BulletGeometry(
                        line_count=3 if n > 8 else 2,
                        last_line_ratio=0.5, text_length=len(b.text),
                    )}
        return {}

    result = _iterative_shrink_to_fit(resume, "b0", max_lines=2, measure=measure, model=model)
    assert result.experience[0].bullets[0].text == clean_rewrite


def test_iterative_shrink_falls_back_to_mechanical_strip_on_llm_error():
    """When the LLM raises, _ensure_clean_ending is used as a safety net."""
    long_text = "one two three four five six seven eight nine ten eleven twelve"
    resume = _make_shrink_resume(long_text)
    model = _ErrorCleanModel()

    def measure(r):
        for sec in r.experience:
            for b in sec.bullets:
                if b.id == "b0":
                    n = len(b.text.split())
                    return {"b0": BulletGeometry(
                        line_count=3 if n > 8 else 2,
                        last_line_ratio=0.5, text_length=len(b.text),
                    )}
        return {}

    result = _iterative_shrink_to_fit(resume, "b0", max_lines=2, measure=measure, model=model)
    final_text = result.experience[0].bullets[0].text
    # LLM failed — mechanical ending must still produce a period-terminated bullet.
    assert final_text.endswith("."), f"expected period termination, got: {final_text!r}"
    assert len(final_text.split()) == 8, f"expected 8 words after popping, got: {final_text!r}"


def test_mechanical_trim_uses_llm_for_cleanup():
    """llm_clean_truncated is called in _mechanical_trim_orphans; result replaces word-popped text."""
    danger_text = "a" * 130  # in orphan danger zone
    clean_rewrite = "Shortened cleanly by LLM."
    resume = _make_shrink_resume(danger_text)
    model = _StubCleanModel(clean_rewrite)

    result = _mechanical_trim_orphans(resume, ["b0"], model=model)
    assert result.experience[0].bullets[0].text == clean_rewrite
