"""AC-8 (model pluggability) + AC-10 (retry budget cap)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.generate import TailoringResponse
from harness.judges.fabrication_audit import _AuditReport
from harness.models import EchoModelClient, ModelResponse
from harness.orchestrator import run
from harness.schema import Resume, autopopulate_bullet_ids

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
