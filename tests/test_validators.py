"""Validator unit tests — AC-3 (source_attribution), AC-4 (field_lock), AC-6 (jd_coverage).
PageFit (AC-2) is exercised in test_page_fit.py because it requires WeasyPrint native libs.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from harness.schema import Resume, autopopulate_bullet_ids
from harness.validators.field_lock import FieldLockValidator
from harness.validators.source_attribution import SourceAttributionValidator
from harness.validators.schema_check import SchemaCheckValidator

_RESUME_PATH = Path(__file__).resolve().parent.parent / "evals/fixtures/resume/me.json"


def _load_input() -> Resume:
    raw = autopopulate_bullet_ids(json.loads(_RESUME_PATH.read_text()))
    return Resume.model_validate(raw)


def _make_tailored(input_resume: Resume) -> Resume:
    raw = input_resume.model_dump()
    for key in ("experience", "education", "projects"):
        for section in raw[key]:
            for i, b in enumerate(section["bullets"]):
                b["id"] = f"{section['id']}-out-{i}"
                b["source_ids"] = [f"{section['id']}-b{i}"]
    return Resume.model_validate(raw)


def test_schema_check_passes_on_valid_resume():
    r = _load_input()
    assert SchemaCheckValidator().run(r).passed


def test_source_attribution_pass():
    inp = _load_input()
    out = _make_tailored(inp)
    res = SourceAttributionValidator().run(output=out, input_resume=inp)
    assert res.passed
    assert res.score == 1.0


def test_source_attribution_catches_empty_source_ids():
    inp = _load_input()
    out = _make_tailored(inp)
    raw = out.model_dump()
    raw["experience"][0]["bullets"][0]["source_ids"] = []
    out_bad = Resume.model_validate(raw)
    res = SourceAttributionValidator().run(output=out_bad, input_resume=inp)
    assert not res.passed
    assert "empty source_ids" in res.errors[0]


def test_source_attribution_catches_unknown_id():
    inp = _load_input()
    out = _make_tailored(inp)
    raw = out.model_dump()
    raw["experience"][0]["bullets"][0]["source_ids"] = ["fabricated-id"]
    out_bad = Resume.model_validate(raw)
    res = SourceAttributionValidator().run(output=out_bad, input_resume=inp)
    assert not res.passed
    assert "unknown source_ids" in res.errors[0]


def test_field_lock_pass_when_unchanged():
    inp = _load_input()
    out = _make_tailored(inp)
    assert FieldLockValidator().run(output=out, input_resume=inp).passed


def test_field_lock_catches_employer_drift():
    inp = _load_input()
    out = _make_tailored(inp)
    raw = out.model_dump()
    raw["experience"][0]["employer"] = "Definitely-Different Lab"
    out_bad = Resume.model_validate(raw)
    res = FieldLockValidator().run(output=out_bad, input_resume=inp)
    assert not res.passed
    assert any("employer" in e for e in res.errors)
