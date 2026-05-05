"""AC-2 PageFit: requires WeasyPrint + system libs. Skipped if libs unavailable."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_RESUME_PATH = Path(__file__).resolve().parent.parent / "evals/fixtures/resume/me.json"

try:
    from harness.validators.page_fit import PageFitValidator, measure_bullet_geometry
    from harness.schema import Resume, autopopulate_bullet_ids
    _AVAILABLE = True
except OSError:  # pragma: no cover
    _AVAILABLE = False


pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="weasyprint native libs unavailable")


def _load() -> "Resume":
    raw = autopopulate_bullet_ids(json.loads(_RESUME_PATH.read_text()))
    return Resume.model_validate(raw)


def test_baseline_resume_fits_one_page():
    r = _load()
    res = PageFitValidator().run(r)
    assert res.payload["pages"] == 1, f"pages={res.payload['pages']}"
    assert res.payload["overflow_bullet_ids"] == []


def test_overflow_emits_bullet_ids():
    r = _load()
    raw = r.model_dump()
    # Inflate bullets to force overflow
    raw["experience"][0]["bullets"] = [
        {"id": f"exp-0-stuffed-{i}", "text": "x" * 400, "source_ids": ["exp-0-b0"]}
        for i in range(40)
    ]
    big = Resume.model_validate(raw)
    res = PageFitValidator().run(big)
    assert res.payload["pages"] > 1, "expected overflow but got 1 page"
    # Overflow ids are whatever bullets land on page 2+ (the inflated exp-0 ones eat page 1,
    # so the *later* bullets get pushed off — either pattern is valid evidence of overflow).
    assert res.payload["overflow_bullet_ids"], (
        f"pages={res.payload['pages']} but overflow_bullet_ids was empty"
    )


def test_measure_bullet_geometry_returns_per_bullet_geometry():
    """measure_bullet_geometry returns line_count + last_line_ratio for every bullet."""
    r = _load()
    geom = measure_bullet_geometry(r)
    # At least one bullet must be measured.
    assert geom, "expected non-empty geometry for the baseline resume"
    # Every measured bullet has plausible values.
    for bid, g in geom.items():
        assert g.line_count >= 1, f"{bid}: line_count={g.line_count}"
        assert 0.0 <= g.last_line_ratio <= 1.0, f"{bid}: ratio={g.last_line_ratio}"
        assert g.text_length > 0, f"{bid}: text_length={g.text_length}"
