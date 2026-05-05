"""Unit tests for the combined orphan/overflow bullet fixer."""
from __future__ import annotations

import pytest

from harness.judges.orphan_fixer import (
    DANGER_MAX,
    DANGER_MIN,
    SINGLE_LINE_MAX,
    OrphanCategory,
    classify_bullets,
    detect_orphans,
    fix_bullets,
    llm_clean_truncated,
)
from harness.models import ModelResponse
from harness.schema import Resume
from harness.validators.page_fit import BulletGeometry


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_resume(bullets: list[dict]) -> Resume:
    return Resume.model_validate({
        "contact": {
            "name": "Test User", "email": "t@t.com",
            "phone": "555-0000", "location": "Chicago, IL", "links": [],
        },
        "summary": "Summary.",
        "experience": [{
            "id": "exp-0",
            "title": "Engineer",
            "employer": "Acme",
            "start_date": "Jan 2020",
            "end_date": "Jan 2023",
            "location": "Chicago, IL",
            "bullets": bullets,
        }],
        "education": [],
        "projects": [],
        "skills": ["Python"],
    })


class _StubModel:
    def __init__(self, result: dict):
        self._result = result

    def generate_structured(self, *, system, prompt, schema):
        return ModelResponse(data=schema.model_validate(self._result), model_name="stub")


class _ErrorModel:
    def generate_structured(self, **kw):
        raise RuntimeError("stub error")


# ── detect_orphans ────────────────────────────────────────────────────────────

def test_detect_orphans_identifies_danger_zone():
    danger = "a" * (DANGER_MIN + 5)       # 126 chars — in zone
    clean_short = "a" * SINGLE_LINE_MAX   # 120 chars — just outside
    clean_long = "a" * (DANGER_MAX + 50)  # 215 chars — full 2-liner
    resume = _make_resume([
        {"id": "b0", "text": danger, "source_ids": ["src"]},
        {"id": "b1", "text": clean_short, "source_ids": ["src"]},
        {"id": "b2", "text": clean_long, "source_ids": ["src"]},
    ])
    assert detect_orphans(resume) == ["b0"]


def test_detect_orphans_boundary_inclusive():
    at_min = "a" * DANGER_MIN   # exactly 121
    at_max = "a" * DANGER_MAX   # exactly 165
    resume = _make_resume([
        {"id": "b0", "text": at_min, "source_ids": ["src"]},
        {"id": "b1", "text": at_max, "source_ids": ["src"]},
    ])
    assert set(detect_orphans(resume)) == {"b0", "b1"}


# ── fix_bullets — rewrites ────────────────────────────────────────────────────

def test_fix_bullets_applies_orphan_rewrite():
    orphan_text = "a" * 130
    new_text = "b" * 100  # ≤120 — clean 1-liner
    resume = _make_resume([{"id": "b0", "text": orphan_text, "source_ids": ["src"]}])
    model = _StubModel({"rewrites": [{"id": "b0", "text": new_text}], "drop_ids": []})

    result = fix_bullets(resume, orphan_ids=["b0"], overflow_candidate_ids=[], model=model)

    assert result.experience[0].bullets[0].text == new_text


def test_fix_bullets_non_rewritten_bullets_unchanged():
    resume = _make_resume([
        {"id": "b0", "text": "a" * 130, "source_ids": ["src"]},
        {"id": "b1", "text": "untouched bullet text", "source_ids": ["src"]},
    ])
    model = _StubModel({"rewrites": [{"id": "b0", "text": "b" * 100}], "drop_ids": []})

    result = fix_bullets(resume, orphan_ids=["b0"], overflow_candidate_ids=[], model=model)

    texts = {b.id: b.text for b in result.experience[0].bullets}
    assert texts["b1"] == "untouched bullet text"


# ── fix_bullets — drops ───────────────────────────────────────────────────────

def test_fix_bullets_drop_ids_removes_overflow_candidates():
    resume = _make_resume([
        {"id": "b0", "text": "Keep this one around for sure.", "source_ids": ["src"]},
        {"id": "b1", "text": "Low-value overflow candidate.", "source_ids": ["src"]},
    ])
    model = _StubModel({
        "rewrites": [{"id": "b0", "text": "Keep this one around for sure."}],
        "drop_ids": ["b1"],
    })

    result = fix_bullets(resume, orphan_ids=[], overflow_candidate_ids=["b1"], model=model)

    ids = [b.id for b in result.experience[0].bullets]
    assert ids == ["b0"]
    assert "b1" not in ids


def test_fix_bullets_does_not_drop_orphan_ids():
    """TYPE A orphan bullets in drop_ids are ignored — they must never be dropped."""
    orphan_text = "a" * 130
    resume = _make_resume([{"id": "b0", "text": orphan_text, "source_ids": ["src"]}])
    # LLM misbehaves and puts the orphan in drop_ids instead of rewrites.
    model = _StubModel({"rewrites": [], "drop_ids": ["b0"]})

    result = fix_bullets(resume, orphan_ids=["b0"], overflow_candidate_ids=[], model=model)

    # Orphan bullet is protected — it stays in the resume with its original text.
    assert len(result.experience[0].bullets) == 1
    assert result.experience[0].bullets[0].text == orphan_text


# ── fix_bullets — overflow deduplication ─────────────────────────────────────

def test_fix_bullets_overflow_candidate_already_in_orphans_not_duplicated():
    """A bullet id in both lists must not appear twice in the LLM prompt."""
    orphan_text = "a" * 130
    new_text = "c" * 100
    resume = _make_resume([{"id": "b0", "text": orphan_text, "source_ids": ["src"]}])
    # b0 is in both orphan and overflow — fix_bullets should deduplicate
    model = _StubModel({"rewrites": [{"id": "b0", "text": new_text}], "drop_ids": []})

    result = fix_bullets(
        resume, orphan_ids=["b0"], overflow_candidate_ids=["b0"], model=model,
    )
    assert result.experience[0].bullets[0].text == new_text


# ── fix_bullets — error handling ──────────────────────────────────────────────

def test_fix_bullets_returns_original_on_model_error():
    original_text = "a" * 130
    resume = _make_resume([{"id": "b0", "text": original_text, "source_ids": ["src"]}])

    result = fix_bullets(resume, orphan_ids=["b0"], overflow_candidate_ids=[], model=_ErrorModel())

    assert result.experience[0].bullets[0].text == original_text


def test_fix_bullets_no_op_when_both_lists_empty():
    resume = _make_resume([{"id": "b0", "text": "Normal bullet.", "source_ids": ["src"]}])
    model = _ErrorModel()  # would raise if called — proves we skip the LLM call

    result = fix_bullets(resume, orphan_ids=[], overflow_candidate_ids=[], model=model)

    assert result.experience[0].bullets[0].text == "Normal bullet."


# ── fix_bullets — force_expand_ids (Bug #3 rescue path) ──────────────────────

def test_fix_bullets_force_expand_treats_clean_bullet_as_orphan():
    """force_expand_ids lets the rescue path expand bullets outside the danger zone."""
    short_text = "a" * 50  # well under SINGLE_LINE_MAX, not a natural orphan
    long_text = "b" * 220  # ≥200 — full 2-liner expansion
    resume = _make_resume([{"id": "b0", "text": short_text, "source_ids": ["src"]}])
    model = _StubModel({"rewrites": [{"id": "b0", "text": long_text}], "drop_ids": []})

    result = fix_bullets(
        resume, orphan_ids=[], overflow_candidate_ids=[], model=model,
        force_expand_ids=["b0"],
    )
    assert result.experience[0].bullets[0].text == long_text


def test_fix_bullets_force_expand_protects_from_drop():
    """force_expand bullets share orphan-protection: cannot be dropped."""
    short_text = "a" * 50
    resume = _make_resume([{"id": "b0", "text": short_text, "source_ids": ["src"]}])
    # Misbehaving model puts force_expand bullet in drop_ids — must be ignored.
    model = _StubModel({"rewrites": [], "drop_ids": ["b0"]})

    result = fix_bullets(
        resume, orphan_ids=[], overflow_candidate_ids=[], model=model,
        force_expand_ids=["b0"],
    )
    assert len(result.experience[0].bullets) == 1
    assert result.experience[0].bullets[0].text == short_text


# ── classify_bullets — geometry-based classification ─────────────────────────

def test_classify_bullets_one_line_is_clean():
    geom = {"b0": BulletGeometry(line_count=1, last_line_ratio=0.5, text_length=100)}
    assert classify_bullets(geom) == {"b0": OrphanCategory.CLEAN}


def test_classify_bullets_two_line_orphan():
    geom = {"b0": BulletGeometry(line_count=2, last_line_ratio=0.15, text_length=140)}
    assert classify_bullets(geom) == {"b0": OrphanCategory.TWO_LINE_ORPHAN}


def test_classify_bullets_three_line_orphan():
    geom = {"b0": BulletGeometry(line_count=3, last_line_ratio=0.10, text_length=270)}
    assert classify_bullets(geom) == {"b0": OrphanCategory.THREE_LINE_ORPHAN}


def test_classify_bullets_three_line_full():
    geom = {"b0": BulletGeometry(line_count=3, last_line_ratio=0.80, text_length=350)}
    assert classify_bullets(geom) == {"b0": OrphanCategory.THREE_LINE_FULL}


def test_classify_bullets_two_line_full_is_clean():
    geom = {"b0": BulletGeometry(line_count=2, last_line_ratio=0.60, text_length=200)}
    assert classify_bullets(geom) == {"b0": OrphanCategory.CLEAN}


def test_classify_bullets_four_line_orphan_is_three_line_orphan():
    """4+ lines with tiny tail still classifies as THREE_LINE_ORPHAN (must trim)."""
    geom = {"b0": BulletGeometry(line_count=4, last_line_ratio=0.05, text_length=400)}
    assert classify_bullets(geom) == {"b0": OrphanCategory.THREE_LINE_ORPHAN}


# ── fix_bullets — orphan_categories tagging ──────────────────────────────────

def test_fix_bullets_three_line_tag_in_prompt():
    """When orphan_categories specifies THREE_LINE_ORPHAN, the prompt tag reflects it."""
    captured = {}

    class _CapturingModel:
        def generate_structured(self, *, system, prompt, schema):
            captured["prompt"] = prompt
            captured["system"] = system
            return ModelResponse(
                data=schema.model_validate({"rewrites": [{"id": "b0", "text": "x" * 100}], "drop_ids": []}),
                model_name="stub",
            )

    resume = _make_resume([{"id": "b0", "text": "a" * 270, "source_ids": ["src"]}])
    fix_bullets(
        resume, orphan_ids=["b0"], overflow_candidate_ids=[], model=_CapturingModel(),
        orphan_categories={"b0": OrphanCategory.THREE_LINE_ORPHAN},
    )
    assert "THREE_LINE_ORPHAN" in captured["prompt"]
    assert "MUST trim to ≤240 chars" in captured["system"]


# ── llm_clean_truncated ───────────────────────────────────────────────────────

def test_llm_clean_truncated_returns_rewrite():
    """Stub model returns _CleanRewrite with a clean ending; function returns its text."""
    clean_text = "Created and executed A/B tests on content messaging to drive a 80% increase in audience engagement."
    assert len(clean_text) <= 120
    model = _StubModel({"text": clean_text})
    result = llm_clean_truncated(
        original="Created and executed A/B tests on content messaging to drive a 80% increase in audience engagement across digital",
        truncated="Created and executed A/B tests on content messaging to drive a 80% increase in audience engagement across",
        target_max_chars=120,
        model=model,
    )
    assert result == clean_text


def test_llm_clean_truncated_returns_none_on_error():
    """When the model raises, the function returns None so callers can fall back."""
    result = llm_clean_truncated(
        original="Some full bullet text that was originally written.",
        truncated="Some full bullet text that was originally",
        target_max_chars=120,
        model=_ErrorModel(),
    )
    assert result is None


def test_llm_clean_truncated_returns_none_when_llm_exceeds_budget():
    """If the LLM returns text exceeding target_max_chars, return None (budget respected)."""
    over_budget = "x" * 200  # > target_max_chars=120
    model = _StubModel({"text": over_budget})
    result = llm_clean_truncated(
        original="Some full bullet.",
        truncated="Some full",
        target_max_chars=120,
        model=model,
    )
    assert result is None
