"""Orphan bullet fixer — detects and rewrites danger-zone bullets.

At 10.5pt Times New Roman with 0.4in margins on US Letter, one line fits
roughly 123 characters. A bullet in the range 121–165 chars renders as 1 line
plus a few orphan words — it looks worse than either a clean 1-liner or a
full 2-liner.

Detection: character count heuristic (121–165 = danger zone).
Fix: one targeted LLM call that also handles overflow candidates.
"""
from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, Field

from harness.models import ModelClient
from harness.schema import Resume, iter_bullets
from harness.validators.page_fit import BulletGeometry

SINGLE_LINE_MAX = 120  # ≤ this → fits on one line (Times NR 10.5pt, 0.4in margins)
DANGER_MIN = 121       # 121–165 → orphan zone: >1 line but second line is < half full
DANGER_MAX = 165

LAST_LINE_ORPHAN_RATIO = 0.30  # last line < 30% of content width = orphan tail


class OrphanCategory(str, Enum):
    CLEAN = "clean"
    TWO_LINE_ORPHAN = "two_line_orphan"
    THREE_LINE_ORPHAN = "three_line_orphan"
    THREE_LINE_FULL = "three_line_full"


def classify_bullets(geometry: dict[str, BulletGeometry]) -> dict[str, OrphanCategory]:
    """Categorize each bullet by its rendered geometry."""
    out: dict[str, OrphanCategory] = {}
    for bid, g in geometry.items():
        if g.line_count <= 1:
            out[bid] = OrphanCategory.CLEAN
        elif g.line_count == 2:
            if g.last_line_ratio < LAST_LINE_ORPHAN_RATIO:
                out[bid] = OrphanCategory.TWO_LINE_ORPHAN
            else:
                out[bid] = OrphanCategory.CLEAN
        else:  # 3+ lines: every wasted line costs page space
            if g.last_line_ratio < LAST_LINE_ORPHAN_RATIO:
                out[bid] = OrphanCategory.THREE_LINE_ORPHAN
            else:
                out[bid] = OrphanCategory.THREE_LINE_FULL
    return out


def detect_orphans_geometry(resume: Resume) -> dict[str, OrphanCategory]:
    """Render and categorize bullets; returns only non-CLEAN bullets.

    Returns {} if rendering fails — callers should fall back to detect_orphans.
    """
    from harness.validators.page_fit import measure_bullet_geometry
    geom = measure_bullet_geometry(resume)
    if not geom:
        return {}
    classified = classify_bullets(geom)
    return {bid: cat for bid, cat in classified.items() if cat is not OrphanCategory.CLEAN}

_BIAS_HIGH = (
    "PAGE IS {pct}% FULL — for [TWO_LINE_ORPHAN] bullets, strongly prefer expanding to ≥200 chars. "
    "Only trim to ≤120 as a last resort when there is genuinely no detail to add. "
    "Never put TYPE A bullets in drop_ids — they must appear in rewrites."
)
_BIAS_LOW = (
    "PAGE IS {pct}% FULL — for [TWO_LINE_ORPHAN] bullets, prefer trimming to ≤120 chars."
)
_BIAS_NEUTRAL = (
    "PAGE IS {pct}% FULL — for [TWO_LINE_ORPHAN] bullets, prefer expansion to ≥200 chars "
    "when the bullet has authentic content to expand into. Only trim to ≤120 if the bullet "
    "is genuinely complete as a 1-liner — do NOT trim merely to reduce length."
)

_SYSTEM_TEMPLATE = """\
You are post-processing a one-page resume. Fix two types of bullets in one pass:

TYPE A — ORPHAN BULLETS (all must be fixed). Each bullet has an action tag:
  • [TWO_LINE_ORPHAN] = renders as 2 lines with a tiny tail. Either:
      - trim to ≤120 chars (clean 1-liner), OR
      - expand to ≥200 chars (full 2-liner) — add method, context, or scope.
{bias_instruction}
  • [THREE_LINE_ORPHAN] = renders as 3+ lines with a tiny tail. MUST trim to ≤240 chars
    (clean 2-liner). Do NOT expand. Every wasted line costs page space.
  • [THREE_LINE_FULL] = renders as 3+ lines with a full last line. MUST trim to ≤240 chars
    (clean 2-liner). Do NOT expand.

TYPE B — OVERFLOW CANDIDATES (trim or drop to recover page space):
The resume overflows onto page 2. These are the lowest-value bullets.
For each, try trimming to ≤120 chars first — saving one line often fixes the overflow.
Only add a bullet to drop_ids if trimming would gut its core meaning entirely.
Prefer keeping a shorter bullet over dropping it.

Rules (all types):
- Do NOT change meaning, facts, or quantitative claims.
- Do NOT add fabricated details.
- 1-liner targets are ≤120 chars. 2-liner targets are ≥200 and ≤240 chars.
"""

_PROMPT = """\
Fix the following resume bullets.

TYPE A — ORPHAN BULLETS (action tag in [brackets] — see system prompt):
{orphan_json}

TYPE B — OVERFLOW CANDIDATES (trim to ≤120 chars, or drop if trimming guts the meaning):
{overflow_json}

Return a JSON object with exactly two keys:
  "rewrites": array of {{id, text}} for every bullet you are keeping (rewritten text)
  "drop_ids": array of bullet ids you are dropping (TYPE B only, when trim is not viable)

Every TYPE A bullet must appear in "rewrites".
Every TYPE B bullet must appear in either "rewrites" or "drop_ids".
Count characters carefully — respect each bullet's action tag.
"""


class _BulletRewrite(BaseModel):
    id: str = Field(description="Bullet id")
    text: str = Field(description="Rewritten bullet text")


class _FixResult(BaseModel):
    rewrites: list[_BulletRewrite] = Field(
        description="Kept bullets with rewritten text (≤120 or ≥200 chars each)."
    )
    drop_ids: list[str] = Field(
        default_factory=list,
        description="IDs of TYPE B bullets to drop (only when trimming guts the meaning).",
    )


def detect_orphans(resume: Resume) -> list[str]:
    """Return bullet ids in the danger zone (121–165 chars)."""
    return [b.id for b in iter_bullets(resume) if DANGER_MIN <= len(b.text) <= DANGER_MAX]


def fix_bullets(
    resume: Resume,
    orphan_ids: list[str],
    overflow_candidate_ids: list[str],
    model: ModelClient,
    *,
    page_utilization_pct: int = 100,
    force_expand_ids: list[str] | None = None,
    orphan_categories: dict[str, OrphanCategory] | None = None,
) -> Resume:
    """Rewrite orphan bullets and trim/drop overflow candidates in one LLM call.

    page_utilization_pct biases the orphan fixer:
      - util < 95 (below the 95% page-utilization target) → prefer expansion
      - 95 ≤ util < 100 (at target) → neutral, let content richness decide
      - util ≥ 100 (saturates when pages > 1 — i.e. overflow) → prefer trimming
    Bias applies only to TWO_LINE_ORPHAN; THREE_LINE_* bullets must always trim.

    force_expand_ids bullets are treated as TWO_LINE_ORPHAN (expansion-eligible),
    even if they are not in the danger zone. Used by the post-processing rescue
    path to grow short bullets back into a 2-liner after shrinkage.

    orphan_categories — when provided, drives per-bullet action tags in the
    prompt. When None (back-compat), all orphan_ids are treated as TWO_LINE_ORPHAN.
    """
    force_expand = list(force_expand_ids or [])
    if not orphan_ids and not overflow_candidate_ids and not force_expand:
        return resume

    all_bullet_map = {b.id: b.text for b in iter_bullets(resume)}

    # Treat force_expand_ids as orphans for prompt + drop-protection purposes.
    combined_orphan_ids: list[str] = list(orphan_ids)
    seen_orphan = set(combined_orphan_ids)
    for bid in force_expand:
        if bid in all_bullet_map and bid not in seen_orphan:
            combined_orphan_ids.append(bid)
            seen_orphan.add(bid)

    cats = dict(orphan_categories or {})
    for bid in combined_orphan_ids:
        cats.setdefault(bid, OrphanCategory.TWO_LINE_ORPHAN)
    for bid in force_expand:
        cats[bid] = OrphanCategory.TWO_LINE_ORPHAN

    orphan_bullets = [{"id": bid, "tag": cats[bid].value.upper(), "text": all_bullet_map[bid]}
                      for bid in combined_orphan_ids if bid in all_bullet_map]
    overflow_bullets = [{"id": bid, "text": all_bullet_map[bid]}
                        for bid in overflow_candidate_ids
                        if bid in all_bullet_map and bid not in seen_orphan]

    if not orphan_bullets and not overflow_bullets:
        return resume

    has_two_line = any(cats.get(bid) is OrphanCategory.TWO_LINE_ORPHAN
                       for bid in combined_orphan_ids)
    if has_two_line and page_utilization_pct >= 100:
        bias = _BIAS_LOW.format(pct=page_utilization_pct)
        _bias_name = "LOW (trim-favored, page at/over limit)"
    elif has_two_line and page_utilization_pct < 95:
        bias = _BIAS_HIGH.format(pct=page_utilization_pct)
        _bias_name = "HIGH (expand-favored, below 95% target)"
    else:
        bias = _BIAS_NEUTRAL.format(pct=page_utilization_pct)
        _bias_name = "NEUTRAL (soft expand-lean, near target)"
    import logging as _logging
    _logging.getLogger(__name__).info(
        "[fix_bullets] bias=%s util=%d%% orphans=%d (TWO_LINE=%d THREE_LINE=%d) overflow_cands=%d",
        _bias_name, page_utilization_pct, len(combined_orphan_ids),
        sum(1 for bid in combined_orphan_ids if cats.get(bid) is OrphanCategory.TWO_LINE_ORPHAN),
        sum(1 for bid in combined_orphan_ids
            if cats.get(bid) in (OrphanCategory.THREE_LINE_ORPHAN, OrphanCategory.THREE_LINE_FULL)),
        len(overflow_bullets),
    )

    system = _SYSTEM_TEMPLATE.format(bias_instruction=bias)
    prompt = _PROMPT.format(
        orphan_json=json.dumps(orphan_bullets, indent=2),
        overflow_json=json.dumps(overflow_bullets, indent=2),
    )

    try:
        resp = model.generate_structured(system=system, prompt=prompt, schema=_FixResult)
        result = resp.data
    except Exception:
        return resume

    rewrite_map = {rw.id: rw.text for rw in result.rewrites}
    # TYPE A (orphan) bullets must never be dropped — guard against LLM putting them in drop_ids.
    orphan_set = set(combined_orphan_ids)
    drop_set = set(result.drop_ids) - orphan_set

    data = resume.model_dump()
    for key in ("experience", "education", "projects"):
        for section in data[key]:
            section["bullets"] = [
                {**b, "text": rewrite_map[b["id"]]} if b["id"] in rewrite_map
                else b
                for b in section["bullets"]
                if b["id"] not in drop_set
            ]
    return Resume.model_validate(data)


class _CleanRewrite(BaseModel):
    text: str = Field(description="Cleanly-ending bullet within the char budget")


def llm_clean_truncated(
    original: str,
    truncated: str,
    target_max_chars: int,
    model: ModelClient,
) -> str | None:
    """Have the LLM rewrite a mechanically-truncated bullet to end cleanly.

    Returns None if the model call fails. Caller should fall back to the
    truncated text in that case.
    """
    system = (
        f"A resume bullet was mechanically shortened and may now end mid-clause. "
        f"Rewrite it to end with proper punctuation, preserving all numbers, "
        f"percentages, and facts from the original. "
        f"The rewrite must be ≤ {target_max_chars} chars."
    )
    prompt = (
        f"ORIGINAL (full meaning): {original}\n"
        f"TRUNCATED (what was cut to): {truncated}\n\n"
        f"Return a clean rewrite ≤ {target_max_chars} chars that ends at a "
        f"natural clause boundary with proper punctuation."
    )
    try:
        resp = model.generate_structured(system=system, prompt=prompt, schema=_CleanRewrite)
        result = resp.data
        if len(result.text) <= target_max_chars:
            return result.text
        # LLM ignored the budget — fall back
        return None
    except Exception:
        return None
