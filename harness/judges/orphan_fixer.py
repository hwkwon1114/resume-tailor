"""Orphan bullet fixer — detects and rewrites danger-zone bullets.

At 10.5pt Times New Roman with 0.4in margins on US Letter, one line fits
roughly 98 characters. A bullet in the range 96–130 chars renders as 1 line
plus a few orphan words — it looks worse than either a clean 1-liner or a
full 2-liner.

Detection: character count heuristic (96–130 = danger zone).
Fix: one targeted LLM call to rewrite only the flagged bullets.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from harness.models import ModelClient
from harness.schema import Resume, iter_bullets

SINGLE_LINE_MAX = 120  # ≤ this → fits on one line (Times NR 10.5pt, 0.4in margins)
DANGER_MIN = 121       # 121–165 → orphan zone: >1 line but second line is < half full
DANGER_MAX = 165

_SYSTEM = """\
You are fixing resume bullet points that have an "orphan word" problem.
At 10.5pt Times New Roman with 0.4in margins, one line fits ~123 characters.
A bullet in the 121–165 character range renders as one full line plus a tiny
second line with only 1–3 words — this looks bad on a resume.

For each bullet, rewrite it to EITHER:
  a) ≤ 120 characters: trim words from the end until it fits cleanly on one line.
     Prefer cutting filler phrases ("ensuring that...", "in order to...").
     KEEP all numbers, percentages, and key outcomes.
  b) ≥ 200 characters: expand with method, context, or impact detail so the
     second line is meaningfully full (at least half a line of real content).

Rules:
- Do NOT change the meaning, facts, or quantitative claims.
- Do NOT add fabricated details.
- Prefer option (b) — expand to a full two-liner — add method, context, or scope detail using only facts already in the bullet. Only fall back to option (a) if there is genuinely no additional detail to add.
- Return only the rewritten text strings, one per bullet, in the same order.
"""

_PROMPT = """\
Rewrite the following bullets to fix orphan words (each must be ≤120 OR ≥200 characters):

{bullets_json}

Return a JSON array of strings, one rewritten bullet per item, in the same order.
Count characters carefully before returning.
"""


class _FixResult(BaseModel):
    bullets: list[str] = Field(description="Rewritten bullet texts in original order.")


def detect_orphans(resume: Resume) -> list[str]:
    """Return bullet ids in the danger zone (96–130 chars)."""
    return [b.id for b in iter_bullets(resume) if DANGER_MIN <= len(b.text) <= DANGER_MAX]


def fix_orphans(resume: Resume, orphan_ids: list[str], model: ModelClient) -> Resume:
    """Rewrite danger-zone bullets via one LLM call. Returns updated Resume."""
    if not orphan_ids:
        return resume

    id_set = set(orphan_ids)
    bullets_to_fix = [(b.id, b.text) for b in iter_bullets(resume) if b.id in id_set]
    if not bullets_to_fix:
        return resume

    bullets_json = json.dumps([{"id": bid, "text": text} for bid, text in bullets_to_fix], indent=2)
    prompt = _PROMPT.format(bullets_json=bullets_json)

    try:
        resp = model.generate_structured(system=_SYSTEM, prompt=prompt, schema=_FixResult)
        rewrites = resp.data.bullets
    except Exception:
        return resume  # if the call fails, keep originals

    if len(rewrites) != len(bullets_to_fix):
        return resume  # mismatched response, keep originals

    rewrite_map = {bid: new_text for (bid, _), new_text in zip(bullets_to_fix, rewrites)}

    data = resume.model_dump()
    for key in ("experience", "education", "projects"):
        for section in data[key]:
            for bullet in section["bullets"]:
                if bullet["id"] in rewrite_map:
                    bullet["text"] = rewrite_map[bullet["id"]]
    return Resume.model_validate(data)
