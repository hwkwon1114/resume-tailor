"""Mechanical pre-processing: JD key-term extraction + per-bullet JD-overlap ranking.

Hand the model a pre-digested problem so its job is selection + reframing, not
discovery. Reuses the same tokenization as the JDCoverage validator so the
ranking and the floor speak the same language.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from harness.schema import Resume, iter_bullets
from harness.validators.jd_coverage import tokenize

DEFAULT_TOP_TERMS = 40
DEFAULT_TOP_BULLETS = 12  # cap injected ranking so the prompt doesn't bloat


@dataclass(slots=True)
class BulletScore:
    bullet_id: str
    section: str
    score: float
    text: str


def extract_jd_key_terms(jd: str, *, top_n: int = DEFAULT_TOP_TERMS) -> list[str]:
    """Pull the most informative terms from the JD (frequency-weighted, stopword-filtered).

    Returns lowercase tokens sorted by frequency desc, length tiebreak desc (so
    longer / more specific terms win on ties).
    """
    tokens = list(_tokens_with_repeats(jd))
    counts = Counter(tokens)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    return [t for t, _ in ranked[:top_n]]


def rank_bullets_by_jd(resume: Resume, jd: str) -> list[BulletScore]:
    """Score every input bullet by token-overlap fraction with the JD."""
    jd_kw = tokenize(jd)
    if not jd_kw:
        # No JD keywords → all bullets are equally (un)relevant.
        return [_score_bullet(b, set(), section_for(resume, b.id)) for b in iter_bullets(resume)]
    out: list[BulletScore] = []
    for b in iter_bullets(resume):
        out.append(_score_bullet(b, jd_kw, section_for(resume, b.id)))
    out.sort(key=lambda bs: (-bs.score, bs.bullet_id))
    return out


def section_for(resume: Resume, bullet_id: str) -> str:
    for e in resume.experience:
        if any(b.id == bullet_id for b in e.bullets):
            return f"experience/{e.employer}"
    for ed in resume.education:
        if any(b.id == bullet_id for b in ed.bullets):
            return f"education/{ed.school}"
    for p in resume.projects:
        if any(b.id == bullet_id for b in p.bullets):
            return f"project/{p.name}"
    return "unknown"


def render_ranking_block(scores: list[BulletScore], *, top_n: int = DEFAULT_TOP_BULLETS) -> str:
    """Markdown-ish list the model can read directly."""
    lines = [
        "JD-RELEVANCE PRE-RANKING (count of distinct JD key terms each bullet covers; higher = more relevant):"
    ]
    for bs in scores[:top_n]:
        snippet = bs.text.strip().replace("\n", " ")
        if len(snippet) > 110:
            snippet = snippet[:107] + "..."
        lines.append(f"  {bs.bullet_id}  hits={int(bs.score):>2}  [{bs.section}]  {snippet!r}")
    if len(scores) > top_n:
        lines.append(f"  ... {len(scores) - top_n} lower-ranked bullets omitted; strong drop candidates.")
    return "\n".join(lines)


def render_keyword_block(jd_terms: list[str]) -> str:
    """Tell the model exactly which JD terms to mirror."""
    return (
        "JD KEY TERMS (mirror these in bullets/skills WHERE you have authentic input support — "
        "do NOT invent experience to use a term):\n  " + ", ".join(jd_terms)
    )


def _score_bullet(bullet, jd_kw: set[str], section: str) -> BulletScore:
    """Score = raw count of distinct JD terms appearing in the bullet.

    Raw count is more informative for the model than the |∩|/|jd| ratio
    (which compresses everything into a tiny range because most bullets
    only cover a handful of the JD's many tokens).
    """
    bullet_kw = tokenize(bullet.text)
    overlap = bullet_kw & jd_kw
    return BulletScore(bullet_id=bullet.id, section=section, score=float(len(overlap)), text=bullet.text)


def _tokens_with_repeats(text: str):
    import re

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = ENGLISH_STOP_WORDS
    for raw in re.split(r"\W+", text or ""):
        t = raw.lower().strip()
        if len(t) < 3 or t in stop or t.isdigit():
            continue
        yield t
