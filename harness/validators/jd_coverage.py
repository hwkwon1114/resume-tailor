"""JDCoverage — keyword overlap between JD and tailored resume.

Uses sklearn's frozen English stopword set so coverage scores are
reproducible across machines. Tokens: lowercase \\W+ split, length>=3,
non-numeric.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from harness.schema import Resume, iter_bullets

_TOKEN_RE = re.compile(r"\W+", flags=re.UNICODE)
_STOPWORDS: frozenset[str] = ENGLISH_STOP_WORDS  # type: ignore[assignment]

# EEO/ADA legal boilerplate and company-culture filler that repeats in JD footers.
# These terms inflate the JD token count and pollute key-term rankings without
# providing any useful signal for resume tailoring.
_HR_BOILERPLATE: frozenset[str] = frozenset({
    # ADA / EEO / legal language (appears in every large company's JD footer)
    "accommodation", "accommodations", "disability", "disabilities",
    "qualified", "protected", "veteran", "prohibited", "discrimination",
    "orientation", "regardless", "applicants", "candidates",
    "ability", "abilities", "applicant", "candidate",
    # Amazon-specific boilerplate
    "amazecon", "amazonians",
    # Generic filler / common verbs that appear in nearly every JD
    "acumen", "affinity", "advancing", "adjustment", "adjustments", "activate",
    "achieve", "achieving", "advice", "align", "alignment", "aligned",
    "addressing", "adopted", "alternative", "alternatives", "annual",
    "approach", "approaches", "applying", "aptitude", "attention",
    "awareness", "balance", "bar", "assignments",
})


def tokenize(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _TOKEN_RE.split(text or ""):
        t = raw.lower().strip()
        if len(t) < 3:
            continue
        if t in _STOPWORDS or t in _HR_BOILERPLATE:
            continue
        if t.isdigit():
            continue
        out.add(t)
    return out


def resume_keywords(resume: Resume) -> set[str]:
    chunks: list[str] = [resume.summary, " ".join(resume.skills)]
    for b in iter_bullets(resume):
        chunks.append(b.text)
    for e in resume.experience:
        chunks.extend([e.employer, e.title])
    for ed in resume.education:
        chunks.extend([ed.school, ed.degree])
    for p in resume.projects:
        chunks.extend([p.name, p.description])
    return tokenize(" ".join(chunks))
