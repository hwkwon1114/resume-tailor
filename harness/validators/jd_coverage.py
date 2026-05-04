"""JDCoverage — keyword overlap between JD and tailored resume.

Uses sklearn's frozen English stopword set so coverage scores are
reproducible across machines. Tokens: lowercase \\W+ split, length>=3,
non-numeric.
"""
from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from harness.schema import Resume, iter_bullets
from harness.validators import ValidationResult

_TOKEN_RE = re.compile(r"\W+", flags=re.UNICODE)
_STOPWORDS: frozenset[str] = ENGLISH_STOP_WORDS  # type: ignore[assignment]
# Calibrated 2026-05-03 against the new prompt: real engineering resumes against
# engineering JDs typically clear ~0.45-0.55 after honest tailoring. Was 0.5.
_DEFAULT_FLOOR = 0.45

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


class JDCoverageValidator:
    name = "jd_coverage"

    def __init__(self, floor: float = _DEFAULT_FLOOR) -> None:
        self.floor = floor

    def run(self, *, output: Resume, jd: str) -> ValidationResult:
        jd_kw = tokenize(jd)
        if not jd_kw:
            return ValidationResult(name=self.name, passed=True, score=1.0)
        out_kw = resume_keywords(output)
        covered = jd_kw & out_kw
        missing = sorted(jd_kw - out_kw)
        coverage = len(covered) / len(jd_kw)
        return ValidationResult(
            name=self.name,
            passed=True,  # pass/fail now decided by JDCoverageJudge (LLM); this provides feedback terms only
            score=coverage,
            errors=[],
            payload={
                "coverage": coverage,
                "covered_keywords": sorted(covered),
                "missing_keywords": missing,
                "missing_keywords_top": missing[:25],
            },
        )


def _ensure_iterable(_x) -> Iterable[str]:  # pragma: no cover — guard for future use
    return _x or ()
