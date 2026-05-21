"""LLM-based JD coverage judge.

Replaces the mechanical token-overlap score with a semantic evaluation:
the model reads the JD and the tailored resume and returns a 0–1 score
plus a list of uncovered requirements.

Threshold: 0.5 (half of the JD's key requirements must be addressed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from harness.models import ModelClient
from harness.schema import Resume, iter_bullets

THRESHOLD = 0.5

_SYSTEM = """\
You are a recruiter evaluating how well a tailored resume covers a job description.
Be realistic and calibrated: a good entry-level or mid-level resume that honestly
addresses the role should score 0.6–0.8. Only score above 0.9 if nearly every
requirement is explicitly addressed. Score below 0.4 only if coverage is genuinely
poor.
"""

_PROMPT = """\
JOB DESCRIPTION:
{jd}

RESUME TEXT:
{resume_text}

Score how well the resume covers the JD's key requirements, skills, and terminology.
Return JSON with:
  score: float 0.0–1.0
  uncovered_requirements: list of specific JD requirements or skills NOT addressed (max 10 items)
"""


class _CoverageReport(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    uncovered_requirements: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class JDCoverageResult:
    score: float
    passed: bool
    uncovered_requirements: list[str] = field(default_factory=list)


def _resume_text(resume: Resume) -> str:
    chunks = [resume.summary, " ".join(resume.skills)]
    chunks += [b.text for b in iter_bullets(resume)]
    chunks += [e.title for e in resume.experience]
    chunks += [p.name + " " + p.description for p in resume.projects]
    return "\n".join(c for c in chunks if c.strip())


class JDCoverageJudge:
    def run(self, *, output_resume: Resume, jd: str, model: ModelClient) -> JDCoverageResult:
        prompt = _PROMPT.format(jd=jd.strip(), resume_text=_resume_text(output_resume))
        # Let exceptions propagate — the orchestrator's gate distinguishes
        # "judge errored" from "judge said pass" via the `is not None` check
        # in `jd_passed = jd_cov_result is not None and jd_cov_result.passed`.
        # A silent-pass-on-exception conflated those two states.
        resp = model.generate_structured(system=_SYSTEM, prompt=prompt, schema=_CoverageReport)
        report: _CoverageReport = resp.data
        return JDCoverageResult(
            score=round(report.score, 3),
            passed=report.score >= THRESHOLD,
            uncovered_requirements=report.uncovered_requirements[:10],
        )
