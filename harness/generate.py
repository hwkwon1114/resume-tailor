"""Generator — calls the model with structured output to produce a tailored Resume.

Feedback policy: replace across attempts; aggregate within an attempt.

One-shot improvements (Rev 2):
  - Pre-ranks input bullets by JD-overlap and injects the ranking into the prompt.
  - Pre-extracts JD key terms so the model doesn't have to discover them.
  - Wraps the output in a TailoringResponse with a `reasoning` field so the
    model has to write out its Phase-1 mapping before bullets — improves
    bullet-level fidelity without changing the canonical Resume schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from jinja2 import Template
from pydantic import BaseModel, Field

from harness.models import ModelClient, ModelResponse
from harness.prompts import load_prompt
from harness.ranking import (
    extract_jd_key_terms,
    rank_bullets_by_jd,
    render_keyword_block,
    render_ranking_block,
)
from harness.schema import Resume

_FEEDBACK_TEMPLATE = Template(load_prompt("feedback_template"))


class TailoringResponse(BaseModel):
    """LLM output envelope: forces the model to write its Phase-1 mapping
    BEFORE the resume, then emit the structured resume."""

    reasoning: str = Field(
        description=(
            "Phase 1 strategic mapping: identify the company's core hiring problem in 1 sentence, "
            "then list 3-5 bullets of the form 'JD priority X -> input bullet Y is the closest match'. "
            "Then state which input bullets you will drop and why. This field is for your own thinking; "
            "the user will not see it. Be concrete: cite input bullet ids."
        )
    )
    resume: Resume


@dataclass(slots=True)
class FeedbackMessage:
    schema_errors: list[str] = field(default_factory=list)
    source_attribution_errors: list[str] = field(default_factory=list)
    field_lock_errors: list[str] = field(default_factory=list)
    page_fit_overflow: list[str] = field(default_factory=list)
    page_fit_underutilized: str | None = None
    jd_coverage_score: float | None = None
    jd_coverage_missing: list[str] = field(default_factory=list)
    fabrication_flagged: list[dict] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.schema_errors
            or self.source_attribution_errors
            or self.field_lock_errors
            or self.page_fit_overflow
            or self.page_fit_underutilized
            or self.jd_coverage_missing
            or self.fabrication_flagged
        )

    def render(self) -> str:
        return _FEEDBACK_TEMPLATE.render(
            schema_errors=self.schema_errors,
            source_attribution_errors=self.source_attribution_errors,
            field_lock_errors=self.field_lock_errors,
            page_fit_overflow=self.page_fit_overflow,
            page_fit_underutilized=self.page_fit_underutilized,
            jd_coverage_score=self.jd_coverage_score,
            jd_coverage_missing=self.jd_coverage_missing,
            fabrication_flagged=self.fabrication_flagged,
        )


def generate(
    *,
    jd: str,
    input_resume: Resume,
    feedback: FeedbackMessage | None,
    model: ModelClient,
) -> ModelResponse[Resume]:
    """Call the model with pre-digested context. Feedback REPLACES across attempts."""
    system = load_prompt("system_generate")

    # Pre-process: hand the model a sorted bullet ranking + the JD key terms.
    # Cuts the model's discovery work, which is the main first-shot failure mode.
    jd_terms = extract_jd_key_terms(jd)
    rankings = rank_bullets_by_jd(input_resume, jd)

    sections = [
        "JOB DESCRIPTION:\n" + jd.strip(),
        render_keyword_block(jd_terms),
        render_ranking_block(rankings),
        "INPUT RESUME (JSON):\n" + json.dumps(input_resume.model_dump(), indent=2),
    ]
    if feedback is not None and not feedback.is_empty():
        sections.append("FEEDBACK FROM PREVIOUS ATTEMPT:\n" + feedback.render())
    prompt = "\n\n".join(sections)

    resp = model.generate_structured(system=system, prompt=prompt, schema=TailoringResponse)
    envelope: TailoringResponse = resp.data
    # Unwrap: callers consume Resume; reasoning travels in metadata for the trajectory log.
    return ModelResponse(
        data=envelope.resume,
        usage=resp.usage,
        model_name=resp.model_name,
        latency_ms=resp.latency_ms,
        metadata={**resp.metadata, "reasoning": envelope.reasoning},
    )
