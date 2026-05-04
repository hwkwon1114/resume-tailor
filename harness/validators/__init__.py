"""Mechanical validators (pure Python, no LLM calls).

Each validator returns a `ValidationResult` and never mutates the resume.
The orchestrator decides what to do with results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.schema import Resume


@dataclass(slots=True)
class ValidationResult:
    name: str
    passed: bool
    score: float = 1.0  # 1.0 = perfect, 0.0 = total failure (interpretation per validator)
    errors: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


def run_all(*, output: Resume, input_resume: Resume, jd: str) -> list[ValidationResult]:
    """Run mechanical validators on an output resume."""
    from harness.validators.field_lock import FieldLockValidator
    from harness.validators.page_fit import PageFitValidator
    from harness.validators.schema_check import SchemaCheckValidator
    from harness.validators.source_attribution import SourceAttributionValidator

    return [
        SchemaCheckValidator().run(output),
        SourceAttributionValidator().run(output=output, input_resume=input_resume),
        FieldLockValidator().run(output=output, input_resume=input_resume),
        PageFitValidator().run(output=output),
    ]
