"""SchemaCheck — defensive Pydantic re-validation.

The model client already calls Resume.model_validate, but a separate
validator stage gives a uniform interface for the orchestrator's report.
"""
from __future__ import annotations

from pydantic import ValidationError

from harness.schema import Resume
from harness.validators import ValidationResult


class SchemaCheckValidator:
    name = "schema_check"

    def run(self, output: Resume) -> ValidationResult:
        try:
            Resume.model_validate(output.model_dump())
            return ValidationResult(name=self.name, passed=True)
        except ValidationError as exc:
            return ValidationResult(
                name=self.name,
                passed=False,
                score=0.0,
                errors=[str(exc)],
            )
