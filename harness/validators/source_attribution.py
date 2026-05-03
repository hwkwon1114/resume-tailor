"""SourceAttribution — every output bullet must cite valid input bullet ids."""
from __future__ import annotations

from harness.schema import Resume, iter_bullet_ids, iter_bullets
from harness.validators import ValidationResult


class SourceAttributionValidator:
    name = "source_attribution"

    def run(self, *, output: Resume, input_resume: Resume) -> ValidationResult:
        valid_ids = set(iter_bullet_ids(input_resume))
        errors: list[str] = []
        bad_bullets: list[str] = []
        total = 0
        ok = 0
        for b in iter_bullets(output):
            total += 1
            if not b.source_ids:
                errors.append(f"bullet {b.id!r} has empty source_ids")
                bad_bullets.append(b.id)
                continue
            unknown = [sid for sid in b.source_ids if sid not in valid_ids]
            if unknown:
                errors.append(f"bullet {b.id!r} cites unknown source_ids {unknown}")
                bad_bullets.append(b.id)
                continue
            ok += 1
        score = (ok / total) if total else 0.0
        return ValidationResult(
            name=self.name,
            passed=not errors,
            score=score,
            errors=errors,
            payload={"bad_bullets": bad_bullets, "total_bullets": total},
        )
