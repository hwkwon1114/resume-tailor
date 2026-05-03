"""FieldLock — locked fields must be byte-identical between input and output."""
from __future__ import annotations

from harness.schema import Resume, locked_fields
from harness.validators import ValidationResult


class FieldLockValidator:
    name = "field_lock"

    def run(self, *, output: Resume, input_resume: Resume) -> ValidationResult:
        in_locked = locked_fields(input_resume)
        out_locked = locked_fields(output)
        errors: list[str] = []
        diffs: dict[str, dict[str, str]] = {}
        # Only check entries that appear in the output — the model is allowed to DROP
        # input entries (that's the selection step). We only enforce that whatever
        # the model KEPT has byte-identical locked fields, and that it didn't INVENT
        # new entries that weren't in the input.
        for key, out_val in out_locked.items():
            in_val = in_locked.get(key)
            if in_val is None:
                errors.append(f"locked field invented in output: {key} = {out_val!r}")
                diffs[key] = {"input": "<missing>", "output": out_val}
            elif out_val != in_val:
                errors.append(f"locked field drift on {key}: {in_val!r} -> {out_val!r}")
                diffs[key] = {"input": in_val, "output": out_val}
        return ValidationResult(
            name=self.name,
            passed=not errors,
            score=1.0 if not errors else max(0.0, 1.0 - len(errors) / max(len(in_locked), 1)),
            errors=errors,
            payload={"diffs": diffs},
        )
