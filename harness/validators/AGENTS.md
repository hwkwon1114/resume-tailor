<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# validators

## Purpose
Pure-Python mechanical validators. Each one inspects a generated `Resume` and returns a `ValidationResult(name, passed, score, errors, payload)` without mutating anything. Validators are cheap, deterministic, and free of LLM calls — they're the harness's first-line defence and the only checks that can fail-fast a retry loop without burning Gemini quota.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | `ValidationResult` dataclass + `run_all(output, input_resume, jd)` orchestrating the four core validators. |
| `schema_check.py` | `SchemaCheckValidator` — re-validates the output `Resume` against the Pydantic schema. |
| `source_attribution.py` | `SourceAttributionValidator` — every output `Bullet.source_ids` must reference real input-bullet ids. The fabrication-detection guardrail. |
| `field_lock.py` | `FieldLockValidator` — employer, dates, school, degree, gpa must match input byte-for-byte. Returns the dotted path of any field that drifted. |
| `page_fit.py` | `PageFitValidator` — renders the resume via `render.pdf` and asserts page count == 1. Single source of truth for "fits on one page" (the renderer is shared with the user-facing PDF). |
| `jd_coverage.py` | Lightweight token-overlap JD coverage check (mechanical companion to the LLM-based JD coverage judge in `harness/judges/`). |

## For AI Agents

### Working In This Directory
- **Never mutate the resume from a validator.** Return errors; let the orchestrator decide what to do. This is why validators are safe to compose in any order.
- New validators should follow the same signature: `run(*, output, input_resume, jd) -> ValidationResult`. Add them to `run_all()` if they should run on every attempt.
- `score` semantics are per-validator. Most are 0/1 (passed/failed). Keep `passed` consistent with `score > 0.5` if you use a continuous score.
- `payload` is the structured side-channel — put per-bullet diagnostics there so the feedback template can format them.

### Testing Requirements
- Each validator gets pass-case and fail-case tests in `tests/test_validators.py` (or its own file for larger ones like `test_page_fit.py`).
- `PageFitValidator` tests must use a real WeasyPrint render — don't stub the renderer, since the validator's whole point is to use the same render as the UI.

### Common Patterns
- Errors are human-readable strings (they get surfaced into LLM feedback). Be specific: `"experience.exp-1.employer drifted: 'Foo' → 'Foo Corp'"` beats `"field lock failed"`.
- Validator names (the `.name` field of `ValidationResult`) are matched against in `orchestrator.py` to decide which failures trigger a retry vs. which are informational.

## Dependencies

### Internal
- `harness.schema` — all validators take `Resume` instances.
- `render.pdf` — `PageFitValidator` calls it directly. **Do not stub** in tests for this validator.

### External
- **pydantic** (schema re-validation).
- **weasyprint** (transitively via `render.pdf` for `PageFitValidator`).

<!-- MANUAL: -->
