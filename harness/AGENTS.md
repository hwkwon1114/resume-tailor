<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# harness

## Purpose
The deterministic harness wrapping Gemini for resume tailoring. Owns the schema, the generate→validate→judge→retry orchestration loop, JD-aware bullet ranking, PDF/text intake parsing, and the model-client abstraction. All semantic work happens here; rendering (PDF) and evaluation (fixtures) sit outside.

## Key Files
| File | Description |
|------|-------------|
| `schema.py` | Pydantic `Resume` / `Bullet` / `ExperienceEntry` / `EducationEntry` / `Project` models. `Bullet.source_ids` is the provenance chain. `locked_fields()` + `iter_bullets()` + `autopopulate_bullet_ids()` are the load-bearing helpers. |
| `orchestrator.py` | The main loop: generate → drop-empty + skill boost → mechanical validators → page-fit trim → JD coverage judge → voice check → orphan fixer. Replaces (not appends) feedback across retries. `DEFAULT_MAX_RETRIES = 3`. |
| `generate.py` | Single-shot generator: builds the Gemini prompt from system prompt + JD + input resume + retry feedback, parses TailoringResponse JSON back into a `Resume`. |
| `ranking.py` | TF-IDF-based JD term extraction + per-bullet JD-relevance scoring. Used by orchestrator's page-fit trim to drop the lowest-ranked bullets first. |
| `pdf_intake.py` | PDF (via pdfplumber) or plain-text → structured `Resume`. Calls Gemini under the hood for the text→schema step. |
| `__init__.py` | Empty package marker. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `models/` | `ModelClient` protocol + `EchoModelClient` stub + `GeminiSubprocessClient` (see `models/AGENTS.md`). |
| `prompts/` | Prompt templates as `.txt` files (see `prompts/AGENTS.md`). |
| `validators/` | Pure-Python mechanical validators — schema, source attribution, field lock, page fit, JD coverage tokens (see `validators/AGENTS.md`). |
| `judges/` | LLM-backed quality judges and the orphan fixer (see `judges/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- **The provenance chain is the safety net.** Every output `Bullet.source_ids` must list real input bullet ids. `SourceAttributionValidator` will fail and force a retry otherwise. Never silently filter or rewrite `source_ids` outside the LLM step.
- **Field lock is also a safety net.** Employer names, dates, and degree titles are byte-comparable locked fields (`locked_fields()` in `schema.py`). The LLM is not allowed to rewrite them; `FieldLockValidator` enforces this.
- The orchestrator owns the retry policy. Validators and judges only return results — they don't mutate the resume or decide whether to retry.
- Page fitting is mechanical (drop lowest-ranked bullets), not an LLM retry. This keeps the cost bounded.
- Orphan fixing (1–3 word spillovers onto a second line) is *only* invoked on the final candidate, and only when needed. It's an LLM call, so guard it.

### Testing Requirements
- Tests use `EchoModelClient` stubbed responses — pass canned JSON via the client and assert orchestrator behaviour. See `tests/test_orchestrator.py` for patterns.
- New validators should ship with both pass-case and fail-case unit tests in `tests/test_validators.py`.

### Common Patterns
- **No Pydantic defaults on Gemini-bound schemas.** google-genai issue #699 rejects them. `Bullet.source_ids: list[str]` is required; the input-resume builder stamps `[]` explicitly via `autopopulate_bullet_ids()`.
- Bullet ids are deterministic: `{kind}-{section_idx}-b{bullet_idx}` (e.g. `exp-0-b2`). Existing ids are preserved.
- Logging uses module-level `log = logging.getLogger(__name__)`; the entry points (`app.py`, `tailor.py`) configure root logging.

## Dependencies

### Internal
- `render/` — orchestrator may invoke `render.pdf.render_to_document` indirectly via `PageFitValidator` (single source of truth for page count).

### External
- **pydantic** — schema validation and serialization.
- **scikit-learn** — TF-IDF in `ranking.py`.
- **pdfplumber** — PDF text extraction in `pdf_intake.py`.
- **google-genai** — type definitions only; actual calls go through the `gemini` CLI subprocess.

<!-- MANUAL: -->
