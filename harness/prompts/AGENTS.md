<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-16 -->

# prompts

## Purpose
Prompt templates kept as plain `.txt` files so prompt iteration shows up as readable git diffs (not buried inside Python string literals). Loaded at runtime by `load_prompt(name)`.

## Key Files
| File | Description |
|------|-------------|
| `system_generate.txt` | The main generator system prompt — rules for tailoring, source-attribution requirements, JSON output shape, length budgets. |
| `system_fabrication_judge.txt` | System prompt for the fabrication audit (per-bullet support scoring, generalization-vs-specification rules, common fabrication patterns). |
| `feedback_template.txt` | Retry feedback template — filled by the orchestrator with validator errors and uncovered JD requirements, then injected into the next generation attempt. |
| `__init__.py` | `load_prompt(name)` — reads `{name}.txt` from this directory. |

## For AI Agents

### Working In This Directory
- Treat prompts as source code: small, scoped edits with a clear rationale. Don't bundle prompt rewrites with refactors — the diff should make the change obvious.
- The generator prompt and the schema in `harness/schema.py` must agree on field shapes. If you add a field to the schema, mirror it in the prompt's output-shape section.
- The feedback template is filled with `.format(...)` style substitution. Keep placeholder names stable; renames cascade into `harness/orchestrator.py`.

### Testing Requirements
- No prompt-only unit tests exist — coverage is via end-to-end evals in `evals/` and orchestrator tests in `tests/test_orchestrator.py`.
- If you change the generator prompt, at minimum run `make evals` (stub mode) and inspect `make evals-live` results on the standard fixtures before merging.

### Common Patterns
- Files are loaded as UTF-8. Embed no binary content.
- Keep one logical concern per file rather than monolithic mega-prompts.

## Dependencies

### Internal
- `harness/generate.py` reads `system_generate.txt` and `feedback_template.txt`.
- `harness/judges/fabrication_audit.py` reads `system_fabrication_judge.txt`.
- `harness/judges/jd_coverage_judge.py` uses an inline `_SYSTEM` constant (no file).

### External
- None.

<!-- MANUAL: -->
