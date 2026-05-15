<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# tests

## Purpose
Pytest unit and integration tests. No tests in here call Gemini for real — everything uses `EchoModelClient` with canned responses. The contract is "anyone can run `make test` without auth or network." Live-LLM behaviour is exercised separately by `evals/`.

## Key Files
| File | Description |
|------|-------------|
| `test_schema.py` | Pydantic schema validation, `autopopulate_bullet_ids`, `locked_fields`, `iter_bullets`. |
| `test_validators.py` | Schema check, source attribution, field lock, JD coverage validators — pass cases and failure modes. |
| `test_page_fit.py` | `PageFitValidator` with real WeasyPrint renders against fixture resumes. |
| `test_orchestrator.py` | The big integration test. Drives the full generate→validate→judge loop with a stub model client, asserts retry trajectory and final metrics. ~23KB — the heaviest test file. |
| `test_orphan_fixer.py` | Orphan detection (heuristic + geometry) and LLM rewrite contracts. |
| `test_voice_check.py` | Stylistic drift signal — mechanical, no LLM. |
| `test_fabrication_audit.py` | Content-level fabrication heuristic. |

## For AI Agents

### Working In This Directory
- **`EchoModelClient` is the LLM substitute.** Construct it with a queue of canned `ModelResponse(text=...)` objects, then assert the orchestrator consumes them in order. Look at `test_orchestrator.py` for the established patterns.
- **`PageFitValidator` tests use a real renderer.** Do not stub `render.pdf` for these — the whole point of the validator is to share the renderer with the UI.
- New tests live in `test_<module>.py` matching the module under test. Keep one file per harness module unless the test surface gets large enough to split.
- `pyproject.toml` sets `addopts = "-q"` and `testpaths = ["tests", "evals"]`, so `evals/test_*.py` would also run if you add one (currently none exist).

### Testing Requirements
- `make test` must stay LLM-free. If you write a test that needs Gemini, gate it behind a `RUN_LIVE_*` env var and skip by default.
- New validators / judges / orchestrator branches should ship with both happy-path and failure-path tests.

### Common Patterns
- Build small in-memory resumes via `Resume.model_validate(autopopulate_bullet_ids({...}))` rather than loading fixtures, when you want to isolate a single case.
- For orchestrator tests, the trick is queueing one canned response per expected LLM call (generate + JD coverage judge + optional orphan fix). Off-by-one in the queue length is the most common test bug.

## Dependencies

### Internal
- `harness/` (everything), `render/pdf.py` for `test_page_fit.py`.

### External
- **pytest** (≥8.0) — declared in `pyproject.toml` dev group.
- **pytest-cov** — available but not currently used in default targets.

<!-- MANUAL: -->
