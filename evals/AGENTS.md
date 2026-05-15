<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# evals

## Purpose
End-to-end eval harness with a content-hashed cache. Runs every fixture (JD + resume pair) through the full orchestrator and writes a markdown report with per-fixture metrics, retry trajectory, and cache annotations. Defaults to **stub mode** (uses `EchoModelClient` with canned responses) so wiring can be sanity-checked without burning Gemini quota; set `RUN_LIVE_EVALS=1` to hit the real model.

## Key Files
| File | Description |
|------|-------------|
| `run_evals.py` | Entry point. Loads fixtures, runs the orchestrator on each, applies the cache, writes `evals/report.md`. |
| `cache.py` | Content-hashed cache keyed by `(fixture_name, jd_hash, resume_hash, prompt_hash)`. `cache_key`, `get`, `put`, and a `--clear` CLI. Lets you re-run evals after non-prompt code changes without re-calling Gemini. |
| `__init__.py` | Empty package marker. |
| `report.md` *(generated)* | Eval output — committed selectively to track regressions. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `fixtures/` | JD texts + canonical input resume + fixture loader (see `fixtures/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- **Default to stub mode.** `make evals` (no env var) runs against `EchoModelClient`. Use `make evals-live` (or `RUN_LIVE_EVALS=1`) only when you need real LLM behaviour — e.g. validating a prompt change.
- The cache key includes prompt content. If you edit any file in `harness/prompts/`, the cache will (correctly) invalidate on next run. Don't bypass this; the whole point of the cache is correctness w.r.t. inputs.
- `clear-cache` (Make target) is the supported nuke. There is no per-fixture invalidation.
- The eval runner is wired through pytest discovery (`testpaths = ["tests", "evals"]` in `pyproject.toml`) — `test_*.py` files placed here will run under `make test`.

### Testing Requirements
- Adding a fixture: drop it in `fixtures/jd/<name>.txt`, register it in `fixtures/__init__.py`'s `all_baselines()` or `temptation_fixture()`, run `make evals` to confirm it loads.
- Changing the report shape: update both `run_evals.py` and any tests asserting on report content (currently none — be cautious).

### Common Patterns
- **Stub-mode canned responses** live alongside the fixture set so the stub and the real Gemini are tested against the same JD/resume pair.
- Live-mode results are noisy run-to-run; use the cache to keep CI deterministic when prompts haven't changed.

## Dependencies

### Internal
- `harness/orchestrator.py` — the system under test.
- `harness/models/` — both `EchoModelClient` (stub mode) and `GeminiSubprocessClient` (live mode).
- `harness/schema.py`, `render/pdf.py` — schema validation + page-fit checks.

### External
- **python-dotenv** — `.env` loading for live mode.

<!-- MANUAL: -->
