<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# jd

## Purpose
Plain-text job descriptions used as inputs to the eval harness. One JD per file; the filename (sans `.txt`) is the fixture name passed to `load_fixture(...)`. The set is small and intentional — each JD exercises a specific code path in the harness.

## Key Files
| File | Description |
|------|-------------|
| `swe-1.txt` | Strong-match software engineering JD — baseline "happy path" eval. |
| `ml-1.txt` | Strong-match ML/data JD — exercises the ML-leaning bullets in `resume/me.json`. |
| `mismatch-1.txt` | Deliberately weak-match JD — verifies the harness doesn't fabricate to force coverage. |
| `temptation-1.txt` | Mentions techs absent from the resume (rust, wasm, cap'n proto). Paired with `forbidden_terms` in the fixture loader to catch fabrication regressions. |

## For AI Agents

### Working In This Directory
- Add a new fixture: drop `name.txt` here, then register it in `../__init__.py` (`all_baselines()` or a dedicated helper). Without registration, the eval runner won't pick it up.
- Keep JDs around 1KB. Large JDs slow live evals without adding signal.
- Files are read as UTF-8. Don't include BOMs or non-printable characters.

### Testing Requirements
- After adding/editing a JD, run `make evals` (stub mode) to confirm the loader works and the cache key updates.

### Common Patterns
- Trailing whitespace is preserved (the harness does no normalization). Trim manually before saving.
- JD content should be realistic — paste a real posting, then strip identifying employer info.

## Dependencies

### Internal
- Loaded by `evals/fixtures/__init__.py` via `load_fixture(name)`.

### External
- None.

<!-- MANUAL: -->
