<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# fixtures

## Purpose
The canonical JD + resume fixture set the evals run against. One small, well-curated input resume (`resume/me.json`) is paired with several JDs (`jd/*.txt`) chosen to exercise different code paths: a strong-match SWE role, a strong-match ML role, a deliberate mismatch (to test that the harness doesn't fabricate), and a "temptation" JD that mentions techs not in the resume (to test that fabrication-prevention holds under pressure).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | `Fixture` dataclass + `load_fixture(name, resume_filename, forbidden_terms)`. Also `all_baselines()` and `temptation_fixture()` helpers used by the eval runner. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `jd/` | One `.txt` per JD fixture. Files: `swe-1.txt`, `ml-1.txt`, `mismatch-1.txt`, `temptation-1.txt`. |
| `resume/` | `me.json` — canonical input resume in harness schema format. |

## For AI Agents

### Working In This Directory
- **The `temptation-1` fixture has `forbidden_terms`** like `rust`, `webassembly`, `wasm` — these aren't in `resume/me.json`. The eval asserts the tailored output never introduces them. If a fabrication-prevention regression lands, this is the fixture that catches it.
- New JD fixtures: drop in `jd/<name>.txt`, then register in `all_baselines()` or as a dedicated helper in `__init__.py`. Keep the JD around 1KB — large JDs slow live evals without adding signal.
- Don't edit `resume/me.json` casually — every baseline fixture is paired with it, so changing it perturbs all metrics. If you need a different resume, add a new file under `resume/` and pass `resume_filename` to `load_fixture`.

### Testing Requirements
- After adding a fixture, run `make evals` (stub mode) to confirm the loader picks it up. Stub-mode canned responses for the new fixture may need to be added to `evals/run_evals.py`.

### Common Patterns
- JD text is read as UTF-8. Strip trailing whitespace; the harness doesn't normalize.
- Resume JSON: bullet ids may be omitted — `autopopulate_bullet_ids()` stamps them deterministically at load time.

## Dependencies

### Internal
- `harness/schema.py` — `Resume` + `autopopulate_bullet_ids()`.

### External
- None.

<!-- MANUAL: -->
