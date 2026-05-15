<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# resume

## Purpose
Canonical input resumes (harness-schema JSON) paired with the JD fixtures next door. Currently a single file (`me.json`) is shared across all baseline fixtures so eval results are comparable JD-to-JD.

## Key Files
| File | Description |
|------|-------------|
| `me.json` | The canonical input resume — harness schema (`Resume` with `contact`, `summary`, `experience`, `education`, `projects`, `skills`). Bullet ids may be omitted; they're stamped deterministically by `autopopulate_bullet_ids()` at load time. |

## For AI Agents

### Working In This Directory
- **Edit `me.json` cautiously.** Every baseline fixture is paired with it, so perturbing it changes all eval metrics simultaneously. If you need a different resume profile, add a new file here (e.g. `senior-ml.json`) and pass `resume_filename` to `load_fixture(...)`.
- Schema rules from `harness/schema.py` apply: bullet ids must be unique, `source_ids` is required on every bullet (input-side defaults to `[]`), and section ids must be unique.
- Don't commit personally identifying info you don't want public.

### Testing Requirements
- After editing, run `python -c "from evals.fixtures import all_baselines; all_baselines()"` to confirm the loader still parses it.
- `make evals` (stub mode) is the fast end-to-end smoke test.

### Common Patterns
- Bullet ids can be omitted in source; `autopopulate_bullet_ids()` stamps `{kind}-{section_idx}-b{bullet_idx}`.
- `summary` is a single string; long summaries get aggressively rewritten by the LLM and may be hard to recognize in output.

## Dependencies

### Internal
- Loaded by `evals/fixtures/__init__.py`, validated via `harness/schema.py:Resume`.

### External
- None.

<!-- MANUAL: -->
