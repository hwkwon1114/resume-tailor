<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# resume-tailor

## Purpose
Local Streamlit app that tailors a one-page resume to a job description by orchestrating the Gemini CLI behind a deterministic Python harness. The harness wraps LLM generation with mechanical validators (schema, source attribution, field lock, page fit), an LLM-based JD coverage judge, and an orphan-fixer pass, then renders the result to a one-page PDF via WeasyPrint.

## Key Files
| File | Description |
|------|-------------|
| `app.py` | Streamlit UI — JD textarea + JSON/PDF/text resume input → tailored PDF with live quality scores. |
| `tailor.py` | CLI runner around `harness.orchestrator.run` (JD file or stdin → tailored PDF). |
| `convert_resume.py` | One-off utility to convert a custom resume JSON shape into the harness schema. |
| `Makefile` | `app`, `tailor`, `test`, `evals`, `evals-live`, `clear-cache` targets. Exports `DYLD_FALLBACK_LIBRARY_PATH` for WeasyPrint on macOS. |
| `pyproject.toml` | Python 3.12, uv-managed deps (pydantic, streamlit, weasyprint, jinja2, scikit-learn, pdfplumber). pytest config + ruff settings. |
| `README.md` | User-facing setup and usage docs. |
| `resume_converted.json` | Default input resume used by `tailor.py` when `--resume` is omitted. |
| `MASTER RESUME !.pdf` | Source-of-truth resume PDF (intake fixture). |
| `.env.example` | Env var template (loaded by `app.py` and `tailor.py` via python-dotenv). |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `harness/` | LLM generation, validators, judges, retry orchestrator (see `harness/AGENTS.md`). |
| `render/` | HTML template + WeasyPrint PDF renderer (see `render/AGENTS.md`). |
| `evals/` | End-to-end eval runner with content-hashed cache and fixture set (see `evals/AGENTS.md`). |
| `tests/` | Pytest unit + integration tests (see `tests/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Python 3.12 + uv. Use `.venv/bin/python` (not system Python). All Makefile targets assume `.venv` exists.
- WeasyPrint on macOS requires `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`. The Makefile sets this; if you invoke Python directly, export it yourself.
- LLM access is the Gemini CLI (`gemini -p ...`), not an HTTP API key. The harness shells out via `harness.models.gemini_subprocess.GeminiSubprocessClient`. Tests use `EchoModelClient` stub.
- Three input formats are supported (JSON, PDF, text). PDF/text go through `harness.pdf_intake` which itself calls Gemini to parse into the schema.
- **Provenance is sacred**: every output `Bullet.source_ids` must reference real input bullet ids. Breaking this trips `SourceAttributionValidator` and forces a retry. Don't strip or auto-fill source_ids without understanding the fabrication-detection consequences.

### Testing Requirements
- `make test` runs the full pytest suite without LLM calls (everything is stubbed via `EchoModelClient`).
- `make evals` runs end-to-end fixtures in stub mode; `make evals-live` actually calls Gemini.
- Any change to the schema or validators should add or update a test in `tests/`.

### Common Patterns
- **No defaults on Pydantic schemas exposed to google-genai** — google-genai issue #699 rejects schemas with `default=`. See the docstring on `harness/schema.py`. Use `Field(default_factory=...)` only on input-side models that never round-trip to Gemini.
- Prompts live as `.txt` files in `harness/prompts/` so iteration is git-diffable.
- The orchestrator's retry loop **replaces** feedback rather than appending, to bound prompt growth.

## Dependencies

### External
- **google-genai** (≥1.20) — type definitions; runtime LLM access is via the `gemini` CLI subprocess.
- **pydantic** (≥2.9) — schema + validation.
- **streamlit** (≥1.40) — UI.
- **weasyprint** (≥63) — HTML → PDF; needs Pango/Cairo from Homebrew on macOS.
- **jinja2** — HTML template rendering in `render/`.
- **scikit-learn** — TF-IDF for JD term extraction in `harness/ranking.py`.
- **pdfplumber** — PDF text extraction in `harness/pdf_intake.py`.
- **python-dotenv** — `.env` loading in `app.py` / `tailor.py`.

<!-- MANUAL: Custom project notes can be added below -->
