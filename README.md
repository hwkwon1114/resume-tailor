# Resume Tailor

Local Streamlit app: paste a job description + upload your resume → get a one-page tailored resume PDF.

Runs entirely through the **Gemini CLI** using your Google account — no API keys or billing required.

## How it works

Each generation attempt runs this pipeline:

1. **Generate** — Gemini writes a tailored resume from your JD + input resume + any retry feedback
2. **Mechanical validators** — schema check, source attribution (every bullet must cite its source), field lock (employer/dates can't change), page fit
3. **JD Coverage Judge** — Gemini scores 0–1 how well the resume semantically covers the JD; uncovered requirements feed back into the next retry
4. **Orphan Fixer** — detects bullets that spill 1–3 words onto a second line and rewrites them to either a clean single line or a full two-liner
5. **Page trim** — mechanically drops lowest-JD-ranked bullets if the resume overflows one page

Retries happen when schema, field lock, source attribution, or JD coverage fail. Page fitting is handled deterministically, not by retrying the LLM.

## Setup

### 1. Install the Gemini CLI

```bash
brew install gemini-cli        # macOS via Homebrew
# OR
npm i -g @google/gemini-cli   # via npm
```

Log in with your Google account (this is what provides model access):

```bash
gemini
# Follow the browser auth flow on first launch
```

Verify it works:

```bash
gemini -p "hello"
# → Hello
```

### 2. Install Python dependencies

This project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. WeasyPrint needs Pango/Cairo from Homebrew on macOS.

```bash
uv python install 3.12
uv venv --python 3.12
uv sync
brew install pango gdk-pixbuf libffi   # one-time, for WeasyPrint PDF rendering
```

### 3. Prepare your resume

The app accepts three input formats:

| Tab | Format | Notes |
|-----|--------|-------|
| Upload JSON | Harness schema JSON | See `convert_resume.py` to convert from other formats |
| Upload PDF | Any text-based PDF | Gemini extracts and structures it automatically |
| Paste text | Plain text copied from any source | Gemini parses it into structured data |

If you have a resume in a different JSON format, run:

```bash
# Edit convert_resume.py to match your schema, then:
uv run python convert_resume.py
# Outputs resume_converted.json — upload via the JSON tab
```

### 4. Run

```bash
make app          # launches Streamlit at http://localhost:8501
make test         # run unit tests (no LLM calls)
make evals        # stub-mode end-to-end eval
make evals-live   # live Gemini CLI eval on all fixtures
```

## Which Gemini model is used

The app calls the local `gemini` binary (whichever model it's configured to use). By default the Gemini CLI uses your account's current default model. You can check with `gemini -p "what model are you?"`.

The generation and JD coverage judge both use the same Gemini CLI subprocess. To use a specific model, set it in the Gemini CLI settings (`~/.gemini/settings.json`).

## Repo layout

```
harness/
  generate.py            # Generator — calls Gemini CLI with TailoringResponse schema
  orchestrator.py        # Main loop: generate → validate → jd-judge → orphan-fix
  schema.py              # Pydantic Resume / Bullet models
  ranking.py             # JD term extraction + bullet pre-ranking
  pdf_intake.py          # PDF/text → Resume (for upload/paste input)
  models/
    base.py              # ModelClient protocol + EchoModelClient stub
    gemini_subprocess.py # Gemini CLI subprocess client (primary)
  validators/            # SchemaCheck, SourceAttribution, FieldLock, PageFit, JDCoverage
  judges/
    jd_coverage_judge.py # LLM semantic JD coverage score
    orphan_fixer.py      # Detect + rewrite danger-zone bullets
    voice_check.py       # Mechanical stylistic drift (informational)
  prompts/               # system_generate.txt, feedback_template.txt, system_judge.txt
render/
  template.html          # FAANGPath-style HTML/CSS resume template
  pdf.py                 # WeasyPrint render
evals/
  fixtures/              # JD + resume fixtures for eval runs
  run_evals.py           # End-to-end eval runner
  cache.py               # Content-hashed eval cache
tests/                   # Unit + integration tests
app.py                   # Streamlit UI
convert_resume.py        # One-time utility: custom JSON → harness schema
```

## Quality metrics shown in the UI

| Metric | What it measures |
|--------|-----------------|
| Pages | Must be 1 |
| JD coverage | Gemini's semantic score of how well the resume covers the JD (0–1, floor 0.5) |
| Schema valid | Pydantic validation passed |
| Source attribution | Every output bullet cites the input bullet(s) it derives from |
| Voice drift | Stylistic shift from the original (informational) |

## Quota

The Gemini CLI uses your Google account quota, which is significantly higher than the API key free tier. A typical tailoring run makes 2 Gemini calls per attempt (generate + JD coverage judge) plus 1 optional orphan-fix call. With 3 retries max, that's at most 9 calls per resume.
