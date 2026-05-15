<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-10 | Updated: 2026-05-10 -->

# render

## Purpose
HTML/CSS → PDF rendering layer. A single Jinja2 template (`template.html`) styled FAANGPath-style is fed a validated `Resume` and rendered by WeasyPrint. Crucially, **this same renderer is used by both the UI (download) and the `PageFitValidator`** — there is exactly one source of truth for "what fits on one page."

## Key Files
| File | Description |
|------|-------------|
| `pdf.py` | WeasyPrint wrapper. Exposes `render_html(resume)`, `render_to_document(resume)`, and `render_to_pdf(resume)`. Defines `ALLOWED_FONTS` for font-substitution detection (catches silent fallback to glyph-missing fonts). Box-tree access lets validators tag bullets by their rendered geometry. |
| `template.html` | Jinja2 template — header, summary, experience, education, projects, skills. CSS targets one-page US Letter with conservative margins. Each bullet gets a `data-bullet-id` attribute so `PageFitValidator` and the geometry-aware orphan fixer can map rendered boxes back to schema ids. |
| `__init__.py` | Empty package marker. |

## For AI Agents

### Working In This Directory
- **The renderer is shared with the validator.** Any change that affects page count (margins, font size, line height, section spacing) will move the validator's pass/fail boundary. Re-run `make test` after edits.
- WeasyPrint on macOS depends on Homebrew Pango/Cairo. The Makefile sets `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`; if you invoke `render_to_pdf` directly from a one-off script, set it yourself.
- `ALLOWED_FONTS` is an explicit whitelist. If you add a new font-family to the CSS, also add it here — otherwise the renderer's font-substitution detector will (correctly) complain on every render.
- Keep the template `data-bullet-id` attribute on every rendered bullet. The orphan fixer's geometry mode relies on it to map line-spillovers back to specific schema bullets.

### Testing Requirements
- Visual changes: render a fixture and eyeball the PDF before merging — there's no snapshot test, so the human review is the gate.
- Structural changes (new sections, new attributes): update `tests/test_page_fit.py` if the change affects line counts or layout.

### Common Patterns
- Jinja2 autoescape is on (`select_autoescape(["html"])`). User-supplied resume strings are safe to render directly.
- Inline styles in `template.html`, not a separate CSS file — keeping the template self-contained makes WeasyPrint's font/layout debugging easier.

## Dependencies

### Internal
- `harness/schema.py` — `Resume` is the only input.
- Consumed by `harness/validators/page_fit.py`, `harness/judges/orphan_fixer.py` (geometry mode), `app.py`, `tailor.py`.

### External
- **weasyprint** — PDF rendering.
- **jinja2** — HTML templating.

<!-- MANUAL: -->
