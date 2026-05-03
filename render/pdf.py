"""WeasyPrint renderer with bullet-id tagging and font-substitution detection.

`render_to_document(...)` is the single source of truth for both PageFit
validation and the final UI download — the validator's box-tree walk and the
user's PDF come from the same render.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from harness.schema import Resume

_TEMPLATE_DIR = Path(__file__).parent
_ENV = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape(["html"]))

# Allowed: explicit names we render with + CSS generic family names (sans-serif,
# serif, monospace). The generics are NOT substitution — they're declared in our
# template's font-family list as the legitimate fallback chain.
ALLOWED_FONTS = {
    "Helvetica", "Arial", "Helvetica Neue", "Liberation Sans", "DejaVu Sans",
    "Times New Roman", "Times", "Times Roman", "Liberation Serif", "DejaVu Serif",
    "sans-serif", "serif", "monospace", "system-ui", "ui-sans-serif",
}


def render_html(resume: Resume) -> str:
    template = _ENV.get_template("template.html")
    return template.render(resume=resume)


def render_to_pdf(resume: Resume) -> bytes:
    return HTML(string=render_html(resume)).write_pdf()


def render_to_document(resume: Resume, *, return_font_info: bool = False):
    """Render to a weasyprint Document so callers can walk pages/box tree.

    When return_font_info=True, also returns a (font_substituted: bool,
    substituted_fonts: set[str]) tuple for the PageFit validator.
    """
    doc = HTML(string=render_html(resume)).render()
    if not return_font_info:
        return doc
    substituted: set[str] = set()
    for page in doc.pages:
        for box in _walk_boxes(page._page_box):
            for fname in _box_fonts(box):
                if fname and fname not in ALLOWED_FONTS and not _font_is_allowed_subset(fname):
                    substituted.add(fname)
    return doc, bool(substituted), substituted


def _font_is_allowed_subset(fname: str) -> bool:
    fl = fname.lower()
    return any(allowed.lower() in fl for allowed in ALLOWED_FONTS)


def _walk_boxes(box):
    yield box
    children = getattr(box, "children", None) or ()
    for child in children:
        yield from _walk_boxes(child)


def _box_fonts(box):
    style = getattr(box, "style", None)
    if style is None:
        return ()
    family = style.get("font_family", ())
    return [f for f in family if isinstance(f, str)]
