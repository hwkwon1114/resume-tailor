"""PageFit — render to PDF and detect overflow.

Returns the set of overflowing bullet ids (extracted via data-bullet-id
attributes on rendered <li> boxes) plus a font-substitution flag. Pure —
does not mutate the resume; PostJudgePruning is a separate orchestrator
stage.
"""
from __future__ import annotations

from harness.schema import Resume
from harness.validators import ValidationResult


class PageFitValidator:
    name = "page_fit"

    def run(self, output: Resume) -> ValidationResult:
        # Lazy import: weasyprint pulls in cairo/pango at import time.
        from render.pdf import render_to_document

        try:
            doc, font_substituted, substituted_fonts = render_to_document(output, return_font_info=True)
        except Exception as exc:  # noqa: BLE001 — surface render errors as a validator failure
            return ValidationResult(
                name=self.name,
                passed=False,
                score=0.0,
                errors=[f"render failed: {exc}"],
            )

        page_count = len(doc.pages)
        overflow_ids: list[str] = []
        if page_count > 1:
            overflow_ids = _collect_overflow_bullet_ids(doc)

        errors: list[str] = []
        if page_count > 1:
            errors.append(
                f"output rendered to {page_count} pages; bullets on pages 2+: {overflow_ids}"
            )
        if font_substituted:
            errors.append(
                f"font substitution detected ({sorted(substituted_fonts)}); "
                "rendered PDF will not match validator measurements"
            )

        return ValidationResult(
            name=self.name,
            passed=not errors,
            score=1.0 if page_count == 1 else 1.0 / page_count,
            errors=errors,
            payload={
                "pages": page_count,
                "overflow_bullet_ids": overflow_ids,
                "font_substituted": font_substituted,
            },
        )


def _collect_overflow_bullet_ids(doc) -> list[str]:
    """Walk pages 2+ and collect every data-bullet-id attribute on rendered boxes."""
    ids: list[str] = []
    seen: set[str] = set()
    for page in doc.pages[1:]:
        for box in _walk_boxes(page._page_box):
            element = getattr(box, "element", None)
            if element is None:
                continue
            attrs = getattr(element, "attrib", {})
            bid = attrs.get("data-bullet-id") if attrs else None
            if bid and bid not in seen:
                ids.append(bid)
                seen.add(bid)
    return ids


def _walk_boxes(box):
    yield box
    children = getattr(box, "children", None) or ()
    for child in children:
        yield from _walk_boxes(child)
