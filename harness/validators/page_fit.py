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

        utilization = 1.0
        if page_count == 1:
            utilization = _measure_utilization(doc)

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
        if page_count == 1 and utilization < 0.90:
            errors.append(
                f"page only {utilization:.0%} utilized — include a 3rd/4th experience role or add more bullets"
            )

        return ValidationResult(
            name=self.name,
            passed=page_count == 1 and not font_substituted,
            score=1.0 if page_count == 1 else 1.0 / page_count,
            errors=errors,
            payload={
                "pages": page_count,
                "overflow_bullet_ids": overflow_ids,
                "font_substituted": font_substituted,
                "page_utilization_pct": round(utilization * 100),
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


def _measure_utilization(doc) -> float:
    """Return fraction of page height occupied by content (0.0–1.0).

    Uses TextBox instances (boxes with a non-empty .text attribute) — the
    atomic rendered text units. Container/layout boxes (BlockBox, LineBox,
    etc.) are skipped because many span the full page height even when the
    page is sparsely filled, causing false 100% readings.
    """
    import logging
    _log = logging.getLogger(__name__)

    page_box = doc.pages[0]._page_box
    page_height = page_box.height
    if not page_height:
        return 1.0
    last_y = 0.0
    last_text = ""
    for box in _walk_boxes(page_box):
        text = getattr(box, "text", None)
        if not text or not text.strip():
            continue
        h = getattr(box, "height", 0) or 0
        y2 = (getattr(box, "position_y", 0) or 0) + h
        if y2 > last_y:
            last_y = y2
            last_text = text.strip()[:50]
    # Also record first content y to understand coordinate origin
    first_y = min(
        (getattr(b, "position_y", 0) or 0) + (getattr(b, "height", 0) or 0)
        for b in _walk_boxes(page_box)
        if getattr(b, "text", None) and (getattr(b, "text", None) or "").strip()
    ) if last_y > 0 else 0.0

    util = min(last_y / page_height, 1.0)
    _log.info(
        "utilization: %.0f%% (page_h=%.1f page_pos_y=%.1f first_y=%.1f last_y=%.1f last_text=%r)",
        util * 100, page_height,
        getattr(page_box, "position_y", -1),
        first_y, last_y, last_text,
    )
    return util


def _walk_boxes(box):
    yield box
    children = getattr(box, "children", None) or ()
    for child in children:
        yield from _walk_boxes(child)
