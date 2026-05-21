"""Unit tests for schema-compaction shared by both Gemini transport clients."""
from __future__ import annotations

from harness.models._text_utils import compact_schema_dict as _compact_schema_dict


def test_compact_schema_strips_title_keys():
    """All `title` keys must be removed at every depth.

    Pydantic emits `title` on every property and class definition; the LLM
    consumer already has the same signal via property names and $def keys.
    Stripping is ~18% off the real TailoringResponse schema in production.
    """
    schema = {
        "title": "Root",
        "properties": {
            "field_a": {"title": "Field A", "type": "string"},
            "nested": {
                "title": "Nested",
                "properties": {"x": {"title": "X", "type": "integer"}},
            },
        },
        "$defs": {
            "Item": {"title": "Item", "type": "object", "properties": {
                "id": {"title": "Id", "type": "string"},
            }},
        },
    }

    compacted = _compact_schema_dict(schema)

    def has_title_anywhere(obj):
        if isinstance(obj, dict):
            if "title" in obj:
                return True
            return any(has_title_anywhere(v) for v in obj.values())
        if isinstance(obj, list):
            return any(has_title_anywhere(v) for v in obj)
        return False

    assert not has_title_anywhere(compacted), (
        "every `title` key must be stripped recursively — top-level, nested "
        "properties, and $defs entries all included"
    )


def test_compact_schema_preserves_load_bearing_fields():
    """Only `title` should be removed; descriptions, types, required arrays,
    defaults, $defs, anyOf — everything semantically load-bearing — must survive.
    """
    schema = {
        "title": "Foo",
        "description": "Root description — semantically load-bearing",
        "type": "object",
        "properties": {
            "id": {"title": "Id", "type": "string", "description": "Bullet id"},
            "phone": {
                "title": "Phone",
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
            },
            "tags": {"title": "Tags", "type": "array", "items": {"type": "string"}},
        },
        "required": ["id"],
        "$defs": {
            "Bar": {"title": "Bar", "type": "object", "properties": {
                "name": {"title": "Name", "type": "string"},
            }},
        },
    }

    compacted = _compact_schema_dict(schema)

    assert compacted["description"] == "Root description — semantically load-bearing"
    assert compacted["type"] == "object"
    assert compacted["properties"]["id"]["type"] == "string"
    assert compacted["properties"]["id"]["description"] == "Bullet id"
    assert compacted["properties"]["phone"]["anyOf"] == [
        {"type": "string"}, {"type": "null"},
    ]
    assert compacted["properties"]["phone"]["default"] is None
    assert compacted["properties"]["tags"]["items"] == {"type": "string"}
    assert compacted["required"] == ["id"]
    assert "Bar" in compacted["$defs"]
    assert compacted["$defs"]["Bar"]["properties"]["name"]["type"] == "string"


def test_compact_schema_does_not_strip_title_as_value():
    """`title` only matters as a key, not as a value embedded in a string field.

    Edge case: a description that mentions the word "title" must remain intact.
    """
    schema = {
        "properties": {
            "kind": {
                "title": "Kind",
                "description": "Set this to the document title or heading.",
                "type": "string",
            },
        },
    }
    compacted = _compact_schema_dict(schema)
    assert "title" not in compacted["properties"]["kind"]
    assert "title" in compacted["properties"]["kind"]["description"]


def test_compact_schema_on_real_tailoring_response_reduces_size():
    """Sanity check on the production schema: stripping titles produces a
    measurable size reduction without breaking the schema's validity.

    Locks an empirical baseline so any future Pydantic behavior change that
    might dilute the savings is caught.
    """
    import json

    from harness.generate import TailoringResponse

    original = TailoringResponse.model_json_schema()
    compacted = _compact_schema_dict(original)

    original_size = len(json.dumps(original))
    compacted_size = len(json.dumps(compacted))
    savings_pct = (original_size - compacted_size) * 100 / original_size

    # Empirical baseline: ~18% reduction. Floor at 10% to absorb minor Pydantic
    # version drift while still catching a regression that would defeat the
    # purpose of the optimization.
    assert savings_pct >= 10, (
        f"compaction should remove ≥10% of schema size; got {savings_pct:.1f}% "
        f"(original={original_size}, compacted={compacted_size})"
    )
    # And the structural skeleton must still be present.
    assert "properties" in compacted
    assert "$defs" in compacted
    assert "required" in compacted
