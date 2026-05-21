"""Shared text utilities for Gemini transport clients."""
from __future__ import annotations

import re
from typing import Any


def strip_fences(text: str) -> str:
    """Strip markdown code fences + ANSI escape codes from CLI output."""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return text.strip()


def compact_schema_dict(obj: Any) -> Any:
    """Recursively strip Pydantic's auto-generated `title` keys from a JSON schema.

    Pydantic emits a `title` on every property and class definition, defaulting
    to the field name in TitleCase. They are informational for human readers
    but content-free for an LLM — the property key already conveys the same
    signal. Stripping is ~18% off the TailoringResponse schema with no
    semantic loss.
    """
    if isinstance(obj, dict):
        return {k: compact_schema_dict(v) for k, v in obj.items() if k != "title"}
    if isinstance(obj, list):
        return [compact_schema_dict(v) for v in obj]
    return obj
