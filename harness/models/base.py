"""ModelClient protocol + ModelResponse wrapper + EchoModelClient stub.

The wrapper carries usage/model/latency so v2 routing can decide without a
contract change. v1 stays single-model and the metadata is used only for
observability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ModelOutputError(RuntimeError):
    """Raised when a model returns malformed output (parse/validation failure)."""


@dataclass(slots=True)
class ModelResponse[T]:
    data: T
    usage: dict[str, int] = field(default_factory=dict)
    model_name: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelClient(Protocol):
    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[Any],
    ) -> ModelResponse[Any]: ...

    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
    ) -> ModelResponse[str]: ...


class EchoModelClient:
    """Test-only stub. Returns a caller-supplied canned response.

    The orchestrator and downstream code import only `ModelClient` (the
    protocol); this stub is what unit tests inject so AC-8 (pluggability)
    holds without burning Gemini quota.
    """

    def __init__(self, structured_response: Any = None, text_response: str = "") -> None:
        self._structured = structured_response
        self._text = text_response
        self.calls: list[dict[str, Any]] = []

    def set_structured(self, response: Any) -> None:
        self._structured = response

    def set_text(self, response: str) -> None:
        self._text = response

    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[Any],
    ) -> ModelResponse[Any]:
        self.calls.append({"kind": "structured", "system": system, "prompt": prompt, "schema": schema})
        if self._structured is None:
            raise ModelOutputError("EchoModelClient has no structured_response set")
        return ModelResponse(data=self._structured, usage={"input_tokens": 0, "output_tokens": 0}, model_name="echo")

    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
    ) -> ModelResponse[str]:
        self.calls.append({"kind": "text", "system": system, "prompt": prompt})
        return ModelResponse(data=self._text, usage={"input_tokens": 0, "output_tokens": 0}, model_name="echo")
