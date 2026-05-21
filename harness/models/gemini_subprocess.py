"""GeminiSubprocessClient — calls the local Gemini CLI as a subprocess.

Uses your Google account auth (whatever `gemini` is logged into) so no API key
needed. Structured output is handled the same way as the DeepSeek client:
inject the JSON schema into the prompt, strip markdown, parse, and validate.

Requirements:
  - Gemini CLI installed and logged in: `gemini` works in your terminal.
  - GEMINI_CLI_BIN env var or default /opt/homebrew/bin/gemini.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

from pydantic import BaseModel, ValidationError

from harness.models._text_utils import compact_schema_dict, strip_fences
from harness.models.base import ModelOutputError, ModelResponse

GEMINI_BIN = os.environ.get("GEMINI_CLI_BIN", "/opt/homebrew/bin/gemini")
DEFAULT_TIMEOUT = 120


def _read_timeout_env() -> int:
    raw = os.environ.get("GEMINI_CLI_TIMEOUT")
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ModelOutputError(
            f"GEMINI_CLI_TIMEOUT must be an integer (seconds), got {raw!r}"
        ) from exc
    if value <= 0:
        raise ModelOutputError(f"GEMINI_CLI_TIMEOUT must be positive, got {value}")
    return value


def _run(prompt: str) -> str:
    """Pipe prompt to `gemini` via stdin and return stdout."""
    resolved = _read_timeout_env()
    try:
        result = subprocess.run(
            [GEMINI_BIN],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=resolved,
        )
    except subprocess.TimeoutExpired as exc:
        raise ModelOutputError(f"Gemini CLI timed out after {resolved}s") from exc
    except FileNotFoundError as exc:
        raise ModelOutputError(
            f"Gemini CLI not found at {GEMINI_BIN}. "
            "Install with: brew install gemini-cli or npm i -g @google/gemini-cli"
        ) from exc
    if result.returncode != 0:
        err = (result.stderr or "").strip()
        raise ModelOutputError(f"Gemini CLI exited {result.returncode}: {err}")
    return result.stdout.strip()


class GeminiSubprocessClient:
    """ModelClient that delegates to the local `gemini` CLI binary."""

    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[BaseModel],
    ) -> ModelResponse[BaseModel]:
        schema_blob = json.dumps(compact_schema_dict(schema.model_json_schema()), indent=2)
        full_prompt = (
            f"{system}\n\n"
            f"{prompt}\n\n"
            f"Output a single JSON object conforming to this schema "
            f"(no prose, no markdown fences, no extra text):\n{schema_blob}"
        )
        t0 = time.perf_counter()
        raw_text = _run(full_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        clean = strip_fences(raw_text)
        try:
            raw = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ModelOutputError(
                f"Gemini CLI JSON parse failed: {exc!r} | text={clean[:300]!r}"
            ) from exc
        try:
            data = schema.model_validate(raw)
        except ValidationError as exc:
            raise ModelOutputError(f"Gemini CLI schema validation failed: {exc}") from exc

        return ModelResponse(
            data=data,
            model_name="gemini-cli/subprocess",
            latency_ms=latency_ms,
        )

    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
    ) -> ModelResponse[str]:
        full_prompt = f"{system}\n\n{prompt}"
        t0 = time.perf_counter()
        text = _run(full_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ModelResponse(
            data=strip_fences(text),
            model_name="gemini-cli/subprocess",
            latency_ms=latency_ms,
        )
