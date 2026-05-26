"""ClaudeSubprocessClient — calls the local `claude` CLI as a subprocess.

Uses your Claude Pro/Max subscription auth (whatever Claude Code is logged
into) so no API key needed. Token usage counts against your subscription
quota; the JSON envelope's `total_cost_usd` is the API-equivalent display
value, not an actual charge.

Key differences from GeminiSubprocessClient:
  - Claude CLI supports `--json-schema` natively, so structured-output
    validation happens CLI-side (no fence-stripping or post-hoc parsing).
  - System prompt goes via `--system-prompt`, user prompt via stdin.
  - We disable tools and slash commands — this is a pure LLM call, not
    an agent invocation. Without `--tools ""` Claude could decide to
    read project files mid-call, which would be wrong for a black-box
    model client.

Requirements:
  - `claude` in PATH and authenticated (Claude Code's normal OAuth flow).
  - CLAUDE_CLI_BIN env var to override the binary path if needed.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

from pydantic import BaseModel, ValidationError

from harness.models._text_utils import compact_schema_dict
from harness.models.base import ModelOutputError, ModelResponse

CLAUDE_BIN = os.environ.get("CLAUDE_CLI_BIN", "claude")
DEFAULT_MODEL = os.environ.get("CLAUDE_CLI_MODEL", "sonnet")
DEFAULT_TIMEOUT = 180


def _read_timeout_env() -> int:
    raw = os.environ.get("CLAUDE_CLI_TIMEOUT")
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ModelOutputError(
            f"CLAUDE_CLI_TIMEOUT must be an integer (seconds), got {raw!r}"
        ) from exc
    if value <= 0:
        raise ModelOutputError(f"CLAUDE_CLI_TIMEOUT must be positive, got {value}")
    return value


def _run(*, system: str, user_prompt: str, json_schema: str | None) -> dict:
    """Invoke `claude -p` and return the parsed JSON envelope.

    Returns the full envelope dict; callers extract `result` (text) or
    `structured_output` (structured) as needed.
    """
    resolved = _read_timeout_env()
    cmd = [
        CLAUDE_BIN, "-p",
        "--model", DEFAULT_MODEL,
        "--output-format", "json",
        "--no-session-persistence",
        "--system-prompt", system,
        "--tools", "",
        "--disable-slash-commands",
    ]
    if json_schema is not None:
        cmd += ["--json-schema", json_schema]

    try:
        result = subprocess.run(
            cmd,
            input=user_prompt,
            capture_output=True,
            text=True,
            timeout=resolved,
        )
    except subprocess.TimeoutExpired as exc:
        raise ModelOutputError(f"Claude CLI timed out after {resolved}s") from exc
    except FileNotFoundError as exc:
        raise ModelOutputError(
            f"Claude CLI not found at {CLAUDE_BIN!r}. "
            "Install Claude Code (https://claude.com/claude-code) and sign in."
        ) from exc

    if result.returncode != 0:
        err = (result.stderr or "").strip()
        raise ModelOutputError(f"Claude CLI exited {result.returncode}: {err}")

    try:
        envelope = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(
            f"Claude CLI JSON envelope parse failed: {exc!r} | "
            f"stdout={result.stdout[:300]!r}"
        ) from exc

    if envelope.get("is_error"):
        # The CLI returns errors (auth, rate-limit, etc.) inside `result`.
        raise ModelOutputError(f"Claude CLI error: {envelope.get('result')!r}")

    return envelope


class ClaudeSubprocessClient:
    """ModelClient that delegates to the local `claude` CLI binary.

    Uses Claude Code's OAuth auth → consumes your Pro/Max subscription
    quota, not Anthropic API billing.
    """

    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[BaseModel],
    ) -> ModelResponse[BaseModel]:
        schema_blob = json.dumps(compact_schema_dict(schema.model_json_schema()))
        t0 = time.perf_counter()
        envelope = _run(system=system, user_prompt=prompt, json_schema=schema_blob)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw = envelope["structured_output"]
        try:
            data = schema.model_validate(raw)
        except ValidationError as exc:
            raise ModelOutputError(f"Claude CLI schema validation failed: {exc}") from exc

        return ModelResponse(
            data=data,
            model_name=f"claude-cli/{DEFAULT_MODEL}",
            latency_ms=latency_ms,
        )

    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
    ) -> ModelResponse[str]:
        t0 = time.perf_counter()
        envelope = _run(system=system, user_prompt=prompt, json_schema=None)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ModelResponse(
            data=(envelope.get("result") or "").strip(),
            model_name=f"claude-cli/{DEFAULT_MODEL}",
            latency_ms=latency_ms,
        )
