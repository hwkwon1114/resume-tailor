"""GeminiAcpClient — persistent JSON-RPC channel to `gemini --acp`.

The Gemini CLI's `--acp` flag starts an Agent Client Protocol server
over stdin/stdout. We spawn it ONCE per Python process, run the
initialize handshake once, then send a session/new + session/prompt
pair per generate call. Skips the ~10s of node startup + auth
handshake that subprocess-per-call invocation pays every time.

Fresh session per generate call — the agent's context window otherwise
accumulates across calls and the model treats subsequent prompts as a
conversation, leaking prior content into responses.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import threading
import time

from pydantic import BaseModel, ValidationError

from harness.models._text_utils import compact_schema_dict, strip_fences
from harness.models.base import ModelOutputError, ModelResponse

log = logging.getLogger(__name__)

GEMINI_BIN = os.environ.get("GEMINI_CLI_BIN", "/opt/homebrew/bin/gemini")
DEFAULT_PROMPT_TIMEOUT = 300  # ACP isn't subject to subprocess timeout walls
PROTOCOL_VERSION = 1


class GeminiAcpClient:
    """ModelClient that talks to a persistent `gemini --acp` subprocess."""

    def __init__(self, *, prompt_timeout: int = DEFAULT_PROMPT_TIMEOUT) -> None:
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._next_id = 1
        self._id_lock = threading.Lock()
        self._pending: dict[int, queue.Queue] = {}
        self._notifications: dict[str, list[dict]] = {}
        self._notif_lock = threading.Lock()
        self._initialized = False
        self._prompt_timeout = prompt_timeout
        atexit.register(self._shutdown)

    def _ensure_started(self) -> None:
        if self._initialized:
            return
        self._proc = subprocess.Popen(
            [GEMINI_BIN, "--acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        # Handshake — fires once per process lifetime.
        self._call("initialize", {"protocolVersion": PROTOCOL_VERSION}, timeout=30)
        self._initialized = True

    def _read_loop(self) -> None:
        assert self._proc is not None
        for line in self._proc.stdout:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                log.warning("[acp] non-JSON stdout line: %r", line[:200])
                continue
            if "id" in obj and ("result" in obj or "error" in obj):
                q = self._pending.get(obj["id"])
                if q is not None:
                    q.put(obj)
            elif obj.get("method") == "session/update":
                params = obj.get("params", {})
                sid = params.get("sessionId")
                update = params.get("update")
                if sid is not None and update is not None:
                    with self._notif_lock:
                        self._notifications.setdefault(sid, []).append(update)

    def _call(self, method: str, params: dict, *, timeout: int) -> dict:
        if self._proc is None or self._proc.stdin is None:
            raise ModelOutputError("ACP subprocess not running")
        with self._id_lock:
            request_id = self._next_id
            self._next_id += 1
            q: queue.Queue = queue.Queue()
            self._pending[request_id] = q
        try:
            msg = json.dumps({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": request_id,
            })
            self._proc.stdin.write(msg + "\n")
            self._proc.stdin.flush()
            try:
                resp = q.get(timeout=timeout)
            except queue.Empty as exc:
                raise ModelOutputError(
                    f"ACP {method} timed out after {timeout}s"
                ) from exc
        finally:
            self._pending.pop(request_id, None)
        if "error" in resp:
            raise ModelOutputError(f"ACP {method}: {resp['error'].get('message', resp['error'])}")
        return resp.get("result", {})

    def _prompt(self, text: str) -> str:
        self._ensure_started()
        # Fresh session per call — prevents context bleed across prompts.
        sess = self._call(
            "session/new",
            {"cwd": os.getcwd(), "mcpServers": []},
            timeout=30,
        )
        session_id = sess.get("sessionId")
        if not session_id:
            raise ModelOutputError(f"session/new returned no sessionId: {sess}")
        with self._notif_lock:
            self._notifications[session_id] = []
        self._call(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
            timeout=self._prompt_timeout,
        )
        with self._notif_lock:
            updates = list(self._notifications.pop(session_id, []))
        chunks = [
            u["content"]["text"]
            for u in updates
            if u.get("sessionUpdate") == "agent_message_chunk"
            and isinstance(u.get("content"), dict)
        ]
        return "".join(chunks)

    def generate_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[BaseModel],
    ) -> ModelResponse[BaseModel]:
        schema_blob = json.dumps(compact_schema_dict(schema.model_json_schema()), indent=2)
        full = (
            f"{system}\n\n{prompt}\n\n"
            "Output a single JSON object conforming to this schema "
            "(no prose, no markdown fences, no extra text):\n"
            f"{schema_blob}"
        )
        t0 = time.perf_counter()
        raw = self._prompt(full)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        clean = strip_fences(raw)
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise ModelOutputError(
                f"ACP JSON parse failed: {exc!r} | text={clean[:300]!r}"
            ) from exc
        try:
            data = schema.model_validate(parsed)
        except ValidationError as exc:
            raise ModelOutputError(f"ACP schema validation failed: {exc}") from exc
        return ModelResponse(data=data, model_name="gemini-cli/acp", latency_ms=latency_ms)

    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
    ) -> ModelResponse[str]:
        t0 = time.perf_counter()
        text = self._prompt(f"{system}\n\n{prompt}")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ModelResponse(data=strip_fences(text), model_name="gemini-cli/acp", latency_ms=latency_ms)

    def _shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except Exception:
            pass
