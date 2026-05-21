"""Unit tests for the GeminiAcpClient — persistent JSON-RPC channel.

Heavy use of subprocess.Popen mocks: a fake process exposes pipe-shaped
stdin/stdout so the client's JSON-RPC framing, handshake sequence, and
response-correlation logic can be exercised without hitting the real
Gemini CLI.
"""
from __future__ import annotations

import io
import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from harness.models.base import ModelOutputError


class _Schema(BaseModel):
    score: float


class _FakeProc:
    """Minimal subprocess-like that scripts JSON-RPC interactions.

    Caller writes line-delimited JSON to .stdin (a real io.StringIO).
    Caller-supplied `responder` builds JSON responses written to .stdout,
    which is also a real io.StringIO advanced by an internal reader thread.
    """

    def __init__(self, responder):
        # Initialize state BEFORE starting the reader thread to avoid races.
        self.stdin = io.StringIO()
        self._out_buffer = []
        self._out_cond = threading.Condition()
        self._responder = responder
        self.returncode = None
        self._terminated = False
        self._reader_thread = threading.Thread(target=self._loop, daemon=True)
        self._reader_thread.start()

        class _StdoutFile:
            def __init__(inner_self):
                inner_self._idx = 0
            def __iter__(inner_self):
                while True:
                    with self._out_cond:
                        while inner_self._idx >= len(self._out_buffer) and not self._terminated:
                            self._out_cond.wait(timeout=0.5)
                        if inner_self._idx >= len(self._out_buffer):
                            return
                        line = self._out_buffer[inner_self._idx]
                        inner_self._idx += 1
                    yield line
            def close(inner_self):
                pass
        self.stdout = _StdoutFile()
        self.stderr = io.StringIO()

    def _loop(self):
        last = 0
        while not self._terminated:
            stdin_value = self.stdin.getvalue()
            new = stdin_value[last:]
            last = len(stdin_value)
            for line in new.splitlines():
                if not line.strip():
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Caller's responder may return [response, notification, ...]
                # Each is enqueued as a line on stdout.
                replies = self._responder(req) or []
                with self._out_cond:
                    for r in replies:
                        self._out_buffer.append(json.dumps(r) + "\n")
                    self._out_cond.notify_all()
            threading.Event().wait(0.01)

    def terminate(self):
        self._terminated = True
        with self._out_cond:
            self._out_cond.notify_all()

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.terminate()


def _handshake_responder(session_id="sess-1"):
    """Stock responder: answers initialize + session/new with canned data."""
    def respond(req):
        method = req.get("method")
        rid = req.get("id")
        if method == "initialize":
            return [{"jsonrpc": "2.0", "id": rid,
                     "result": {"protocolVersion": 1, "agentInfo": {"name": "fake", "version": "0"}}}]
        if method == "session/new":
            return [{"jsonrpc": "2.0", "id": rid,
                     "result": {"sessionId": session_id, "modes": {}}}]
        return []
    return respond


def _prompt_responder(session_id, response_text):
    """Responder that handles initialize, session/new, and session/prompt
    by streaming response_text via agent_message_chunk + final stopReason result.
    """
    base = _handshake_responder(session_id)

    def respond(req):
        method = req.get("method")
        rid = req.get("id")
        if method == "session/prompt":
            # Emit a thought chunk (should be ignored), then the message text, then result.
            return [
                {"jsonrpc": "2.0", "method": "session/update",
                 "params": {"sessionId": session_id, "update": {
                     "sessionUpdate": "agent_thought_chunk",
                     "content": {"type": "text", "text": "thinking..."}}}},
                {"jsonrpc": "2.0", "method": "session/update",
                 "params": {"sessionId": session_id, "update": {
                     "sessionUpdate": "agent_message_chunk",
                     "content": {"type": "text", "text": response_text}}}},
                {"jsonrpc": "2.0", "id": rid,
                 "result": {"stopReason": "end_turn", "_meta": {}}},
            ]
        return base(req)
    return respond


def test_generate_text_full_handshake_and_prompt():
    """Lazy init does initialize+session/new, then session/prompt extracts message text."""
    from harness.models.gemini_acp import GeminiAcpClient

    requests_seen: list[dict] = []
    def responder(req):
        requests_seen.append(req)
        return _prompt_responder("s-1", "hello, world")(req)

    with patch("harness.models.gemini_acp.subprocess.Popen") as popen:
        popen.return_value = _FakeProc(responder)
        client = GeminiAcpClient()
        resp = client.generate_text(system="you are helpful", prompt="say hello")

    assert resp.data == "hello, world"
    methods = [r["method"] for r in requests_seen]
    assert methods == ["initialize", "session/new", "session/prompt"], (
        f"expected handshake then prompt, got {methods}"
    )
    # session/prompt must carry the combined system+prompt as a text chunk.
    prompt_req = requests_seen[-1]
    prompt_chunks = prompt_req["params"]["prompt"]
    assert prompt_chunks == [{"type": "text", "text": "you are helpful\n\nsay hello"}]


def test_generate_structured_parses_and_validates_response():
    """Structured output: client strips fences, parses JSON, runs Pydantic validate."""
    from harness.models.gemini_acp import GeminiAcpClient

    response_json = '{"score": 0.87}'
    with patch("harness.models.gemini_acp.subprocess.Popen") as popen:
        popen.return_value = _FakeProc(_prompt_responder("s-1", response_json))
        client = GeminiAcpClient()
        resp = client.generate_structured(system="judge", prompt="score it", schema=_Schema)

    assert isinstance(resp.data, _Schema)
    assert resp.data.score == 0.87


def test_initialization_only_happens_once_across_calls():
    """Two generate calls reuse the same process and handshake — no re-init."""
    from harness.models.gemini_acp import GeminiAcpClient

    requests_seen: list[dict] = []
    def responder(req):
        requests_seen.append(req)
        return _prompt_responder("s-1", "ok")(req)

    with patch("harness.models.gemini_acp.subprocess.Popen") as popen:
        popen.return_value = _FakeProc(responder)
        client = GeminiAcpClient()
        client.generate_text(system="s", prompt="p1")
        client.generate_text(system="s", prompt="p2")

    method_counts = {}
    for r in requests_seen:
        method_counts[r["method"]] = method_counts.get(r["method"], 0) + 1

    assert method_counts.get("initialize") == 1, (
        f"initialize must fire exactly once across calls, got {method_counts}"
    )
    # session/new MAY happen per-call (to avoid context bleed) or once total —
    # both are valid designs. Lock the contract chosen:
    assert method_counts.get("session/prompt") == 2
    assert popen.call_count == 1


def test_fresh_session_per_call_to_avoid_context_bleed():
    """Each generate call uses a NEW session id — prevents the model from
    treating prior calls' content as conversation context."""
    from harness.models.gemini_acp import GeminiAcpClient

    session_ids_issued: list[str] = []
    sessions_used: list[str] = []
    next_session = iter(["s-0", "s-1", "s-2"])

    def responder(req):
        method = req["method"]
        rid = req["id"]
        if method == "initialize":
            return [{"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": 1}}]
        if method == "session/new":
            sid = next(next_session)
            session_ids_issued.append(sid)
            return [{"jsonrpc": "2.0", "id": rid, "result": {"sessionId": sid}}]
        if method == "session/prompt":
            sid = req["params"]["sessionId"]
            sessions_used.append(sid)
            return [
                {"jsonrpc": "2.0", "method": "session/update",
                 "params": {"sessionId": sid, "update": {
                     "sessionUpdate": "agent_message_chunk",
                     "content": {"type": "text", "text": "ok"}}}},
                {"jsonrpc": "2.0", "id": rid,
                 "result": {"stopReason": "end_turn", "_meta": {}}},
            ]
        return []

    with patch("harness.models.gemini_acp.subprocess.Popen") as popen:
        popen.return_value = _FakeProc(responder)
        client = GeminiAcpClient()
        client.generate_text(system="s", prompt="p1")
        client.generate_text(system="s", prompt="p2")

    assert sessions_used[0] != sessions_used[1], (
        f"second call must use a different session id to avoid context bleed; "
        f"got both {sessions_used}"
    )


def test_jsonrpc_error_raises_model_output_error():
    """A JSON-RPC error response surfaces as ModelOutputError, not a silent pass."""
    from harness.models.gemini_acp import GeminiAcpClient

    def responder(req):
        if req["method"] == "session/prompt":
            return [{"jsonrpc": "2.0", "id": req["id"],
                     "error": {"code": -32603, "message": "model crashed"}}]
        return _handshake_responder()(req)

    with patch("harness.models.gemini_acp.subprocess.Popen") as popen:
        popen.return_value = _FakeProc(responder)
        client = GeminiAcpClient()
        with pytest.raises(ModelOutputError, match="model crashed|ACP"):
            client.generate_text(system="s", prompt="p")
