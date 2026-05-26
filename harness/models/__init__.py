import os

from harness.models.base import EchoModelClient, ModelClient, ModelOutputError, ModelResponse

__all__ = [
    "EchoModelClient", "ModelClient", "ModelOutputError", "ModelResponse",
    "get_default_model", "get_default_judge_model",
]


def _build(transport: str) -> ModelClient:
    if transport == "gemini-acp":
        from harness.models.gemini_acp import GeminiAcpClient
        return GeminiAcpClient()
    if transport == "gemini-subprocess":
        from harness.models.gemini_subprocess import GeminiSubprocessClient
        return GeminiSubprocessClient()
    if transport == "claude-subprocess":
        from harness.models.claude_subprocess import ClaudeSubprocessClient
        return ClaudeSubprocessClient()
    raise ValueError(
        f"Unknown transport {transport!r}. Expected one of: "
        "gemini-acp, gemini-subprocess, claude-subprocess."
    )


def get_default_model() -> ModelClient:
    # MODEL_TRANSPORT picks the LLM used for generation + intake parsing.
    # Defaults to gemini-acp (persistent JSON-RPC channel — ~10s startup
    # amortized across all calls, no subprocess 120s timeout wall).
    return _build(os.environ.get("MODEL_TRANSPORT", "gemini-acp"))


def get_default_judge_model() -> ModelClient:
    # JUDGE_TRANSPORT picks the LLM used for jd-coverage + fabrication
    # judging. Falls back to MODEL_TRANSPORT when unset — so the hybrid
    # setup (e.g. Gemini for generation, Claude for judging) is just
    # JUDGE_TRANSPORT=claude-subprocess with no other env vars set.
    return _build(os.environ.get("JUDGE_TRANSPORT") or os.environ.get("MODEL_TRANSPORT", "gemini-acp"))
