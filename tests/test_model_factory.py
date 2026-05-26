"""Dispatch tests for get_default_model / get_default_judge_model.

These tests do not invoke any LLM — they only assert that the right
ModelClient subclass comes out of the factory given the env vars. The
actual client implementations are exercised by their own test files.
"""
from __future__ import annotations

import pytest

from harness.models import get_default_judge_model, get_default_model
from harness.models.claude_subprocess import ClaudeSubprocessClient
from harness.models.gemini_acp import GeminiAcpClient
from harness.models.gemini_subprocess import GeminiSubprocessClient


@pytest.fixture(autouse=True)
def _clear_transport_env(monkeypatch):
    """Clear MODEL_TRANSPORT and JUDGE_TRANSPORT so each test starts clean."""
    monkeypatch.delenv("MODEL_TRANSPORT", raising=False)
    monkeypatch.delenv("JUDGE_TRANSPORT", raising=False)


def test_default_model_is_gemini_acp_when_no_env_var():
    assert isinstance(get_default_model(), GeminiAcpClient)


def test_model_transport_gemini_subprocess(monkeypatch):
    monkeypatch.setenv("MODEL_TRANSPORT", "gemini-subprocess")
    assert isinstance(get_default_model(), GeminiSubprocessClient)


def test_model_transport_claude_subprocess(monkeypatch):
    monkeypatch.setenv("MODEL_TRANSPORT", "claude-subprocess")
    assert isinstance(get_default_model(), ClaudeSubprocessClient)


def test_unknown_transport_raises(monkeypatch):
    monkeypatch.setenv("MODEL_TRANSPORT", "deepseek-v9")
    with pytest.raises(ValueError, match="Unknown transport"):
        get_default_model()


def test_judge_falls_back_to_model_transport(monkeypatch):
    """When JUDGE_TRANSPORT is unset, the judge follows MODEL_TRANSPORT."""
    monkeypatch.setenv("MODEL_TRANSPORT", "claude-subprocess")
    assert isinstance(get_default_judge_model(), ClaudeSubprocessClient)


def test_judge_overrides_model_for_hybrid_setup(monkeypatch):
    """The hybrid setup: Gemini for generation, Claude for judging.
    JUDGE_TRANSPORT takes precedence over MODEL_TRANSPORT for the judge.
    """
    monkeypatch.setenv("MODEL_TRANSPORT", "gemini-acp")
    monkeypatch.setenv("JUDGE_TRANSPORT", "claude-subprocess")
    assert isinstance(get_default_model(), GeminiAcpClient)
    assert isinstance(get_default_judge_model(), ClaudeSubprocessClient)
