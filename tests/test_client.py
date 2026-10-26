"""Tests for the LLM client."""

import subprocess
from pathlib import Path

import pytest

from unipred.llm.client import LLMClient


def _fake_codex(calls):
    """A stand-in for subprocess.run that behaves like `codex exec`."""

    def _run(command, **kwargs):
        calls.append((command, kwargs["input"]))
        reply_path = Path(command[command.index("-o") + 1])
        reply_path.write_text("ADD: Pick\nDEL:\nWHY: fake", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    return _run


def test_codex_provider_and_cache(tmp_path, monkeypatch):
    """The codex provider runs codex exec read-only with the model and
    effort; replies are cached per request and attempt."""
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_codex(calls))
    client = LLMClient("codex", "gpt-5.6-luna", 0.2, 100, 30, 2,
                       cache_dir=str(tmp_path), reasoning_effort="low")
    assert client.complete("sys", "user", "effect_proposal") == \
        "ADD: Pick\nDEL:\nWHY: fake"
    command, prompt = calls[0]
    assert command[:2] == ["codex", "exec"]
    assert command[command.index("--model") + 1] == "gpt-5.6-luna"
    assert 'model_reasoning_effort="low"' in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert prompt.startswith("sys\n\nuser")
    client.complete("sys", "user", "effect_proposal")
    assert len(calls) == 1 and client.usage["cached"] == 1
    client.complete("sys", "user", "effect_proposal", attempt=1)
    assert len(calls) == 2


def test_codex_failure(tmp_path, monkeypatch):
    """A failing codex exec raises after the retries."""
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.
        CompletedProcess(command, 1, "", "not logged in"))
    monkeypatch.setattr("time.sleep", lambda _: None)
    client = LLMClient("codex", "gpt-5.6-luna", 0.2, 100, 30, 2)
    with pytest.raises(RuntimeError, match="not logged in"):
        client.complete("sys", "user", "effect_proposal")


def test_unknown_provider_and_missing_key(monkeypatch):
    """Unknown providers and missing API keys are errors."""
    with pytest.raises(ValueError):
        LLMClient("other", "m", 0.2, 100, 30, 1)
    monkeypatch.setattr("unipred.llm.client._load_env_files", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient("openai", "gpt-4o", 0.2, 100, 30, 1)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        client.complete("sys", "user", "pddl_completion")
