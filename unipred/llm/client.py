"""A small chat-completion client for the LLM providers used by UniPred.

Providers: "openai" (OPENAI_API_KEY), "qwen" (DASHSCOPE_API_KEY, through
the OpenAI-compatible DashScope endpoint), "gemini" (GEMINI_API_KEY or
GOOGLE_API_KEY), "codex" (the local Codex CLI, ``codex exec``, with its own
login) and "mock" (no network; see unipred/llm/mock.py). Keys are read from
the environment, after loading ``.env.local`` and ``.env`` from the working
directory and the repository root.

Responses are cached on disk under ``CFG.llm_cache_dir``, keyed by the
provider, model, sampling parameters, prompt and attempt number, so a rerun
with the same seed replays the same answers without new API calls. Every
request and its reply can also be appended to a JSON-lines log (log_path).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from predicators.settings import CFG

_REPO_ROOT = Path(__file__).resolve().parents[2]
_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_KEY_ENV_VARS = {
    "openai": ("OPENAI_API_KEY", ),
    "qwen": ("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}


def _load_env_files() -> None:
    from dotenv import load_dotenv  # pylint: disable=import-outside-toplevel
    for directory in (Path.cwd(), _REPO_ROOT):
        for name in (".env.local", ".env"):
            path = directory / name
            if path.exists():
                load_dotenv(path, override=False)


def _api_key(provider: str) -> str:
    _load_env_files()
    for var in _KEY_ENV_VARS[provider]:
        key = os.environ.get(var)
        if key:
            return key
    raise RuntimeError(f"No API key for provider '{provider}': set one of "
                       f"{', '.join(_KEY_ENV_VARS[provider])} in the "
                       "environment or in .env.local.")


class LLMClient:
    """Chat completions with retries, an on-disk cache and usage counts."""

    def __init__(self,
                 provider: str,
                 model: str,
                 temperature: float,
                 max_tokens: int,
                 timeout: float,
                 retry_attempts: int,
                 base_url: str = "",
                 cache_dir: str = "",
                 mock_responses: str = "",
                 reasoning_effort: str = "low",
                 codex_command: str = "codex",
                 log_path: str = "") -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.base_url = base_url
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._mock_responses = mock_responses
        self.reasoning_effort = reasoning_effort
        self._codex_command = codex_command
        self._log_path = Path(log_path) if log_path else None
        self._backend: Any = None
        self.usage = {
            "calls": 0,
            "cached": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0
        }
        if self.provider not in ("openai", "qwen", "gemini", "codex", "mock"):
            raise ValueError(f"Unknown LLM provider: {provider}")

    @classmethod
    def from_settings(cls,
                      temperature: Optional[float] = None,
                      log_path: str = "") -> LLMClient:
        """Create a client from the llm_* settings."""
        return cls(provider=CFG.llm_provider,
                   model=CFG.llm_model,
                   temperature=(CFG.llm_temperature
                                if temperature is None else temperature),
                   max_tokens=CFG.llm_max_tokens,
                   timeout=CFG.llm_timeout,
                   retry_attempts=CFG.llm_retry_attempts,
                   base_url=CFG.llm_base_url,
                   cache_dir=CFG.llm_cache_dir,
                   mock_responses=CFG.llm_mock_responses,
                   reasoning_effort=CFG.llm_reasoning_effort,
                   codex_command=CFG.llm_codex_command,
                   log_path=log_path)

    def complete(self,
                 system: str,
                 user: str,
                 task: str,
                 attempt: int = 0) -> str:
        """Return the model's reply to a system and a user message.

        The task names the kind of request (for example
        "pddl_completion"); the mock provider answers by task. Retries of
        the same request must pass increasing attempt numbers, so that
        they are cached separately.
        """
        request = {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "system": system,
            "user": user,
            "task": task,
            "attempt": attempt,
        }
        cache_path = self._cache_path(request)
        cached = False
        if cache_path is not None and cache_path.exists():
            cached = True
            self.usage["cached"] += 1
            with open(cache_path, "r", encoding="utf-8") as f:
                response = json.load(f)["response"]
        else:
            response = self._call_with_retries(system, user, task)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({**request, "response": response}, f, indent=1)
        if self._log_path is not None:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({
                        "task": task,
                        "attempt": attempt,
                        "cached": cached,
                        "system": system,
                        "user": user,
                        "response": response
                    }) + "\n")
        return response

    def _cache_path(self, request: Dict[str, Any]) -> Optional[Path]:
        if self._cache_dir is None or self.provider == "mock":
            return None
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
        return self._cache_dir / self.provider / f"{digest}.json"

    def _call_with_retries(self, system: str, user: str, task: str) -> str:
        for attempt in range(self.retry_attempts):
            try:
                return self._call(system, user, task)
            except Exception as err:  # pylint: disable=broad-except
                if self.provider == "mock" or \
                        attempt == self.retry_attempts - 1:
                    raise
                wait = 2**attempt
                logging.warning(f"LLM request failed ({err}); retrying in "
                                f"{wait} s.")
                time.sleep(wait)
        raise RuntimeError("retry_attempts must be positive")

    def _call(self, system: str, user: str, task: str) -> str:
        self.usage["calls"] += 1
        if self.provider == "mock":
            from unipred.llm.mock import mock_response  # pylint: disable=import-outside-toplevel
            return mock_response(system, user, task, self._mock_responses)
        if self.provider in ("openai", "qwen"):
            return self._call_openai(system, user)
        if self.provider == "codex":
            return self._call_codex(system, user)
        return self._call_gemini(system, user)

    def _call_openai(self, system: str, user: str) -> str:
        if self._backend is None:
            from openai import OpenAI  # pylint: disable=import-outside-toplevel
            base_url = self.base_url or (_QWEN_BASE_URL if self.provider
                                         == "qwen" else None)
            self._backend = OpenAI(api_key=_api_key(self.provider),
                                   base_url=base_url,
                                   timeout=self.timeout)
        response = self._backend.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "system",
                "content": system
            }, {
                "role": "user",
                "content": user
            }])
        if response.usage is not None:
            self.usage["prompt_tokens"] += response.usage.prompt_tokens
            self.usage["completion_tokens"] += \
                response.usage.completion_tokens
        return response.choices[0].message.content or ""

    def _call_gemini(self, system: str, user: str) -> str:
        import google.generativeai as genai  # pylint: disable=import-outside-toplevel
        if self._backend is None:
            genai.configure(api_key=_api_key("gemini"))
            self._backend = genai
        # The system prompt is part of the model object, so create one per
        # request.
        model = genai.GenerativeModel(self.model, system_instruction=system)
        response = model.generate_content(
            user,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens),
            request_options={"timeout": self.timeout})
        metadata = getattr(response, "usage_metadata", None)
        if metadata is not None:
            self.usage["prompt_tokens"] += metadata.prompt_token_count
            self.usage["completion_tokens"] += metadata.candidates_token_count
        return response.text or ""

    def _call_codex(self, system: str, user: str) -> str:
        """Ask the local Codex CLI. It runs as an agent, so it gets an empty
        working directory and a read-only sandbox, and is told to answer
        with text only."""
        prompt = (f"{system}\n\n{user}\n\nAnswer with the reply text only. "
                  "Do not run commands, read files or edit files.")
        with tempfile.TemporaryDirectory() as workdir:
            reply_path = Path(workdir) / "reply.txt"
            command = [
                self._codex_command, "exec", "--model", self.model, "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "--sandbox", "read-only", "--skip-git-repo-check",
                "--ephemeral", "--ignore-user-config", "--color", "never",
                "-C", workdir, "-o",
                str(reply_path), "-"
            ]
            result = subprocess.run(command,
                                    input=prompt,
                                    capture_output=True,
                                    text=True,
                                    timeout=self.timeout,
                                    check=False)
            if result.returncode != 0 or not reply_path.exists():
                raise RuntimeError(f"codex exec failed with code "
                                   f"{result.returncode}: "
                                   f"{result.stderr.strip()[-500:]}")
            return reply_path.read_text(encoding="utf-8")
