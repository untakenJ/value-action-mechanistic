"""OpenAI-compatible HTTP client for five-factor LLM-as-judge scoring."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from psycho_llm_behavioral.judge_prompt import (
    build_judge_messages,
    parse_judge_response,
    reverse_score,
    sample_keying,
    selected_statements,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


class JudgeAPIError(RuntimeError):
    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    try:
        return int(value) if value else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    try:
        return float(value) if value else default
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc


@dataclass(frozen=True)
class JudgeConfig:
    api_key: str | None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    thinking: str = "disabled"
    reasoning_effort: str = "high"

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("JUDGE_MODEL", "deepseek-v4-pro"),
            temperature=_env_float("JUDGE_TEMPERATURE", 0.0),
            max_tokens=_env_int("JUDGE_MAX_TOKENS", 512),
            timeout_seconds=_env_float("JUDGE_TIMEOUT_SECONDS", 60.0),
            max_attempts=_env_int("JUDGE_MAX_ATTEMPTS", 3),
            thinking=os.environ.get("JUDGE_THINKING", "disabled").lower(),
            reasoning_effort=os.environ.get("JUDGE_REASONING_EFFORT", "high"),
        )

    def validate(self) -> None:
        if not self.api_key:
            raise ValueError(
                "Missing Judge API key. Set DEEPSEEK_API_KEY or JUDGE_API_KEY in .env."
            )
        if self.thinking not in {"enabled", "disabled", "omit"}:
            raise ValueError("JUDGE_THINKING must be 'enabled', 'disabled', or 'omit'")
        if self.max_tokens <= 0 or self.max_attempts <= 0 or self.timeout_seconds <= 0:
            raise ValueError("Judge token, attempt, and timeout settings must be positive")

    def public_dict(self) -> dict:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "thinking": self.thinking,
            "reasoning_effort": self.reasoning_effort,
        }


class JudgeClient:
    def __init__(
        self,
        config: JudgeConfig,
        *,
        opener: Callable | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        config.validate()
        self.config = config
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep

    @property
    def endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def rate(self, prompt: dict, response_record: dict) -> dict:
        response_id = response_record["response_id"]
        keying = sample_keying(self.config.model, response_id)
        messages = build_judge_messages(
            prompt,
            response_record["raw_response"],
            keying,
        )
        last_error: str | None = None
        last_status = "api_error"
        last_raw: str | None = None
        metadata: dict = {}

        for attempt in range(self.config.max_attempts):
            try:
                payload = self._request(messages)
                raw_text, metadata = self._extract_response(payload)
                last_raw = raw_text
                scores, parse_error = parse_judge_response(raw_text)
                if scores is not None:
                    return {
                        "response_id": response_id,
                        "subject_model": response_record["subject_model"],
                        "prompt_id": response_record["prompt_id"],
                        "run_number": response_record["run_number"],
                        "judge_model": self.config.model,
                        "keying": keying,
                        "statements": selected_statements(keying),
                        "raw_scores": scores,
                        "scores": reverse_score(scores, keying),
                        "raw_response": raw_text,
                        "provider_metadata": metadata,
                        "status": "success",
                        "error_message": None,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                last_status = "parse_error"
                last_error = parse_error
            except JudgeAPIError as exc:
                last_status = "api_error"
                last_error = str(exc)
                if not exc.transient:
                    break

            if attempt < self.config.max_attempts - 1:
                self._sleep(min(2 ** attempt, 30))

        return {
            "response_id": response_id,
            "subject_model": response_record["subject_model"],
            "prompt_id": response_record["prompt_id"],
            "run_number": response_record["run_number"],
            "judge_model": self.config.model,
            "keying": keying,
            "statements": selected_statements(keying),
            "raw_scores": None,
            "scores": None,
            "raw_response": last_raw,
            "provider_metadata": metadata,
            "status": last_status,
            "error_message": last_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _request(self, messages: list[dict[str, str]]) -> dict:
        body: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if self.config.thinking != "omit":
            body["thinking"] = {"type": self.config.thinking}
        if self.config.thinking == "enabled":
            body["reasoning_effort"] = self.config.reasoning_effort

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.config.timeout_seconds) as response:
                data = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise JudgeAPIError(
                f"Judge HTTP {exc.code}: {detail}",
                transient=exc.code == 429 or exc.code >= 500,
            ) from exc
        except urllib.error.URLError as exc:
            raise JudgeAPIError(f"Judge network error: {exc.reason}", transient=True) from exc
        except TimeoutError as exc:
            raise JudgeAPIError("Judge request timed out", transient=True) from exc

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise JudgeAPIError("Judge API returned non-JSON response", transient=True) from exc
        if not isinstance(payload, dict):
            raise JudgeAPIError("Judge API returned an unexpected payload", transient=True)
        if payload.get("error"):
            raise JudgeAPIError(f"Judge API error: {payload['error']}", transient=False)
        return payload

    @staticmethod
    def _extract_response(payload: dict) -> tuple[str, dict]:
        try:
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeAPIError("Judge API response has no completion choice", transient=True) from exc

        metadata = {
            "id": payload.get("id"),
            "model": payload.get("model"),
            "created": payload.get("created"),
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage"),
        }
        return content, metadata
