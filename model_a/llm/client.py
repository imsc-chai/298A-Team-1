"""The single LLM entry point for every model (CLAUDE.md: all calls go through here).

No mock or offline fallback: if Ollama is unreachable we raise, because a silently faked
answer would be scored as if the model produced it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, TypeVar

import requests
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]


class LLMError(RuntimeError):
    """Ollama unreachable, HTTP error, or malformed response envelope."""


class LLMParseError(LLMError):
    """The model's JSON failed validation twice (original call + one retry)."""


@dataclass(frozen=True)
class LLMSettings:
    """Identical for M1 and M3 (CLAUDE.md rule 4). Recorded with every result."""

    model: str = "qwen3.5:9b"
    temperature: float = 0.6
    max_tokens: int = 8192
    seed: int = 42
    # Ollama's default context is small and truncates long prompts silently, so set it explicitly.
    num_ctx: int = 32768
    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    timeout_s: float = 900.0

    def __post_init__(self) -> None:
        # OLLAMA_HOST is often set as "0.0.0.0:11434" for the server; normalize it for the client.
        host = self.host if "://" in self.host else f"http://{self.host}"
        object.__setattr__(self, "host", host.rstrip("/"))

    def options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CallRecord:
    purpose: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    attempt: int


@dataclass
class Telemetry:
    """Per-task accumulator; reset() between tasks."""

    calls: list[CallRecord] = field(default_factory=list)

    def reset(self) -> None:
        self.calls = []

    def summary(self) -> dict[str, int]:
        return {
            "llm_calls": len(self.calls),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "latency_ms": sum(c.latency_ms for c in self.calls),
        }


def _http_transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
    except requests.RequestException as exc:
        raise LLMError(f"Ollama request to {url} failed: {exc}") from exc
    if resp.status_code != 200:
        raise LLMError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


class OllamaClient:
    def __init__(self, settings: LLMSettings | None = None, transport: Transport | None = None) -> None:
        self.settings = settings or LLMSettings()
        self._transport = transport or _http_transport
        self.telemetry = Telemetry()

    def check_available(self) -> None:
        """Fail before a run starts, not 40 minutes in, if the server or model is missing."""
        url = f"{self.settings.host}/api/tags"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama not reachable at {self.settings.host}: {exc}") from exc
        names = {m.get("name") for m in resp.json().get("models", [])}
        if self.settings.model not in names:
            raise LLMError(f"Model {self.settings.model!r} not pulled on {self.settings.host}; have {sorted(names)}")

    def chat(self, messages: list[dict[str, str]], purpose: str, json_mode: bool = True, attempt: int = 1) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
            "options": self.settings.options(),
        }
        if json_mode:
            payload["format"] = "json"
        start = time.perf_counter()
        data = self._transport(f"{self.settings.host}/api/chat", payload, self.settings.timeout_s)
        latency_ms = int((time.perf_counter() - start) * 1000)
        if "message" not in data:
            raise LLMError(f"Unexpected Ollama response (no 'message'): {str(data)[:300]}")
        self.telemetry.calls.append(
            CallRecord(
                purpose=purpose,
                input_tokens=int(data.get("prompt_eval_count", 0)),
                output_tokens=int(data.get("eval_count", 0)),
                latency_ms=latency_ms,
                attempt=attempt,
            )
        )
        return data["message"].get("content", "")

    def chat_json(self, messages: list[dict[str, str]], schema: type[T], purpose: str) -> T:
        """Ask for JSON and validate it; on failure retry once with the error, then raise."""
        content = self.chat(messages, purpose, json_mode=True, attempt=1)
        try:
            return _validate(content, schema)
        except (json.JSONDecodeError, ValidationError) as first_error:
            retry_messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid for the required JSON schema. "
                        f"Error: {_short_error(first_error)}\nReply again with ONLY the corrected JSON object."
                    ),
                },
            ]
            content = self.chat(retry_messages, purpose, json_mode=True, attempt=2)
            try:
                return _validate(content, schema)
            except (json.JSONDecodeError, ValidationError) as second_error:
                raise LLMParseError(
                    f"{purpose}: invalid JSON after retry: {_short_error(second_error)} | content={content[:300]!r}"
                ) from second_error


def _validate(content: str, schema: type[T]) -> T:
    return schema.model_validate(json.loads(content))


def _short_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:400]
