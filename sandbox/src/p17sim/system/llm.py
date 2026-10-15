"""Local model client with profiling and an egress log.

Every model call goes through `Chat.chat` or `embed`. Each call records:
- wall time, model load time, prompt (prefill) tokens and time, output (decode) tokens and time;
- the component that made it (plan, answer, act, ...), so latency can be split by component;
- an egress entry: the host it went to and how many bytes of persona text it carried.
A call to any host other than localhost counts toward "raw-data export" (it must stay 0 on the
local path).

Model settings follow OpenJarvis's Ollama engine at commit e97088f: thinking off, 8,192-token
context, temperature 0.6, up to 8,192 output tokens (pinchbench-qwen-9b.toml).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class CallRecord:
    component: str
    model: str
    host: str
    wall_s: float
    load_s: float = 0.0
    prompt_tokens: int = 0
    prefill_s: float = 0.0
    output_tokens: int = 0
    decode_s: float = 0.0
    bytes_out: int = 0
    persona_bytes_out: int = 0
    error: str | None = None


@dataclass
class Profile:
    """Collects model calls and timed blocks for one task (or one whole run)."""
    calls: list[CallRecord] = field(default_factory=list)
    blocks: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self.blocks[name] = self.blocks.get(name, 0.0) + time.perf_counter() - t

    def summary(self) -> dict[str, Any]:
        by_comp: dict[str, float] = {}
        for c in self.calls:
            by_comp[f"llm:{c.component}"] = by_comp.get(f"llm:{c.component}", 0.0) + c.wall_s
        return {
            "llm_calls": len(self.calls),
            "llm_wall_s": round(sum(c.wall_s for c in self.calls), 3),
            "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "prefill_s": round(sum(c.prefill_s for c in self.calls), 3),
            "decode_s": round(sum(c.decode_s for c in self.calls), 3),
            "load_s": round(sum(c.load_s for c in self.calls), 3),
            "ttft_s": [round(c.load_s + c.prefill_s, 3) for c in self.calls],
            "nonlocal_calls": sum(c.host not in LOCAL_HOSTS for c in self.calls),
            "raw_export_bytes": sum(c.persona_bytes_out for c in self.calls if c.host not in LOCAL_HOSTS),
            "errors": [c.error for c in self.calls if c.error],
            "time_by_component_s": {k: round(v, 3) for k, v in {**self.blocks, **by_comp}.items()},
        }


class PersonaText:
    """Knows the persona's raw text, so the egress log can count how much of it a request carries."""

    def __init__(self, texts: list[str] | None = None):
        self.snippets = [t for t in (texts or []) if len(t) >= 12]

    def bytes_in(self, payload: str) -> int:
        return sum(len(s.encode()) for s in self.snippets if s in payload)


class Chat:
    def __init__(self, model: str = "qwen3.5:4b", host: str = "http://localhost:11434", temperature: float = 0.6,
                 max_tokens: int = 8192, num_ctx: int = 8192, think: bool = False, seed: int | None = None,
                 timeout: int = 900):
        self.model, self.host, self.temperature = model, host, temperature
        self.max_tokens, self.num_ctx, self.think, self.seed, self.timeout = max_tokens, num_ctx, think, seed, timeout
        self.hostname = urllib.parse.urlparse(host).hostname or host
        self.profile = Profile()
        self.persona_text = PersonaText()

    def settings(self) -> dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "num_ctx": self.num_ctx, "think": self.think, "seed": self.seed, "host": self.hostname}

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None, component: str = "llm",
             json_mode: bool = False, max_tokens: int | None = None) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self.temperature, "num_predict": max_tokens or self.max_tokens,
                                   "num_ctx": self.num_ctx}
        if self.seed is not None:
            options["seed"] = self.seed
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "think": self.think,
                                   "options": options}
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["format"] = "json"
        body = json.dumps(payload).encode()
        rec = CallRecord(component=component, model=self.model, host=self.hostname, wall_s=0.0,
                         bytes_out=len(body), persona_bytes_out=self.persona_text.bytes_in(body.decode()))
        t = time.perf_counter()
        try:
            req = urllib.request.Request(f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
        except Exception as exc:  # recorded and re-raised so the runner logs a real error
            rec.wall_s, rec.error = time.perf_counter() - t, f"{type(exc).__name__}: {exc}"
            self.profile.calls.append(rec)
            raise
        rec.wall_s = time.perf_counter() - t
        rec.load_s = data.get("load_duration", 0) / 1e9
        rec.prompt_tokens = data.get("prompt_eval_count", 0)
        rec.prefill_s = data.get("prompt_eval_duration", 0) / 1e9
        rec.output_tokens = data.get("eval_count", 0)
        rec.decode_s = data.get("eval_duration", 0) / 1e9
        self.profile.calls.append(rec)
        msg = data.get("message", {})
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"id": tc.get("id", f"call_{i}"), "name": fn.get("name", ""), "arguments": args})
        return {"content": msg.get("content", "") or "", "tool_calls": calls, "raw_message": msg}


def embed(texts: list[str], model: str = "all-minilm", host: str = "http://localhost:11434",
          profile: Profile | None = None, persona_text: PersonaText | None = None) -> list[list[float]]:
    """Local sentence embeddings (all-minilm, 23M parameters) through Ollama."""
    body = json.dumps({"model": model, "input": texts}).encode()
    hostname = urllib.parse.urlparse(host).hostname or host
    rec = CallRecord(component="embed", model=model, host=hostname, wall_s=0.0, bytes_out=len(body),
                     persona_bytes_out=(persona_text or PersonaText()).bytes_in(body.decode()))
    t = time.perf_counter()
    req = urllib.request.Request(f"{host}/api/embed", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    rec.wall_s = time.perf_counter() - t
    rec.load_s = data.get("load_duration", 0) / 1e9
    rec.prompt_tokens = data.get("prompt_eval_count", 0)
    if profile is not None:
        profile.calls.append(rec)
    return data["embeddings"]


class ScriptedChat(Chat):
    """Test double: returns queued replies and records calls without a server."""

    def __init__(self, replies: list[dict[str, Any]]):
        super().__init__(model="scripted")
        self.replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None, component="llm", json_mode=False, max_tokens=None):
        self.seen.append(messages)
        self.profile.calls.append(CallRecord(component=component, model="scripted", host="localhost", wall_s=0.0))
        reply = self.replies.pop(0) if self.replies else {"content": "", "tool_calls": []}
        return {"content": reply.get("content", ""), "tool_calls": reply.get("tool_calls", []), "raw_message": reply}
