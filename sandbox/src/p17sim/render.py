"""Surface renderer: an LLM writes the text; the skeleton keeps every label.

Rendered text must contain the asserted fact value and any canary verbatim, or the event is
retried and finally marked render_failed. Results are cached so re-runs are byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from p17sim.rng import derive_seed
from p17sim.schema import RENDERED_TYPES, Event, EventType, Persona

MAX_ATTEMPTS = 3
_REQUIRED_LINE = "It must include these exact strings verbatim:"


class Backend(Protocol):
    name: str

    def generate(self, prompt: str, seed: int) -> str: ...


class FakeBackend:
    """Offline backend for tests and CI. Can drop the required strings for the first N attempts."""

    def __init__(self, fail_first: int = 0):
        self.name = "fake"
        self.fail_first = fail_first
        self.calls: dict[str, int] = {}

    def generate(self, prompt: str, seed: int) -> str:
        n = self.calls.get(prompt, 0)
        self.calls[prompt] = n + 1
        required = []
        for line in prompt.splitlines():
            if line.startswith(_REQUIRED_LINE):
                required = re.findall(r'"([^"]+)"', line)
        body = prompt.splitlines()[0]  # event-specific, so fake text is not near-duplicate across events
        return body if n < self.fail_first else f"{body} {' '.join(required)}".strip()


class OllamaBackend:
    """Local Ollama server. Use a non-Qwen family (default llama3.2) so the renderer differs from the agent."""

    def __init__(self, model: str | None = None, host: str = "http://localhost:11434"):
        self.model = model or os.environ.get("P17_RENDER_MODEL", "llama3.2")
        self.name = f"ollama:{self.model}"
        self.host = host

    def generate(self, prompt: str, seed: int) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0.9, "seed": seed % 2**31}}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["response"].strip()


class GeminiBackend:
    """Gemini via the google-genai SDK. Needs GEMINI_API_KEY, or Vertex AI env settings."""

    def __init__(self, model: str | None = None):
        from google import genai

        self.model = model or os.environ.get("P17_RENDER_MODEL", "gemini-2.5-flash")
        self.name = f"gemini:{self.model}"
        self._client = genai.Client()

    def generate(self, prompt: str, seed: int) -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(temperature=0.9, seed=seed % 2**31))
        return (resp.text or "").strip()


def required_strings(event: Event) -> list[str]:
    return [s for s in (event.fact_value, event.canary) if s]


def build_prompt(event: Event, persona: Persona) -> str:
    contacts = {c.contact_id: c for c in persona.contacts}
    sender = contacts.get(event.payload.get("from", ""))
    topic = event.payload.get("topic", "")
    if event.type == EventType.FILE:
        ask = f"Write a short personal note (2-4 sentences) that {persona.name} saved about {topic}."
    elif event.type == EventType.FACT_UPDATE:
        ask = (f"Write a short message (1-3 sentences) from {sender.name} ({sender.relation}) to {persona.name} "
               f"saying the {topic} has changed from {event.payload['old_value']} to {event.fact_value}.")
    else:
        ask = f"Write a short message (1-3 sentences) from {sender.name} ({sender.relation}) to {persona.name} about {topic}."
    lines = [ask]
    required = required_strings(event)
    if required:
        lines.append(_REQUIRED_LINE + " " + "; ".join(f'"{r}"' for r in required))
    if event.canary:
        lines.append("Mention the reference code naturally, for example 'ref " + event.canary + "'.")
    lines.append("Use only the names given. Do not invent email addresses, phone numbers or other names. "
                 "Output only the text.")
    return "\n".join(lines)


def render_event(event: Event, persona: Persona, backend: Backend, cache_dir: Path) -> Event:
    prompt = build_prompt(event, persona)
    key = hashlib.sha256(f"{event.event_id}|{hashlib.sha256(prompt.encode()).hexdigest()}|{backend.name}"
                         .encode()).hexdigest()
    path = cache_dir / f"{key}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        return event.model_copy(update=cached)
    required = required_strings(event)
    result = {"text": None, "render_status": "render_failed"}
    for attempt in range(MAX_ATTEMPTS):
        text = backend.generate(prompt, derive_seed(event.event_id, attempt))
        if all(r in text for r in required):
            result = {"text": text, "render_status": "ok"}
            break
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return event.model_copy(update=result)


def render_events(events: list[Event], personas: dict[str, Persona], backend: Backend,
                  cache_dir: Path, workers: int = 8) -> list[Event]:
    def one(e: Event) -> Event:
        if e.type in RENDERED_TYPES and e.render_status == "pending":
            return render_event(e, personas[e.persona_id], backend, cache_dir)
        return e

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, events))
