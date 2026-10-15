"""v2 surface renderer: style-aware prompts, two model families, the same fact check and cache as v1.

Each contact has a writing style (formality, English variety, a quirk). Each message has a target
length drawn from the Enron length distribution. Facts, distractors, updates and sensitive items say
exactly whose value it is, so the label stays true after rendering. A persona's renderer is fixed in
its PersonaExtra; phi3 renders only held-out test and OOD personas.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from p17sim.gen2.personas import PersonaExtra
from p17sim.render import OllamaBackend, render_events
from p17sim.schema import EventType, Persona

_REQUIRED_LINE = "It must include these exact strings verbatim:"  # same marker the v1 FakeBackend parses


class CappedOllama(OllamaBackend):
    """v1's Ollama backend with an output cap and a longer timeout. Without the cap, phi3 sometimes
    keeps writing until the 120 s timeout. A 1,024-token context keeps 4 parallel requests in memory.
    The cache key (backend name) is unchanged."""

    def __init__(self, model: str, max_tokens: int = 400, timeout: int = 300):
        super().__init__(model=model)
        self.max_tokens, self.timeout = max_tokens, timeout

    def generate(self, prompt: str, seed: int) -> str:
        import json
        import urllib.request

        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0.9, "seed": seed % 2**31,
                                       "num_predict": self.max_tokens, "num_ctx": 1024}}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["response"].strip()


def _style_line(style: dict[str, str] | None) -> str:
    if not style:
        return ""
    quirk = "" if style["quirk"] == "none" else f", with {style['quirk']}"
    return f"Write it in a {style['formality']} tone, in {style['variety']}{quirk}."


def build_prompt(event, persona: Persona, extra: PersonaExtra) -> str:
    contacts = {c.contact_id: c for c in persona.contacts}
    sender = contacts.get(event.payload.get("from", ""))
    topic = event.payload.get("topic", "")
    words = event.payload.get("words", 30)
    who = f"from {sender.name} ({sender.relation}) to {persona.name}" if sender else f"that {persona.name} saved"
    prov = event.provenance.split(":")[0]
    if event.type == EventType.FILE:
        ask = f"Write a personal note of about {words} words that {persona.name} saved about {topic}."
        if prov == "fact":
            ask += f" The note records that the {topic} is {event.fact_value}."
    elif event.type == EventType.FACT_UPDATE:
        ask = (f"Write a message of about {words} words {who} saying the {topic} has changed from "
               f"{event.payload['old_value']} to {event.fact_value}.")
    elif prov == "distractor":
        owner = event.payload["owner"]
        ask = (f"Write a message of about {words} words {who} mentioning that {owner}'s {topic} is "
               f"{event.fact_value}. Make it clear this is {owner}'s, not {persona.name}'s own.")
    elif prov in ("fact", "fact_repeat"):
        ask = f"Write a message of about {words} words {who} that tells them their {topic} is {event.fact_value}."
    elif prov == "sensitive":
        ask = (f"Write a private message of about {words} words {who} about this {topic} matter: "
               f"{event.fact_value}.")
    else:
        reply = "a reply in an ongoing conversation " if event.payload.get("reply") else ""
        ask = f"Write {reply}a message of about {words} words {who} about {topic}."
    lines = [ask]
    required = [s for s in (event.fact_value, event.canary) if s]
    if required:
        lines.append(_REQUIRED_LINE + " " + "; ".join(f'"{r}"' for r in required))
    if event.canary:
        lines.append(f"Work the reference '{event.canary}' into the text naturally.")
    style = extra.styles.get(sender.contact_id) if sender else None
    if style:
        lines.append(_style_line(style))
    lines.append("Use only the names given. Do not invent email addresses, phone numbers or other names. "
                 "Output only the message text: no quotes around it, no title, no preface.")
    return "\n".join(lines)


def render_cohort(events, personas: list[Persona], extras: list[PersonaExtra], cache: Path,
                  workers: int = 4, backends: dict[str, object] | None = None):
    """Render every pending event with its persona's renderer, one model at a time so only one
    model sits in memory on an 8 GB machine. Retries, the fact check and the cache are v1's."""
    by_pid = {p.persona_id: p for p in personas}
    x_by_pid = {x.persona_id: x for x in extras}
    groups: dict[str, list] = defaultdict(list)
    for e in events:
        groups[x_by_pid[e.persona_id].renderer].append(e)
    out = []
    for renderer in sorted(groups):
        backend = (backends or {}).get(renderer) or CappedOllama(model=renderer.split(":", 1)[1])
        out += render_events(groups[renderer], by_pid, backend, cache, workers,
                             prompt_fn=lambda e, p: build_prompt(e, p, x_by_pid[p.persona_id]))
    order = {e.event_id: i for i, e in enumerate(events)}
    return sorted(out, key=lambda e: order[e.event_id])
