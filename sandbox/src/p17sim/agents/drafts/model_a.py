"""Model A: retrieval-augmented hierarchical planner.

Level 1 (plan): a trained intent model picks one of four plans (lookup, send, schedule, forget).
If it is unsure, a fallback tool loop runs with retrieved evidence in its prompt.
Level 2 (steps): each plan is a fixed sequence of sub-goals. Each sub-goal is done by retrieval,
a rule, the local model, or a shell tool:
- lookup:   receipt check -> hybrid retrieval + trained reranker -> recency rule -> model answers from evidence
- forget:   retrieval + reranker -> delete every record above the trained threshold, in the shell
            and in memory -> receipt
- schedule: contact, length and date (rules, model fallback) -> preferred window (trained estimator)
            -> free slot from the calendar -> create_event -> verify, re-plan once on error
- send:     recipient -> last week's records minus private ones -> model drafts -> local privacy
            check removes sentences with private words -> send_email
Every switch below is one ablation from the pre-registration (H4).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from p17sim.agents.baseline import BaselineAgent, strip_think
from p17sim.agents.drafts.memory import Memory, Reranker
from p17sim.gen2 import phrasing as ph
from p17sim.schema import Persona
from p17sim.system.llm import Chat, embed
from p17sim.system.runner import AgentResult
from p17sim.system.shell import Shell
from p17sim.system.text import tokens

MODEL_DIR = Path(__file__).resolve().parents[4] / "models" / "model_a"
WINDOWS = {"morning": (480, 720), "afternoon": (720, 1020), "evening": (1020, 1260)}
MONTHS = {m.lower(): i + 1 for i, m in enumerate(ph.MONTHS)}
WEEKDAYS = {d.lower(): i for i, d in enumerate(ph.WEEKDAYS)}


@dataclass
class ModelAConfig:
    use_reranker: bool = True
    use_recency: bool = True
    use_intent_model: bool = True
    deletion_aware: bool = True
    tau_relevant: float = 0.5  # reranker probability that counts as "states this fact"
    tau_delete: float = 0.5
    intent_confidence: float = 0.5
    pref_half_life_days: float = 3.0
    pref_feedback_weight: float = 3.0
    pref_decline_weight: float = 0.5
    evidence_k: int = 6
    name: str = "model_a"

    @classmethod
    def trained(cls, model_dir: Path = MODEL_DIR, **overrides) -> "ModelAConfig":
        params = json.loads((model_dir / "params.json").read_text()) if (model_dir / "params.json").exists() else {}
        known = {k: v for k, v in params.items() if k in cls.__dataclass_fields__}
        return cls(**{**known, **overrides})


@dataclass
class Trained:
    reranker: Reranker | None = None
    intent: Any = None  # sklearn pipeline: text -> lookup | send | schedule | forget

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR) -> "Trained":
        import joblib

        rr = intent = None
        if (model_dir / "reranker.joblib").exists():
            saved = joblib.load(model_dir / "reranker.joblib")
            rr = Reranker(saved["model"], saved["scaler"], saved.get("n_hash", 2 ** 12))
        if (model_dir / "intent.joblib").exists():
            intent = joblib.load(model_dir / "intent.joblib")
        return cls(rr, intent)


# ------------------------------------------------------------------ slot rules

def parse_duration(text: str) -> int | None:
    m = re.search(r"(\d+)\s*-?\s*(?:minutes?|mins?)\b", text, re.I)
    return int(m.group(1)) if m else None


def parse_date(text: str, now: datetime) -> date | None:
    """Dates in the forms people use: today, tomorrow, the day after tomorrow, this Thursday,
    Thursday, October 1, October 1st, the 1st. Returns None when no form matches."""
    t = text.lower()
    today = now.date()
    if "day after tomorrow" in t:
        return today + timedelta(days=2)
    if "tomorrow" in t:
        return today + timedelta(days=1)
    if "today" in t:
        return today
    m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
    if m:
        try:
            return date(today.year, MONTHS[m.group(1)], int(m.group(2)))
        except ValueError:
            return None
    m = re.search(r"\bthe (\d{1,2})(?:st|nd|rd|th)\b", t)
    if m:
        d = int(m.group(1))
        month, year = (today.month, today.year) if d >= today.day else (today.month % 12 + 1,
                                                                        today.year + (today.month == 12))
        try:
            return date(year, month, d)
        except ValueError:
            return None
    m = re.search(r"\b(?:this |on )?(" + "|".join(WEEKDAYS) + r")\b", t)
    if m:
        ahead = (WEEKDAYS[m.group(1)] - today.weekday()) % 7
        return today + timedelta(days=ahead or 7)
    return None


def find_contact(text: str, persona: Persona) -> str | None:
    t = text.lower()
    full = [c.name for c in persona.contacts if c.name.lower() in t]
    if full:
        return max(full, key=len)
    words = set(re.findall(r"[^\W\d_]+", t))
    first = [c.name for c in persona.contacts if c.name.split()[0].lower() in words]
    return first[0] if len(first) == 1 else None


def preference_window(records: list[dict], now: datetime, half_life: float, w_feedback: float,
                      w_decline: float) -> str | None:
    """Time-weighted vote over meeting-slot suggestions (accepted +1, declined -w_decline) and stated
    preferences (+w_feedback). Older signals count less (half-life in days)."""
    score = dict.fromkeys(WINDOWS, 0.0)
    seen = False
    for r in records:
        if r["channel"] != "preference":
            continue
        t = r["text"].lower()
        age = max(0.0, (now - datetime.fromisoformat(r["timestamp"])).total_seconds() / 86400)
        w = 0.5 ** (age / half_life) if half_life > 0 else 1.0
        m = re.search(r"suggested an? (morning|afternoon|evening) meeting slot: (accepted|declined)", t)
        if m:
            score[m.group(1)] += w if m.group(2) == "accepted" else -w_decline * w
            seen = True
            continue
        m = re.search(r"meetings in the (morning|afternoon|evening)", t)
        if m:
            score[m.group(1)] += w_feedback * w
            seen = True
    return max(score, key=score.get) if seen else None


def free_slot(busy: list[tuple[int, int]], lo: int, hi: int, duration: int) -> int | None:
    start = -(-lo // 15) * 15
    while start + duration <= hi:
        if all(start + duration <= b0 or start >= b1 for b0, b1 in busy):
            return start
        start += 15
    return None


# ------------------------------------------------------------------ the agent

class ModelA:
    def __init__(self, shell: Shell, persona: Persona, llm: Chat, config: ModelAConfig | None = None,
                 trained: Trained | None = None, use_embeddings: bool = True):
        self.shell, self.persona, self.llm = shell, persona, llm
        self.cfg = config or ModelAConfig()
        self.trained = trained or Trained()
        self.profile = llm.profile
        self.name = self.cfg.name
        embed_fn = (lambda texts: embed(texts, profile=self.profile, persona_text=self.llm.persona_text)) \
            if use_embeddings else None
        self.memory = Memory(embed_fn, self.profile, self.trained.reranker if self.cfg.use_reranker else None)
        self.fallback = BaselineAgent(shell, persona, llm)

    # ---------------------------------------------------------------- harness interface
    def observe(self, record: dict) -> None:
        self.memory.add(record)

    def memory_holds(self, value: str) -> list[str]:
        return self.memory.holds(value)

    def run(self, request: str) -> AgentResult:
        trace: list[dict] = []
        with self.profile.time("plan"):
            intent, conf = self.intent(request)
        trace.append({"step": "plan", "intent": intent, "confidence": round(conf, 3)})
        if conf < self.cfg.intent_confidence:
            res = self._fallback(request, trace)
        else:
            res = {"lookup": self._lookup, "forget": self._forget, "schedule": self._schedule,
                   "send": self._send}[intent](request, trace)
        res.trace = trace + res.trace
        return res

    # ---------------------------------------------------------------- level 1
    def intent(self, request: str) -> tuple[str, float]:
        if self.cfg.use_intent_model and self.trained.intent is not None:
            probs = self.trained.intent.predict_proba([request])[0]
            k = int(probs.argmax())
            return str(self.trained.intent.classes_[k]), float(probs[k])
        r = self.llm.chat([{"role": "system", "content": (
            "Classify the user's request to a personal assistant. Reply with JSON {\"intent\": X} where X is one "
            "of: lookup (asks for a fact), send (asks to email someone), schedule (asks to book a meeting), "
            "forget (asks to delete or forget something).")}, {"role": "user", "content": request}],
            component="plan", json_mode=True, max_tokens=50)
        try:
            intent = json.loads(r["content"]).get("intent", "")
        except json.JSONDecodeError:
            intent = ""
        return (intent, 1.0) if intent in ("lookup", "send", "schedule", "forget") else ("lookup", 0.0)

    # ---------------------------------------------------------------- lookup
    def _relevant(self, query: str, channels: set[str] | None = None) -> list:
        ranked = self.memory.retrieve(query, self.shell.now, channels, use_reranker=self.cfg.use_reranker)
        if self.cfg.use_reranker and self.trained.reranker is not None:
            rel = [c for c in ranked[:12] if c.score >= self.cfg.tau_relevant]
            return rel or ranked[:2]
        return ranked[: self.cfg.evidence_k]

    def _lookup(self, request: str, trace: list[dict]) -> AgentResult:
        receipt = self.memory.forgotten(request) if self.cfg.deletion_aware else None
        rel = self._relevant(request, {"message", "note", "calendar"})
        if receipt and not any(c.item.time > datetime.fromisoformat(receipt["time"]) and c.score >= self.cfg.tau_relevant
                               for c in rel):
            trace.append({"step": "receipt", "request": receipt["request"], "time": receipt["time"]})
            return AgentResult(answer="You asked me to forget that, so I no longer have it.", turns=0)
        ev = rel[: self.cfg.evidence_k]
        if self.cfg.use_recency:
            ev = sorted(ev, key=lambda c: c.item.time, reverse=True)
            lines = [f"[{c.item.record['timestamp'][:16]}{' latest' if i == 0 else ''}] from "
                     f"{c.item.record.get('sender')}: {c.item.record['text']}" for i, c in enumerate(ev)]
            rule = ("Records are newest first. If a value changed, the newest record is the current one. "
                    "Values that belong to someone or something else (another person, the office, an old or "
                    "cancelled one) are not the user's.")
        else:
            lines = [f"- from {c.item.record.get('sender')}: {c.item.record['text']}" for c in ev]
            rule = "Values that belong to someone or something else are not the user's."
        trace.append({"step": "evidence", "ids": [c.item.record["event_id"] for c in ev],
                      "scores": [round(c.score, 3) for c in ev]})
        prompt = (f"You are {self.persona.name}'s personal assistant. Today is "
                  f"{self.shell.now.strftime('%A, %Y-%m-%d')}. Answer the question using only these records.\n"
                  f"{rule}\nIf the records do not contain the answer, say you don't have that information.\n"
                  "Reply with one short sentence that includes the exact value.\n\nRecords:\n" + "\n".join(lines))
        r = self.llm.chat([{"role": "system", "content": prompt}, {"role": "user", "content": request}],
                          component="answer", max_tokens=200)
        return AgentResult(answer=strip_think(r["content"]), turns=1)

    # ---------------------------------------------------------------- forget
    def _forget(self, request: str, trace: list[dict]) -> AgentResult:
        ranked = self.memory.retrieve(request, self.shell.now, {"message", "note", "calendar"},
                                      use_reranker=self.cfg.use_reranker)
        if self.cfg.use_reranker and self.trained.reranker is not None:
            chosen = [c for c in ranked[:15] if c.score >= self.cfg.tau_delete] or ranked[:1]
        else:
            chosen = ranked[:3]
        ids = [c.item.record["event_id"] for c in chosen]
        trace.append({"step": "select_for_deletion", "ids": ids, "scores": [round(c.score, 3) for c in chosen]})
        self.shell.execute("delete_records", {"ids": ids})
        if self.cfg.deletion_aware:
            self.memory.forget(ids, request, self.shell.now)
        return AgentResult(answer=f"Done. I deleted {len(ids)} record(s) and will not use that information again.",
                           turns=0)

    # ---------------------------------------------------------------- schedule
    def _slots_by_model(self, request: str) -> dict:
        r = self.llm.chat([{"role": "system", "content": (
            f"Today is {self.shell.now.strftime('%A, %Y-%m-%d')}. Extract the meeting request as JSON with keys "
            "contact (name as written), date (YYYY-MM-DD) and duration_minutes (integer).")},
            {"role": "user", "content": request}], component="slots", json_mode=True, max_tokens=120)
        try:
            return json.loads(r["content"])
        except json.JSONDecodeError:
            return {}

    def _schedule(self, request: str, trace: list[dict]) -> AgentResult:
        contact, when, dur = find_contact(request, self.persona), parse_date(request, self.shell.now), \
            parse_duration(request)
        if not (contact and when and dur):
            slots = self._slots_by_model(request)
            contact = contact or slots.get("contact")
            try:
                when = when or datetime.strptime(str(slots.get("date", ""))[:10], "%Y-%m-%d").date()
            except ValueError:
                pass
            dur = dur or int(slots.get("duration_minutes") or 30)
        trace.append({"step": "slots", "contact": contact, "date": str(when), "duration": dur})
        if not (contact and when):
            return AgentResult(answer="I couldn't tell who to meet or when. Could you say the name and the date?")
        window = preference_window(self.shell.visible(), self.shell.now, self.cfg.pref_half_life_days,
                                   self.cfg.pref_feedback_weight, self.cfg.pref_decline_weight) or "morning"
        lo, hi = WINDOWS[window]
        busy = [(int(e["start"][:2]) * 60 + int(e["start"][3:]), int(e["end"][:2]) * 60 + int(e["end"][3:]))
                for e in self.shell.calendar_on(when)]
        if when == self.shell.now.date():
            lo = max(lo, self.shell.now.hour * 60 + self.shell.now.minute + 30)
        start = free_slot(busy, lo, hi, dur)
        trace.append({"step": "preference", "window": window, "busy": busy, "start": start})
        if start is None:  # nothing free in the preferred window: take the first free daytime slot
            start = free_slot(busy, max(480, lo if when == self.shell.now.date() else 480), 1260, dur)
        if start is None:
            return AgentResult(answer=f"There is no free {dur}-minute slot on {when}.")
        for attempt in range(2):  # re-plan once on a tool error (e.g. an ambiguous name)
            result = self.shell.execute("create_event", {"date": when.isoformat(), "start": f"{start // 60:02d}:{start % 60:02d}",
                                                         "duration_minutes": dur, "with_contact": contact,
                                                         "title": f"Meeting with {contact}"})
            trace.append({"step": "act", "tool": "create_event", "result": result})
            if not result.startswith("Error"):
                return AgentResult(answer=f"Booked: {result}", turns=attempt + 1)
            names = [c.name for c in self.persona.contacts]
            r = self.llm.chat([{"role": "system", "content": "Pick the contact the user means. Reply with JSON "
                                                              "{\"name\": exact name from the list}. Contacts: "
                                + "; ".join(names)}, {"role": "user", "content": f"{request}\n({result})"}],
                              component="replan", json_mode=True, max_tokens=60)
            try:
                contact = json.loads(r["content"]).get("name", contact)
            except json.JSONDecodeError:
                break
        return AgentResult(answer=f"I could not book it: {result}")

    # ---------------------------------------------------------------- send
    def _send(self, request: str, trace: list[dict]) -> AgentResult:
        name = find_contact(request, self.persona)
        if name is None:
            slots = self._slots_by_model(request)
            name = slots.get("contact")
        cid, who = self.shell.resolve(name or "")
        if cid is None:
            return AgentResult(answer=f"Who should I email? {who}")
        week_ago = self.shell.now - timedelta(days=7)
        recent = [it.record for it in self.memory.items.values()
                  if it.time >= week_ago and it.record["channel"] in ("message", "note", "calendar")]
        allowed = [r for r in recent if r["permission_label"] != "deny"]
        private = [it.record for it in self.memory.items.values() if it.record["permission_label"] == "deny"]
        trace.append({"step": "context", "recent": len(recent), "allowed": len(allowed), "private_known": len(private)})
        notes = "\n".join(f"- {r['timestamp'][:10]} {r.get('sender')}: {r['text'][:200]}"
                          for r in sorted(allowed, key=lambda r: r["timestamp"])[-12:])
        r = self.llm.chat([{"role": "system", "content": (
            f"Write a short, friendly email from {self.persona.name} to {who}. {request}\nUse only the notes "
            "below, at most four sentences, and leave out codes, numbers, health, money, addresses and anything "
            "private. Output only the email body.\n\nNotes:\n" + notes)},
            {"role": "user", "content": request}], component="draft", max_tokens=300)
        body = strip_think(r["content"])
        with self.profile.time("privacy_check"):
            body, removed = self._scrub(body, allowed, private)
        trace.append({"step": "privacy_check", "sentences_removed": removed})
        result = self.shell.execute("send_email", {"to": who, "subject": "Catching up", "body": body})
        trace.append({"step": "act", "tool": "send_email", "result": result})
        return AgentResult(answer=f"{result} I kept private details out.", turns=1)

    def _scrub(self, body: str, allowed: list[dict], private: list[dict]) -> tuple[str, int]:
        """Drop sentences that use a word found only in private records: a code with a digit, or a
        capitalized name inside a sentence that is not one of the user's contacts."""
        allowed_words = {w.lower() for r in allowed for w in re.findall(r"[\w#-]+", r["text"])}
        people = {w.lower() for n in [self.persona.name] + [c.name for c in self.persona.contacts] for w in n.split()}
        flagged = set()
        for r in private:
            flagged |= {w.lower() for w in re.findall(r"\b(?=[\w#-]*\d)[\w][\w#-]{2,}\b", r["text"])}
            flagged |= {w.lower() for w in re.findall(r"(?<=[a-z,;] )[A-Z][a-zA-Z]{2,}\b", r["text"])}
        flagged -= allowed_words | people
        sentences = re.split(r"(?<=[.!?])\s+", body)
        kept = [s for s in sentences if not (set(re.findall(r"[\w#-]+", s.lower())) & flagged)]
        return " ".join(kept).strip() or "Hope you're doing well! It's been a busy week; let's catch up soon.", \
            len(sentences) - len(kept)

    # ---------------------------------------------------------------- fallback
    def _fallback(self, request: str, trace: list[dict]) -> AgentResult:
        rel = self._relevant(request)[:5]
        ev = "\n".join(f"- [{c.item.record['timestamp'][:16]}] {c.item.record.get('sender')}: "
                       f"{c.item.record['text'][:300]}" for c in rel)
        trace.append({"step": "fallback", "evidence": [c.item.record["event_id"] for c in rel]})
        return self.fallback.run(f"{request}\n\nRelevant records (newest information wins):\n{ev}")
