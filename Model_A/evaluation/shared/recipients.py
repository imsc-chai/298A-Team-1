"""Bind outbound destinations from the original request, never LLM output."""
from __future__ import annotations

import copy
import re
from email.utils import parseaddr

from .core import GuardPlanError


def address_key(value: str, channel: str) -> str:
    if channel == "email":
        if "," in value or ";" in value:
            return ""
        address = parseaddr(value)[1]
        return address.casefold() if re.fullmatch(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+", address) else ""
    digits = re.sub(r"\D", "", value) if re.fullmatch(r"\+?[\d ()-]+", value) else ""
    return digits if 7 <= len(digits) <= 15 else ""


class RecipientBinding:
    """Conservative resolution of one contact/address from trusted user text.

    Explicit destinations take priority over a named sender, then unique contact
    mentions. Quoted message content cannot override these roles. Ambiguous or
    missing destinations fail closed. Contacts are frozen before planning so a
    planner cannot acquire release privileges by changing its contact snapshot.
    """

    DESTINATION = re.compile(
        r"\b(?:reply|respond|write|email|sms|text|message|response)\s+(?:back\s+)?to\s+|"
        r"(?:^|\bplease\s+|\band\s+)(?:email|text|message|notify|tell|inform)\s+", re.I)
    SENDER = re.compile(r"\s+(?:(?:from|at|with|of)\s+[^:;.!?\n]{1,100}?\s+)?"
                        r"(?:texted|emailed|messaged|asked|wrote|wants|requested|sent|is asking)\b", re.I)
    STOP = re.compile(r"[;!?\n]|\b(?:about|regarding|letting|saying|asking|because|to let)\b", re.I)

    def __init__(self, contacts: list[dict]):
        self.contacts = {c["id"]: copy.deepcopy(c) for c in contacts}
        self.targets = copy.deepcopy(self.contacts)
        self.matches: list[tuple[int, int, str]] = []
        self.contact_id: str | None = None
        self.target: dict | None = None
        self.source = "unresolved"
        self.candidates: set[str] = set()
        self.explicit_channels: set[str] = set()
        self.query = ""

    @staticmethod
    def _aliases(contact: dict) -> set[str]:
        parts = re.findall(r"[^\W\d_]+", contact.get("name", "").casefold())
        titles = {"dr", "doctor", "mr", "mrs", "ms", "prof", "professor", "nurse", "principal", "aunt", "uncle"}
        core = parts[1:] if parts and parts[0] in titles else parts
        aliases = {" ".join(parts), " ".join(core), contact["id"]}
        if core:
            aliases.update({core[0], core[-1]})
            if parts[0] in titles:
                aliases.add(parts[0] + " " + core[-1])
        return {v for v in aliases if v}

    def _references(self, text: str) -> list[tuple[int, int, str]]:
        found = []
        for identifier, contact in self.contacts.items():
            for alias in self._aliases(contact):
                expression = r"[\s.'’\-]+".join(re.escape(p) for p in alias.split())
                for match in re.finditer(r"(?<!\w)" + expression + r"(?!\w)", text, re.I):
                    found.append((match.start(), match.end(), identifier))
        for channel, pattern in [("email", r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}"),
                                 ("phone", r"\+\d[\d ()-]{6,}\d")]:
            for match in re.finditer(pattern, text):
                raw = match.group()
                ids = [i for i, c in self.contacts.items() if address_key(str(c.get(channel, "")), channel)
                       and address_key(str(c.get(channel, "")), channel) == address_key(raw, channel)]
                if not ids:
                    identifier = "address:" + channel + ":" + address_key(raw, channel)
                    self.targets[identifier] = {"id": None, "name": raw, channel: raw}
                    ids = [identifier]
                found.extend((match.start(), match.end(), i) for i in ids)
        # Full names/addresses disambiguate contained first-name matches. Equal
        # spans with different IDs remain ambiguous, rather than picking first.
        return [ref for ref in set(found) if not any(
            other[0] <= ref[0] and ref[1] <= other[1] and other[:2] != ref[:2] for other in found)]

    def bind(self, query: str) -> None:
        self.query = query
        self.target, self.contact_id = None, None
        self.explicit_channels = set()
        # Mask quoted messages but retain offsets for role resolution.
        quoted = r'"[^"\n]*"|“[^”]*”|(?<!\w)\x27(?:[^\x27]|\x27(?=\w))*\x27(?!\w)|‘[^’]*’'
        role_text = re.sub(quoted, lambda m: " " * len(m.group()), query)
        references = self._references(role_text)
        destinations = list(self.DESTINATION.finditer(role_text))
        direct = set()
        for cue in destinations:
            tail = role_text[cue.end():]
            stop = self.STOP.search(tail)
            end = cue.end() + (stop.start() if stop else len(tail))
            direct.update(i for start, finish, i in references if cue.end() <= start and finish <= end)
        senders = {i for start, end, i in references if self.SENDER.match(role_text[end:])}
        if destinations:
            self.candidates, self.source = direct, "explicit_destination"
            # "Chloe texted ... reply to her" is grounded only when there is a
            # single named sender and no other unquoted contact antecedent.
            if (not direct and len(senders) == 1 and {i for _, _, i in references} == senders
                    and all(re.match(r"(?:him|her|them)\b", role_text[c.end():], re.I) for c in destinations)):
                self.candidates, self.source = senders, "named_sender_pronoun"
        elif senders:
            self.candidates, self.source = senders, "named_sender"
        else:
            self.candidates = {i for start, _, i in references if not re.search(
                r"\b(?:about|regarding|concerning|with)\s*$", role_text[:start], re.I)}
            self.source = "unique_contact_mention"
        if len(self.candidates) == 1:
            key = next(iter(self.candidates))
            self.target = copy.deepcopy(self.targets[key])
            self.contact_id = self.target["id"]
            # Explicit address text also pins the communication channel.
            for channel in ("email", "phone"):
                address = self.target.get(channel)
                pattern = r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}" if channel == "email" else r"\+\d[\d ()-]{6,}\d"
                if address and any(address_key(m.group(), channel) == address_key(address, channel)
                                   for m in re.finditer(pattern, role_text)):
                    self.explicit_channels.add(channel)

    def summary(self) -> dict:
        if self.target:
            return {"status": "bound", "source": self.source, "contact_id": self.contact_id,
                    "evidence_id": "contact_" + self.contact_id if self.contact_id else "user_request",
                    "name": self.target.get("name"), "email": self.target.get("email"),
                    "phone": self.target.get("phone")}
        return {"status": "ambiguous" if self.candidates else "unresolved", "source": self.source,
                "candidates": sorted(self.candidates)}

    def validate(self, address: str, tool: str) -> dict:
        if not self.target:
            raise GuardPlanError("Intended recipient is ambiguous or unresolved; name one recipient or provide an explicit address")
        channel = "email" if tool == "draft_email" else "phone"
        expected = self.target.get(channel)
        if not expected or self.explicit_channels and channel not in self.explicit_channels:
            raise GuardPlanError("Draft channel does not match the intended recipient's requested/available address")
        if not address_key(address, channel) or address_key(address, channel) != address_key(expected, channel):
            raise GuardPlanError("Recipient mismatch: user request identifies " + str(self.target.get("name"))
                                 + "; expected to=" + str(expected) + ". Repair the draft for that recipient before privacy release")
        return {"contact_id": self.contact_id, "address": expected, "source": self.source,
                "evidence_id": "contact_" + self.contact_id if self.contact_id else "user_request"}
