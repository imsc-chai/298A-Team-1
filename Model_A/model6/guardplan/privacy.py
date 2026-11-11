from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .core import GuardPlanError, MemoryRecord


@dataclass
class Category:
    name: str
    tokens: list[str]
    allowed: set[str] = field(default_factory=set)
    substitute: str = "[confidential information omitted]"


def token_aliases(token: str) -> set[str]:
    """Recognize values with/without their descriptive prefix; no task gold used."""
    aliases = {token}
    value = re.sub(r"^(?:confirmation|ticket|chart|diagnosis|gate code|keybox|package locker code|"
                   r"wi-fi password|invoice|tax id|acct|claim)\s*#?\s*", "", token, flags=re.I)
    if len(value) >= 4:
        aliases.add(value)
    aliases.update(re.findall(r"\$[\d,]+\.\d{2}(?:/hr)?|\b[A-Z]{2,}[-\d][A-Z\d-]+\b", token))
    if token.lower().startswith("gate code"):
        aliases.add(value.lstrip("#"))
        aliases.add("#" + value.lstrip("#"))
    provider = re.match(r"^(?:Dr\.?|Doctor)\s+(.+)$", token, flags=re.I)
    if provider:
        name = provider.group(1).strip()
        surname = name.split()[-1]
        aliases.update({name, "Dr. " + surname, "Dr " + surname, "Doctor " + surname})
    return aliases


# Bounded category-disclosure recognition, separate from the dataset's exact
# forbidden tokens. These noun phrases reveal health context even when they do
# not repeat a diagnosis or provider name. This is not a general semantic judge.
HEALTH_DISCLOSURE_PATTERNS = (
    r"\b(?:medical|health(?:[ -]?(?:care|related))?|clinical|doctor(?:['’]s)?|"
    r"physician(?:['’]s)?|therap(?:y|ist(?:['’]s)?)|physio(?:therapy)?|"
    r"physical[ -]+therapy|rehab(?:ilitation)?|pediatric|paediatric|dental|dentist(?:['’]s)?)"
    r"[ -]+(?:(?:follow[ -]?up|check[ -]?up)[ -]+)?"
    r"(?:appointment|appt|visit|session|consultation|check[ -]?up|treatment)s?\b",
    r"\b(?:medical|health|mental[ -]+health|clinical|pediatric|paediatric|dental)"
    r"[ -]+(?:condition|issue|concern|problem|reason|treatment|care|procedure)s?\b",
)

HEALTH_VISIT = re.compile(r"\b(seeing|visiting|consulting|see|visit|consult)\s+(?:(?:my|a|the|our)\s+)?"
                          r"(?:doctor|physician|therapist|dentist|clinician)\b", re.I)

STRUCTURAL_FIELDS = {"id", "persona_id", "source", "type", "date", "timestamp", "valid_from", "superseded_by",
                     "deleted", "busy", "start_time", "end_time", "duration_minutes", "weekday", "weekdays",
                     "strict_conflict", "is_sensitive"}


class Privacy:
    def __init__(self, policy: dict, profile: dict, typing_enabled=True, release_enabled=True):
        self.typing_enabled, self.release_enabled = typing_enabled, release_enabled
        self.categories: dict[str, Category] = {}
        for item in policy.get("sensitive_categories", []):
            name = item["category"]
            if item.get("forbidden_cross_tokens"):
                clients = {c["client_id"]: c for c in profile.get("active_clients", [])}
                for client_id, tokens in item["forbidden_cross_tokens"].items():
                    email = clients.get(client_id, {}).get("contact_email")
                    allowed = {c["id"] for c in profile.get("contacts", []) if email and c.get("email") == email}
                    self.categories[name + ":" + client_id] = Category(
                        name + ":" + client_id, tokens, allowed, "[other client's confidential details]")
            else:
                self.categories[name] = Category(name, list(item.get("forbidden_tokens", [])),
                                                 set(item.get("allowed_contacts", [])),
                                                 item.get("redaction_substitute", "[confidential information omitted]"))
        health = [c for c in self.categories.values()
                  if re.search(r"health|medical|pediatric|paediatric|patient_hipaa", c.name, re.I)]
        self.category_patterns: list[tuple[re.Pattern, str]] = []
        if health:
            if len(health) == 1:
                category = health[0]
            else:
                # Generic "medical appointment" cannot establish whether it is
                # a child's or a patient's information. Never union their ACLs.
                allowed = set.intersection(*(c.allowed for c in health))
                substitutes = {c.substitute for c in health}
                substitute = next(iter(substitutes)) if len(substitutes) == 1 else "[confidential information omitted]"
                category = Category("_derived_health_disclosure", [], allowed, substitute)
                self.categories[category.name] = category
            providers = {alias for c in health for token in c.tokens
                         if re.match(r"^(?:Dr\.?|Doctor)\s+", token, re.I) for alias in token_aliases(token)}
            patterns = list(HEALTH_DISCLOSURE_PATTERNS)
            if providers:
                provider = "(?:" + "|".join(re.escape(v) for v in sorted(providers, key=len, reverse=True)) + ")"
                suffix = r"\s+(?:with|at)\s+" + provider + r"\b"
                # Treat appointment + provider as one disclosure, rather than
                # substituting the same noun phrase twice in different roles.
                patterns = ["(?:" + p + ")(?:(?:" + suffix + "))?" for p in patterns]
                patterns.insert(0, re.escape(category.substitute) + suffix)
            patterns.append(HEALTH_VISIT.pattern)
            self.category_patterns = [(re.compile(pattern, re.I), category.name) for pattern in patterns]
        self.categories["unclassified_sensitive"] = Category("unclassified_sensitive", [])
        self.handles: dict[str, tuple[str, str]] = {}
        self._patterns = []
        for cat in self.categories.values():
            for token in cat.tokens:
                for alias in token_aliases(token):
                    self._patterns.append((alias, cat.name))
        self._patterns.sort(key=lambda pair: len(pair[0]), reverse=True)

    def handle(self, value: str, category: str) -> str:
        identifier = hashlib.sha256((category + "\0" + value).encode()).hexdigest()[:12]
        name = re.sub(r"\W+", "_", category).upper()
        handle = "⟨" + name + "_" + identifier + "⟩"
        self.handles[handle] = (value, category)
        return handle

    def type_text(self, text: str) -> str:
        if not self.typing_enabled:
            return text
        # One regex pass prevents a shorter alias from corrupting a new handle.
        patterns = self._patterns + [(raw, category) for raw, category in self.handles.values() if raw]
        patterns.sort(key=lambda pair: len(pair[0]), reverse=True)
        if not patterns and not self.category_patterns:
            return text
        lookup = {token.casefold(): cat for token, cat in reversed(patterns)}
        alternatives = [pattern.pattern for pattern, _ in self.category_patterns]
        alternatives.extend(re.escape(token) for token, _ in patterns)
        pattern = r"⟨[^⟨⟩]+⟩|" + "|".join(alternatives)
        def replace(match):
            value = match.group()
            if value.startswith("⟨"):
                return value
            for expression, category in self.category_patterns:
                if expression.fullmatch(value):
                    return self.handle(value, category)
            return self.handle(value, lookup[value.casefold()])
        return re.sub(pattern, replace, text, flags=re.I)

    def type_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.type_text(value)
        if isinstance(value, list):
            return [self.type_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self.type_value(v) for k, v in value.items()}
        return value

    def classify(self, text: str, marked: bool = False) -> str:
        for token, category in self._patterns:
            if token.casefold() in text.casefold():
                return category
        for pattern, category in self.category_patterns:
            if pattern.search(text):
                return category
        if marked:
            health = next((c for c in self.categories if "health" in c or "medical" in c), None)
            return health or "unclassified_sensitive"
        return "none"

    def record(self, record: MemoryRecord) -> dict:
        result = record.to_dict()
        if not self.typing_enabled:
            return result
        result["text"] = self.type_text(record.text)
        result["data"] = self.type_value(record.data)
        sensitive = record.sensitivity != "none" or record.data.get("is_sensitive")
        if sensitive:
            category = record.sensitivity if record.sensitivity in self.categories else "unclassified_sensitive"
            health = bool(re.search(r"health|medical|pediatric|paediatric|patient_hipaa", category, re.I))
            if record.data or health or result["text"] == record.text:
                result["text"] = self.handle(record.text, category)
            # Inherit the record's category for EVERY non-structural value, not
            # just a short list of title/body fields or known token matches.
            result["data"] = {
                key: value if key in STRUCTURAL_FIELDS or value is None or value == "" or value == [] or value == {} else self.handle(
                    value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), category)
                for key, value in record.data.items()}
        return result

    @staticmethod
    def _substitute(raw: str, cat: Category) -> str:
        visit = HEALTH_VISIT.fullmatch(raw)
        if visit:
            return ("attending a " if visit.group(1).lower().endswith("ing") else "attend a ") + cat.substitute
        return cat.substitute

    def resolve_internal(self, value: Any) -> Any:
        """Only on-device tools can resolve handles independent of a recipient."""
        return self._replace(value, None, internal=True)

    def release(self, value: Any, recipient_id: str | None) -> Any:
        if not self.typing_enabled or not self.release_enabled:
            return copy.deepcopy(value)
        released = self._replace(copy.deepcopy(value), recipient_id)
        # Protect generated category wording even if a caller did not first
        # type it, or it arose while a handle was being resolved.
        released = self._redact_category_phrases(released, recipient_id)
        self.check(released, recipient_id)
        return released

    def _redact_category_phrases(self, value: Any, recipient_id: str | None) -> Any:
        if isinstance(value, dict):
            return {k: self._redact_category_phrases(v, recipient_id) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_category_phrases(v, recipient_id) for v in value]
        if not isinstance(value, str):
            return value
        for pattern, category in self.category_patterns:
            cat = self.categories[category]
            if recipient_id not in cat.allowed:
                # Approved substitutes are stable under repeated release.
                parts = re.split("(" + re.escape(cat.substitute) + ")", value, flags=re.I)
                value = "".join(part if part.casefold() == cat.substitute.casefold() else
                                pattern.sub(lambda m: self._substitute(m.group(), cat), part) for part in parts)
        return value

    def _replace(self, value: Any, recipient_id: str | None, internal=False) -> Any:
        if isinstance(value, dict):
            return {k: self._replace(v, recipient_id, internal) for k, v in value.items()}
        if isinstance(value, list):
            return [self._replace(v, recipient_id, internal) for v in value]
        if not isinstance(value, str):
            return value
        def replace(match):
            handle = match.group()
            if handle not in self.handles:
                raise GuardPlanError("Unknown privacy handle")
            raw, category = self.handles[handle]
            cat = self.categories[category]
            return raw if internal or recipient_id in cat.allowed else self._substitute(raw, cat)
        return re.sub(r"⟨[^⟨⟩]+⟩", replace, value)

    def check(self, args: Any, recipient_id: str | None) -> None:
        if not self.typing_enabled or not self.release_enabled:
            return
        self.check_forbidden_tokens(args, recipient_id)
        self.check_category_disclosures(args, recipient_id)
        text = json.dumps(args, ensure_ascii=False).casefold()
        for raw, category in self.handles.values():
            if not raw:
                continue
            cat = self.categories[category]
            remaining = text.replace(cat.substitute.casefold(), "")
            if raw.casefold() in remaining and recipient_id not in cat.allowed:
                raise GuardPlanError("Release blocked: forbidden value for recipient")

    def check_forbidden_tokens(self, args: Any, recipient_id: str | None) -> None:
        """Independent exact-token/alias detector; does not count phrase rules."""
        text = json.dumps(args, ensure_ascii=False).casefold()
        for token, category in self._patterns:
            if token.casefold() in text and recipient_id not in self.categories[category].allowed:
                raise GuardPlanError("Release blocked: forbidden value for recipient")

    def check_category_disclosures(self, args: Any, recipient_id: str | None) -> None:
        """Independent bounded phrase detector, including generated wording."""
        text = json.dumps(args, ensure_ascii=False)
        for pattern, category in self.category_patterns:
            cat = self.categories[category]
            # An explicit policy substitute is approved by the user, even if it
            # contains category words. Other occurrences remain checked.
            remaining = re.sub(re.escape(cat.substitute), "", text, flags=re.I)
            if recipient_id not in cat.allowed and pattern.search(remaining):
                raise GuardPlanError("Release blocked: health-related disclosure for recipient")

    def audit(self) -> list[dict]:
        # Actual values remain in process memory, never in planner logs.
        return [{"handle": handle, "category": cat} for handle, (_, cat) in self.handles.items()]
