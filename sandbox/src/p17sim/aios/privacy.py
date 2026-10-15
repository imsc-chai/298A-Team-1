"""Privacy layer: where data may be processed, which tools may run, what leaves the device, and how
deletion reaches every copy.

- Data classes: public < personal < sensitive. A record with permission `deny` is sensitive, any other
  user record is personal. A node's output carries the highest class of its inputs and sources.
- Location policy: local nodes may process anything; cloud nodes only public data.
- Tool policy: allow / ask / deny per tool, from the persona's permission records unless overridden.
- Egress gateway: every outbound request goes through `post`. Hosts off the allowlist are blocked;
  every attempt is logged with its size and how many bytes of persona text it carried.
- Deletion manager: removes records from the table, the FTS index and the vectors, then redacts what
  they said from every other column that can hold user text, and writes a receipt.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from typing import Any

from p17sim.aios.store import Store
from p17sim.system.text import contains_value

CLASSES = ["public", "personal", "sensitive"]
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
REDACTED = "[deleted]"
# Tool permission labels from the trace -> policy decision.
FROM_LABEL = {"send": "allow", "read": "allow", "draft": "ask", "deny": "deny"}
DEFAULT_TOOLS = {"calendar.read": "allow", "mail.read": "allow", "files.read": "allow",
                 "calendar.create": "allow", "mail.send": "ask", "memory.forget": "allow"}


class PolicyError(Exception):
    pass


class EgressBlocked(PolicyError):
    pass


def max_class(*classes: str) -> str:
    return max((c for c in classes if c), key=CLASSES.index, default="public")


def classify_record(rec: dict) -> str:
    return "sensitive" if rec.get("permission_label") == "deny" else "personal"


class Policy:
    def __init__(self, store: Store, pid: str):
        self.store, self.pid = store, pid

    def tool(self, name: str) -> str:
        override = self.store.policy(self.pid, "tool").get(name)
        if override:
            return override
        perm = next((r["permission"] for r in self.store._rows(
            "SELECT permission FROM records WHERE persona_id = ? AND channel = 'permission' AND text LIKE ?",
            (self.pid, f"{name}:%"))), None)
        return FROM_LABEL.get(perm, DEFAULT_TOOLS.get(name, "ask")) if perm else DEFAULT_TOOLS.get(name, "ask")

    def set_tool(self, name: str, value: str) -> None:
        if value not in ("allow", "ask", "deny"):
            raise ValueError(f"policy value must be allow, ask or deny, not {value!r}")
        self.store.set_policy(self.pid, "tool", name, value)

    def tools(self) -> dict[str, str]:
        return {t: self.tool(t) for t in sorted(set(DEFAULT_TOOLS) | set(self.store.policy(self.pid, "tool")))}

    @staticmethod
    def may_process(location: str, data_class: str) -> tuple[bool, str]:
        if location == "local":
            return True, "local processing"
        if data_class == "public":
            return True, "public data may be processed in the cloud"
        return False, f"{data_class} data may only be processed on this device"


class EgressGateway:
    def __init__(self, store: Store, allow_hosts: set[str] | None = None):
        self.store = store
        self.allow_hosts = set(allow_hosts or LOCAL_HOSTS)

    @staticmethod
    def _send(url: str, body: bytes, timeout: int) -> bytes:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def post(self, url: str, body: bytes, purpose: str, persona_id: str | None = None,
             persona_texts: list[str] | None = None, timeout: int = 900) -> bytes:
        host = urllib.parse.urlparse(url).hostname or ""
        text = body.decode(errors="ignore")
        persona_bytes = sum(len(t.encode()) for t in (persona_texts or []) if len(t) >= 12 and t in text)
        allowed = host in self.allow_hosts
        self.store.log_egress({"ts": datetime.now().isoformat(timespec="seconds"), "persona_id": persona_id,
                               "host": host, "purpose": purpose, "bytes": len(body),
                               "persona_bytes": persona_bytes, "allowed": allowed})
        if not allowed:
            raise EgressBlocked(f"outbound request to {host} blocked by the egress policy")
        return self._send(url, body, timeout)


_CODE = re.compile(r"\b(?=[\w#-]*\d)[\w][\w#-]{2,}\b")
_NAME = re.compile(r"(?<=[a-z0-9,;:] )[A-Z][a-zA-Z]{2,}(?: [A-Z][a-zA-Z]{2,})*")


class DeletionManager:
    def __init__(self, store: Store):
        self.store = store

    def _tokens(self, pid: str, texts: list[str], kept_texts: list[str]) -> list[str]:
        """What to scrub from other stores: the deleted text itself, plus codes and mid-sentence names
        that appear only in deleted records (a contact named in other records is kept)."""
        kept = " ".join(kept_texts)
        found = set()
        for t in texts:
            found |= set(_CODE.findall(t)) | set(_NAME.findall(t))
        distinctive = sorted((tok for tok in found if not contains_value(kept, tok)), key=len, reverse=True)
        return [t for t in texts if len(t) >= 12] + distinctive

    def delete(self, pid: str, record_ids: list[str], request: str, actor: str, ts: str) -> dict:
        rows = [r for r in (self.store.record(i) for i in record_ids) if r and r["persona_id"] == pid]
        texts = [r["text"] for r in rows if not r["deleted"]]
        done = self.store.mark_deleted([r["id"] for r in rows])
        kept = [r["text"] for r in self.store.visible(pid)]
        scrub = self._tokens(pid, texts, kept) + done
        redactions = 0
        for table, column in self.store.text_columns():
            if table == "records":
                continue
            for key, value in self.store.rows_with_text(pid, table, column):
                new = value
                for tok in scrub:
                    new = re.sub(re.escape(tok), REDACTED, new) if tok in new else new
                if new != value:
                    self.store.update_text(table, column, key, new)
                    redactions += 1
        receipt = {"id": f"rcpt-{uuid.uuid4().hex[:10]}", "persona_id": pid, "ts": ts, "request": request,
                   "record_ids": done, "redactions": redactions}
        self.store.add_receipt(receipt)
        self.store.audit_log(pid, ts, actor, "delete", ",".join(done), {"request": request, "redactions": redactions})
        return receipt

    def storage_audit(self, pid: str, value: str) -> list[tuple[str, str, Any]]:
        """Every (table, column, row) that still contains `value`, in any store the system keeps."""
        hits = []
        for table, column in self.store.text_columns():
            for key, text in self.store.rows_with_text(pid, table, column):
                if contains_value(text if isinstance(text, str) else json.dumps(text), value):
                    hits.append((table, column, key))
        return hits
