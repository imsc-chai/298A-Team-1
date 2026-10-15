"""Local storage: one SQLite file per workspace (or in memory for the harness).

Everything the system keeps about a person lives here: records and their full-text index, vectors,
what the assistant sent and booked, workflows, executions and node runs, approvals, the audit log,
deletion receipts, policies and the egress log. `text_columns()` lists every column that can hold
user text, so deletion and the storage audit can cover all of them.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from p17sim.system.text import tokens

SCHEMA = """
CREATE TABLE IF NOT EXISTS personas(id TEXT PRIMARY KEY, body TEXT);
CREATE TABLE IF NOT EXISTS contacts(id TEXT PRIMARY KEY, persona_id TEXT, name TEXT, relation TEXT, email TEXT);
CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, persona_id TEXT, ts TEXT, end_ts TEXT, channel TEXT,
  event_type TEXT, sender TEXT, text TEXT, permission TEXT, data_class TEXT, deleted INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS records_by_persona ON records(persona_id, ts);
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(id UNINDEXED, persona_id UNINDEXED, body,
  tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS vectors(record_id TEXT PRIMARY KEY, persona_id TEXT, dim INTEGER, blob BLOB);
CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY, persona_id TEXT, ts TEXT, to_id TEXT, to_name TEXT,
  subject TEXT, body TEXT, execution_id TEXT, approved INTEGER);
CREATE TABLE IF NOT EXISTS events_created(id TEXT PRIMARY KEY, persona_id TEXT, date TEXT, start_min INTEGER,
  duration INTEGER, with_id TEXT, with_name TEXT, title TEXT, cancelled INTEGER DEFAULT 0, execution_id TEXT,
  ts TEXT);
CREATE TABLE IF NOT EXISTS workflows(id TEXT PRIMARY KEY, persona_id TEXT, name TEXT, source TEXT, goal TEXT,
  graph TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS triggers(id TEXT PRIMARY KEY, persona_id TEXT, workflow_id TEXT, kind TEXT, spec TEXT,
  enabled INTEGER DEFAULT 1, last_fired TEXT);
CREATE TABLE IF NOT EXISTS executions(id TEXT PRIMARY KEY, workflow_id TEXT, persona_id TEXT, generator TEXT,
  trigger TEXT, goal TEXT, status TEXT, started TEXT, finished TEXT, answer TEXT, profile TEXT);
CREATE TABLE IF NOT EXISTS node_runs(id INTEGER PRIMARY KEY, execution_id TEXT, persona_id TEXT, node_id TEXT,
  type TEXT, status TEXT, location TEXT, data_class TEXT, input TEXT, output TEXT, error TEXT, started TEXT,
  wall_s REAL);
CREATE TABLE IF NOT EXISTS approvals(id TEXT PRIMARY KEY, persona_id TEXT, execution_id TEXT, node_id TEXT,
  tool TEXT, request TEXT, status TEXT, ts TEXT, decided_ts TEXT);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, persona_id TEXT, ts TEXT, actor TEXT, action TEXT,
  target TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS deletion_receipts(id TEXT PRIMARY KEY, persona_id TEXT, ts TEXT, request TEXT,
  record_ids TEXT, redactions INTEGER, audit TEXT);
CREATE TABLE IF NOT EXISTS policies(persona_id TEXT, kind TEXT, key TEXT, value TEXT,
  PRIMARY KEY(persona_id, kind, key));
CREATE TABLE IF NOT EXISTS egress(id INTEGER PRIMARY KEY, ts TEXT, persona_id TEXT, host TEXT, purpose TEXT,
  bytes INTEGER, persona_bytes INTEGER, allowed INTEGER);
"""

# Columns that may hold user text (deletion redacts them; the storage audit searches them).
TEXT_COLUMNS = [("records", "text"), ("outbox", "subject"), ("outbox", "body"), ("events_created", "title"),
                ("workflows", "goal"), ("workflows", "graph"), ("executions", "goal"), ("executions", "answer"),
                ("node_runs", "input"), ("node_runs", "output"), ("node_runs", "error"),
                ("approvals", "request"), ("audit", "detail"), ("deletion_receipts", "request")]


def _fts_query(text: str) -> str:
    terms = sorted(set(tokens(text)))
    return " OR ".join(f'"{t}"' for t in terms)


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    def close(self) -> None:
        self.db.close()

    def _x(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.db.execute(sql, tuple(args))
            self.db.commit()
            return cur

    def _rows(self, sql: str, args: Iterable[Any] = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, tuple(args)).fetchall()]

    @staticmethod
    def text_columns() -> list[tuple[str, str]]:
        return list(TEXT_COLUMNS)

    # ---------------------------------------------------------------- personas
    def put_persona(self, persona: dict) -> None:
        self._x("INSERT OR REPLACE INTO personas VALUES (?, ?)", (persona["persona_id"], json.dumps(persona)))
        for c in persona.get("contacts", []):
            self._x("INSERT OR REPLACE INTO contacts VALUES (?, ?, ?, ?, ?)",
                    (c["contact_id"], persona["persona_id"], c["name"], c["relation"], c["email"]))

    def persona(self, pid: str) -> dict | None:
        r = self._rows("SELECT body FROM personas WHERE id = ?", (pid,))
        return json.loads(r[0]["body"]) if r else None

    def contacts(self, pid: str) -> list[dict]:
        return self._rows("SELECT * FROM contacts WHERE persona_id = ? ORDER BY id", (pid,))

    # ---------------------------------------------------------------- records
    def add_record(self, rec: dict, data_class: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,0)",
                            (rec["event_id"], rec["persona_id"], rec["timestamp"], rec.get("end"), rec["channel"],
                             rec.get("event_type"), rec.get("sender"), rec["text"], rec.get("permission_label"),
                             data_class))
            self.db.execute("DELETE FROM records_fts WHERE id = ?", (rec["event_id"],))
            self.db.execute("INSERT INTO records_fts VALUES (?, ?, ?)",
                            (rec["event_id"], rec["persona_id"], f"{rec.get('sender') or ''} {rec['text']}"))
            self.db.commit()

    def record(self, rid: str) -> dict | None:
        r = self._rows("SELECT * FROM records WHERE id = ?", (rid,))
        return r[0] if r else None

    def visible(self, pid: str, before: str | None = None, channels: set[str] | None = None) -> list[dict]:
        rows = self._rows("SELECT * FROM records WHERE persona_id = ? AND deleted = 0 AND (? IS NULL OR ts < ?) "
                          "ORDER BY ts, id", (pid, before, before))
        return [r for r in rows if channels is None or r["channel"] in channels]

    def search(self, pid: str, query: str, channels: set[str] | None = None, limit: int = 8,
               before: str | None = None) -> list[dict]:
        q = _fts_query(query)
        if not q:
            return []
        rows = self._rows(
            "SELECT r.*, bm25(records_fts) AS score FROM records_fts JOIN records r ON r.id = records_fts.id "
            "WHERE records_fts MATCH ? AND records_fts.persona_id = ? AND r.deleted = 0 AND (? IS NULL OR r.ts < ?) "
            "ORDER BY score, r.ts DESC LIMIT ?", (q, pid, before, before, max(limit * 4, 40)))
        rows = [r for r in rows if channels is None or r["channel"] in channels]
        return rows[:limit]

    def mark_deleted(self, ids: list[str]) -> list[str]:
        done = []
        with self.lock:
            for rid in ids:
                cur = self.db.execute("UPDATE records SET deleted = 1 WHERE id = ? AND deleted = 0", (rid,))
                if cur.rowcount:
                    self.db.execute("DELETE FROM records_fts WHERE id = ?", (rid,))
                    self.db.execute("DELETE FROM vectors WHERE record_id = ?", (rid,))
                    done.append(rid)
            self.db.commit()
        return done

    # ---------------------------------------------------------------- vectors
    def put_vector(self, rid: str, pid: str, vec: bytes, dim: int) -> None:
        self._x("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)", (rid, pid, dim, vec))

    def vectors(self, pid: str) -> list[dict]:
        return self._rows("SELECT v.* FROM vectors v JOIN records r ON r.id = v.record_id "
                          "WHERE v.persona_id = ? AND r.deleted = 0", (pid,))

    # ---------------------------------------------------------------- what the assistant did
    def add_outbox(self, row: dict) -> int:
        return self._x("INSERT INTO outbox(persona_id, ts, to_id, to_name, subject, body, execution_id, approved) "
                       "VALUES (?,?,?,?,?,?,?,?)", (row["persona_id"], row["ts"], row["to_id"], row["to_name"],
                                                    row.get("subject", ""), row.get("body", ""),
                                                    row.get("execution_id"), int(row.get("approved", 0)))).lastrowid

    def outbox(self, pid: str, execution_id: str | None = None) -> list[dict]:
        return self._rows("SELECT * FROM outbox WHERE persona_id = ? AND (? IS NULL OR execution_id = ?) ORDER BY id",
                          (pid, execution_id, execution_id))

    def add_event(self, e: dict) -> None:
        self._x("INSERT INTO events_created VALUES (?,?,?,?,?,?,?,?,0,?,?)",
                (e["id"], e["persona_id"], e["date"], e["start_min"], e["duration"], e["with_id"], e["with_name"],
                 e["title"], e.get("execution_id"), e.get("ts")))

    def cancel_event(self, event_id: str) -> bool:
        return bool(self._x("UPDATE events_created SET cancelled = 1 WHERE id = ? AND cancelled = 0",
                            (event_id,)).rowcount)

    def events(self, pid: str, execution_id: str | None = None) -> list[dict]:
        return self._rows("SELECT * FROM events_created WHERE persona_id = ? AND (? IS NULL OR execution_id = ?) "
                          "ORDER BY ts, id", (pid, execution_id, execution_id))

    def calendar_on(self, pid: str, day: date) -> list[dict]:
        out = []
        lo, hi = day.isoformat(), (day + timedelta(days=1)).isoformat()
        for r in self._rows("SELECT * FROM records WHERE persona_id = ? AND channel = 'calendar' AND deleted = 0 "
                            "AND ts >= ? AND ts < ?", (pid, lo, hi)):
            start = datetime.fromisoformat(r["ts"])
            end = datetime.fromisoformat(r["end_ts"]) if r["end_ts"] else start + timedelta(minutes=30)
            out.append({"id": r["id"], "start": start.strftime("%H:%M"), "end": end.strftime("%H:%M"),
                        "title": r["text"], "source": "calendar"})
        for e in self._rows("SELECT * FROM events_created WHERE persona_id = ? AND date = ? AND cancelled = 0",
                            (pid, day.isoformat())):
            s, d = e["start_min"], e["duration"]
            out.append({"id": e["id"], "start": f"{s // 60:02d}:{s % 60:02d}",
                        "end": f"{(s + d) // 60:02d}:{(s + d) % 60:02d}", "title": e["title"], "source": "assistant"})
        return sorted(out, key=lambda e: (e["start"], e["id"]))

    # ---------------------------------------------------------------- workflows and executions
    def save_workflow(self, wf: dict, pid: str, ts: str) -> None:
        self._x("INSERT OR REPLACE INTO workflows VALUES (?,?,?,?,?,?,?)",
                (wf["id"], pid, wf["name"], wf["source"], wf.get("goal", ""), json.dumps(wf), ts))

    def workflow(self, wid: str) -> dict | None:
        r = self._rows("SELECT graph FROM workflows WHERE id = ?", (wid,))
        return json.loads(r[0]["graph"]) if r else None

    def workflows(self, pid: str) -> list[dict]:
        return self._rows("SELECT id, name, source, goal, ts FROM workflows WHERE persona_id = ? ORDER BY ts DESC",
                          (pid,))

    def start_execution(self, eid: str, wid: str, pid: str, generator: str, trigger: str, goal: str, ts: str) -> None:
        self._x("INSERT INTO executions VALUES (?,?,?,?,?,?,'running',?,NULL,NULL,NULL)",
                (eid, wid, pid, generator, trigger, goal, ts))

    def finish_execution(self, eid: str, status: str, answer: str, profile: dict, ts: str) -> None:
        self._x("UPDATE executions SET status = ?, answer = ?, profile = ?, finished = ? WHERE id = ?",
                (status, answer, json.dumps(profile), ts, eid))

    def set_execution_status(self, eid: str, status: str) -> None:
        self._x("UPDATE executions SET status = ? WHERE id = ?", (status, eid))

    def execution(self, eid: str) -> dict | None:
        r = self._rows("SELECT * FROM executions WHERE id = ?", (eid,))
        if not r:
            return None
        e = r[0]
        e["profile"] = json.loads(e["profile"]) if e["profile"] else None
        return e

    def executions(self, pid: str, limit: int = 50) -> list[dict]:
        return self._rows("SELECT id, workflow_id, generator, trigger, goal, status, started, finished, answer "
                          "FROM executions WHERE persona_id = ? ORDER BY started DESC, id DESC LIMIT ?", (pid, limit))

    def add_node_run(self, row: dict) -> int:
        return self._x("INSERT INTO node_runs(execution_id, persona_id, node_id, type, status, location, data_class, "
                       "input, output, error, started, wall_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (row["execution_id"], row["persona_id"], row["node_id"], row["type"], row["status"],
                        row["location"], row["data_class"], row["input"], row["output"], row.get("error"),
                        row["started"], row["wall_s"])).lastrowid

    def node_runs(self, eid: str) -> list[dict]:
        return self._rows("SELECT * FROM node_runs WHERE execution_id = ? ORDER BY id", (eid,))

    # ---------------------------------------------------------------- approvals, audit, receipts
    def add_approval(self, row: dict) -> None:
        self._x("INSERT INTO approvals VALUES (?,?,?,?,?,?,'pending',?,NULL)",
                (row["id"], row["persona_id"], row["execution_id"], row["node_id"], row["tool"],
                 json.dumps(row["request"]), row["ts"]))

    def approval(self, aid: str) -> dict | None:
        r = self._rows("SELECT * FROM approvals WHERE id = ?", (aid,))
        if not r:
            return None
        r[0]["request"] = json.loads(r[0]["request"])
        return r[0]

    def approvals(self, pid: str, status: str | None = None) -> list[dict]:
        rows = self._rows("SELECT * FROM approvals WHERE persona_id = ? AND (? IS NULL OR status = ?) ORDER BY ts DESC",
                          (pid, status, status))
        for r in rows:
            r["request"] = json.loads(r["request"])
        return rows

    def decide(self, aid: str, status: str, ts: str) -> None:
        self._x("UPDATE approvals SET status = ?, decided_ts = ? WHERE id = ?", (status, ts, aid))

    def audit_log(self, pid: str, ts: str, actor: str, action: str, target: str, detail: Any) -> None:
        self._x("INSERT INTO audit(persona_id, ts, actor, action, target, detail) VALUES (?,?,?,?,?,?)",
                (pid, ts, actor, action, target, json.dumps(detail, default=str)[:4000]))

    def audits(self, pid: str, limit: int = 200) -> list[dict]:
        return self._rows("SELECT * FROM audit WHERE persona_id = ? ORDER BY id DESC LIMIT ?", (pid, limit))

    def add_receipt(self, row: dict) -> None:
        self._x("INSERT INTO deletion_receipts VALUES (?,?,?,?,?,?,?)",
                (row["id"], row["persona_id"], row["ts"], row["request"], json.dumps(row["record_ids"]),
                 row["redactions"], json.dumps(row.get("audit", []))))

    def receipts(self, pid: str) -> list[dict]:
        rows = self._rows("SELECT * FROM deletion_receipts WHERE persona_id = ? ORDER BY ts DESC", (pid,))
        for r in rows:
            r["record_ids"] = json.loads(r["record_ids"])
            r["audit"] = json.loads(r["audit"])
        return rows

    # ---------------------------------------------------------------- policy and egress
    def set_policy(self, pid: str, kind: str, key: str, value: str) -> None:
        self._x("INSERT OR REPLACE INTO policies VALUES (?,?,?,?)", (pid, kind, key, value))

    def policy(self, pid: str, kind: str) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self._rows("SELECT key, value FROM policies WHERE persona_id = ? "
                                                         "AND kind = ?", (pid, kind))}

    def log_egress(self, row: dict) -> None:
        self._x("INSERT INTO egress(ts, persona_id, host, purpose, bytes, persona_bytes, allowed) "
                "VALUES (?,?,?,?,?,?,?)", (row["ts"], row.get("persona_id"), row["host"], row["purpose"],
                                           row["bytes"], row["persona_bytes"], int(row["allowed"])))

    def egress_log(self, pid: str | None = None, limit: int = 500) -> list[dict]:
        return self._rows("SELECT * FROM egress WHERE (? IS NULL OR persona_id = ?) ORDER BY id DESC LIMIT ?",
                          (pid, pid, limit))

    # ---------------------------------------------------------------- triggers
    def add_trigger(self, row: dict) -> None:
        self._x("INSERT OR REPLACE INTO triggers VALUES (?,?,?,?,?,1,NULL)",
                (row["id"], row["persona_id"], row["workflow_id"], row["kind"], json.dumps(row["spec"])))

    def triggers(self, pid: str) -> list[dict]:
        rows = self._rows("SELECT * FROM triggers WHERE persona_id = ? AND enabled = 1", (pid,))
        for r in rows:
            r["spec"] = json.loads(r["spec"])
        return rows

    def trigger_fired(self, tid: str, ts: str) -> None:
        self._x("UPDATE triggers SET last_fired = ? WHERE id = ?", (ts, tid))

    # ---------------------------------------------------------------- raw access for deletion and audits
    def rows_with_text(self, pid: str, table: str, column: str) -> list[tuple[Any, str]]:
        key = "rowid"
        sql = f"SELECT {key} AS k, {column} AS v FROM {table} WHERE persona_id = ? AND {column} IS NOT NULL"
        if table == "records":
            sql += " AND deleted = 0"
        return [(r["k"], r["v"]) for r in self._rows(sql, (pid,))]

    def update_text(self, table: str, column: str, key: Any, value: str) -> None:
        self._x(f"UPDATE {table} SET {column} = ? WHERE rowid = ?", (value, key))
