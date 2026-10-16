# Personal AI OS (complete system, baseline generator) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `development/reference/executing-plans-guide.md` to implement this plan task-by-task.

**Goal:** Build everything in the Personal AI OS except the learned models: SQLite storage, the
privacy layer, an n8n-inspired workflow engine, generators (baseline + templates + pending slots),
the harness on the real runtime, training-data exports, a FastAPI backend and a no-build web front end.

**Architecture:** A per-persona `Workspace` holds a SQLite store, a privacy layer (data classes,
location policy, tool policy, egress gateway, deletion manager, audit) and an `Engine` that runs
`Workflow` graphs of typed nodes. Generators turn a goal into a workflow; the baseline agent builds
one step by step (each model step and tool call becomes a node). The harness and the API both drive
the same `Workspace`.

**Tech Stack:** Python 3.12, sqlite3 (FTS5), pydantic 2, FastAPI + uvicorn (+ httpx for tests),
Ollama (`qwen3.5:4b`, `all-minilm`), vanilla JS + SVG front end, pytest.

**Design:** `docs/plans/2026-10-15-personal-ai-os-design.md`

**Working rules:**
- All commands run from `sandbox/`.
- Branch `savitha/research-pipeline`; new files under `src/p17sim/aios/`.
- `baseline/` is never touched.
- One commit per task; no Claude co-author lines.
- Tick `PROGRESS.md` after each task.

---

### Task 0: Dependencies

**Files:** Modify: `pyproject.toml`, `uv.lock`

**Step 1:**

```bash
~/Library/Python/3.13/bin/uv add "fastapi>=0.115" "uvicorn>=0.30" && ~/Library/Python/3.13/bin/uv add --dev "httpx>=0.27"
~/Library/Python/3.13/bin/uv cache clean
```

**Step 2:** `.venv/bin/python -c "import fastapi, uvicorn, httpx; print('ok')"` → `ok`

**Step 3:** Commit: `Add FastAPI, uvicorn and httpx`.

### Task 1: Data value fix (package locker)

**Files:**
- Modify: `src/p17sim/gen2/skeleton.py` (`_value`, `package_locker` branch)
- Test: `tests/test_gen2.py`

**Step 1: Failing test.**

```python
def test_package_locker_value_is_the_bare_code():
    import numpy as np
    from p17sim.gen2.skeleton import _value
    v = _value("package_locker", np.random.default_rng(0))
    assert re.fullmatch(r"[A-F]-\d{1,2}", v)
```

**Step 2:** Run it → FAIL (`locker E-34`).

**Step 3:** Change the return to `f"{rng.choice(list('ABCDEF'))}-{int(rng.integers(1, 40))}"`. The
same random draws are used, so every other value is unchanged.

**Step 4:**
1. Run the test → PASS.
2. `p17sim generate2` and copy the manifest to `manifests/`.
3. Restart `render2`; only the package-locker texts render again.

**Step 5:** Commit.

### Task 2: Store (SQLite)

**Files:**
- Create: `src/p17sim/aios/__init__.py`, `src/p17sim/aios/store.py`
- Test: `tests/aios/test_store.py`

Schema (`SCHEMA` constant, executed in `Store.__init__`):

```sql
CREATE TABLE IF NOT EXISTS personas(id TEXT PRIMARY KEY, body TEXT);
CREATE TABLE IF NOT EXISTS contacts(id TEXT PRIMARY KEY, persona_id TEXT, name TEXT, relation TEXT, email TEXT);
CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, persona_id TEXT, ts TEXT, end_ts TEXT, channel TEXT,
  event_type TEXT, sender TEXT, text TEXT, permission TEXT, data_class TEXT, deleted INTEGER DEFAULT 0);
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(id UNINDEXED, persona_id UNINDEXED, body,
  tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS vectors(record_id TEXT PRIMARY KEY, persona_id TEXT, dim INTEGER, blob BLOB);
CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY, persona_id TEXT, ts TEXT, to_id TEXT, to_name TEXT,
  subject TEXT, body TEXT, execution_id TEXT, approved INTEGER);
CREATE TABLE IF NOT EXISTS events_created(id TEXT PRIMARY KEY, persona_id TEXT, date TEXT, start_min INTEGER,
  duration INTEGER, with_id TEXT, with_name TEXT, title TEXT, cancelled INTEGER DEFAULT 0, execution_id TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS workflows(id TEXT PRIMARY KEY, persona_id TEXT, name TEXT, source TEXT, goal TEXT,
  graph TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS triggers(id TEXT PRIMARY KEY, persona_id TEXT, workflow_id TEXT, kind TEXT, spec TEXT,
  enabled INTEGER DEFAULT 1, last_fired TEXT);
CREATE TABLE IF NOT EXISTS executions(id TEXT PRIMARY KEY, workflow_id TEXT, persona_id TEXT, generator TEXT,
  trigger TEXT, goal TEXT, status TEXT, started TEXT, finished TEXT, answer TEXT, profile TEXT);
CREATE TABLE IF NOT EXISTS node_runs(id INTEGER PRIMARY KEY, execution_id TEXT, node_id TEXT, type TEXT,
  status TEXT, location TEXT, data_class TEXT, input TEXT, output TEXT, error TEXT, started TEXT, wall_s REAL);
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
```

`Store(path=":memory:")` methods:
- `add_record(rec, data_class)` inserts into `records` and `records_fts`;
- `search(pid, query, channels, limit, before)` uses FTS5 `bm25` (query terms OR-joined; tokens from
  `system/text.tokens`) and returns dicts;
- `record(id)`, `visible(pid, before)`, `calendar_on(pid, date)` (records + events_created);
- `put_vector`/`vectors(pid)`;
- `add_outbox`, `outbox(pid, execution_id=None)`; `add_event`, `cancel_event`, `events(pid, execution_id=None)`;
- `save_workflow`/`workflow`/`workflows`; `start_execution`/`finish_execution`/`execution`/`executions`;
  `add_node_run`/`node_runs`;
- `add_approval`/`approvals`/`decide`; `audit_log`/`audits`; `add_receipt`/`receipts`;
- `set_policy`/`policy`; `log_egress`/`egress_log`;
- `text_columns()` lists every (table, column) holding user text, used by deletion and the audit.

Tests:
1. Insert three records; FTS search returns the matching one first.
2. A record deleted via `mark_deleted` disappears from `search` and `visible`.
3. `calendar_on` merges routine entries and created events, sorted.
4. A file-backed store persists across reopen (`tmp_path / "w.db"`).
5. `text_columns()` includes `node_runs.input`, `node_runs.output`, `outbox.body` and `approvals.request`.

Commit: `Add the SQLite store for the Personal AI OS`.

### Task 3: Privacy layer

**Files:**
- Create: `src/p17sim/aios/privacy.py`
- Test: `tests/aios/test_privacy.py`

Contents:
- `CLASSES = ["public", "personal", "sensitive"]`; `max_class(*cls)`;
  `classify_record(rec) -> "sensitive" if permission == "deny" else "personal"`.
- `Policy(store, pid)`:
  - `tool(name) -> "allow" | "ask" | "deny"`: defaults from the persona's permission records
    (`mail.send: draft` → ask; `send` → allow; `deny` → deny; reads → allow);
  - `may_process(location, data_class) -> (bool, reason)`: local always; cloud only for public;
  - overrides are stored in `policies`.
- `EgressGateway(store, allow_hosts={"localhost","127.0.0.1"})`:
  - `post(url, body: bytes, purpose, persona_id, persona_texts) -> bytes`;
  - a host not on the allowlist raises `EgressBlocked` and is logged with `allowed=0`;
  - allowed calls are logged with bytes and persona bytes.
  - `system/llm.Chat` and `embed` get an optional `gateway` argument and use `gateway.post` when it
    is given, so every model call is logged.
- `DeletionManager(store)`:
  - `delete(pid, record_ids, request, actor) -> receipt`:
    1. mark the records deleted; drop their FTS rows and vectors;
    2. compute redaction tokens: record ids, each deleted text, and tokens in deleted texts that
       contain a digit or are capitalized mid-sentence, minus tokens found in non-deleted records of
       the persona;
    3. replace those tokens in every `text_columns()` value of that persona with `[deleted]`;
    4. write a receipt and an audit row.
  - `storage_audit(pid, value) -> list[(table, column, row_id)]` uses `text.contains_value` over
    every text column, including `records` rows with `deleted=0` only.
- `AuditLog` helper: `log(pid, actor, action, target, detail)`.

Tests:
1. A deny record is classified sensitive.
2. Cloud + personal is refused; cloud + public is allowed.
3. `draft` → ask, `deny` → deny.
4. The gateway blocks `https://api.example.com`, logs it, and raises.
5. After deleting a record that says "gate code 4821" and also appears in a node run's output and an
   approval request, `storage_audit(pid, "4821")` is empty, and the receipt counts the redactions.
6. A token shared with a kept record (e.g. a contact's name) is **not** redacted.

Commit.

### Task 4: Workflow model, expressions and nodes

**Files:**
- Create: `src/p17sim/aios/workflow.py`, `src/p17sim/aios/nodes.py`
- Test: `tests/aios/test_workflow.py`

`workflow.py`:

```python
class Node(BaseModel):
    id: str; type: str; params: dict[str, Any] = {}; label: str = ""
class Edge(BaseModel):
    source: str; target: str
class Workflow(BaseModel):
    id: str; name: str; source: str; goal: str = ""; nodes: list[Node]; edges: list[Edge] = []
    trigger: dict[str, Any] | None = None   # {"kind": "chat"|"time"|"event", ...}
    def order(self) -> list[Node]: ...       # topological; ValueError on a cycle or unknown edge end
def resolve(value, outputs: dict[str, Any]) -> Any:   # "{{ n1.items.0.text }}" -> value; strings mix
```

`nodes.py`: a registry `NODES: dict[str, NodeSpec]` with
`NodeSpec(type, category, location, tool | None, reversible, params_schema, run)`, where
`run(ctx, params) -> (output, data_class)`. Node types:

| Category | Types |
|---|---|
| trigger | `trigger.chat`, `trigger.time`, `trigger.event` |
| context | `context.search` (q, kind, limit), `context.read`, `context.calendar` (date), `context.contacts`, `context.preferences` (key, days) |
| reason | `reason.llm` (prompt, json: bool) — local Ollama via the gateway |
| logic | `logic.if` (left, op, right), `logic.merge` |
| action | `action.send_email` (tool mail.send, irreversible), `action.create_event` (tool calendar.create, reversible), `action.cancel_event`, `action.delete_records` (tool memory.forget, irreversible) |
| human | `human.approval` (message) |
| output | `output.respond` (text) |
| cloud | `cloud.llm` (location cloud; demo of the location policy; needs `P17_CLOUD_URL`) |

`ctx` = `RunContext(workspace, execution_id, persona, now, profile)`.

Tests:
1. `order()` on a 4-node chain; a cycle raises.
2. `resolve("{{n1.items.0.text}}")` returns the string; a missing path raises `KeyError`.
3. Each context node returns rows from a store fixture.
4. `action.create_event` writes `events_created`.
5. `reason.llm` with a `ScriptedChat` returns its content.

Commit.

### Task 5: Engine

**Files:**
- Create: `src/p17sim/aios/engine.py`
- Test: `tests/aios/test_engine.py`

`Engine(workspace).run(workflow, trigger, approver) -> ExecutionResult` (execution id, status
`succeeded|failed|waiting_approval|blocked`, answer, outputs, node runs). For each node in `order()`:
1. Resolve the parameters.
2. Input data class = max of the referenced upstream outputs' classes.
3. Location check (`Policy.may_process`) → on refusal, status `blocked`, logged, stop.
4. If a tool is set, `Policy.tool`:
   - `deny` → fail;
   - `ask` → `approver(request)`, which returns `approved | rejected | pending`. Harness approver:
     `approved`, counts one intervention. App approver: `pending` stores an approval and stops with
     `waiting_approval`; `Engine.resume(approval_id, decision)` runs the stored node.
5. Run the node with a timer; write `node_runs` (input and output JSON truncated to 4 KB) and audit.
6. On an exception: roll back the reversible actions done so far (cancel the events), mark failed.

The answer is the last `output.respond` text. `Engine.step(execution_id, node)` runs a single node
inside an open execution; the baseline generator uses it.

Tests:
1. A chain search → llm → respond succeeds; there are 3 node runs with data class personal.
2. `cloud.llm` fed by a search is blocked and leaves an audit row.
3. `mail.send` with policy ask + harness approver → sent and one intervention.
4. With the app approver → `waiting_approval`, then `resume` sends.
5. Create event then a node that raises → the event is cancelled (rollback).
6. Deny → failed with a policy error.

Commit.

### Task 6: Workspace runtime (clock, ingest, triggers)

**Files:**
- Create: `src/p17sim/aios/workspace.py`
- Test: `tests/aios/test_workspace.py`

`Workspace(store, persona, extra, gateway, model)`:
- `ingest(record)`: classify, store, fire event triggers.
- `advance_to(ts, records)`: replays records `< ts`; time triggers fire at their times.
- `now`; `add_trigger(workflow, kind, spec)`.
- `goal(text, generator, approver) -> ExecutionResult`.
- `tools()` returns an `EngineTools` object with the shell interface (`tool_schemas`,
  `tool_descriptions`, `execute`). Each `execute(name, args)` maps to an engine node
  (`search_records` → `context.search`, ...) and runs through `Engine.step`, so policies, logging
  and data classes apply to the baseline too.

`system/shell.py` keeps its tool names and schemas, which `EngineTools` reuses.

Tests:
1. Replay a fixture persona to day 25: `visible` count = records before then.
2. A daily 08:00 time trigger fires once per day when advancing two days.
3. An event trigger "message from X" fires on ingest.
4. `EngineTools.execute("search_records", ...)` writes a node run.

Commit.

### Task 7: Generators

**Files:**
- Create: `src/p17sim/aios/generators.py`, `src/p17sim/aios/templates.py`
- Test: `tests/aios/test_generators.py`

```python
class Generator(Protocol):
    name: str; status: str  # "ready" | "pending training"
    def run(self, goal: str, ws: Workspace, approver) -> ExecutionResult: ...
```

- `BaselineGenerator`: opens an execution with workflow `[trigger.chat]` and runs
  `agents.baseline.BaselineAgent` with `ws.tools()`.
  - Each model call becomes a `reason.agent_step` node (output: content and tool calls); each tool
    call becomes its node.
  - It ends with `output.respond`. The workflow graph saved = the nodes in order, chained.
- `TemplateGenerator(template_id)`: instantiates a template from `templates.py`:
  - `morning_briefing`: time 08:00 → calendar today → search messages from the last day → llm
    summary → respond;
  - `weekly_recap(to)`: search the week → llm draft → approval → send;
  - `book_meeting(contact, date, duration)`: calendar → llm choose slot (json) → create_event → respond.
- `PendingGenerator("model_a" | "model_b" | "model_c")`: `status = "pending training"`; `run` raises
  `NotTrained`.
- `RoutineReference` (B slot reference): a calendar title that repeats on the same weekday within
  ±45 min at least 3 times → a proposed weekly trigger.
- `SuggestionReference` (C slot reference): per preference key, the option with the most acceptances
  in the last 7 days.

Tests:
1. The baseline with a `ScriptedChat` (search then answer) produces an execution whose nodes are
   `trigger.chat, reason.agent_step, context.search, reason.agent_step, output.respond`.
2. Templates validate (`order()` ok) and run with a scripted LLM.
3. A pending generator raises `NotTrained`.
4. The routine reference finds a planted weekly routine.
5. The suggestion reference picks the most-accepted option.

Commit.

### Task 8: Harness on the runtime

**Files:**
- Modify: `src/p17sim/system/runner.py` (use `Workspace`; read `outbox`/`events_created` for checkers),
  `src/p17sim/agents/reference.py` (oracle and null over `ws.tools()`),
  `src/p17sim/agents/registry.py` (generators)
- Move: `src/p17sim/agents/model_a.py`, `memory.py`, `train.py` → `src/p17sim/agents/drafts/`, with
  their tests marked `skip("Model A pending")`
- Test: `tests/test_system.py` (update)

Steps:
1. The oracle and null sanity tests run on the runtime → oracle 100%, null 0% on utility tasks; the
   deletion storage audit uses `DeletionManager.storage_audit`, which covers every table.
2. The baseline mechanics test with a `ScriptedChat` still passes.
3. Full suite green.

Commit.

### Task 9: Training-data exports

**Files:**
- Create: `src/p17sim/aios/export.py`; CLI `p17sim export-training`
- Test: `tests/aios/test_export.py`

- **Model A:** for each task of train and val personas, one row:
  `{persona_id, split, ts, goal, context_ids (visible record ids before ts), gold_workflow (Workflow JSON
  built from labels: recall/update → search with the fact name → respond with the current value;
  deletion probe → respond "I don't have that"; privacy → search week (non-deny) → llm draft → send to
  contact; scheduling → calendar(date) → create_event(slot computed from the true preference and busy
  times) → respond), checker}`.
  Forget instructions get a gold workflow too (search → delete_records with the fact's record ids).
  - Validation: each gold workflow, executed with a scripted LLM that echoes the gold text, passes
    its checker.
- **Model B:** per persona, a time-ordered sequence of `{ts, weekday, minute, channel, title_or_topic,
  routine_label}`.
- **Model C:** `{persona_id, ts, weekday, hour, key, offered, accepted, true_value}`.
- `data/training/manifest.json` holds row counts, splits and sha256 values.

Tests:
1. Only train and val personas are exported.
2. Gold workflows pass their checkers in the runtime.
3. Model C rows: accepted == (offered == true_value) except for persona noise, and the noise rate is
   within ±0.1 of `pref_noise`.

Commit.

### Task 10: Backend API

**Files:**
- Create: `src/p17sim/aios/api.py`; CLI `p17sim serve` (replaces `system/app.py`, which is deleted)
- Test: `tests/aios/test_api.py` (FastAPI `TestClient`)

Routes (JSON):

```
GET  /api/personas
POST /api/ws/{pid}/load            {day, time}      -> state
POST /api/ws/{pid}/advance         {minutes}        -> fired triggers
GET  /api/ws/{pid}/state
GET  /api/ws/{pid}/records         ?channel&q&limit
GET  /api/ws/{pid}/calendar        ?date
POST /api/ws/{pid}/goal            {text, generator}
GET  /api/ws/{pid}/workflows       ; POST (save) ; GET /api/templates
POST /api/ws/{pid}/workflows/{wid}/run
GET  /api/ws/{pid}/executions      ; GET /api/ws/{pid}/executions/{eid}
POST /api/ws/{pid}/executions/{eid}/undo
GET  /api/ws/{pid}/approvals       ; POST /api/ws/{pid}/approvals/{aid} {decision}
GET  /api/ws/{pid}/policy          ; PUT  {tool, value}
GET  /api/ws/{pid}/privacy         -> egress, receipts, audit tail
POST /api/ws/{pid}/storage-audit   {value}
POST /api/ws/{pid}/delete          {record_ids, request}
GET  /api/generators               ; GET /api/ws/{pid}/suggestions
GET  /api/benchmark                ; GET /api/profile
GET  /                              -> static/index.html
```

Workspaces are cached per persona. One model request runs at a time (a lock), because only one
model fits in 8 GB.

Tests:
1. Personas list.
2. Load day 25 → state.
3. A goal with a scripted generator returns an execution with node runs.
4. A template run with mail ask → approval pending → decide approve → outbox has 1 mail.
5. Delete → storage audit empty.
6. A policy update is reflected.
7. `/` serves HTML.

Commit.

### Task 11: Front end

**Files:**
- Create: `src/p17sim/aios/static/index.html`, `app.js`, `app.css`
- Test: `tests/aios/test_frontend.py` (the page loads; every `fetch('/api/...')` path in `app.js`
  matches a route)

- **Header:** persona select, day/time, Load, Advance +1 h / +1 day, generator select.
- **Tabs:**
  - Assistant;
  - Workflows (SVG canvas: nodes laid out by topological depth, edges as curves, category colours,
    data-class and location badges, a click opens the inspector);
  - Executions;
  - Data;
  - Privacy;
  - Approvals;
  - Benchmark.
- Light and dark modes via `prefers-color-scheme`; a 16 px gutter; works at phone width.
- **Manual check:** `p17sim serve`, open `http://localhost:8017`, run "What is my gate code?" on a
  val persona, and see the workflow and its execution.

Commit.

### Task 12: Baseline experiments, profiling, report

Run after the render finishes and Task 1's re-render:
1. `p17sim build-trace2` → 0 violations, 0 findings.
2. `p17sim audit`.
3. `p17sim export-training`.
4. Oracle and null on test personas.
5. `p17sim eval --system baseline --seed {11,22,33} --personas test` and `--personas ood_test`
   (about 8 h).
6. `p17sim profile-models` (with nothing else running).
7. `p17sim report` → RESULTS.md with the baseline column; the Model A/B/C rows say "pending training".

Commit the results JSON and RESULTS.md (data stays out of git).

### Task 13: Docs and checklist

- A `README` section for the system (how to run the API and UI, the CLI table, architecture diagram).
- `research/PIPELINE.md` stages 7–16 updated to the runtime.
- `PROGRESS.md` ticked with evidence.
- The evidence folder refreshed.
