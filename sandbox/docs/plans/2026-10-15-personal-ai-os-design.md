# Design: Personal AI OS — the complete system with the baseline (Models A, B, C pending)

Team 1, Section 21, Topic 17. Date: 2026-10-15. Status: approved by Savitha on 2026-10-15
(backend: FastAPI).

## Goal

Build the whole Personal AI OS except the learned models: storage, privacy layer, workflow engine,
backend API, web front end, the replay harness and the training-data exports. The reproduced
OpenJarvis-style agent fills the workflow-generator slot. When this is done, the only pending work is
training and plugging in Models A, B and C.

## 1. Core idea (inspired by n8n, not using n8n)

n8n runs workflows that people draw. This system runs workflows that are **generated** from a user's
goal and personal context. A **privacy layer** decides where each step's data may be processed and
which tools may run the actions.

| n8n | This system |
|---|---|
| Workflow = graph of nodes | `Workflow` = typed nodes and edges, stored as JSON |
| Trigger nodes | Triggers: chat goal, time (daily/weekly), event (a record arrives) |
| Node library | Context, Reason (local LLM), Logic, Action, Human (approval), Output, Cloud (blocked for personal data) |
| Expressions `{{ $node.x.json.y }}` | `{{ node_id.path }}` references to upstream outputs |
| Executions view | Every node run is logged with input, output, time, data class and where it was processed |
| A person builds the workflow | A **generator** builds it |

Generators:
- **baseline** (now): the OpenJarvis `native_openhands`-style agent. Each model step and tool call
  becomes a node as it runs, so its run appears as a workflow.
- **template** (now): hand-built workflows (morning briefing, weekly recap, meeting booking) for
  demos, triggers and gold examples.
- **model_a / model_b / model_c** (pending): A generates a full graph from a goal; B proposes
  recurring workflows and triggers from routines; C chooses which proactive workflow to suggest and when.
- Non-learned references for B and C, so their slots show something and there is a bar to beat:
  frequency-based routine detection (B) and a most-accepted-option policy (C).

## 2. Privacy layer

1. **Data classes.** Each value moving between nodes carries a class: `public` < `personal` <
   `sensitive`.
   - Records are classified at ingest: `permission_label = deny` → sensitive; any other user record →
     personal.
   - A node's output class is the highest class among its inputs and its own sources.
2. **Processing location.** Each node declares `local` or `cloud`. Policy: personal and sensitive
   data may be processed only locally; cloud nodes may receive only public data.
   - The engine blocks a violating node before it runs and logs the attempt.
3. **Egress gateway.** All outbound HTTP goes through one client.
   - It uses a host allowlist (localhost by default) and blocks every other host.
   - It logs host, bytes, and bytes of persona text.
4. **Tool policy.** For each persona and each tool, the policy is `allow` / `ask` / `deny`, defaulting
   to the persona's permission records.
   - `ask` → an approval request. In the app it waits for the user; in the harness a simulated user
     approves and one intervention is counted.
   - `deny` → the node fails with a policy error.
   - Actions are tagged reversible (create_event) or irreversible (send_email). A failed execution
     rolls back its reversible actions, and "undo execution" does the same.
5. **Audit log.** Every node run, tool call, policy decision, approval and deletion.
6. **Deletion manager.** Deleting records removes them from:
   - the records table and the full-text index;
   - the vector table;
   - caches;
   - execution logs, by redacting record ids, record text, and code-like or name-like tokens that
     appear only in the deleted records.

   It then writes a receipt, and a storage audit can search every store for a value.
7. **Hook for Model D.** A learned safety critic can be added later as an extra policy check before
   action nodes. Not built now.

## 3. Storage

One SQLite file per workspace (standard library; FTS5 checked available, SQLite 3.45.3). The harness
uses an in-memory database with the same schema.

Tables:
- `personas`, `contacts`;
- `records` + `records_fts`, `vectors`;
- `outbox`, `events_created`;
- `workflows`, `triggers`;
- `executions`, `node_runs`;
- `approvals`, `audit`, `deletion_receipts`;
- `policies`, `egress`.

## 4. Backend

FastAPI + uvicorn, small new dependencies. Routes:
- personas;
- workspace load, replaying to a moment;
- clock advance, which fires triggers;
- records and calendar browsing and search;
- goals (generate → run);
- workflows: list, get, save, run, templates;
- executions, with node runs;
- approvals: list, decide;
- policy: get, set;
- privacy: egress, receipts, audit, storage audit;
- deletion;
- generators, with their status;
- suggestions (B/C slots and their references);
- benchmark results and model profile.

## 5. Front end

One page served by the backend: vanilla JS and SVG, no build step, works offline. Tabs:
1. **Assistant:** a goal box. The generated workflow appears, runs, and shows the answer.
2. **Workflows:** an n8n-style canvas, a node inspector, templates, and Run.
3. **Executions:** a list, and per-node input/output, time, data class and location badges.
4. **Data:** inbox, calendar, notes and preferences, with the persona's clock.
5. **Privacy:** the policy table, the egress log, deletion receipts, the audit log and the storage-audit search.
6. **Approvals:** pending actions, with approve or reject.
7. **Benchmark:** harness results, generator status and the model profile.

A persona switcher and clock controls sit at the top.

## 6. Data ready for training

- Fix found in the smoke test: `package_locker` values become the bare code (`E-34`, not
  `locker E-34`), so a correct "your locker is E-34" passes. Only those messages are re-rendered.
- Exports (`p17sim export-training`), from train and val personas only, with a manifest and hashes:
  - **Model A:** goal, time, persona, the ids of the context visible at that time, the **gold
    workflow** (built from ground truth) and the checker.
  - **Model B:** per-persona event sequences with routine labels.
  - **Model C:** suggestion logs (context features, the option offered, accepted/declined) plus the
    true preference, for regret.

## 7. Harness on the real system

The replay harness drives the workspace runtime, not the old in-memory shell, so it measures the
complete system: storage, privacy layer and engine. The checkers read the `outbox` and
`events_created` tables. The oracle must pass 100% and the null agent 0% on utility tasks. Baseline
runs: seeds 11, 22 and 33 on test and OOD personas, days 25–30, plus model profiling and `RESULTS.md`
with the baseline column filled.

## 8. Existing code

- **Reused:** the generator v2, the audit, the baseline agent loop (its tools are now
  engine-backed), the checkers, metrics, stats, report and profiling.
- **`system/shell.py`:** becomes the engine's tool layer over the store.
- **Model A draft:** moved to `agents/drafts/`, unwired and marked pending.

## 9. Testing

- Unit tests per module.
- API tests (FastAPI TestClient).
- Privacy tests:
  - a cloud node with personal input is blocked;
  - a non-allowlisted host is blocked;
  - a `deny` tool fails;
  - `ask` creates an approval;
  - after deletion, a storage audit finds the value in no table, including node runs.
- End-to-end: goal → workflow → execution → checked result.
- Harness sanity with the oracle and null agents.
- Front-end smoke test: the page and the API calls it uses respond.

## Out of scope (needs the team's GCP access)

GCS/BigQuery layers, the Airflow DAG and a public deployment (Checkpoint 3). A Dockerfile and the DAG
code can be added later with a local fallback.

## Addendum (2026-10-15): cloud and GPU now in scope

The team has GPU access and can get cloud accounts. The out-of-scope items above (GCS/BigQuery, Airflow,
deployment) are now planned, along with GPU rendering, training, evaluation at scale and a deployed demo.
See `docs/REQUIREMENTS.md` version 2.0:
- §2.4: deployment planes and the privacy boundary;
- §3.12–3.17: the new requirements;
- D52–D74: the decisions;
- §10: roadmap and access checklist.

The device-plane design in this document is unchanged.
