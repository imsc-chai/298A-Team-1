# Personal AI OS: sandbox, system and harness (Topic 17)

A local-first personal assistant. Workflows are **generated** from a user's goal and personal
context, not drawn by hand as in n8n. A **privacy layer** decides where each step's data may be
processed and which tools may run the actions.

Everything runs on one laptop (Apple M1, 8 GB):
- the model (`qwen3.5:4b` through Ollama);
- embeddings (`all-minilm`);
- storage (SQLite);
- the backend (FastAPI) and the web front end.

The data is synthetic, so no real person's mail is used. Models A, B and C are the only parts still to build:
their slots exist, report "pending training", and their training data is exported.

Requirements and design decisions: [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) ·
Design: [docs/plans/2026-10-15-personal-ai-os-design.md](docs/plans/2026-10-15-personal-ai-os-design.md) ·
Research pipeline: [research/PIPELINE.md](research/PIPELINE.md) ·
Pre-registration: [research/PREREGISTRATION.md](research/PREREGISTRATION.md) ·
Checklist: [PROGRESS.md](PROGRESS.md)

## Architecture

```
                         ┌──────────────────────── this laptop ────────────────────────┐
  browser ── HTTP ──►    │  FastAPI backend (aios/api.py)  ◄── web front end (aios/static)
                         │        │
                         │        ▼
  goal ──────────────►   │  Generator ── builds a ──► Workflow (typed nodes + edges)
                         │  baseline | template | model_a/b/c (pending)
                         │        │
                         │        ▼
                         │  Engine (aios/engine.py): runs nodes in order
                         │        │  before every node:
                         │        ├─► Privacy layer (aios/privacy.py)
                         │        │     data class   public < personal < sensitive
                         │        │     location     local: anything · cloud: public only
                         │        │     tool policy  allow | ask (approval) | deny
                         │        │     egress gateway: allowlist, every byte logged
                         │        ▼
                         │  Nodes (aios/nodes.py): context · reason (local LLM) · logic ·
                         │                        action · human · output · cloud
                         │        │
                         │        ▼
                         │  Store (aios/store.py, SQLite): records + FTS5, vectors, outbox,
                         │    events, workflows, executions, node runs, approvals, audit,
                         │    deletion receipts, policies, egress log
                         │  Deletion manager: removes a record and every copy of what it said
                         └─────────────────────────────────────────────────────────────┘
  Harness (system/runner.py) replays a persona's 30 days through the same workspace and scores it.
```

## Run it

Setup, once: Python 3.12, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
uv sync
ollama pull qwen3.5:4b && ollama pull llama3.2 && ollama pull phi3:mini && ollama pull all-minilm
```

Data pipeline (v2, debiased; v1 is unchanged and still reproduces Workbook 1):

```bash
p17sim generate2                 # 36 personas, 30 days each, seed 2026 -> data/v2
p17sim render2                   # local LLMs write the message text (cached, resumable)
p17sim build-trace2              # clean + validate + leakage checks -> ../data/trace_v2
p17sim audit                     # bias and shortcut audit, v1 vs v2
p17sim export-training           # training data for Models A, B, C (train + val personas only)
```

The system:

```bash
p17sim serve                     # http://localhost:8017  (API docs at /docs)
```

Evaluation:

```bash
p17sim eval --system oracle --personas test             # harness sanity: must pass 100%
p17sim eval --system baseline --seed 11 --personas test # also seeds 22, 33; and --personas ood_test
p17sim profile-models                                    # load time, memory, TTFT, tokens/s per model
p17sim report                                            # research/results/RESULTS.md + figures
pytest                                                   # all tests
```

## The web app

| Tab | What you can do |
|---|---|
| Assistant | Type a goal, pick a generator, and watch the workflow it builds run, node by node |
| Workflows | n8n-style canvas of saved and template workflows; click a node to inspect it |
| Executions | Every run, with each node's input, output, data class and location; undo booked events |
| Data | The persona's messages, notes, calendar, preferences and instructions; full-text search; delete |
| Privacy | Tool policy (allow/ask/deny), where data may go, egress log, deletion receipts, storage audit, audit log |
| Approvals | Actions waiting for the user's OK |
| Benchmark | Generator status, the B/C slot references, harness runs, the model profile and the results report |

Set the persona and the moment in their month at the top. "+1 h" and "+1 day" move the clock; records
arrive and triggers fire as it moves.

## Plugging in Models A, B and C

| Model | Slot | Interface | Training data |
|---|---|---|---|
| A: retrieval-augmented hierarchical planner | `model_a` generator | `run(goal, ws, approver) -> ExecutionResult`, usually `ws.engine.run(workflow, "model_a", approver=approver)` with a generated `Workflow` | `model_a_goal_to_workflow.{train,val}.jsonl`: goal, time, visible context ids, evidence ids, gold workflow (each checked) |
| B: routine/preference model | `model_b` | proposes `{kind: "time", at, weekdays}` triggers with a workflow (the reference is `routine_reference`) | `model_b_event_sequences.{train,val}.jsonl` |
| C: contextual bandit | `model_c` | picks a suggestion from context (the reference is `suggestion_reference`) | `model_c_suggestion_log.{train,val}.jsonl` |

Register a trained model in `aios/generators.py` (replace its `PendingGenerator`) and in
`agents/registry.py` (`system()`), then evaluate it with `p17sim eval --system model_a ...`.
The pre-registered ablations for Model A are listed in `agents/registry.py` (`ABLATIONS`).
`agents/drafts/` holds an early, untrained Model A draft; it is not wired in.

## Privacy guarantees and how they are tested

| Guarantee | Test |
|---|---|
| Personal and sensitive data is processed only locally | `test_engine.py::test_cloud_node_with_personal_input_is_blocked` |
| Nothing leaves for a host off the allowlist | `test_privacy.py::test_gateway_blocks_hosts_off_the_allowlist` |
| "ask" actions wait for approval; "deny" actions never run | `test_engine.py` (ask, reject, deny) |
| A failed workflow rolls back what it booked | `test_engine.py::test_failure_rolls_back_reversible_actions` |
| Deletion reaches every store: records, index, vectors, node runs, approvals, outbox | `test_privacy.py::test_deletion_removes_the_value_from_every_store`, `test_generators.py::test_forget_template_leaves_no_copy_anywhere` |
| The harness's storage audit after every deletion request | `system/runner.py` (`deletion_audit` rows) |

## Code map

```
src/p17sim/
  gen2/        v2 data: personas, skeleton, phrasing, renderer, cleaning/export, audit, Enron calibration
  aios/        the system: store, privacy, workflow, nodes, engine, workspace, generators, templates,
               export, api, static/ (web front end)
  agents/      baseline (OpenJarvis native_openhands re-implementation), reference (oracle, null),
               registry, drafts/ (pending Model A)
  system/      harness (runner), checkers, local model client (llm), shell tool schemas, text utils
  analysis/    metrics, statistics, model profiling, report
  (v1 modules: personas, skeleton, render, trace, validate, leakage, splits, eda — Workbook 1)
```
