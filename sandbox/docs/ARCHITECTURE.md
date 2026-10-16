# Personal AI OS — Architecture at a glance

One picture of what the whole system does. It has two planes:
- the **device plane**: the user's own computer, where all personal data lives (built and tested);
- the **cloud plane**: data pipeline, GPU training and the hosted demo, on synthetic data only (planned).

A hard privacy boundary separates them. Requirements and design decisions are in
[REQUIREMENTS.md](REQUIREMENTS.md); how to run everything is in [../README.md](../README.md).

```text
╔═══════════════════ DEVICE PLANE · the user's own computer · personal data lives only here ═══════════════════╗
┌──────────────────────────────────────────────── CLIENT TIER ─────────────────────────────────────────────────┐
│ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                     Web app (browser) · localhost:8017 · vanilla JS + SVG, no build                      │ │
│ │ • User view:   Assistant (goal → workflow → answer) · Approvals · Data (search, delete)                  │ │
│ │ • Owner view:  Workflows (n8n-style canvas) · Executions (node by node) · Privacy center · Benchmark     │ │
│ └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │  REST / JSON
                                                        ▼
┌────────────────────────────────────────────────── API TIER ──────────────────────────────────────────────────┐
│ ┌───────────────────────────────────────────────────┐  ┌───────────────────────────────────────────────────┐ │
│ │               FastAPI backend :8017               │  │         CLI  p17sim  (same runtime, no UI)        │ │
│ │ personas · load day/time · advance clock          │  │ generate2 · render2 · build-trace2 · audit        │ │
│ │ goals · workflows · executions · undo             │  │ export-training (Models A, B, C)                  │ │
│ │ approvals · policy · privacy · storage audit      │  │ eval (harness) · profile-models · report          │ │
│ │ delete · generators · suggestions · results       │  │ serve (starts the API + web app)                  │ │
│ └───────────────────────────────────────────────────┘  └───────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │  goal + the person's context (records visible at that moment)
                                                        ▼
┌─────────────────── INTELLIGENCE TIER · workflow generators (turn a goal into a workflow) ────────────────────┐
│ ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐ │
│ │ Baseline agent [built]│  │   Templates [built]   │  │   Model A [pending]   │  │  Model B / C [pending]  │ │
│ │ OpenJarvis loop       │  │ morning briefing      │  │ retrieval-augmented   │  │ B: routines →           │ │
│ │ (native_openhands)    │  │ weekly recap (asks)   │  │ hierarchical planner  │  │    recurring triggers   │ │
│ │ qwen3.5:4b picks one  │  │ book a meeting        │  │ goal → full workflow  │  │ C: what to suggest,     │ │
│ │ tool at a time; each  │  │ forget something      │  │                       │  │    and when             │ │
│ │ step = a node         │  │                       │  │                       │  │                         │ │
│ └───────────────────────┘  └───────────────────────┘  └───────────────────────┘  └─────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │  workflow = typed nodes + edges (JSON)
                                                        ▼
┌──────────────────── ORCHESTRATION & PRIVACY TIER · every step is checked before it runs ─────────────────────┐
│ ┌────────────────────────────────┐  ┌────────────────────────────────┐  ┌──────────────────────────────────┐ │
│ │        Workflow engine         │  │         Privacy layer          │  │        Workspace runtime         │ │
│ │ runs nodes in order            │  │ data class on every value      │  │ one per persona                  │ │
│ │ if/else branches               │  │ cloud steps: public only       │  │ simulated clock (a month)        │ │
│ │ pauses for approval            │  │ tools: allow / ask / deny      │  │ time + event triggers            │ │
│ │ undoes bookings on error       │  │ egress: localhost only,        │  │ Model D check (planned)          │ │
│ │ logs every node run            │  │   every byte logged            │  │ audit log of decisions           │ │
│ │                                │  │ deletion: all copies +         │  │                                  │ │
│ │                                │  │   a receipt                    │  │                                  │ │
│ └────────────────────────────────┘  └────────────────────────────────┘  └──────────────────────────────────┘ │
│                                                                                                              │
│ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │                                               Node library                                               │ │
│ │ trigger (chat · time · event)  ·  context (search · read · recent · calendar · contacts · preferences)   │ │
│ │ reason (local LLM)  ·  logic (if · merge)  ·  action (send email · create/cancel event · delete records) │ │
│ │ human (approval)  ·  output (answer)  ·  cloud (blocked whenever the data is personal)                   │ │
│ └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │  model calls (through the egress gateway)  ·  reads and writes
                                                        ▼
┌────────────────────────────────────────── MODEL TIER · local only ───────────────────────────────────────────┐
│ ┌───────────────────────────────────────────────────┐  ┌───────────────────────────────────────────────────┐ │
│ │                   Ollama :11434                   │  │           Renderers (data creation only)          │ │
│ │ qwen3.5:4b  · the agent's model (4.2B, 3.3 GB)    │  │ llama3.2 3B · writes synthetic messages           │ │
│ │ all-minilm · sentence embeddings (23M)            │  │ phi3 mini  · held-out writing style               │ │
│ │ device builds of Models A–D (later)               │  │ never see a real person's data                    │ │
│ └───────────────────────────────────────────────────┘  └───────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        │  SQL
                                                        ▼
┌─────────────────── DATA TIER · SQLite, one file per persona (in memory for the benchmark) ───────────────────┐
│ ┌────────────────────────────────┐  ┌────────────────────────────────┐  ┌──────────────────────────────────┐ │
│ │         Personal data          │  │            Activity            │  │            Governance            │ │
│ │ • records: mail, notes,        │  │ • outbox (sent mail)           │  │ • tool policies                  │ │
│ │   calendar, preferences,       │  │ • events created               │  │ • approvals                      │ │
│ │   instructions                 │  │ • workflows                    │  │ • audit log                      │ │
│ │ • full-text index (FTS5)       │  │ • executions + node runs       │  │ • deletion receipts              │ │
│ │ • vectors                      │  │                                │  │ • egress log                     │ │
│ └────────────────────────────────┘  └────────────────────────────────┘  └──────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

                          ▲ signed model updates (registry release channel)
                          ▼ optional aggregate counts only, off by default — never text, ids or values
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PRIVACY BOUNDARY: personal records, text and embeddings NEVER cross this line (enforced by the egress gateway)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╔═════════════════════════════ CLOUD PLANE (planned) · synthetic data only · GCP ══════════════════════════════╗
┌┄┄┄┄┄┄┄┄┄┄┄ DATA PIPELINE TIER · Airflow :8080 (DAGs p17_build on demand, p17_daily on a schedule) ┄┄┄┄┄┄┄┄┄┄┄┐
┆ generate → render (GPU) → validate + leakage → bronze → Presidio PII → silver → dbt → tests → marts          ┆
┆                                                                                                              ┆
┆ ┌────────────────────────────────┐  ┌────────────────────────────────┐  ┌──────────────────────────────────┐ ┆
┆ │       Cloud Storage lake       │  │       BigQuery warehouse       │  │       Quality & monitoring       │ ┆
┆ │ • bronze: raw JSONL            │  │ • p17_gold: records, tasks,    │  │ • gates stop the DAG and         │ ┆
┆ │   (write-once)                 │  │   training marts               │  │   name the failing check         │ ┆
┆ │ • silver: cleaned Parquet      │  │ • p17_labels (restricted)      │  │ • retries, email alerts          │ ┆
┆ │                                │  │ • p17_eval: runs, metrics      │  │ • Looker Studio dashboards       │ ┆
┆ └────────────────────────────────┘  └────────────────────────────────┘  └──────────────────────────────────┘ ┆
└┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘
                                                        │
                                                        │  versioned training marts (train + val personas only)
                                                        ▼
┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ ML PLATFORM TIER ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐
┆ ┌────────────────────────────────┐  ┌────────────────────────────────┐  ┌──────────────────────────────────┐ ┆
┆ │     GPU (team / Vertex AI)     │  │          MLflow :5000          │  │         Release channel          │ ┆
┆ │ • vLLM :8000: rendering        │  │ • experiment tracking          │  │ • quantized, signed builds       │ ┆
┆ │   and evaluation               │  │ • model registry + gates:      │  │ • Ollama on the device           │ ┆
┆ │ • baseline 9B FP16             │  │   candidate → validated        │  │   pulls them                     │ ┆
┆ │   (the paper's setup)          │  │   → released                   │  │ • checksum checked first         │ ┆
┆ │ • training: Models A–D         │  │                                │  │                                  │ ┆
┆ └────────────────────────────────┘  └────────────────────────────────┘  └──────────────────────────────────┘ ┆
└┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘
                                                        │
                                                        │  same app container
                                                        ▼
┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ DEMO TIER ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐
┆ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐ ┆
┆ │                                         Cloud Run + IAP sign-in                                          │ ┆
┆ │ the same web app and API, in 'demo' mode: synthetic personas only (refuses anything else)                │ ┆
┆ │ model calls go to a vLLM GPU endpoint · reachable by the team, instructor and ISA                        │ ┆
┆ └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘ ┆
└┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

Legend:  solid boxes = built and tested · dashed boxes = planned · [pending] = slot ready, model not trained
```

## Reading the picture, tier by tier

| Tier | In one sentence | Code |
|---|---|---|
| Client | A web page where you type what you want and see exactly what the assistant did. | `aios/static/` |
| API | The doorway: every button in the page is one call to this local server; the CLI runs the same steps without a page. | `aios/api.py`, `cli.py` |
| Intelligence | Turns your goal into a workflow: today the reproduced baseline agent or a template; later Models A, B, C. | `aios/generators.py`, `aios/templates.py`, `agents/baseline.py` |
| Orchestration & privacy | Runs the workflow one node at a time; before each node it checks where the data may go and whether the tool is allowed, asks you when needed, and logs everything. | `aios/engine.py`, `aios/privacy.py`, `aios/workspace.py`, `aios/nodes.py` |
| Model | The language models, running on this laptop through Ollama. | `system/llm.py` |
| Data | One SQLite file per person: their records, what the assistant did, and the governance trail. | `aios/store.py` |
| Data pipeline (planned) | Builds the synthetic datasets at scale on a schedule and loads them into a warehouse, stopping on bad data. | `gen2/` (steps) + Airflow, dbt (planned) |
| ML platform (planned) | Trains and evaluates the models on GPUs, tracks every run, and releases checked models to the device. | `aios/export.py` (training data) + MLflow, vLLM (planned) |
| Demo (planned) | The same app on the internet, with synthetic people only, behind a sign-in. | Dockerfile, Cloud Run (planned) |

## What happens when you ask something

The names, times and values below are examples.

**1. "What is my gate code?"** (the Assistant tab, baseline generator)

```text
 You ─► Web app ─► FastAPI ─► Workspace for Mia, at Thu 25 Sep 09:00 (only her past records are visible)
   1  Baseline generator asks qwen3.5:4b (on this laptop): which tool?       → search_records("gate code")
   2  Engine, before running it:  location = local ✓   tool mail.read = allow ✓
   3  context.search node ─► SQLite full-text search ─► 8 records, tagged "personal"
   4  The model reads them and answers                 → "Your gate code is 4821."
   5  Logged: 5 node runs (trigger, model step, search, model step, answer), each with
      input, output, data class and location, plus audit entries
   6  Egress log: both model calls went to localhost:11434 → 0 bytes left the device
   7  The answer and the workflow graph appear in the Assistant tab
```

**2. "Email Ben a summary of my week"** (an action that needs your OK)

```text
   1-4  as above: the agent gathers last week's records and drafts the email
   5    Engine: tool mail.send = ask  ─► the step waits in the Approvals tab
   6    You press Approve ─► action.send_email node ─► outbox; audit: who approved, when
        (Reject ─► nothing is sent. If a later step fails, any meeting it booked is cancelled: rollback.)
```

**3. "Please forget my parking spot"** (deletion)

```text
   1  The generator finds the records that say it and calls delete_records
   2  Deletion manager: marks them deleted, drops them from the search index and vectors,
      and redacts what they said from every other store (node runs, approvals, outbox, audit)
   3  Writes a receipt;  Privacy tab ─► "storage audit" for "P3-217" ─► found nowhere
```

**The benchmark** uses the same runtime. The harness replays each synthetic person's month, asks the
hidden questions at their times, and lets checkers score the answers and actions. Every run is written to
`research/results/runs/`; `p17sim report` turns the runs into `RESULTS.md`.

## Mapping to a classic three-tier diagram

| Classic tier | Here |
|---|---|
| Client tier | Web app (device) and the hosted demo (cloud, synthetic) |
| Middleware tier | API + generators + workflow engine + privacy layer + workspace runtime; Ollama as the model service |
| Database tier | SQLite per person on the device; Cloud Storage + BigQuery for synthetic data in the cloud |
| Messaging (Kafka in the example) | Not needed on one device: the engine calls nodes directly, and triggers fire from the workspace clock. Airflow schedules the cloud pipeline. |
