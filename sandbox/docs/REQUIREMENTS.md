# Personal AI OS — Requirements Specification

**Project:** Topic 17, Privacy-Preserving Personal AI Operating System (DATA 298A/B, Section 21, Team 1)
**Document owner:** Savitha Vijayarangan · **Version:** 1.0 · **Date:** 2026-10-15
**Status of the system on this date:**
- Built and tested: everything except Models A, B, C and D (136 automated tests pass).
- Data render in progress.

Related documents:
- Design: [plans/2026-10-15-personal-ai-os-design.md](plans/2026-10-15-personal-ai-os-design.md).
- Hypotheses and metrics, fixed before any result: [../research/PREREGISTRATION.md](../research/PREREGISTRATION.md).
- Research stages: [../research/PIPELINE.md](../research/PIPELINE.md).
- How to run: [../README.md](../README.md).

## How to read this document

Every requirement has an ID, a priority and a status.

| Field | Values |
|---|---|
| ID | `FR-<area>-<n>` functional, `NFR-<area>-<n>` non-functional, `MR-<model>-<n>` model, `DR-<n>` data, `IR-<n>` interface |
| Priority | **M** must (298A thin slice or a course rule) · **S** should · **C** could (later or optional) |
| Status | **Built** (implemented and verified) · **Partial** · **Pending** (not built yet) · **Planned 298B** |
| Verified by | the automated test, command or document that shows it holds |
| Decision | the design decision behind it (section 11): 2–3 options compared, the chosen one marked ✅ |

Test paths are relative to `sandbox/tests/`.

---

## 1. Introduction

### 1.1 Purpose
This document lists everything the Personal AI OS must do and how well it must do it:
- the data pipeline;
- the privacy layer;
- the workflow engine;
- the generators;
- the learned models;
- the evaluation harness;
- the backend, front end and tooling.

It is the reference for building, testing, presenting and grading the system.

### 1.2 Problem
Personal assistants help most when they can read a person's mail, calendar and notes. Today that
means sending this content to a cloud provider. The system must do the same work while keeping raw
personal data on the person's own device, and must prove it with measurements.

### 1.3 Scope

| In scope | Out of scope (for now) |
|---|---|
| Synthetic 30-day personal-data traces (no real people) | Real user accounts, real mail servers, a real-user pilot |
| Local model inference on one laptop | GPU servers, vLLM, FP16 weights |
| Generated workflows run under a privacy layer | A general-purpose automation product |
| Four learned models compared with one reproduced baseline | Models beyond A–D |
| Local SQLite storage, FastAPI backend, web front end | Public cloud deployment (needs the team's GCP access) |
| Benchmark harness, metrics, statistics, report | Mobile apps |

### 1.4 Stakeholders

| Stakeholder | Interest |
|---|---|
| End user: a student or professional (simulated by personas) | Help with mail, calendar and notes without exporting personal data |
| Project team (5 members) | A working system and a fair comparison of four methods |
| Instructor and ISA | Checkpoint rubric: live demos, Linear evidence, reproducibility |
| Future maintainers (298B) | Clear slots and interfaces for Models B, C, D |

### 1.5 Definitions

| Term | Meaning |
|---|---|
| Persona | A synthetic person with contacts, routines, preferences and secrets |
| Record | One item in a persona's month: message, note, calendar entry, notification, preference signal, instruction, permission |
| Workflow | A graph of typed nodes (trigger, context, reason, logic, action, human, output, cloud) and edges |
| Generator | Turns a goal plus context into a workflow (baseline, template, Model A) |
| Execution | One run of a workflow, with a log row per node |
| Data class | `public` < `personal` < `sensitive`, attached to every value moving between nodes |
| Egress | Any network request; "raw-data export" = bytes of persona text sent to a non-local host |
| Harness | The replay that runs a persona's 30 days and scores a generator on hidden tasks |
| Canary | A unique string planted in a secret, so a leak can be detected exactly |

---

## 2. System overview

### 2.1 Context

```
 User (browser) ──► Web front end ──► FastAPI backend ──► Workspace (one per persona)
                                                            ├─ Generator (baseline | template | Model A/B/C)
                                                            ├─ Workflow engine ──► Nodes ──► Tools
                                                            ├─ Privacy layer (classes, location, tool policy,
                                                            │                 egress gateway, deletion, audit)
                                                            └─ SQLite store
 Local model server (Ollama) ◄── every model call goes through the egress gateway
 Harness / CLI ──► the same Workspace, in memory, scored by hidden checkers
```

### 2.2 Component inventory

| # | Component | Code | Responsibility | Status |
|---|---|---|---|---|
| C1 | v1 data generator | `p17sim/{personas,skeleton,render,trace,...}.py` | Workbook 1 trace (10 personas); kept byte-identical | Built |
| C2 | v2 data generator | `p17sim/gen2/` | Debiased personas, 30-day events, rendering, cleaning, export, checks | Built |
| C3 | Real-world calibration | `gen2/calibrate.py`, `reference_stats.json` | Aggregate statistics from a 9,300-row Enron sample | Built |
| C4 | Bias and shortcut audit | `gen2/audit.py` | Same measurements on v1 and v2 | Built |
| C5 | Store | `aios/store.py` | SQLite tables, FTS5 search, every user-text column listed | Built |
| C6 | Privacy layer | `aios/privacy.py` | Data classes, location and tool policy, egress gateway, deletion manager | Built |
| C7 | Workflow model | `aios/workflow.py` | Nodes, edges, branches, references, ordering | Built |
| C8 | Node library | `aios/nodes.py` | 20 node types and their contracts | Built |
| C9 | Engine | `aios/engine.py` | Runs workflows under the privacy layer; approvals, rollback, logs | Built |
| C10 | Workspace runtime | `aios/workspace.py` | Clock, replay, triggers, engine-backed agent tools | Built |
| C11 | Generators | `aios/generators.py`, `templates.py` | Baseline, templates, model slots, B/C references | Built (models pending) |
| C12 | Baseline agent | `agents/baseline.py` | OpenJarvis `native_openhands` loop re-implemented | Built |
| C13 | Reference agents | `agents/reference.py` | Oracle and null generators for harness sanity | Built |
| C14 | Harness | `system/runner.py`, `checkers.py` | Replay, deletion audits, scoring, resumable runs | Built |
| C15 | Training exports | `aios/export.py` | Data for Models A, B, C with checked gold workflows | Built |
| C16 | Analysis | `analysis/` | Metrics, statistics, model profiling, report and figures | Built |
| C17 | Backend API | `aios/api.py` | 28 routes; one workspace per persona | Built |
| C18 | Front end | `aios/static/` | 7-tab single page, workflow canvas | Built |
| C19 | CLI | `p17sim/cli.py` | Commands for every pipeline step | Built |
| C20 | Job guard | `scripts/guarded_run.py` | Restarts long jobs when the model server bloats or disk runs low | Built |
| C21 | Model A | slot `model_a` | Retrieval-augmented hierarchical planner | Pending |
| C22 | Model B | slot `model_b` | Routine and preference model | Planned 298B |
| C23 | Model C | slot `model_c` | Contextual-bandit suggestion policy | Planned 298B |
| C24 | Model D | hook before action nodes | Safety critic | Planned 298B |
| C25 | Cloud pipeline | — | GCS/BigQuery layers, Airflow DAG, deployment | Pending (needs GCP access) |

### 2.3 Services, tools and models used

| Kind | Name | Version / size | Role | Runs |
|---|---|---|---|---|
| Model server | Ollama | 0.35.1 | Serves all local models over localhost | Laptop |
| LLM (agent) | `qwen3.5:4b` | 4.2B, Q4_K_M, 3.3 GB | Baseline and future generators; same settings for all | Laptop |
| LLM (renderer) | `llama3.2` | 3.2B, Q4_K_M | Writes message text for all personas | Laptop |
| LLM (held-out renderer) | `phi3:mini` | 3.8B, Q4_0 | Writes text for half of the test and OOD personas | Laptop |
| Embeddings | `all-minilm` | 23M | Sentence vectors (for Model A) | Laptop |
| Database | SQLite | 3.45.3 with FTS5 | All workspace storage | Laptop |
| Web framework | FastAPI + uvicorn | 0.143 / 0.54 | Backend API | Laptop |
| Libraries | pydantic, numpy, pandas, scikit-learn, SciPy, matplotlib, Faker, datasketch | pinned in `uv.lock` | Data, analysis, tests | Laptop |
| Baseline source | OpenJarvis | commit `e97088f` | Agent loop being reproduced | Reference only |
| Benchmark | PinchBench, 23 tasks | snapshot `0c419e45` | Published-number reproduction (`baseline/`) | Reference only |
| Reference data | Enron corpus sample | 9,300 rows, aggregates only | Timing and length calibration | One-time fetch |

**Agent tools** (identical for every generator, defined in `system/shell.py`, executed as engine nodes):

| Tool | Node | Permission | Reversible |
|---|---|---|---|
| `search_records(query, kind, limit)` | `context.search` | `mail.read` | — |
| `read_record(id)` | `context.read` | `mail.read` | — |
| `list_contacts()` | `context.contacts` | none | — |
| `list_calendar(date)` | `context.calendar` | `calendar.read` | — |
| `create_event(date, start, duration_minutes, with_contact, title)` | `action.create_event` | `calendar.create` | yes |
| `cancel_event(event_id)` | `action.cancel_event` | `calendar.create` | — |
| `send_email(to, subject, body)` | `action.send_email` | `mail.send` | no |
| `delete_records(ids)` | `action.delete_records` | `memory.forget` | no |

---

## 3. Functional requirements

### 3.1 Data generation (C1–C4)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-DAT-1 | The v1 generator shall still produce byte-identical events, trace, labels and splits for seed 2026. | M | Built | rebuild + `cmp` (PROGRESS §A), `test_skeleton.py` | [D46](#d46) |
| FR-DAT-2 | The v2 generator shall produce the same output for the same seed. | M | Built | `test_gen2.py::test_same_seed_same_trace` | [D02](#d02) |
| FR-DAT-3 | Personas shall be sampled per person: routines (start, weekdays, jitter, skip rate), 4–14 contacts, activity level, chronotype, preference noise, 1–4 secrets. | M | Built | `gen2/personas.py`, audit `heterogeneity.*` | [D03](#d03) |
| FR-DAT-4 | Names shall come from 16 locales in 6 regions; region and name list shall be balanced within each split. | M | Built | `test_gen2.py::test_each_split_is_balanced_by_region_and_name_list` | [D03](#d03) |
| FR-DAT-5 | Message timing and length shall be calibrated to aggregate real-mail statistics; no real text shall be stored. | M | Built | `gen2/calibrate.py`, `reference_stats.json` | [D04](#d04) |
| FR-DAT-6 | Facts shall have repeats, near-miss distractors and up to two updates or a deletion before they are asked about. | M | Built | `gen2/skeleton.py::fact_events`; leakage check 3 | [D02](#d02) |
| FR-DAT-7 | Preference signals shall cover all 30 days, with up to two preference changes. | M | Built | `test_gen2.py::test_pref_value_follows_every_change` | [D03](#d03) |
| FR-DAT-8 | Scheduling tasks shall use natural dates and always have a clash-free slot. | M | Built | `test_gen2.py::test_scheduling_tasks_have_a_free_slot`, `test_when_phrase_names_the_right_date` | [D02](#d02) |
| FR-DAT-9 | Requests shall have several wordings; held-out wordings shall appear only in the test window of test and OOD personas. | M | Built | `test_gen2.py::test_heldout_wordings_stay_in_the_test_window` | [D07](#d07) |
| FR-DAT-10 | Canaries shall take 12 natural shapes, none matching the v1 pattern, and none shall look like a phone number. | M | Built | `test_gen2.py::test_audit_finds_v1_shortcuts_absent_in_v2` | [D06](#d06) |
| FR-DAT-11 | Text shall be written by local models with per-contact styles; a second model family (phi3) shall be held out for test and OOD personas. | M | Built | `test_gen2.py::test_second_renderer_only_in_held_out_splits` | [D05](#d05) |
| FR-DAT-12 | Rendering shall keep each required fact verbatim (3 tries, then flagged and imputed) and be cached and resumable. | M | Built | `render.py`, `test_render.py` | [D05](#d05) |
| FR-DAT-13 | Cleaning shall remove model artifacts (preambles, quotes, markdown, meta-notes, placeholders), duplicates and real-looking identifiers. | M | Built | `test_gen2.py::test_v2_cleaner_strips_renderer_artifacts`, `test_trace.py` | [D08](#d08) |
| FR-DAT-14 | Export shall validate schema, ids, per-persona time windows, deletion targets and PII, and stop on any violation. | M | Built | `test_gen2.py::test_full_pipeline_has_no_violations_or_leaks`, `test_validator_names_a_corrupted_timestamp` | [D09](#d09) |
| FR-DAT-15 | Export shall run six leakage checks (persona overlap, held-out wording, evidence before task, canary ownership, answer in task, near duplicates) and stop on any finding. | M | Built | `gen2/trace.py::leakage` | [D09](#d09) |
| FR-DAT-16 | The audit shall report representation, heterogeneity, realism, text diversity and shortcut measurements for v1 and v2 side by side. | M | Built | `p17sim audit` | [D47](#d47) |
| FR-DAT-17 | PII detection shall add Microsoft Presidio to the pattern rules. | S | Pending | Workbook 1 §2.1 promise | [D08](#d08) |

### 3.2 Storage (C5)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-STO-1 | All workspace data shall live in one SQLite database per workspace (file in the app, memory in the harness). | M | Built | `test_store.py::test_file_store_persists` | [D10](#d10) |
| FR-STO-2 | Full-text search shall rank records by BM25 and return only the persona's non-deleted records from before the current time. | M | Built | `test_store.py::test_search_ranks_*`, `test_deleted_and_future_records_are_not_visible`, `test_search_respects_persona_and_channel` | [D11](#d11) |
| FR-STO-3 | The calendar view shall merge routine entries with events the assistant created, excluding cancelled ones. | M | Built | `test_store.py::test_calendar_merges_routines_and_created_events` | [D10](#d10) |
| FR-STO-4 | The store shall keep workflows, executions, node runs, approvals, audit entries, deletion receipts, policies, egress entries and triggers. | M | Built | `aios/store.py` schema | [D10](#d10), [D12](#d12) |
| FR-STO-5 | The store shall list every column that can hold user text, for deletion and audits. | M | Built | `test_store.py::test_text_columns_cover_every_place_user_text_can_land` | [D16](#d16) |

### 3.3 Privacy layer (C6)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-PRV-1 | Every record shall be classified on arrival: `deny` permission → sensitive; other user records → personal. | M | Built | `test_privacy.py::test_classes_order_and_classification` | [D14](#d14) |
| FR-PRV-2 | Each value passed between nodes shall carry the highest class of its inputs and sources. | M | Built | `test_engine.py::test_chain_runs_and_carries_the_data_class` | [D14](#d14) |
| FR-PRV-3 | A cloud node shall never receive personal or sensitive data; the engine shall block it before it runs and log the attempt. | M | Built | `test_engine.py::test_cloud_node_with_personal_input_is_blocked` | [D13](#d13) |
| FR-PRV-4 | Each tool shall have a policy of allow, ask or deny, defaulting from the persona's permission records and overridable by the user. | M | Built | `test_privacy.py::test_tool_policy_follows_permission_records_and_overrides` | [D17](#d17) |
| FR-PRV-5 | Every outbound request (including every model call) shall go through the egress gateway, which shall block hosts off the allowlist and log host, bytes and bytes of persona text. | M | Built | `test_privacy.py::test_gateway_*`; `system/llm.py` | [D15](#d15) |
| FR-PRV-6 | Deleting records shall remove them from records, the full-text index and vectors, and redact what they said from every other store (node runs, approvals, outbox, executions, audit) without removing words still used by kept records. | M | Built | `test_privacy.py::test_deletion_removes_the_value_from_every_store` | [D16](#d16) |
| FR-PRV-7 | Every deletion shall produce a receipt (records, number of redactions, time, who asked). | M | Built | `test_api.py::test_delete_then_storage_audit_is_empty` | [D16](#d16) |
| FR-PRV-8 | A storage audit shall find every place a value is still stored. | M | Built | `privacy.py::DeletionManager.storage_audit` | [D16](#d16) |
| FR-PRV-9 | Every node run, tool decision, block, approval, rollback, policy change and deletion shall be written to the audit log. | M | Built | `test_engine.py` (audit assertions) | [D13](#d13) |
| FR-PRV-10 | A learned safety critic (Model D) shall be able to veto an action node before it runs. | S | Planned 298B | — | [D43](#d43) |

### 3.4 Workflow model, nodes and engine (C7–C9)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-WF-1 | A workflow shall be a graph of typed nodes and edges, stored as JSON; cycles and edges to unknown nodes shall be rejected. | M | Built | `test_workflow.py::test_order_is_topological_and_cycles_fail` | [D19](#d19) |
| FR-WF-2 | Node parameters shall reference upstream outputs with `{{ node.path }}`; a reference that fails to resolve shall fail the node with a clear error. | M | Built | `test_workflow.py::test_references_resolve_whole_values_and_inside_text` | [D20](#d20) |
| FR-WF-3 | `logic.if` edges shall carry a true/false branch; nodes on the branch not taken shall be skipped and logged as skipped. | M | Built | `test_engine.py::test_if_branches_skip_the_other_side` | [D19](#d19) |
| FR-WF-4 | The node library shall provide trigger (chat, time, event), context (search, read, recent, calendar, contacts, preferences), reason (local LLM, agent step), logic (if, merge), action (send email, create/cancel event, delete records), human (approval), output (respond) and cloud (LLM) nodes. | M | Built | `aios/nodes.py`; `test_workflow.py` | [D18](#d18) |
| FR-WF-5 | Context nodes shall only see records from before the current time and from this persona. | M | Built | `test_workflow.py::test_search_does_not_see_the_future` | [D22](#d22) |
| FR-WF-6 | Contact names shall be resolved exactly, by unique part or by close match; ambiguous or unknown names shall return a message, not a guess. | M | Built | `test_workflow.py::test_contact_resolution_reports_ambiguity` | [D48](#d48) |
| FR-ENG-1 | The engine shall check the location policy and then the tool policy before every node. | M | Built | `test_engine.py` (blocked, deny) | [D13](#d13) |
| FR-ENG-2 | An `ask` action shall go to an approver: the harness approver approves and counts one intervention; the app approver parks it until the user decides; `resume` shall then continue the workflow. | M | Built | `test_engine.py::test_ask_policy_*` | [D17](#d17) |
| FR-ENG-3 | A rejected action shall not run. | M | Built | `test_engine.py::test_rejected_action_does_not_run` | [D17](#d17) |
| FR-ENG-4 | When a workflow fails, the engine shall undo the reversible actions it already took. | M | Built | `test_engine.py::test_failure_rolls_back_reversible_actions` | [D21](#d21) |
| FR-ENG-5 | Every node run shall be logged with input, output, status, data class, location and wall time. | M | Built | `test_engine.py` | [D13](#d13) |
| FR-ENG-6 | Agent loops shall run step by step inside one execution, so their tool calls and model calls become workflow nodes; a failed step shall be reported to the agent without rolling back earlier steps. | M | Built | `test_engine.py::test_stepwise_execution_records_an_agent_run_as_a_workflow` | [D24](#d24) |
| FR-ENG-7 | The delete step shall log only record ids, never the deleted text. | M | Built | `test_generators.py::test_forget_template_leaves_no_copy_anywhere` | [D16](#d16) |

### 3.5 Workspace runtime (C10)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-WS-1 | A workspace shall replay a persona's records in time order up to a given moment (simulated clock). | M | Built | `test_workspace.py::test_replay_delivers_only_the_past` | [D22](#d22) |
| FR-WS-2 | Benchmark questions shall never be stored as history; user instructions shall be handed to a handler when they arrive. | M | Built | `test_workspace.py::test_instructions_reach_the_handler_and_questions_can_be_hidden` | [D22](#d22) |
| FR-WS-3 | Time triggers (daily or weekday) shall fire once per passing of their time; event triggers shall fire on matching arrivals. | M | Built | `test_workspace.py::test_time_trigger_*`, `test_event_trigger_*` | [D22](#d22) |
| FR-WS-4 | Agents shall get the standard tool interface, with every call executed as an engine node. | M | Built | `test_workspace.py::test_agent_tools_run_as_engine_nodes` | [D24](#d24) |

### 3.6 Generators and templates (C11–C13)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-GEN-1 | All generators shall share one interface: `run(goal, workspace, approver) → execution result`. | M | Built | `aios/generators.py` | [D49](#d49) |
| FR-GEN-2 | The baseline generator shall run the reproduced OpenJarvis loop; each model call and tool call shall appear as a workflow node. | M | Built | `test_generators.py::test_baseline_run_becomes_a_workflow` | [D23](#d23) |
| FR-GEN-3 | A model-server failure shall end the execution as failed with the error, not crash the system. | M | Built | `test_generators.py::test_baseline_failure_is_recorded_not_raised` | [D21](#d21) |
| FR-GEN-4 | Templates shall exist for morning briefing (time trigger), weekly recap (with approval), book meeting and forget (with approval), and their prompts shall receive upstream data. | M | Built | `test_generators.py::test_templates_are_valid_graphs`, `test_template_prompts_receive_the_upstream_data`, `test_weekly_recap_waits_for_approval_then_sends` | [D18](#d18) |
| FR-GEN-5 | The Model A, B and C slots shall report "pending training" and refuse to run until trained. | M | Built | `test_generators.py::test_pending_models_say_so`, `test_api.py::test_pending_models_return_409` | [D49](#d49) |
| FR-GEN-6 | Non-learned references shall fill the B slot (weekly routines by counting) and the C slot (most-accepted option). | S | Built | `test_generators.py::test_routine_reference_*`, `test_suggestion_reference_*` | [D41](#d41) |
| FR-GEN-7 | An oracle generator (reads ground truth) and a null generator shall exist for harness sanity. | M | Built | `test_system.py::test_oracle_*`, `test_null_*` | [D28](#d28) |

### 3.7 Evaluation harness and analysis (C14, C16)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-EVA-1 | The harness shall run any generator on the complete runtime (store, privacy layer, engine). | M | Built | `test_system.py::test_baseline_generator_runs_on_the_harness` | [D50](#d50) |
| FR-EVA-2 | A deletion request shall be handled when it happens, followed by a storage audit of every store for every value of that fact. | M | Built | `test_system.py::test_null_fails_utility_tasks` (audit fails), oracle (passes) | [D16](#d16) |
| FR-EVA-3 | Checkers shall see only the answer and the emails and events made by that task's execution, and shall match values regardless of spacing or punctuation. | M | Built | `test_system.py::test_value_matching_ignores_spacing_but_not_digits` | [D28](#d28) |
| FR-EVA-4 | The oracle shall pass 100% and the null generator 0% of utility tasks. | M | Built | `test_system.py` | [D28](#d28) |
| FR-EVA-5 | Runs shall be written row by row and resume per persona after a stop. | M | Built | `system/runner.py::evaluate` | [D36](#d36) |
| FR-EVA-6 | Each run shall record the git commit, model digest, settings, seed, hardware and personas. | M | Built | `agents/registry.py::run_meta` | [D44](#d44) |
| FR-EVA-7 | The analysis shall compute the seven metrics, cluster bootstrap CIs, McNemar with Holm, generalization gaps and fairness slices, and build `RESULTS.md` with figures. | M | Built | `test_system.py::test_report_builds_from_run_files` | [D29](#d29) |
| FR-EVA-8 | Model profiling shall measure, per model, disk size, resident memory, cold load, time to first token, prefill and decode speed, and energy per 1,000 tokens. | M | Built | `p17sim profile-models` | [D30](#d30) |
| FR-EVA-9 | Baseline numbers on the 30-day harness: 3 seeds on test personas and OOD personas. | M | Pending | runs after the render | [D25](#d25) |

### 3.8 Training-data export (C15)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-TRN-1 | Training files shall come from train and val personas only. | M | Built | `test_export.py::test_exports_cover_only_training_personas_and_gold_passes` | [D07](#d07) |
| FR-TRN-2 | Model A rows shall hold goal, time, visible context ids, evidence ids, a gold workflow and the checker. | M | Built | `test_export.py::test_gold_rows_name_their_evidence` | [D31](#d31) |
| FR-TRN-3 | Every gold workflow shall pass its checker when run through the runtime before export. | M | Built | `export.py::verify_gold` | [D31](#d31) |
| FR-TRN-4 | Model B rows shall hold each event with its routine label; Model C rows each suggestion with context, offer, acceptance and the true preference. | M | Built | `test_export.py::test_suggestion_log_matches_the_true_preference_up_to_noise` | [D31](#d31) |
| FR-TRN-5 | The export shall write a manifest with row counts and hashes. | M | Built | `export.py::export_all` | [D44](#d44) |

### 3.9 Backend API (C17)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-API-1 | The API shall provide the routes in §6.1, as JSON. | M | Built | `test_api.py` | [D32](#d32) |
| FR-API-2 | Loading a persona shall replay to the chosen day and time, optionally letting a generator act on past instructions. | M | Built | `test_api.py::test_personas_and_load` | [D22](#d22) |
| FR-API-3 | Only one model request shall run at a time. | M | Built | `aios/api.py` (`Hub.lock`) | [D35](#d35) |
| FR-API-4 | Approvals shall be listable and decidable; deciding shall resume the workflow. | M | Built | `test_api.py::test_template_waits_for_approval_then_sends` | [D17](#d17) |
| FR-API-5 | Policy changes, deletions and storage audits shall be available through the API. | M | Built | `test_api.py::test_delete_*`, `test_policy_update_and_privacy_view` | [D32](#d32) |
| FR-API-6 | Interactive API documentation shall be served at `/docs`. | S | Built | FastAPI default | [D32](#d32) |

### 3.10 Front end (C18)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-UI-1 | Assistant tab: type a goal, pick a generator (templates show their parameters), run, see the answer, status, model steps, confirmations and the workflow graph with per-node details. | M | Built | live check; `test_frontend.py` | [D33](#d33) |
| FR-UI-2 | Workflows tab: n8n-style canvas of saved workflows and templates; clicking a node shows its parameters. | M | Built | live check | [D34](#d34) |
| FR-UI-3 | Executions tab: list of runs; each with its graph, node inputs and outputs, data class and location, sent mail, booked events and Undo. | M | Built | live check | [D34](#d34) |
| FR-UI-4 | Data tab: records by channel up to the current time, full-text search, multi-select delete with confirmation, calendar by date. | M | Built | live check | [D33](#d33) |
| FR-UI-5 | Privacy tab: tool policy editor, location rules, egress log and totals, data classes held, storage-audit search, deletion receipts, audit log. | M | Built | live check | [D33](#d33) |
| FR-UI-6 | Approvals tab: pending actions with their parameters; approve or reject. A badge shows the pending count. | M | Built | live check | [D33](#d33) |
| FR-UI-7 | Benchmark tab: generator status, B/C references, harness runs, model profile, the rendered results report with figures. | M | Built | live check | [D33](#d33) |
| FR-UI-8 | Header: persona, day, time, Load, +1 hour, +1 day, current clock. | M | Built | live check | [D33](#d33) |
| FR-UI-9 | Every API path the page calls shall exist, and every element it uses shall exist. | M | Built | `test_frontend.py` | [D33](#d33) |

### 3.11 Command line and operations (C19, C20)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| FR-OPS-1 | Each pipeline step shall have a CLI command (§6.2). | M | Built | `p17sim --help` | [D51](#d51) |
| FR-OPS-2 | Long model jobs shall run under a guard that restarts them when the model server passes 5 GB or free disk drops below 2.5 GB. | M | Built | `scripts/guarded_run.py` (2 restarts during the render) | [D36](#d36) |
| FR-OPS-3 | A cloud pipeline shall move synthetic benchmark data to GCS and BigQuery on an Airflow schedule, with a failing sanity check on corrupted input. | M (Checkpoint 3) | Pending | needs GCP access | [D38](#d38) |
| FR-OPS-4 | The prototype shall be deployed and reachable beyond one laptop. | M (Checkpoint 3) | Pending | needs a host | [D39](#d39) |
| FR-OPS-5 | A Dockerfile shall build the backend and front end. | S | Pending | — | [D39](#d39) |

---

## 4. Model requirements

All generator models share the baseline's model settings unless a requirement says otherwise:
`qwen3.5:4b`, temperature 0.6, 8,192 output tokens, 8,192-token context, thinking off, local only.

### 4.1 Baseline (reproduced OpenJarvis agent)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| MR-BL-1 | Reproduce Table 1 condition (c) of Saad-Falcon et al. (2026) on the 23-task PinchBench: 5 runs, mean ± SD. | M | Built: 71.5% ± 3.4 vs 87.9% published | `baseline/SETUP.md` (Phase 1) | [D23](#d23) |
| MR-BL-2 | Re-implement the `native_openhands` loop (prompt, native tools, text fallback, 4,000-char observations, 3,000-token trimming, 10 turns). | M | Built | `test_system.py::test_openjarvis_truncation_rule`, `test_baseline_loop_*` | [D23](#d23), [D27](#d27) |
| MR-BL-3 | Run on the 30-day harness with the same tools, shell and tasks as every other generator. | M | Built; runs pending | FR-EVA-9 | [D25](#d25) |

### 4.2 Model A: retrieval-augmented hierarchical planner (298A)

| ID | Requirement | Pri | Status | Decision |
|---|---|---|---|---|
| MR-A-1 | Input: goal + the person's context (records visible at that time). Output: a full workflow graph run by the engine. | M | Pending | [D40](#d40) |
| MR-A-2 | Retrieve relevant records with provenance; resolve conflicting values in favour of the latest. | M | Pending | [D40](#d40) |
| MR-A-3 | Decompose requests into sub-goals (hierarchical plan) and execute them through engine nodes only. | M | Pending | [D40](#d40) |
| MR-A-4 | Be trained, fine-tuned or adapted by the team (a hosted API call does not count), with learning curves, a hyperparameter search and compute reported. | M (course rule) | Pending | [D40](#d40) |
| MR-A-5 | Train only on `model_a_goal_to_workflow.train.jsonl`; make every choice on the val file. | M | Pending (data ready) | [D40](#d40) |
| MR-A-6 | Task success at least the baseline's on test personas (non-inferiority margin −5 pp, H1). | M | Pending | [D40](#d40) |
| MR-A-7 | Deletion compliance 100%: probes and storage audit (H2). | M | Pending | [D40](#d40) |
| MR-A-8 | Privacy leakage no higher than the baseline's; 0 bytes of raw personal data exported (H3). | M | Pending | [D40](#d40) |
| MR-A-9 | Each pre-registered component shall help its target task kind (four ablations, H4). | M | Pending | [D40](#d40) |
| MR-A-10 | Drop on held-out conditions (OOD archetypes, held-out wordings, second renderer, regions) of at most 10 pp (H5). | S | Pending | [D40](#d40) |

### 4.3 Model B: routine and preference model (298B)

| ID | Requirement | Pri | Status | Decision |
|---|---|---|---|---|
| MR-B-1 | Learn routines and changing preferences from event sequences; propose recurring workflows and their triggers. | M | Planned 298B | [D41](#d41) |
| MR-B-2 | Train on `model_b_event_sequences.train.jsonl`. | M | Planned 298B (data ready) | [D41](#d41) |
| MR-B-3 | Beat the counting reference (`routine_reference`) on next-occurrence prediction and on preference accuracy at probe times. Proposed metrics; to be fixed in the 298B pre-registration. | S | Planned 298B | [D41](#d41) |

### 4.4 Model C: contextual-bandit suggestion policy (298B)

| ID | Requirement | Pri | Status | Decision |
|---|---|---|---|---|
| MR-C-1 | Choose which proactive suggestion to make, and when, from context; learn from accept/decline. | M | Planned 298B | [D42](#d42) |
| MR-C-2 | Evaluate by offline replay on `model_c_suggestion_log.*.jsonl`. | M | Planned 298B (data ready) | [D42](#d42) |
| MR-C-3 | Lower preference regret and intervention rate than the most-accepted reference (`suggestion_reference`). Proposed; to be fixed in the 298B pre-registration. | S | Planned 298B | [D42](#d42) |

### 4.5 Model D: safety critic (298B)

| ID | Requirement | Pri | Status | Decision |
|---|---|---|---|---|
| MR-D-1 | Score each planned action for privacy and irreversibility risk before it runs; veto or ask the user. | M | Planned 298B | [D43](#d43) |
| MR-D-2 | Plug into the engine as an extra check before action nodes (FR-PRV-10). | M | Planned 298B | [D43](#d43) |
| MR-D-3 | High recall on unsafe actions at a fixed false-refusal rate. The target is set in the 298B pre-registration. | S | Planned 298B | [D43](#d43) |

---

## 5. Non-functional requirements

### 5.1 Privacy and security

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| NFR-PRV-1 | Raw personal data shall never leave the device on the local path: 0 bytes to non-local hosts. | M | Built (enforced and measured) | egress log; `raw_export_bytes` metric | [D15](#d15) |
| NFR-PRV-2 | No real person's identifier shall be in any dataset: example domains, 555-01xx numbers, synthetic names; PII scan = 0. | M | Built | `pii_remaining = 0` in the cleaning log | [D01](#d01) |
| NFR-PRV-3 | Deleted information shall not remain anywhere the system stores text. | M | Built | FR-PRV-6, FR-ENG-7 | [D16](#d16) |
| NFR-PRV-4 | Labels (answers) shall never reach a generator; they stay with the harness. | M | Built | `system/runner.py` | [D28](#d28) |
| NFR-PRV-5 | Secrets (API keys) shall never be committed or stored in the trace. | M | Built | `.gitignore` (`.env`) | [D37](#d37) |
| NFR-SEC-1 | The backend shall listen on 127.0.0.1 only. | M | Built | `aios/api.py::serve` | [D32](#d32) |
| NFR-SEC-2 | The front end shall escape every value it shows. | M | Built | `app.js` (`esc`) | [D33](#d33) |
| NFR-SEC-3 | The figure route shall not serve files outside the figures folder. | M | Built | `aios/api.py::figure` | [D32](#d32) |

### 5.2 Performance and cost (pre-registered, H7)

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| NFR-PERF-1 | p95 latency per task ≤ 600 s on the 8 GB laptop. | M | To measure | report (`latency_p95_s`) | [D25](#d25) |
| NFR-PERF-2 | Cost per user-day on the local path ≤ $0.10 (20 W × time × $0.32/kWh). | M | To measure | report (`cost_per_user_day_usd`) | [D30](#d30) |
| NFR-PERF-3 | Non-model work (store, policy, engine) shall add well under a second per node. | S | Built | node `wall_s` in logs (milliseconds in tests) | [D10](#d10) |
| NFR-PERF-4 | The renderer shall cap output (400 tokens) and context (1,024 tokens) so 4 requests fit in memory. | S | Built | `gen2/render.py` | [D05](#d05) |

### 5.3 Resource constraints

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| NFR-RES-1 | Everything shall run on an Apple M1 with 8 GB RAM and limited disk. | M | Built | this machine | [D25](#d25) |
| NFR-RES-2 | Only one model shall be loaded at a time for heavy jobs. | M | Built | render groups by renderer; API lock | [D35](#d35) |
| NFR-RES-3 | Long jobs shall survive model-server memory growth and low disk without losing work. | M | Built | FR-OPS-2 | [D36](#d36) |

### 5.4 Reliability and recoverability

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| NFR-REL-1 | Rendering shall resume from its cache; evaluation shall resume per persona. | M | Built | restarts during the render | [D36](#d36) |
| NFR-REL-2 | A failing node shall fail its execution cleanly, with rollback of reversible actions and a logged error. | M | Built | FR-ENG-4 | [D21](#d21) |
| NFR-REL-3 | A model timeout or crash shall be recorded as a failed task, not stop a run. | M | Built | FR-GEN-3; `system/runner.py::_turn` | [D21](#d21) |
| NFR-REL-4 | Corrupted inputs shall be caught by validation with the check named. | M | Built | FR-DAT-14 | [D09](#d09) |

### 5.5 Reproducibility and research integrity

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| NFR-REP-1 | The same seed shall give the same data (content hashes recorded). | M | Built | FR-DAT-1, FR-DAT-2 | [D02](#d02) |
| NFR-REP-2 | Environments shall be pinned (`uv.lock`, Python 3.12); model digests recorded per run. | M | Built | `uv.lock`, run meta | [D37](#d37) |
| NFR-REP-3 | Hypotheses, metrics and tests shall be fixed before any result; later changes go in dated amendments. | M | Built | `research/PREREGISTRATION.md` (commit `efe545f`) | [D44](#d44) |
| NFR-REP-4 | Splits shall be committed with a hash. | M | Built | `manifests/v2_split_manifest.json` | [D07](#d07) |
| NFR-REP-5 | Nothing shall be tuned on test personas or days 25–30. | M | Built (by construction) | FR-TRN-1, protocol | [D07](#d07) |
| NFR-REP-6 | Every reported number shall be computed from run files by code, not typed in. | M | Built | `analysis/report.py` | [D29](#d29) |

### 5.6 Data quality, fairness and generalization

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| NFR-DQ-1 | v2 shall score lower than v1 on every shortcut test and closer to the real reference on calibrated distributions (H6). | M | Built; final table after the render | `p17sim audit` | [D47](#d47) |
| NFR-DQ-2 | No single wording, canary shape, locale or schedule shall dominate a split. | M | Built | audit rows | [D03](#d03) |
| NFR-FAIR-1 | Results shall be reported by region, name list and archetype, with the largest gap and its CI. | M | Built | report "Fairness slices" | [D29](#d29) |
| NFR-GEN-1 | Results shall be reported on held-out persona types, wordings and the held-out renderer. | M | Built | report "Generalization" | [D07](#d07) |

### 5.7 Usability and accessibility

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| NFR-UX-1 | The web app shall work offline with no build step and no external scripts. | M | Built | `aios/static/` | [D33](#d33) |
| NFR-UX-2 | It shall follow the system's light or dark mode. | S | Built | `app.css` | [D33](#d33) |
| NFR-UX-3 | It shall work at phone width (16 px gutter, columns stack). | S | Built | `app.css` | [D33](#d33) |
| NFR-UX-4 | Identity shall never be shown by colour alone (labels on every node and badge; tables beside charts). | S | Built | canvas labels; report tables | [D33](#d33) |
| NFR-UX-5 | Irreversible actions shall be stated as such and confirmed (delete confirmation; "cannot be recalled" note on undo). | M | Built | `app.js` | [D33](#d33) |

### 5.8 Observability

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| NFR-OBS-1 | Every model call shall record wall time, load time, prompt and output tokens, prefill and decode time, and component. | M | Built | `system/llm.py::CallRecord` | [D24](#d24), [D26](#d26) |
| NFR-OBS-2 | Every execution shall be inspectable node by node after the fact. | M | Built | Executions tab, `node_runs` | [D24](#d24) |
| NFR-OBS-3 | Every privacy decision shall be in the audit log. | M | Built | FR-PRV-9 | [D13](#d13) |

### 5.9 Maintainability and extensibility

| ID | Requirement | Pri | Status | Verified by | Decision |
|---|---|---|---|---|---|
| NFR-MNT-1 | Automated tests shall cover every component; all shall pass before a commit. | M | Built: 136 tests | `pytest` | [D45](#d45) |
| NFR-MNT-2 | A new model shall plug in by implementing the generator interface and registering it, with no change to the engine, privacy layer, harness or UI. | M | Built | `aios/generators.py`, `agents/registry.py` | [D49](#d49) |
| NFR-MNT-3 | A new node type shall be added by registering one `NodeSpec`. | S | Built | `aios/nodes.py` | [D18](#d18) |
| NFR-MNT-4 | Team rules: work on `savitha/*` branches in new files; PRs reviewed before merging to `main`; Linear issue per task. | M | Built (branch); PR and Linear pending | repo ruleset `team1` | [D37](#d37) |

### 5.10 Compliance and course rules

| ID | Requirement | Pri | Status | Decision |
|---|---|---|---|---|
| NFR-CMP-1 | Licences respected: OpenJarvis (Apache 2.0), PinchBench (MIT), Llama 3.2 Community Licence, phi3 (MIT), Qwen3.5 (Apache 2.0); Enron used as aggregates only. | M | Built | [D01](#d01) |
| NFR-CMP-2 | Every demo shall be runnable again on request; anything mocked shall be said out loud. | M | Built (commands in README) | [D44](#d44) |
| NFR-CMP-3 | Written reports shall be the students' own writing. | M | Team responsibility | [D44](#d44) |

---

## 6. Interface requirements

### 6.1 Backend API (IR-1)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/personas` | Personas with split, region and contacts |
| POST | `/api/ws/{pid}/load` | Replay to `{day, time}`; optional `handle_instructions` |
| GET | `/api/ws/{pid}/state` | Clock, records seen, pending approvals, triggers |
| POST | `/api/ws/{pid}/advance` | Move the clock `{minutes}`; fires triggers |
| GET | `/api/ws/{pid}/records` | Records by `channel`, or full-text `q` |
| GET | `/api/ws/{pid}/calendar` | Calendar on `date` |
| POST | `/api/ws/{pid}/goal` | Run `{text, generator, params}`; 409 if the generator is pending |
| GET | `/api/templates` | Template catalog |
| GET/POST | `/api/ws/{pid}/workflows` | List, or save (optionally with a trigger) |
| GET | `/api/ws/{pid}/workflows/{wid}` | One workflow graph |
| POST | `/api/ws/{pid}/workflows/{wid}/run` | Run a saved workflow |
| GET | `/api/ws/{pid}/executions` | Execution list |
| GET | `/api/ws/{pid}/executions/{eid}` | Execution with graph, node runs, sent mail, booked events |
| POST | `/api/ws/{pid}/executions/{eid}/undo` | Cancel the events it booked |
| GET | `/api/ws/{pid}/approvals` | Approvals (`status` filter) |
| POST | `/api/ws/{pid}/approvals/{aid}` | `{decision: approve | reject}`; resumes the workflow |
| GET/PUT | `/api/ws/{pid}/policy` | Tool policy, location rules, egress allowlist |
| GET | `/api/ws/{pid}/privacy` | Egress log and totals, receipts, audit log, data classes |
| POST | `/api/ws/{pid}/storage-audit` | Where a `{value}` is still stored |
| POST | `/api/ws/{pid}/delete` | Delete `{record_ids, request}`; returns the receipt |
| GET | `/api/generators` | Generator catalog and status |
| GET | `/api/ws/{pid}/suggestions` | B/C slot references and model status |
| GET | `/api/benchmark` | Harness runs and `RESULTS.md` |
| GET | `/api/results/figures/{name}` | Report figures |
| GET | `/api/profile` | Model profile |
| GET | `/` and `/static/*` | Web front end |

### 6.2 Command line (IR-2)

| Command | Does |
|---|---|
| `generate`, `render`, `validate`, `leakage`, `sanity`, `eda`, `build-trace`, `validate-trace` | v1 (Workbook 1) pipeline, unchanged |
| `generate2` | v2 personas and events + split manifest |
| `render2` | v2 text with each persona's renderer (cached, resumable) |
| `build-trace2` | Clean, validate, leakage-check and export the v2 trace |
| `audit` | Bias and shortcut audit, v1 vs v2 |
| `export-training` | Training data for Models A, B, C, with a gold check |
| `serve` | Backend and web app on localhost:8017 |
| `eval` | Run a system (`baseline`, `oracle`, `null`; models when trained) on the harness |
| `profile-models` | Measure each local model alone |
| `report` | Build `RESULTS.md` and figures |

### 6.3 Data formats (IR-3)

| Artifact | Format | Key fields |
|---|---|---|
| Trace v2 | JSONL | `event_id, persona_id, timestamp, event_type, text, permission_label, deleted, sender, channel, end` |
| Labels v2 | JSONL (harness only) | provenance, fact id/value, canary, sensitivity, preference, checker, wording, split |
| Split manifest | JSON + sha256 | persona → split, temporal windows, scoring rule |
| Workflow | JSON | `id, name, source, goal, nodes[{id, type, params, label}], edges[{source, target, branch}], trigger` |
| Run file | JSONL + `.meta.json` | one row per task or deletion audit; meta with commit, model digest, settings |
| Training exports | JSONL + manifest | see FR-TRN-2 to FR-TRN-5 |

### 6.4 Generator interface (IR-4)

```python
class Generator(Protocol):
    name: str
    status: str            # "ready" | "pending training"
    def run(self, goal: str, ws: Workspace, approver: Approver = auto_approve, **kw) -> ExecutionResult: ...
```

A model may run a full graph with `ws.engine.run(workflow, name, approver=approver)`, or work step by step
with `ws.engine.open / step / record / close`.

---

## 7. Data requirements

| ID | Requirement | Pri | Status | Decision |
|---|---|---|---|---|
| DR-1 | Dataset v2: 36 personas (12 train, 6 val, 12 test, 6 OOD from 3 held-out archetypes), 30 days each, 10,550 events, seed 2026. | M | Built (render in progress) | [D01](#d01) |
| DR-2 | Splits in two layers: persona-level (train/val/test/OOD), and day windows inside each persona (1–20, 21–24, 25–30). | M | Built | [D07](#d07) |
| DR-3 | Headline scores only on days 25–30 of test personas; OOD reported separately. | M | Built | [D07](#d07) |
| DR-4 | Data stays out of git (`data/` ignored); manifests, reference statistics and code are committed. | M | Built | [D37](#d37) |
| DR-5 | Only synthetic benchmark data may ever go to a cloud pipeline; runtime user data never does. | M | Built (policy); pipeline pending | [D38](#d38) |
| DR-6 | Retention: each dataset version keeps its manifest and hashes; deletions write receipts. | M | Built | [D16](#d16) |

---

## 8. Constraints and assumptions

1. **Hardware:** Apple M1, 8 GB RAM, about 5–12 GB free disk.
   - The paper's 9B model (7.6 GB) does not fit, so `qwen3.5:4b` stands in for every system.
   - This is reported as a limitation.
2. **Speed:** decode is about 18 tokens/s on this machine. Baseline runs take hours, so they run overnight
   under the guard.
3. **Synthetic text:** it may be easier than real mail. The audit and the held-out conditions measure how
   far v2 is from the real reference and how results hold on unseen conditions.
4. **Cloud:** GCS, BigQuery and deployment depend on the team's GCP access, which is not set up on this
   machine.
5. **Tests:** the simulated user approves every "ask" action in the harness, and each approval is counted
   as an intervention.

---

## 9. Traceability: success criteria → requirements → evidence

| Success criterion (Workbook 1 / pre-registration) | Requirements | Evidence |
|---|---|---|
| Model A ≥ reproduced baseline on the 30-day harness (H1) | MR-A-6, MR-BL-3, FR-EVA-1..7 | `RESULTS.md` H1 (after Model A) |
| 100% deletion compliance (H2) | MR-A-7, FR-PRV-6..8, FR-EVA-2, NFR-PRV-3 | deletion probes + storage audits |
| 0 raw personal data exported on the local path (H3) | NFR-PRV-1, FR-PRV-3, FR-PRV-5 | egress log, `raw_export_bytes` |
| Components help their targets (H4) | MR-A-9 | ablations (after Model A) |
| Generalizes to held-out conditions (H5) | MR-A-10, NFR-GEN-1, FR-DAT-9, FR-DAT-11 | report "Generalization" |
| v2 data less biased than v1 (H6) | NFR-DQ-1, FR-DAT-3..11, FR-DAT-16 | `research/results/audit.json` |
| p95 ≤ 600 s, cost ≤ $0.10 per user-day (H7) | NFR-PERF-1, NFR-PERF-2, FR-EVA-8 | report, model profile |
| Live, repeatable demo (course) | FR-UI-*, FR-API-*, NFR-CMP-2 | `p17sim serve` |

## 10. Open items

| Item | Owner | Blocked by |
|---|---|---|
| Baseline runs on the 30-day harness, audit table, model profile, `RESULTS.md` | Savitha | v2 render (in progress) |
| Model A design, training and evaluation | Phase 3 owner | — (data and slot ready) |
| Presidio in the PII scan | Data owner | — |
| GCS/BigQuery/Airflow pipeline; deployed prototype; Dockerfile | Team | GCP access |
| Models B, C, D | 298B | — |
| PR for `savitha/research-pipeline` and its Linear issue | Savitha | review |

---

## 11. Design decisions and trade-offs

Each requirement in sections 3–7 names its decision in the **Decision** column. Each decision below
compares 2–3 options; ✅ marks the one chosen and **Why** gives the reason in this project's situation
(8 GB laptop, synthetic data, privacy as the core claim, course checkpoints). "Pending" decisions are
not settled yet, and the recommended option is marked ➜.

### Data

<a id="d01"></a>**D01 · Data source** — serves FR-DAT-*, NFR-PRV-2, NFR-CMP-1, DR-1

| Option | Upside | Downside |
|---|---|---|
| Real personal mail and calendars (volunteers) | Most realistic | Breaks the privacy promise; consent, ethics review, no ground truth |
| Public corpora text (Enron, Avocado) as the trace | Real writing | Real names and identifiers; corporate only; no labels for deletion, leaks or preferences |
| ✅ Synthetic generator, calibrated with public aggregates | Exact labels, no real people, regenerable, adjustable difficulty | May be easier or more regular than real life (measured by the audit, D47) |

**Why:** privacy tests need planted secrets and exact answers, and no real identifier may enter the data.

<a id="d02"></a>**D02 · Where the ground truth lives** — serves FR-DAT-2, FR-DAT-6, FR-DAT-8, NFR-REP-1

| Option | Upside | Downside |
|---|---|---|
| An LLM writes whole scenarios, labels extracted afterwards | Rich, varied stories | Labels can be wrong; not reproducible; costly |
| ✅ Seeded Python skeleton holds all facts; an LLM only writes the surface text | Labels exact by construction; same seed, same data; text checked for required facts | Text variety limited by prompts |
| Templates only, no LLM | Fast, fully deterministic | Repetitive text a model can pattern-match |

**Why:** correct labels are the base of every metric; the LLM adds realism without being trusted with facts.

<a id="d03"></a>**D03 · How personas vary** — serves FR-DAT-3, FR-DAT-4, FR-DAT-7, NFR-DQ-2

| Option | Upside | Downside |
|---|---|---|
| One fixed schedule per archetype (v1) | Simple | Every member identical: the bias the review found |
| ✅ Per-person sampling from routine libraries, 16 locales, balanced splits | Real spread between people; balanced groups for fairness slices | More parameters to justify |
| Learn persona behaviour from real data | Most realistic | No such data without privacy issues |

**Why:** directly answers the feedback that the synthetic data may be biased toward certain factors.

<a id="d04"></a>**D04 · Real-world calibration** — serves FR-DAT-5

| Option | Upside | Downside |
|---|---|---|
| Hand-picked numbers | No download | Not defensible |
| ✅ Enron corpus sample, aggregates only (hour, weekday, length, reply share) | Real timing and length shapes; no text kept | Corporate 1999–2002 mail; clock offset needed correcting |
| American Time Use Survey | Best source for daily routines | BLS blocks scripted downloads (403); account needed |

**Why:** the only real reference reachable from the laptop; ATUS stays a later improvement for routines.

<a id="d05"></a>**D05 · Text renderers** — serves FR-DAT-11, FR-DAT-12, NFR-PERF-4

| Option | Upside | Downside |
|---|---|---|
| One local model (v1: llama3.2) | Simple, consistent | One writing style; models can learn it |
| ✅ Two local families (llama3.2 for all, phi3 held out for half of test/OOD) | Tests robustness to an unseen writing style; stays local and free | phi3 adds artifacts (handled by cleaning) |
| Cloud model (Gemini Flash) | Higher quality, fast | Costs credits and a key; adds a cloud dependency |

**Why:** a held-out renderer is a measurable generalization test, at no cost.

<a id="d06"></a>**D06 · Secrets and leak detection** — serves FR-DAT-10

| Option | Upside | Downside |
|---|---|---|
| One canary format (`REF-xxxxxxxx`, v1) | Easy to detect | One regex finds them all: a shortcut |
| ✅ 12 natural canary shapes, plus the secret's distinctive words | No single filter catches them; paraphrased leaks still detected | Marker words need care to avoid false alarms |
| No canaries; an LLM judges leaks | Catches subtle leaks | Not deterministic; costs model calls |

**Why:** keeps leak checking exact while removing the shortcut.

<a id="d07"></a>**D07 · Split design** — serves FR-DAT-9, FR-TRN-1, DR-2, DR-3, NFR-REP-4, NFR-REP-5, NFR-GEN-1

| Option | Upside | Downside |
|---|---|---|
| Temporal only (days 1–20/21–24/25–30, Workbook 1) | Matches how an assistant is used | A model can memorize each person |
| Persona only (train/val/test people) | Tests new people | Ignores time order |
| ✅ Both: persona splits, with Workbook 1 day windows inside each persona; plus OOD archetypes and held-out wordings | Tests new people, future days and unseen conditions at once | Fewer tasks per cell |

**Why:** keeps the Workbook 1 commitment and adds the generalization tests the feedback asked for.

<a id="d08"></a>**D08 · PII detection** — serves FR-DAT-13, FR-DAT-17

| Option | Upside | Downside |
|---|---|---|
| ✅ (now) Pattern rules with allowed synthetic forms | Fast, predictable, no install | Misses names and free-text identifiers |
| ➜ (next) Patterns + Microsoft Presidio | Adds named-entity detection | ~1 GB of models; slower; tuning needed |
| LLM-based detection | Best recall on free text | Slow; not deterministic |

**Why:** the data uses only synthetic identifiers by construction; Presidio is promised in Workbook 1 and still pending.

<a id="d09"></a>**D09 · Validation and leakage gate** — serves FR-DAT-14, FR-DAT-15, NFR-REL-4

| Option | Upside | Downside |
|---|---|---|
| Manual inspection | No code | Not repeatable; misses things |
| Warnings only | Never blocks work | Bad data slips through |
| ✅ Automatic checks that stop the export and name the failing check | Bad data can never reach training or scoring; demo-able | A false alarm blocks until fixed |

**Why:** the ISA demo requires a check failing on corrupted input.

<a id="d46"></a>**D46 · Dataset versioning** — serves FR-DAT-1

| Option | Upside | Downside |
|---|---|---|
| Change the v1 generator in place | One code path | Workbook 1 numbers no longer reproducible |
| ✅ Freeze v1, add v2 as a new module | Both reproducible; before/after audit possible | Two generators to maintain |
| Keep v2 on a separate branch | Clean main | Hard to compare and test together |

**Why:** the Workbook 1 trace hash must still reproduce.

<a id="d47"></a>**D47 · How data bias is evaluated** — serves FR-DAT-16, NFR-DQ-1

| Option | Upside | Downside |
|---|---|---|
| Qualitative review | Quick | Not convincing to reviewers |
| Downstream scores only | Measures what matters | Cannot tell data bias from model weakness |
| ✅ Quantitative audit on v1 and v2 (representation, spread, realism, shortcuts) plus downstream held-out tests | Shows exactly which bias moved and by how much | Calibrated rows are partly by construction (stated in limitations) |

**Why:** the feedback was about bias; a side-by-side table answers it with evidence.

### Storage

<a id="d10"></a>**D10 · Storage engine** — serves FR-STO-1, FR-STO-3, FR-STO-4, NFR-PERF-3

| Option | Upside | Downside |
|---|---|---|
| JSONL files (v1 style) | Simple, diff-able | No queries, no indexes, deletion means rewriting files |
| ✅ SQLite with FTS5 (one file per workspace) | In the standard library; transactions, full-text search, one local file | One writer at a time (fine for one user) |
| PostgreSQL + pgvector | Scales; rich vector search | A server to run on an 8 GB laptop; more to deploy |

**Why:** local-first, zero install, and deletion can reach every table in one place.

<a id="d11"></a>**D11 · Text search** — serves FR-STO-2

| Option | Upside | Downside |
|---|---|---|
| Own BM25 in Python | Full control | Slow on large stores; another index to keep in sync on deletion |
| ✅ SQLite FTS5 with its bm25 ranking | Fast, inside the same database, deleted rows leave the index together | Less control over scoring |
| Vector search only | Finds paraphrases | Misses exact codes and names; needs embeddings for everything |

**Why:** exact codes matter in personal data, and one store keeps deletion simple. Vector search is left to Model A.

<a id="d12"></a>**D12 · Vector storage** — serves FR-STO-4 (vectors table, for Model A)

| Option | Upside | Downside |
|---|---|---|
| ✅ Vectors as blobs in SQLite | Deleted with the record; no new dependency | Brute-force search (fine at thousands of records) |
| FAISS index | Very fast | A separate file to keep in sync with deletions |
| Chroma / LanceDB | Built-in persistence | Extra service or dependency; deletion must be audited separately |

**Why:** deletion compliance beats speed at this scale.

### Privacy layer

<a id="d13"></a>**D13 · Where privacy is enforced** — serves FR-PRV-3, FR-PRV-9, FR-ENG-1, FR-ENG-5, NFR-OBS-3

| Option | Upside | Downside |
|---|---|---|
| Instructions in each agent's prompt | No extra code | A model can ignore them; different per model |
| ✅ One engine that checks every node before it runs | Same rules for every model, impossible to bypass from a prompt, every decision logged | Every action must be a node |
| Operating-system sandbox only | Strong isolation | Cannot tell personal from public data |

**Why:** the comparison must change only the model; the rules must be identical and enforced.

<a id="d14"></a>**D14 · Tracking what data flows where** — serves FR-PRV-1, FR-PRV-2

| Option | Upside | Downside |
|---|---|---|
| No tracking | Simple | Cannot decide what may go to the cloud |
| ✅ A data class on every value (public < personal < sensitive), the maximum carried forward | Simple, explainable, enough to block cloud steps | Coarse: a summary of personal data stays personal |
| Fine-grained taint per byte | Precise | Complex; slow; hard to explain in a demo |

**Why:** a conservative rule that is easy to show and test.

<a id="d15"></a>**D15 · Egress control** — serves FR-PRV-5, NFR-PRV-1

| Option | Upside | Downside |
|---|---|---|
| Trust the code | Nothing to build | No proof that nothing left |
| ✅ One gateway for all outbound requests: allowlist, block, and byte accounting | Measurable "0 bytes exported"; blocked attempts logged | Code that bypasses the gateway is not covered (all model calls use it) |
| OS firewall (pf, Little Snitch) | Covers every process | Cannot count persona text; setup per machine |

**Why:** the success criterion is a number (0 bytes); the gateway produces it.

<a id="d16"></a>**D16 · Deletion method** — serves FR-PRV-6, FR-PRV-7, FR-PRV-8, FR-STO-5, FR-ENG-7, FR-EVA-2, NFR-PRV-3, DR-6

| Option | Upside | Downside |
|---|---|---|
| Soft delete (a flag) | Undo possible | The data is still there: fails the audit |
| ✅ Delete the record, its index row and vector, and redact what it said from every other store; write a receipt | Passes a storage audit across every table; provable | Redaction must avoid words still used elsewhere (handled) |
| Crypto-shredding (per-record keys) | Strong guarantee | Key management; logs still hold plaintext copies |

**Why:** the target is 100% deletion compliance, checked by searching every store.

<a id="d17"></a>**D17 · Approving actions** — serves FR-PRV-4, FR-ENG-2, FR-ENG-3, FR-API-4

| Option | Upside | Downside |
|---|---|---|
| Never ask | Fast | Unsafe sends |
| Ask for every action | Safe | Tiresome; high intervention rate |
| ✅ Per-tool allow / ask / deny from the user's permissions, with an approval queue | Asks only where the user wants; intervention rate is measurable | Users must set policies (defaults come from their permissions) |

**Why:** matches the trace's permission labels and the intervention-rate metric.

### Workflow engine and runtime

<a id="d18"></a>**D18 · Workflow runtime** — serves FR-WF-4, FR-GEN-4, NFR-MNT-3

| Option | Upside | Downside |
|---|---|---|
| n8n | Mature editor, many integrations | Built for people drawing workflows; Node.js service; Sustainable Use License; privacy checks would sit outside it |
| LangGraph | Agent graphs with state | Heavy dependency tree; cloud integrations to disable; its own concepts |
| ✅ Own small engine inspired by n8n | Every node passes our privacy checks; tiny; fully tested; workflows can be generated | We maintain it |

**Why:** the project studies *generated* workflows under a privacy layer; the engine is where the privacy claim lives.

<a id="d19"></a>**D19 · Workflow representation** — serves FR-WF-1, FR-WF-3

| Option | Upside | Downside |
|---|---|---|
| Python code per workflow | Flexible | A model cannot safely generate it; hard to inspect |
| ✅ JSON graph of typed nodes and edges | A model can output it; easy to validate, store, draw and audit | Less expressive than code |
| BPMN | Standard | Verbose; poor fit for model output |

**Why:** Model A must produce something the engine can check before running it.

<a id="d20"></a>**D20 · Passing data between nodes** — serves FR-WF-2

| Option | Upside | Downside |
|---|---|---|
| One shared context object | Simple | Unclear what depends on what |
| ✅ n8n-style references `{{ node.path }}` | Dependencies visible; data classes follow references | Paths can break (caught as a clear error) |
| Typed ports per node | Strong checking | Heavier to author and to generate |

<a id="d21"></a>**D21 · Failure handling** — serves FR-ENG-4, FR-GEN-3, NFR-REL-2, NFR-REL-3

| Option | Upside | Downside |
|---|---|---|
| Stop and leave things as they are | Simple | Half-done actions (a booked meeting with no follow-up) |
| ✅ Stop and undo reversible actions; record the failure | Clean state; honest logs | Sent emails cannot be undone (stated to the user) |
| Full sagas with compensation for every action | Complete | Most actions have no real compensation |

<a id="d48"></a>**D48 · Resolving names** — serves FR-WF-6

| Option | Upside | Downside |
|---|---|---|
| Exact match only | Never wrong | Fails on "Ana" for "Ana Silva" |
| ✅ Exact, then unique part, then close match; ambiguity returned as a message | Handles real requests; never guesses between two people | A typo may still fail |
| Let the model choose | Flexible | Can pick the wrong person silently |

<a id="d22"></a>**D22 · Time model** — serves FR-WF-5, FR-WS-1, FR-WS-2, FR-WS-3, FR-API-2

| Option | Upside | Downside |
|---|---|---|
| Real wall clock | Realistic | A 30-day month takes 30 days |
| ✅ Simulated clock that replays records in order | Any moment of the month on demand; no peeking at the future | Not a live mailbox |
| Snapshots per day | Fast loading | Coarse; no triggers between snapshots |

<a id="d49"></a>**D49 · Generator interface** — serves FR-GEN-1, FR-GEN-5, NFR-MNT-2

| Option | Upside | Downside |
|---|---|---|
| Each model with its own API | Freedom per model | Harness and UI change for each model |
| ✅ One `run(goal, workspace, approver)` interface | Models swap without touching engine, harness or UI | Models must express themselves as workflows |
| Message bus between services | Scales | Overkill on one laptop |

<a id="d50"></a>**D50 · What the harness runs** — serves FR-EVA-1

| Option | Upside | Downside |
|---|---|---|
| A light in-memory shell | Fast | Does not test storage, policies or deletion for real |
| ✅ The full runtime with an in-memory database | Measures the real system; same code as the app | Slightly slower |
| The live app over HTTP | End to end | Slow; flaky; harder to resume |

### Models and baseline

<a id="d23"></a>**D23 · Baseline integration** — serves FR-GEN-2, MR-BL-1, MR-BL-2

| Option | Upside | Downside |
|---|---|---|
| Run OpenJarvis itself on our tools | Most faithful | ~GBs of dependencies and a Rust build; disk too small; tool APIs differ |
| ✅ Re-implement its `native_openhands` loop from source (prompt, tools, trimming, turns) | Same behaviour, runs on our tools and policies, small | A re-implementation, documented as such |
| A generic ReAct agent | Easy | Not the published baseline |

<a id="d24"></a>**D24 · Recording agent runs** — serves FR-ENG-6, FR-WS-4, NFR-OBS-1, NFR-OBS-2

| Option | Upside | Downside |
|---|---|---|
| Separate agent logs | Simple | Two kinds of evidence to read |
| ✅ Each model step and tool call becomes a workflow node | Baseline and Model A shown and audited the same way | More node rows |

<a id="d25"></a>**D25 · Local model size** — serves FR-EVA-9, MR-BL-3, NFR-PERF-1, NFR-RES-1

| Option | Upside | Downside |
|---|---|---|
| `qwen3.5:9b` (the paper's model) | Faithful to the paper | 7.6 GB: does not fit in 8 GB RAM or the free disk |
| ✅ `qwen3.5:4b` for every system | Fits; same family; comparison stays controlled | Lower quality than 9B (reported as a limitation) |
| A cloud model | Best quality | Breaks the local-only claim |

<a id="d26"></a>**D26 · Model server** — serves NFR-OBS-1

| Option | Upside | Downside |
|---|---|---|
| ✅ Ollama | Same server as the PinchBench baseline; tool calling; timing fields per call | Memory growth over long runs (handled by D36) |
| llama.cpp server | Lean | More setup; separate model files |
| MLX | Fast on Apple silicon | Different tool-calling path from the baseline |

<a id="d27"></a>**D27 · Model settings** — serves MR-BL-2

| Option | Upside | Downside |
|---|---|---|
| ✅ OpenJarvis defaults: thinking off, 8,192 context, temperature 0.6 | Faithful to the baseline; ~3× fewer tokens | Some reasoning quality lost |
| Thinking on | Better reasoning | Slower; longer runs on the laptop |

<a id="d40"></a>**D40 · Model A method (pending)** — serves MR-A-1 … MR-A-10

| Option | Upside | Downside |
|---|---|---|
| ➜ Retrieval-augmented hierarchical planner with trained parts (pre-registered) | Matches the abstract; ablations map to parts | Must show real training (MR-A-4) |
| LoRA fine-tune of a small LLM on gold workflows | Clearly "trained by the team" | Disk and memory on 8 GB; adds a second model |
| Prompting only | Quick | Does not meet the course rule |

<a id="d41"></a>**D41 · Model B method (298B)** — serves FR-GEN-6, MR-B-*

| Option | Upside | Downside |
|---|---|---|
| ✅ (reference now) Counting repeated calendar entries | Transparent bar to beat | Cannot follow preference changes |
| GRU/LSTM on event sequences | Small; trains locally | Short memory |
| S4 / Mamba | Long histories | Heavier training |

<a id="d42"></a>**D42 · Model C method (298B)** — serves MR-C-*

| Option | Upside | Downside |
|---|---|---|
| ✅ (reference now) Most-accepted option over the last week | Transparent bar to beat | Ignores context |
| LinUCB | Uses context; offline replay evaluation | Linear assumptions |
| Thompson sampling | Good exploration | Harder to explain |

<a id="d43"></a>**D43 · Model D method (298B)** — serves FR-PRV-10, MR-D-*

| Option | Upside | Downside |
|---|---|---|
| Classifier critic (Llama Guard style) | Fast | Generic safety categories, not personal permissions |
| Emulator-based risk scoring (ToolEmu style) | Finds risky sequences | Costly |
| Rules plus a learned scorer | Explainable | Rules need upkeep |

### Evaluation and research integrity

<a id="d28"></a>**D28 · Task scoring** — serves FR-EVA-3, FR-EVA-4, FR-GEN-7, NFR-PRV-4

| Option | Upside | Downside |
|---|---|---|
| LLM judge | Flexible | Costs, varies, needs a cloud call |
| ✅ Deterministic checkers on answers and actions, with an oracle and a null agent | Exact, free, repeatable; sanity-checked | Strict about wording of values (spacing handled) |
| Human grading | Gold standard | Slow; not repeatable |

<a id="d29"></a>**D29 · Statistics** — serves FR-EVA-7, NFR-REP-6, NFR-FAIR-1

| Option | Upside | Downside |
|---|---|---|
| Plain t-test per task | Familiar | Treats a person's tasks as independent |
| ✅ Cluster bootstrap over personas; exact McNemar with Holm for ablations | Respects correlation within a person; exact for paired pass/fail | Wider intervals |
| Mixed-effects models | Most complete | Harder to explain and check |

<a id="d30"></a>**D30 · Cost and energy** — serves FR-EVA-8, NFR-PERF-2

| Option | Upside | Downside |
|---|---|---|
| Measure power with `powermetrics` | Real watts | Needs administrator rights |
| ✅ Measured model time × a fixed 20 W × $0.32/kWh (pre-registered) | Simple, comparable across systems | An estimate |
| Cloud price-per-token equivalent | Easy to compare with APIs | Not what the laptop costs |

<a id="d31"></a>**D31 · Training supervision for Model A** — serves FR-TRN-2, FR-TRN-3, FR-TRN-4

| Option | Upside | Downside |
|---|---|---|
| Plans written by a large LLM | Varied | Can be wrong; needs a cloud call |
| ✅ Gold workflows built from ground truth, each run and checked before export | Correct by test | Fewer styles of plan |
| Reinforcement from checker rewards only | No labels needed | Slow; unstable on a laptop |

<a id="d44"></a>**D44 · Research integrity** — serves FR-EVA-6, FR-TRN-5, NFR-REP-3, NFR-CMP-2, NFR-CMP-3

| Option | Upside | Downside |
|---|---|---|
| Decide hypotheses after seeing results | Flexible | Not credible |
| ✅ Pre-register hypotheses, metrics and tests; dated amendments; run metadata | Credible; examiners can check the order | Less room to change course |

<a id="d45"></a>**D45 · Testing approach** — serves NFR-MNT-1

| Option | Upside | Downside |
|---|---|---|
| Manual testing | Quick at first | Breaks silently |
| ✅ pytest, tests first, scripted model and fake renderer for speed, plus live checks | 136 fast tests; caught real bugs | Model quality itself is not unit-tested (the harness does that) |

### Backend, front end and operations

<a id="d32"></a>**D32 · Backend framework** — serves FR-API-1, FR-API-5, FR-API-6, NFR-SEC-1, NFR-SEC-3

| Option | Upside | Downside |
|---|---|---|
| Standard-library `http.server` | No install | Manual routing and validation |
| ✅ FastAPI + uvicorn | Validation, automatic docs at `/docs`, test client; ~5 MB | One more dependency |
| Flask / Django | Familiar | Flask lacks validation; Django is heavy |

**Why:** chosen with you on 2026-10-15.

<a id="d33"></a>**D33 · Front end** — serves FR-UI-1, FR-UI-4..9, NFR-SEC-2, NFR-UX-1..5

| Option | Upside | Downside |
|---|---|---|
| React + a build step | Rich components | npm, node_modules and a build on a full disk |
| Streamlit | Fast to write | Hard to make an n8n-style canvas; reruns on every click |
| ✅ One page with vanilla JS and SVG, served by the backend | No build, works offline, light | More hand-written code |

<a id="d34"></a>**D34 · Workflow canvas** — serves FR-UI-2, FR-UI-3

| Option | Upside | Downside |
|---|---|---|
| React Flow | Polished editing | Needs React and a build |
| Mermaid | Quick diagrams | Static; no per-node inspection |
| ✅ Own SVG layout by dependency depth | Click a node to inspect it; no dependency | No drag-and-drop editing yet |

<a id="d35"></a>**D35 · Concurrency** — serves FR-API-3, NFR-RES-2

| Option | Upside | Downside |
|---|---|---|
| Parallel model requests | Faster for many users | Two models do not fit in 8 GB |
| ✅ One model request at a time (a lock) | Stable on 8 GB | Requests wait in line |
| A job queue with workers | Scales | Overkill for one user |

<a id="d36"></a>**D36 · Protecting long jobs** — serves FR-EVA-5, FR-OPS-2, NFR-REL-1, NFR-RES-3

| Option | Upside | Downside |
|---|---|---|
| Watch by hand | Nothing to build | The disk filled twice |
| Ollama settings only (keep_alive, fewer parallel slots) | Simple | Did not stop the 10 GB growth |
| ✅ Guard that stops, unloads and resumes when memory or disk crosses a limit | Runs unattended overnight; no lost work | Brief pauses during restarts |

<a id="d37"></a>**D37 · Environment and secrets** — serves NFR-PRV-5, NFR-REP-2, NFR-MNT-4, DR-4

| Option | Upside | Downside |
|---|---|---|
| pip + requirements.txt | Familiar | Versions drift |
| conda | Handles native libraries | Large; slow |
| ✅ uv with a lock file; `.env` and `data/` ignored by git | Exact versions; fast; no secrets or data in git | uv must be installed |

<a id="d51"></a>**D51 · Operations interface** — serves FR-OPS-1

| Option | Upside | Downside |
|---|---|---|
| Notebooks | Interactive | Hidden state; hard to rerun |
| Makefile | Familiar | Poor arguments; macOS/Linux only |
| ✅ One `p17sim` command with sub-commands | Every step repeatable with one line; documented in README | Must be kept in sync with the code |

<a id="d38"></a>**D38 · Cloud pipeline (pending, needs GCP)** — serves FR-OPS-3, DR-5

| Option | Upside | Downside |
|---|---|---|
| Cloud Composer (managed Airflow) | Managed | Base cost would use up the credits |
| ➜ Airflow in Docker on a small GCE VM, GCS + BigQuery | Real scheduler, logs, retries; cheap | We run the VM |
| Cron scripts | Simple | Not an automated, monitored pipeline (the rubric asks for one) |

<a id="d39"></a>**D39 · Deployment (pending)** — serves FR-OPS-4, FR-OPS-5

| Option | Upside | Downside |
|---|---|---|
| ➜ GCE VM with Docker (backend + Ollama) | Same stack as the laptop; reachable | Pay while it runs |
| Cloud Run with a GPU | Scales to zero | Cold starts load the model; GPU quota needed |
| Laptop behind a tunnel | Free | Not "deployed"; laptop must stay on |

Note for both: only synthetic data and code would be deployed, never a real user's data (DR-5).
