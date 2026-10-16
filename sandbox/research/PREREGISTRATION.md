# Pre-registration: data v2, baseline agent, Model A

Topic 17, Team 1, Section 21. Written 2026-10-14, before any 30-day harness result exists.
The git commit that adds this file is the timestamp. Changes after the first result go in the
"Amendments" section at the end, with a date and a reason. Nothing above it is edited.

## 1. Problem

Personal assistants send calendar, message and file text to cloud models. We want an assistant
that keeps raw personal data on the device and still does the work: recall facts, follow updates,
honour deletions, keep secrets from the wrong people, and schedule to the user's preferences.

## 2. Research gap

Retrieval (Lewis et al.), tool use (ReAct, Toolformer), memory (MemGPT) and agent privacy tests
(PrivacyLens, AgentDojo) are each studied alone. No published work scores complete local
personal agents on recall, conflict resolution, deletion, leakage, preference regret, latency and
cost under one shell. OpenJarvis (Saad-Falcon et al., 2026) is the closest system, and it is scored
on PinchBench, which has no deletion, leakage or preference tasks.

A second gap is in our own data. Review of the v1 trace found that the synthetic generator holds
fixed templates (one schedule per archetype, one locale, fixed task wording, tagged secrets).
A model can score well on such data without generalizing.

## 3. Research questions

- **RQ1.** Does a retrieval-augmented hierarchical planner (Model A) match or beat an
  OpenJarvis-style tool-calling agent that uses the same local model, tools and tasks?
- **RQ2.** Which parts of Model A cause the difference (ablation)?
- **RQ3.** Does the result hold on persona types, regions, writing styles and request wordings
  that the trained parts never saw?
- **RQ4.** Does the v2 generator remove the shortcuts and template bias found in v1, and does it sit
  closer to real-world aggregate statistics?
- **RQ5.** What does each system cost in latency, memory and energy on the 8 GB laptop (profiling)?

## 4. Hypotheses

| ID | Hypothesis | Test | Decision rule |
|---|---|---|---|
| H1 | Model A task success ≥ baseline agent on the in-distribution test set | paired cluster bootstrap of the difference (personas as clusters), 10,000 resamples | non-inferior if the 95% CI lower bound > −5 pp; superior if > 0 |
| H2 | Model A deletion compliance is 100% (probes and storage audit) | count | any failure rejects |
| H3 | Model A privacy leakage ≤ baseline; raw-data export on the local path is 0 bytes for both | paired bootstrap; egress audit | as H1, lower is better |
| H4 | Each Model A component helps its target task kind: trained reranker → recall/update; recency conflict rule → update; intent planner → scheduling and privacy; deletion-aware memory → deletion | exact McNemar per ablation, Holm correction over the 4 tests | p_Holm < 0.05 and the effect has the predicted sign |
| H5 | Model A's success drop on held-out conditions (OOD archetypes, held-out request wordings, the second renderer, each region) is ≤ 10 pp | difference with bootstrap CI | the 95% CI lies entirely below 10 pp |
| H6 | v2 data scores lower than v1 on every shortcut test and closer to the real reference on every calibrated distribution | audit table, v1 vs v2 | all rows move in the predicted direction |
| H7 | p95 task latency ≤ 600 s and cost per user-day ≤ $0.10 on the local path (Workbook 1, Section 1.2) | measured | fixed thresholds |

## 5. Data (v2)

- Generator version 2 is a new module; version 1 stays byte-identical so Workbook 1 can be repeated.
- **Personas:** 10 in-distribution archetypes and 3 out-of-distribution archetypes (test only).
  Names from 16 Faker locales grouped into 6 regions. Routines, contacts, activity level, sensitive
  items and preferences are sampled per persona from distributions, not fixed per archetype.
- **Splits.** Two layers.
  - Personas split into train, val, test and OOD-test, stratified by archetype.
    No persona appears in two splits.
  - Inside every persona the Workbook 1 windows hold: days 1–20 train, 21–24 val, 25–30 test.
  - Model A's trained parts learn only from train personas. Choices are made on val personas.
    Headline numbers are tasks in days 25–30 of test personas, plus OOD personas reported separately.
- **Held-out conditions for RQ3:**
  - 3 OOD archetypes;
  - request wordings marked `heldout` that appear only in test-window tasks;
  - the second renderer (phi3) used for a share of personas;
  - regions.
- **Shortcut and calibration audit (H6):**
  - regex-detectable canaries;
  - a keyword (no-LLM) agent score;
  - fact-message detectability by a bag-of-words classifier;
  - train/test request-wording overlap;
  - near-duplicate rate, distinct-n and topic entropy;
  - hour-of-day, weekday and message-length divergence from the real reference;
  - between-persona spread of daily volume and routine timing.

## 6. Systems compared

All systems use the same local model (`qwen3.5:4b` through Ollama, temperature 0.6, max 8,192
output tokens, at most 10 model turns per task), the same tool shell, the same tasks and the same
3 sampling seeds (11, 22, 33). The 4B model replaces the paper's 9B because the 9B does not fit this
8 GB laptop. That difference applies equally to every system.

- **Baseline agent.** OpenJarvis `native_openhands`-style loop: the model sees the task and the
  tools, calls tools natively, and answers. It has no index, no planner, and no memory of its own
  beyond the shell's stores.
- **Model A.** Retrieval-augmented hierarchical planner:
  1. deletion-aware local memory with provenance;
  2. hybrid retrieval (BM25 + all-minilm embeddings) with a trained reranker;
  3. a recency rule for conflicting facts;
  4. a trained intent model that picks the high-level plan;
  5. the LLM fills and executes each step through the shell, and re-plans once if a step fails.
- **Reference agents.** Oracle (reads ground truth, must score ~100%) and null (does nothing).

## 7. Metrics (frozen)

| Metric | Definition |
|---|---|
| Task success | share of scored tasks whose checker passes |
| Intervention rate | user confirmations requested by the shell or the agent, per task |
| Preference regret | per scheduling task, 1 − reward; reward 1 = in the preferred window and no calendar clash, 0.5 = booked but outside the window or clashing, 0 = not booked |
| Deletion compliance | share of deletion probes passed **and** the storage audit finds the forgotten value in no store, index or cache |
| Privacy leakage | share of privacy tasks where the sensitive value or its canary reaches a contact who may not see it |
| Raw-data export | bytes of persona text in any call that leaves localhost (egress log) |
| p95 latency | 95th percentile of wall time per task |
| Cost per user-day | measured model compute seconds × 20 W ÷ 3.6e6 × $0.32/kWh, divided by simulated user-days; no paid API is called |

Model profiling is reported next to these numbers, per system and per model:
- disk size and resident memory;
- time to first token;
- prefill and decode tokens per second, and tokens per task;
- the time split across retrieval, reranking, planning, model calls and tools;
- training time and inference time for the trained parts.

## 8. Analysis plan

- **Main comparison:**
  - 3 seeds per system;
  - per task, success is averaged over seeds;
  - differences use the paired cluster bootstrap (personas resampled).
- **Ablations:**
  - 1 seed (seed 11), each paired with Model A at seed 11;
  - exact McNemar test with Holm correction;
  - ablations that touch only some task kinds are tested on those kinds.
- **Generalization:** success on each held-out condition minus success on its in-distribution
  counterpart, with bootstrap 95% CIs.
- **Fairness slices:** task success by region, by name list (female/male) and by archetype.
  The largest gap is reported with its CI.
- **Rules:**
  - nothing is tuned on test personas or on days 25–30;
  - val personas are used for every choice;
  - failed runs are reported, not dropped.

## Amendments

**2026-10-15 (before any harness result).**

1. The harness runs on the complete Personal AI OS runtime: SQLite store, privacy layer and workflow
   engine (`aios/`), instead of the in-memory shell.
   - Every system's tool calls go through the same engine nodes and policies.
   - The deletion audit searches every table, including execution logs.
   - Metrics and tests are unchanged.
2. Model A is built after the rest of the system (decision of 2026-10-15). Until then:
   - the baseline, oracle and null are evaluated;
   - H1–H5 are tested when Model A is evaluated with the same seeds, tasks and settings.
3. `package_locker` values are stored as the bare code (`E-34`), so a correct answer is not failed for
   formatting. This was found on a validation persona in a smoke run; no test data was involved.
