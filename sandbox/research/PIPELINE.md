# Topic 17 research pipeline: from problem to conclusion

Team 1, Section 21. Each stage says what was done, how, and where the evidence is. Commands run
from `sandbox/`. Numbers in the results stages come from `research/results/RESULTS.md`, which
`p17sim report` builds from the run files; nothing is typed in by hand.

| # | Stage | Evidence |
|---|---|---|
| 1 | Problem | this file, §1 |
| 2 | Literature review | Workbook 1 §1.5 (Table 4) |
| 3 | Research gap | §3, `PREREGISTRATION.md` §2 |
| 4 | Research questions | `PREREGISTRATION.md` §3 |
| 5 | Hypotheses | `PREREGISTRATION.md` §4 (committed before any result) |
| 6 | Dataset | `src/p17sim/gen2/`, `manifests/v2_split_manifest.json`, `research/results/audit.json` |
| 7 | Baseline | `baseline/` (PinchBench reproduction, untouched) and `src/p17sim/agents/baseline.py` |
| 8 | Proposed method | `src/p17sim/agents/model_a.py`, `memory.py` |
| 9 | Training | `src/p17sim/agents/train.py`, `models/model_a/training_report.json` |
| 10 | Validation | val personas; grids and thresholds in the training report |
| 11 | Testing | test and OOD personas, days 25–30, `research/results/runs/` |
| 12 | Metrics | `src/p17sim/analysis/metrics.py` |
| 13 | Controlled comparison | `RESULTS.md` "Main comparison", H1–H3 |
| 14 | Ablation study | `RESULTS.md` "Ablations", H4 |
| 15 | Statistical analysis | `src/p17sim/analysis/stats.py` |
| 16 | System-level evaluation | `RESULTS.md` profiling and latency sections; `research/results/system_eval.json` |
| 17 | Results | `research/results/RESULTS.md` |
| 18 | Limitations | §18 |
| 19 | Conclusion | §19 |

---

## 1. Problem

Personal assistants are useful because they can read a person's mail, calendar and notes, and that
is exactly what most people do not want to send to a cloud provider. The project asks whether a
small model running on a laptop, inside a shell that controls what it may do, can handle that work
well: recall facts, follow changes, forget on request, keep private things private, and book
meetings the way the person likes.

## 2. Literature review

Workbook 1, §1.5, Table 4 reviews 23 papers grouped by the design choice each informs:
- **grounding:** RAG [2];
- **tool use:** ReAct [3], Toolformer [4];
- **memory and feedback:** Reflexion [5], Generative Agents [6], MemGPT [7], trustworthy memory search [11];
- **evaluation:** AgentBench [8], survey [9];
- **planning:** HuggingGPT [13], ADaPT [14];
- **routines:** S4 [15], Mamba [16];
- **suggestions:** LinUCB [17], Thompson sampling [18];
- **safety:** Llama Guard [19], ToolEmu [20], AgentDojo [21];
- **privacy:** ConfAIde [22], PrivacyLens [23];
- **deletion:** machine unlearning [24];
- **the reproduced system:** OpenJarvis [1].

## 3. Research gap

Two gaps.

1. **Method gap.** Each paper covers one piece: retrieval, tool use, memory, planning, privacy tests or
   deletion. None scores complete local personal agents on all of these at once (recall, conflicting
   updates, deletion, leakage, preference regret, latency and cost) under one shell. OpenJarvis [1] is
   the closest system, and its benchmark (PinchBench) has no deletion, leakage or preference tasks.
2. **Data gap.** Real personal mail cannot be used: it would break the privacy promise the project
   makes. Our v1 synthetic trace was reviewed and found to hold a single template per archetype, one
   locale, one wording per task and one canary format, so a model could pass it by matching patterns.

## 4–5. Research questions and hypotheses

See `PREREGISTRATION.md`, committed on 2026-10-14 before any harness result (commit `efe545f`):
- RQ1 to RQ5;
- hypotheses H1 to H7, each with its test and decision rule.

## 6. Dataset

### 6.1 Sources

| Source | Use | Licence or terms |
|---|---|---|
| Team generator v2 (`gen2/`) | the 30-day personal-data traces | team-created; no real identifiers |
| Enron corpus, 9,300-row random sample (Hugging Face parsed copy) | aggregate timing and length statistics only; no text kept | FERC public release |
| llama3.2 3B, phi3 mini (local, Ollama) | write the message text | Llama 3.2 Community Licence; MIT |
| OpenJarvis v1.0.0, PinchBench 23 tasks | baseline reproduction only (`baseline/`) | Apache 2.0; MIT |

### 6.2 How v2 removes the bias found in v1

| Bias or shortcut in v1 | v2 change |
|---|---|
| One schedule per archetype | per-person routines drawn from a library; start, weekdays, jitter (Laplace) and skip rate drawn per person |
| One locale (en_US) | 16 Faker locales in 6 regions; region and name list balanced within each split |
| Exactly 6 contacts and 4 secrets | 4–14 contacts with a relation mix by archetype and Zipf-like activity; 1–4 secrets |
| Poisson(5) messages a day, uniform from 07:00 to 23:00 | per-person volume (lognormal) and weekend factor; Enron hour-of-day; night chronotypes |
| Text 1–3 sentences, one model | Enron-calibrated length; per-contact writing style; two model families, phi3 held out |
| Facts said once, never confusable | repeats; near-miss distractors (someone else's code); up to two updates |
| Preference signals stop at day 20 | signals on all 30 days, with per-person noise and up to two changes |
| "Book ... on day 27" | real dates ("this Thursday", "on October 3rd"); 15–60 minute meetings; no clash with the calendar |
| Canaries all match `REF-[0-9a-f]{8}` | 12 natural formats; leak checks also look for the secret's distinctive words |
| One wording per task | 4–6 seen wordings per task type, plus held-out wordings used only in the test window |

### 6.3 Size, schema and splits

- **Size:** 36 personas and 10,550 events.
  - 12 train, 6 val, 12 test (in-distribution) and 6 OOD-test (3 held-out archetypes).
  - About 5,500 texts are written by the local models.
- **Schema:** Workbook 1's seven fields (`event_id`, `persona_id`, `timestamp`, `event_type`, `text`,
  `permission_label`, `deleted`) plus `sender`, `channel` and `end`. Labels live in a separate file that
  only the harness reads.
- **Splits:** two layers.
  - Personas are split train/val/test/OOD, and no persona appears in two splits.
  - Inside each persona the Workbook 1 day windows hold: days 1–20, 21–24 and 25–30.
  - The manifest and its hash are committed (`manifests/v2_split_manifest.json`).
- **Cleaning:** v1's rules (preambles, quotes, markdown, whitespace, imputation, duplicates, PII
  redaction), plus v2 rules for model meta-notes ("(15 words, casual tone)") and "[Your Name]"
  placeholders.
- **Validation:**
  - schema, unique ids, timestamps inside each persona's 30 days;
  - deletions and updates point to an earlier record;
  - no PII remains.
- **Leakage checks:** these stop the pipeline.
  - no persona in two splits;
  - held-out wordings only in the test window;
  - every task's evidence precedes it;
  - canaries stay with their persona;
  - no task contains its answer;
  - no near-duplicate label text across persona splits.
- **Bias and shortcut audit:** v1 and v2 go through the same measurements (`p17sim audit`), reported
  in RESULTS.md.

## 7. Baseline

Two baseline layers, kept apart.

1. **The published-number reproduction** (`baseline/`, untouched by this work).
   - Paper: OpenJarvis + Qwen3.5-9B on the 23-task PinchBench, Saad-Falcon et al. (2026), Table 1,
     Hermes block, condition (c). The paper reports 87.9%.
   - Our five runs give **71.5% ± 3.4**. The differences are documented in `baseline/SETUP.md`.
2. **The same agent on our 30-day harness.** `agents/baseline.py` re-implements OpenJarvis's
   `native_openhands` loop from its source at commit `e97088f`:
   - its system prompt, native tool calls with the text fallback, 4,000-character observations and
     the 3,000-token truncation rule;
   - temperature 0.6, 8,192 max tokens, 10 turns, thinking off and an 8,192-token context.

   It uses `qwen3.5:4b` because the 9B does not fit the 8 GB laptop. Model A uses the same model, so
   the comparison is controlled.

## 8. Proposed method: Model A

A retrieval-augmented hierarchical planner (`agents/model_a.py`):
- **Level 1:** a trained intent model chooses one of four plans; if unsure, a tool loop with retrieved
  evidence takes over.
- **Level 2:** each plan's sub-goals are done by retrieval, rules, the local model, or shell tools.
  - **Lookup:** deletion-receipt check → hybrid BM25 + all-minilm retrieval → trained reranker →
    recency rule → the model answers from the evidence only.
  - **Forget:** reranker-selected records are deleted in the shell and in memory (index, vectors,
    caches) → receipt.
  - **Schedule:** contact, length and date (rules first, model fallback) → preferred window from a
    trained, time-decayed estimator → first free slot → `create_event` → verify, and re-plan once on error.
  - **Send:** recipient → last week's records minus private ones → model drafts → local check drops
    sentences with words found only in private records → `send_email`.

Everything runs on the laptop. The egress log checks this on every call.

## 9. Training

Run with `p17sim train-a`. Only train personas are used, and held-out wordings are never seen.

| Part | Data | Model | Search |
|---|---|---|---|
| Reranker | fact questions at the times they are asked plus extra times; candidates from hybrid retrieval; label = the record states that fact | logistic regression on 18 numeric features + hashed n-grams of the record and query-record word pairs | C grid; learning curve over training-set size |
| Intent | train-persona requests + 400 seen-template fills | TF-IDF word (1–2) and character (2–5) n-grams + logistic regression | C grid |
| Preference | probe days 21–30, true meeting window | time-decayed vote | half-life × feedback weight × decline weight grid (128 settings) |

Compute time and artifact sizes are in `models/model_a/training_report.json`.

## 10. Validation

- Every choice uses val personas only:
  - the C values;
  - the relevance and deletion thresholds (best F1; recall ≥ 0.95 for deletion);
  - the preference settings;
  - prompt and pipeline fixes during development (smoke runs on val, days 21–24).
- Test personas and days 25–30 were not looked at before the final runs.

## 11. Testing

- **Headline:** tasks in days 25–30 of the 12 test personas.
- **Generalization:** the same window for the 6 OOD personas.
- **Seeds:** 3 sampling seeds (11, 22, 33) for the baseline and Model A, and seed 11 for each ablation.
- **Scoring:** the harness replays every day; deletion requests are handled when they happen.

## 12. Metrics

Defined in `PREREGISTRATION.md` §7 and computed in `analysis/metrics.py`:
- task success;
- intervention rate;
- preference regret;
- deletion compliance (probes **and** storage audit);
- privacy leakage;
- raw-data export bytes;
- p95 latency;
- cost per user-day.

Each comes with profiling: tokens, calls, time to first token, prefill and decode speed, and seconds
per component.

## 13. Controlled comparison

Same model, settings, tools, shell, data, tasks and seeds; only the method differs. Pairs are matched
per task.

## 14. Ablation study

Four ablations, each removing one part, scored on the task kinds that part targets (pre-registered):

| Ablation | Removes | Target kinds |
|---|---|---|
| `model_a-no_reranker` | trained reranker (reciprocal-rank fusion instead) | recall, update |
| `model_a-no_recency` | recency rule and dates in evidence | update |
| `model_a-no_intent` | trained intent model (the LLM picks the plan) | scheduling, privacy |
| `model_a-no_deletion_memory` | purge of Model A's own memory | deletion (probes and storage audit) |

## 15. Statistical analysis

- **Comparisons:** paired cluster bootstrap (10,000 resamples, personas as clusters), because tasks of
  the same person are correlated.
- **Ablations:** exact McNemar with Holm correction.
- **Generalization and fairness:** independent-group bootstrap for gaps.
- **Hypotheses:** decision rules are fixed in the pre-registration.

## 16. System-level evaluation

- **Latency, cost and memory:** the whole system on the laptop, with p95 latency per task, cost per
  user-day, and the model profile (`p17sim profile-models`: load time, resident memory, time to first
  token, prefill and decode speed, energy per 1,000 tokens).
- **Privacy at the system boundary:** the egress log over every model call.
- **Deletion end to end:** storage audits after every deletion request.

## 17. Results

See `research/results/RESULTS.md`.

## 18. Limitations

Written after the results; see RESULTS.md, "Limitations".

## 19. Conclusion

Written after the results; see RESULTS.md, "Conclusion".
