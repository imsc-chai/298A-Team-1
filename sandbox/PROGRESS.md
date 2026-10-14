# Trace v1 — progress checklist (Phase 2: trace, splits, harness)

Owner: Savitha Vijayarangan · Branch: `savitha/trace-v1` (from `main` at `1ec259e`) · Target: Team Meeting 1, Oct 8, 2026

Tick a box only when its evidence exists (a file, a test, a command output). Update the "Last updated" line each time.

Last updated: 2026-10-08 01:10 PDT

## How the pipeline fits together

```
personas (seeded)  ->  skeleton events + labels  ->  LLM renders text  ->  clean + PII scan  ->  trace.jsonl (Workbook schema)
   personas.py           skeleton.py                 render.py (Ollama)     trace.py              labels.jsonl (harness only)
                                                                                                   splits.json, manifest.json
                                                                       ->  validate + leakage  ->  EDA plots + report.md
                                                                           trace.py               trace_eda.py
```

- **Skeleton**: plain Python, no LLM. Writes every event and its ground truth (facts, canaries, task checkers). Same seed, same output.
- **Render**: a local Llama 3.2 model writes the message and file text only. Different model family from the agent (Qwen3.5-9B).
- **Clean**: strips LLM artifacts, imputes failed renders, drops duplicate filler, redacts any real-looking identifier.
- **Trace vs labels**: models read `trace.jsonl`; only the harness reads `labels.jsonl`.
- **Splits** (Workbook 1, Section 2.1): days 1–20 train, 21–24 val, 25–30 test.

## Checklist

### 0. Setup — done
- [x] Clone team repo into `Final_Project/298A-Team-1`
- [x] Create branch `savitha/trace-v1`; all work in the new `sandbox/` folder; no existing file changed (`git diff main -- . ':!sandbox'` is empty)
- [x] Pinned environment: `sandbox/uv.lock`, Python 3.12, `.venv` created with `uv sync`

### 1. Generator (extraction source 1) — done
- [x] Persona specs: Faker names, `example.com`/`example.org` emails, fictional 555-01xx phones
- [x] 30-day skeleton: calendar, messages, files, preferences, conflicts, deletions, sensitive items with canaries, scored tasks
- [x] Workbook temporal windows: train 1–20, val 21–24, test 25–30 (`schema.py`)
- [x] Generate 10 personas, seed 2026: 2,842 events, trace hash `d41497d4…` (`p17sim generate --seed 2026 --n-in 10 --n-ood 0`)
- [x] Determinism, invariant, split and leakage tests pass

### 2. LLM render (extraction source 2) — done
- [x] Renderer with retries, required-string check and cache (`render.py`); 16/16 smoke renders kept every fact
- [x] Render all 1,725 text events with `llama3.2` (resumed from 359 cached; about 20 per minute, finished 01:05)
- [x] Final render: 1,715 ok, 10 `render_failed` after 3 tries; 1,117 events need no text

### 3. Clean into the Workbook schema — done
- [x] Cleaning: LLM preamble, wrapping/stray quotes, markdown, whitespace (`trace.py: clean_text`)
- [x] Missing values: failed renders imputed with template text that keeps the labeled fact; flagged `source=imputed`
- [x] Duplicates: exact duplicate filler messages per persona dropped and counted
- [x] PII scan: non-example emails, non-fictional phones, SSN and card patterns redacted (pattern rules; Presidio not yet)
- [x] Export to the 7 Workbook fields + permission records; labels to a separate file
- [x] Tested end to end on fake text (`tests/test_trace.py`, 8 tests)
- [x] Run on the real render: `p17sim build-trace --data data/v1 --out ../data/trace_v1` → 2,886 records (2,842 − 6 duplicates + 50 permission records); trace hash `5e28d37f…`; rebuild in 11 s is byte-identical
- [ ] Copy cleaning counts and 3 before/after samples into the slides (counts: 495 wrapping quotes, 17 stray quotes, 5 preambles, 2 whitespace, 10 imputed, 6 duplicates, 0 PII left; samples in `Presentation_Evidence_TM1/03_clean_trace/`)

### 4. Validate, split, leakage — done
- [x] Validation: schema, unique ids, timestamps in window, deletions/conflicts point to earlier records, no PII left
- [x] Leakage: no tasks in train, task evidence inside train window, canaries only in their own persona, no task text contains its answer
- [x] Real run shows `0 violations` and `0 findings`; splits train 2,147 / val 299 / test 440; oracle 100%, null 0% on utility tasks
- [x] Corruption demo run: bad timestamp, planted PII, duplicate id each fail with the named check (outputs in `Presentation_Evidence_TM1/06_corruption_demo/`); rehearse it live once

### 5. EDA — real plots done
- [x] Five plots: records per day with split lines, type mix per split, text length, scored tasks per split, cleaning issues
- [x] `eda/report.md` with tables, samples and coverage gaps
- [x] Review real plots; picked `records_per_day.png` (slide 7) and `cleaning_issues.png` (ISA demo)
- [ ] Write the "problems found" list (start below)

### 6. Presentation prep
- [x] Slide plan, key numbers, Q&A, concept explainer: [Team Meeting 1 Prep](https://claude.ai/code/artifact/b3ff2fd0-9c62-4235-aa69-26c44ab3718b)
- [ ] ISA demo runbook section (needs real numbers from step 3)
- [ ] Rehearse slides 5 and 7 with a timer (1.5 + 2.5 min)
- [ ] Rehearse the ISA demo once from a fresh terminal

### 7. Git
- [x] One commit per pipeline step, in execution order (12 commits, authored by Savitha)
- [x] Pushed as a new branch `savitha/trace-v1-steps` (ruleset `team1` blocks updates to an existing branch, but allows creating one)
- [ ] Open a PR from `savitha/trace-v1-steps` to `main`; merge after 1 approving review (required by the ruleset)
- [ ] Open a Linear issue for Phase 2 (one assignee) and link it in the PR as `Fixes 298-xx`
- Note: `savitha/trace-v1` on GitHub holds only commit 1 and can't be deleted (ruleset); ignore it.
- Note: later changes (the real trace, plots) can't be pushed onto this branch either: use a new branch per update, or a PR into this branch.

## Problems found (for the "problems, not only fixes" slide)
- Preference signals stop at day 20, so val/test have none; Model C's bandit needs them in the test window. Fix in Cycle 4 (changes event ids, so it needs a full re-render).
- LLM text adds preambles ("Here is a short message:"), stray quotes, and invented names despite instructions.
- Routines are more regular than real life (start-time spread of only about 7–16 minutes).
- PII scan uses pattern rules; Workbook 2.1 promises Presidio as well.
- 10 of 1,725 renders (0.6%) dropped a required fact 3 times; imputed from a template.
- The LLM wrapped 495 texts (29%) in quotes despite "Output only the text".

## Next cycle (Cycle 4, Oct 8–21)
- [ ] Preference signals across all 30 days; re-render
- [ ] Presidio in the PII scan
- [ ] GCS + BigQuery layers, Airflow DAG on a schedule
- [ ] Tool adapter and harness runs for the baseline and Model A
