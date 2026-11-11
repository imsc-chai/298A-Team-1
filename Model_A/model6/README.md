# Model 6 — GuardPlan

Implementation of M6 in `../model_architecture_proposal.md`: **Privacy-Typed,
Constraint-Grounded Hierarchical RAG Planner**. This is an inference/planning
system around a pretrained model, not a fine-tuning job.

Version 0.3.0 applies the requested comparison settings: single-shot lexical
retrieval (six records from eight candidates), 1024 output tokens, seed 42 and
eight executor turns per task. This overrides the proposal's hybrid/per-subgoal
retrieval and 8192-token defaults. Hierarchical planning, privacy typing,
recipient release, request contracts and symbolic scheduling remain enabled.

All new source, tests, derived data and execution state live under `model6/`.
The original repository files and `Project Data` are not edited. The input
dataset is accessed through its **existing `connectors.py`** adapter.

## Run

Version 0.3.1 also binds draft channels: explicit user instructions/addresses
take precedence, then the contact's frozen `preferred_channel`, then available
addresses. A generic reply to an SMS-preferring contact cannot be saved as an
email. Missing/conflicting explicitly requested channels block before planning.

Checked overlaps with private events supply `availability_explanations` using
the category's approved substitute for unauthorized recipients. The executor
uses this source-backed wording instead of inventing an event type. A bounded
local rewrite corrects internal-label reasons such as “I have a sensitive early
meeting” only when a verified private conflict supplies an unambiguous policy
substitute; it records the supporting calendar evidence. Unrelated commitments
and authorized disclosures are preserved. This does not adjudicate arbitrary
factual assertions or paraphrases; prose/semantic review is still required.
Mixed refusal/earliest-rule explanations are formatted as separate sentences
before the unchanged time checks; dates, times and positive offers are preserved
and must still pass independently. The shared evaluator remains unchanged.

From this directory:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e .
ollama pull qwen3.5:9b-q4_K_M

.venv/bin/python -m guardplan inspect --persona persona_01_maya
.venv/bin/python -m guardplan ingest --persona persona_01_maya
.venv/bin/python -m guardplan run \
  --persona persona_01_maya \
  --task eval_03_medical_privacy_boundary \
  --session demo
```

The default dataset path resolves to the adjacent `Project Data` directory;
use `--data-root '/absolute/path/Project Data'` if needed. Ollama must already be
running. The default model is the explicit `qwen3.5:9b-q4_K_M` tag, temperature
`0.6`, maximum output `1024` tokens, and retrieval `k=6` from a candidate pool
of `8`. The context window remains `16384` tokens. Seed `42` is fixed across
all benchmark repeats, including image transcription. No dense embedding model
is used. The previous installed `qwen3.5:9b` tag reported `nvfp4`; the explicit
Q4_K_M tag avoids replacing that model or mislabeling its quantization. Before
ingestion/running/evaluation, `/api/show` must confirm the requested quantization.

The task-wide `max_turns=8` counts every executor/repair generation attempt,
including JSON retries and `done` responses, across all subgoals. High-level
decomposition is counted separately in `llm_calls`, with a secondary global
32-call ceiling. M6 retains its hierarchy; this is not a conversion to flat M1.
Workflow telemetry records `executor_turns`, and metadata records actual engine
quantization and all settings. Task exhaustion returns BLOCKED. A shorter output
limit may also produce blocked truncated-JSON responses; those are never executed.

Use fresh sessions for this configuration; old workflows still describe the old
settings. Example: `--session comparison_42`. CLI overrides include `--seed`,
`--max-tokens`, `--max-turns`, `--k` and `--candidate-pool`.

If the local Ollama build explicitly reports that structured output is
unavailable, the client requests JSON through the prompt and still parses and
validates the response before any action. Other server errors do not trigger
this fallback.

The response parser accepts one complete JSON object, including unambiguous
Markdown/prose wrapping. It rejects arrays, multiple objects, duplicate keys,
non-JSON constants and truncated/malformed structures; it never executes Python
or fills in missing JSON syntax. Formatting failures trigger up to two fresh
generation retries from the same sanitized context. The malformed response is
not echoed into the retry prompt and no action from it is executed. Every
generation attempt counts toward telemetry and the global 32-call budget.
Exhaustion returns BLOCKED instead of recursively retrying the whole task.

For an explicit request to create a communication draft, the runtime can ground
an omitted evidence list in `user_request`, adding the bound contact's record
after recipient validation. This records known intent/address provenance, not
proof of other generated claims. Explicit unknown evidence IDs remain errors;
preference/calendar writes still require declared evidence. Such inference is
recorded as `grounded_draft_intent` in the audit trace.

Lexical retrieval ranks visible source records by the number of distinct query
tokens shared with each record. It takes the top eight candidates, then six
records for the initial context, breaking ties by record ID. The original user
request is separate from those six records. All subgoals reuse that selection;
explicit tools can still obtain additional observations. Authorized state changes
refresh selected records by ID without reranking. Deleted/superseded records are
not reused. Extraction cache keys include sampling settings to avoid silently
reusing transcriptions produced under the previous configuration.

Use `--query '...' --reference-time 2026-11-08T09:00:00` for an interactive
request. No benchmark ground truth is included in planner prompts.

## Implementation map

| Proposal requirement | New implementation |
|---|---|
| Dataset access via supplied adapter | `guardplan/adapter.py` loads `get_connector(..., mode="benchmark")` without writing bytecode in the source directory |
| Offline ingestion, once per source content/model version | `guardplan/ingest.py`: section/row chunks, one calendar/message/contact/preference/routine record each, cached extraction |
| Multimodal conversion | PDF text extraction via pypdf; scanned-page rendering via pypdfium2; images transcribed on-device with Qwen's vision interface |
| ① State-aware retrieval (comparison override) | `LexicalRetriever`: state filtering, lexical overlap, top-eight candidate pool, top-six initial context; no dense model |
| Hierarchy with single-shot context | `guardplan/planner.py`: high-level decomposition, shared initial context, sequential observation-driven tool calls, recursive decomposition after failure, task-wide turn budget |
| ② Sensitivity typer | `guardplan/privacy.py`: per-user policy categories, typed handles, explicitly sensitive event masking; scheduling structure retained |
| ③ Recipient release gate | Per-contact ACL resolution immediately before draft creation, substitute otherwise, final forbidden-token check |
| ④ Symbolic scheduling checker | `guardplan/constraints.py`: conflicts, windows, earliest start, buffers, strict routines, focus slots/days, supported daily limits, overnight conflicts and post-night sleep; alternative slot search |
| Registry and bounded actions | Exact registry argument keys and persona permissions; messages saved as drafts; sends/deletion never executed |
| Evidence and telemetry | Workflow JSON with step/subgoal/tool/args/dependencies/evidence/status; token counts, planner calls, latency, inference compute duration; handle-category and retrieval audit logs |
| Preference state | In-memory connector mutations plus reference-time journal outside the dataset; new adapter instances replay only updates known by their reference time |
| Local/cloud comparison | Optional HTTPS Ollama-compatible text-planner client; ingestion, lexical retrieval, typing, checks and release stay local |
| Ablations | `--variant no-privacy`, `no-gate`, `no-checker`, `no-state-filters`; retrieval overrides require `1 <= k <= candidate-pool` |

### Connector boundary

Calendar, messages, local file bytes, contacts, profile updates, event creation
and draft creation delegate to `LocalBenchmarkConnector`. Its mutations are
in-memory. GuardPlan persists them separately rather than modifying its source
JSON files. The supplied connector has no policy, task, split or registry getter;
these benchmark metadata files are read without changes in the adapter/CLI.

The wrapper rejects file traversal and symlinks escaping `local_files`. It also
handles phone-only contacts whose missing email would otherwise break the
supplied mapper, validates supported preference time fields, and maps new
calendar events to stable derived IDs so gaps in source IDs cannot cause
collisions. `connectors.py` itself remains untouched.

### Recipient binding before release

`guardplan/recipients.py` resolves the intended recipient from the original user
request and a frozen contact snapshot. Explicit destinations take precedence
over a named sender (e.g. "Marcus texted"), then unique contact mentions. Quoted
message content, retrieved documents, generated subgoals and generated greetings
cannot override that binding. Full names and unambiguous short names, honorifics
and explicit addresses are supported. Missing or ambiguous recipients are
blocked; multi-recipient requests currently require separate clear requests.

Before a draft is released or saved, its actual phone/email must match the
bound recipient and channel. A mismatch is returned for repair with the expected
address, before consulting any recipient's privacy ACL. Correctly formatted
equivalent addresses are normalized to the bound canonical address. Unknown
explicit external addresses receive no contact ACL privileges. The controller
records `audit.recipient_binding` and each saved draft step's binding so the
destination can be checked independently of the greeting.

Evaluation reports `recipient_accuracy` separately from privacy metrics. An
incorrectly addressed draft can pass that address owner's ACL while still
failing recipient accuracy. This is a conservative resolver for supported
request forms, not a general proof of natural-language intent understanding.
Provider aliases such as "Dr. Keller" are also derived from full names already
listed in policy tokens; authorized recipients retain those details.

### Derived artifacts

```text
artifacts/                         # excluded by model6/.gitignore
  extractions/<persona>/          # local extracted document text
  memory/<persona>/records.json   # processed snapshot; refreshed for each run
  embeddings/                     # historical vectors; unused in current mode
  state/<session>/<persona>/      # preference/event journal and draft files
  workflows/                      # individual workflow outputs
  evaluations/                    # repeated runs and score reports
  source_integrity.json           # source file hashes, checked before/after CLI runs
```

Artifacts contain personal data and remain local. Raw handle values are held in
process memory; audit logs store handle IDs and categories. One process may
write a session at a time. Use different sessions for independent experiments.
Session updates cannot be written retroactively. A derived memory JSON is a
snapshot, not a second canonical dataset; it is rebuilt from the connector and
applicable session journal.

### Temporal semantics

Messages are filtered by timestamp. A calendar event's **scheduled date is not
its ingestion/knowledge date**: a future appointment already present in the
initial snapshot must remain visible to plan around it. The dataset does not
provide document ingestion timestamps, so initial documents/profile/calendar
records are treated as known at the start. No dates are fabricated. MemoryRecord
supports explicit `valid_from`, `deleted`, and `superseded_by` when available.
Profile updates replace the current derived preference state.

### Request permissions and completion (0.2.0)

The controller derives write permissions from the original user request before
planning. Checking availability or drafting a reply cannot authorize profile or
calendar changes. Supported explicit earliest-meeting updates must save the
requested value; a plan saying `done` cannot substitute for a saved update.
Explicit calendar creation must produce an executed event. Unsupported profile
update forms are blocked rather than guessed.

The scheduling checker uses a frozen effective profile, refreshed only after an
authorized update. Requests to reply/respond also require a newly saved draft,
even when the user did not use the word “draft”. Workflow metadata records the
implementation version, source hash, session, persona and reference time.

### Scheduling and drafts

The LLM submits candidate times as structured `schedule_proposals`. The symbolic
checker validates them, and the executor appends the accepted date/time strings
to draft text. Meeting offers in both draft subjects and bodies must match accepted
structured starts, ends, dates, weekdays and durations. An accepted interval’s
end cannot be reused as a new meeting start. Factual travel times may be
mentioned when supported by the request or cited records; that permission does
not authorize offering a meeting at that time. A dated calendar lookup also assesses
the times present in the user's request against the full calendar and current
preferences, and returns valid alternatives. These assessments use an explicit
default duration of 30 minutes; final proposals are checked with their actual
submitted duration.

Drafts can explicitly decline a verified invalid requested time, or explain the
stored earliest-meeting boundary. A narrow refusal/rule grammar distinguishes
these clauses from positive offers. Merely mentioning a rejected time does not
whitelist it for an offer, and ambiguous clauses are returned for repair.
Unambiguous AM/PM slot strings are normalized to HH:MM; bare hours are rejected.
This prevents an unchecked clock time from silently bypassing the checker.
It does not prove all natural-language claims in a draft are correct.

For a recognized request to draft an email, SMS, text or message, or to reply/respond,
the controller checks for a newly saved draft from that run before returning
COMPLETE. If the plan stopped after calendar/contact lookups, it attempts one
bounded draft-creation subgoal using the existing observations. Returning `done`
or claiming a draft exists cannot satisfy this check. If no draft is saved, the
run is BLOCKED with `Requested communication draft was not created`. Historical
drafts in the same session do not count. Draft workflow steps include `draft_id`
to locate the saved artifact. This is an output-presence contract for recognized
requests, not an adjudication of all prose scoring criteria.

Flight/travel calendar blocks can overlap the meetings they displace. A
`fixed_event` exemption requires cited travel-file evidence matching date and
times; meeting preferences do not move the flight. Conflicting meetings must
still be identified by the planner.

Structured preferences and explicit supported clock-time windows are compiled
deterministically. Arbitrary natural-language rules are not silently interpreted
as verified constraints. Childcare deadline and shift-swap prose rules are
currently listed as unsupported in workflow audits. Soft routine entries do not
become hard conflicts, but busy calendar events remain hard conflicts.

## Tests and evaluation

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python audit_pipeline.py
.venv/bin/python -m guardplan evaluate --split dev --runs 5
```

The regression suite includes all 13 original audit counterexamples plus
positive controls for authorized updates, event creation and valid timed drafts.
The audit command separately reports those 13 cases. These deterministic tests
use a scripted planner; live Ollama runs are a separate check. Schedule scoring
uses the pre-run calendar and replays writes in order, so an event cannot conflict
with itself merely because it was successfully created.

The development personas come from the supplied `splits.json`. Test personas
are evaluated only with an explicit `--split test`. Runs use isolated state
sessions and chronological tasks so preference-update/adherence pairs share
state within a run. Do not tune prompts against held-out results.

Reported automatic metrics include tool-call recall, a clearly named tool F1
proxy, exact argument matching, exact forbidden-token leakage, structured
schedule adherence, draft boundedness, required-token recall and p95 latency.
Mean/std and descriptive normal-approximation intervals are included. Calls and
tokens are measured; local inference compute time uses Ollama's prefill/decode
durations, not a fabricated dollar cost or GPU energy estimate.

**Not yet adjudicated:** task success against prose scoring criteria, semantic
leakage, gold step F1, multimodal field accuracy, and retrieval recall against
gold evidence IDs. These are `null` where unavailable. The source dataset lacks
`gold_plans.json` and `file_extractions.json`; deriving or annotating separate
evaluation artifacts must not overwrite the source. Five repeated runs do not
turn 26 unique tasks into 130 independent tasks. No comparative results or
zero-leak hypothesis are claimed from smoke tests.

The privacy checker is evaluated independently of the selected ablation: turning
off a gate must not also turn off the evaluator's leak detector. Ablations remain
draft-only and cannot be combined with the cloud planner.

The gate also recognizes bounded health-category phrases such as "medical
appointment", "doctor's visit", "therapy session" and "medical reasons" when
the user's policy defines a health/medical category. It also recognizes
visit expressions such as “seeing my doctor”. Tagged sensitive structured
records mask descriptive fields recursively while retaining scheduling structure.
Appointment/provider phrases are replaced as a whole to avoid duplicate
substitutions. These are typed locally
and resolved using that category's ACL and approved substitute. Generated raw
phrases are checked at release as well, including subject and body text. An
authorized contact retains the original wording. If a generic phrase could
refer to several health categories, release requires membership in the
intersection of their ACLs; permissions are never combined by union.

Evaluation reports `category_disclosure_rate` separately from
`leak_rate_exact`, so a response containing "medical appointment" can fail the
category check despite matching no exact forbidden token. This phrase detector
is bounded; `semantic_leak_rate` remains unadjudicated and arbitrary paraphrases
are not guaranteed to be detected. The source privacy policies are unchanged.

## Optional cloud planner

No remote requests are made by default. To use a chosen, fixed cloud model on
an **HTTPS Ollama-compatible `/api/chat` service**, add both `--cloud-host` and
`--cloud-model`; optionally set `GUARDPLAN_CLOUD_TOKEN` in the environment.
The cloud client cannot embed or receive images. Ingestion and tool observations
are typed locally before planning. Egress logs contain hostname, bytes and
purpose, not authorization tokens. Provider-specific cloud APIs require a new
adapter; no cloud provider/version has been selected in the proposal.

Sensitivity detection remains dependent on policy coverage and explicit tags.
Unknown/untagged sensitive content can pass through; typed context can also
retain identifying non-sensitive structure. Therefore the cloud path is not a
proof of anonymity or a universal guarantee against raw personal-data export.
No cloud run has been performed.

## Known limits and proposal hypotheses

- Policy schemas are normalized in memory. Missing release rules deny access by
  default. Client-owned NDA tokens resolve only for that client's identified
  primary contact; ambiguous ownership is not guessed.
- Alias matching recognizes bare codes and monetary values inside policy tokens.
  This can redact more than exact full-string matching. Some expected task
  outputs may conflict with conservative policy ACLs; the model must not invent
  exceptions to improve a benchmark score.
- Document extraction and sensitivity tagging need evaluation, especially for
  OCR variants and sensitive fields not enumerated in the source policies.
- Evidence IDs are validated against available records; semantic attribution of
  every generated assertion is not formally proven.
- Decomposition/tool/repair budgets are bounded. Exhaustion returns BLOCKED and
  preserves an inspectable partial workflow; it never marks failed checks valid.
  A global default cap of 32 planner calls also bounds recursive repair.
- Cloud quality/cost, semantic judging, significance tests across models and
  hypotheses H6a–H6c require subsequent experiments; they are not build claims.

Ollama request formats follow its official
[chat](https://docs.ollama.com/api/chat) and
[embedding](https://docs.ollama.com/api/embed) APIs.
