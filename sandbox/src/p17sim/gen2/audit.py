"""Bias and shortcut audit: the same measurements on v1 and v2 so the change is visible (H6).

Each row answers one question a reviewer asked about the v1 data:
- Who is in it? (archetypes, regions, locales, name lists)
- Do people behave differently from each other? (volume spread, routine-time spread)
- Does it look like real mail? (hour of day, weekday, length vs the Enron reference)
- Can a model pass without the skill? (single canary shape, train/test wording overlap,
  a keyword agent with no LLM, a bag-of-words classifier that finds fact messages)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from p17sim.gen2.skeleton import reference


@dataclass
class View:
    """A dataset reduced to what the audit needs, so v1 and v2 go through one code path."""
    name: str
    personas: list[dict[str, Any]]  # persona_id, archetype, region, locale, name_list, split
    events: list[dict[str, Any]]  # persona_id, day, minute, weekday, kind, provenance, text, sender, ...
    tasks: list[dict[str, Any]]  # persona_id, day, minute, kind, text, checker, persona_split, window
    canaries: list[str]
    routines: list[dict[str, Any]] = field(default_factory=list)  # persona_id, archetype, name, start_minutes


# ------------------------------------------------------------------ helpers

def _fact_words() -> list[str]:
    from p17sim.gen2.phrasing import FACT_NAMES
    from p17sim.skeleton import FACT_KINDS

    return sorted({k.replace("_", " ") for k in FACT_KINDS} | {n for names, _ in FACT_NAMES.values() for n in names},
                  key=len, reverse=True)


FACT_WORDS = _fact_words()

def _entropy(counts: Counter) -> float:
    p = np.array(list(counts.values()), dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(len(p))) if len(p) > 1 else 0.0


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(a / b), 0)))  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


V1_CANARY = re.compile(r"REF-[0-9a-f]{8}")


def prefix(s: str) -> str:
    """The leading word of a canary ('REF', 'chart', 'loan'): what a filter would key on."""
    m = re.match(r"[A-Za-z]+", s)
    return m.group(0).lower() if m else "<digits>"


def _template(text: str, names: list[str]) -> str:
    t = text.lower()
    for n in sorted(names, key=len, reverse=True):
        t = t.replace(n.lower(), "<x>")
    t = re.sub(r"\d+", "<n>", t)
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------------ measurements

def representation(v: View) -> dict[str, Any]:
    arch = Counter(p["archetype"] for p in v.personas)
    region = Counter(p.get("region", "north_america") for p in v.personas)
    locale = Counter(p.get("locale", "en_US") for p in v.personas)
    names = Counter(p.get("name_list", "unrecorded") for p in v.personas)
    return {"archetypes": len(arch), "archetype_evenness": round(_entropy(arch), 3),
            "regions": len(region), "locales": len(locale), "region_evenness": round(_entropy(region), 3),
            "name_lists": dict(names)}


def heterogeneity(v: View) -> dict[str, Any]:
    per = defaultdict(Counter)
    for e in v.events:
        if e["kind"] == "message" and e.get("provenance", "").startswith("filler:message"):
            per[e["persona_id"]][e["day"]] += 1
    means = np.array([np.mean([c[d] for d in range(1, 31)]) for c in per.values()])
    by_arch = defaultdict(list)
    for r in v.routines:
        by_arch[(r["archetype"], r["name"])].append(np.mean(r["start_minutes"]))
    spreads = [float(np.std(x)) for x in by_arch.values() if len(x) >= 2]
    within = [float(np.std(r["start_minutes"])) for r in v.routines if len(r["start_minutes"]) >= 3]
    return {"daily_volume_cv_across_people": round(float(means.std() / means.mean()), 3),
            "routine_start_spread_across_people_min": round(float(np.mean(spreads)), 1) if spreads else 0.0,
            "routine_start_spread_within_person_min": round(float(np.median(within)), 1) if within else 0.0}


def realism(v: View) -> dict[str, Any]:
    ref = reference()
    hours, wdays, words = np.zeros(24), np.zeros(7), []
    for e in v.events:
        if e["kind"] == "message" and e.get("provenance", "").startswith(("filler:message", "fact", "sensitive")):
            hours[e["minute"] // 60] += 1
            wdays[e["weekday"]] += 1
            if e.get("text"):
                words.append(len(e["text"].split()))
    ref_hours = np.array(ref["hour_of_day_pmf"]) + 1e-9
    ref_wd = np.array(ref["weekday_pmf_mon0"]) + 1e-9
    out = {"hour_js_divergence": round(_js(hours + 1e-9, ref_hours), 3),
           "weekday_js_divergence": round(_js(wdays + 1e-9, ref_wd), 3)}
    if words:
        w = np.log1p(np.array(words))
        out["log_words_mean"] = round(float(w.mean()), 2)
        out["log_words_sd"] = round(float(w.std()), 2)
        out["log_words_mean_gap_to_ref"] = round(abs(float(w.mean()) - ref["words_log_mean"]), 2)
        out["log_words_sd_ratio_to_ref"] = round(float(w.std()) / ref["words_log_sd"], 2)
    return out


def text_diversity(v: View) -> dict[str, Any]:
    texts = [e["text"] for e in v.events if e["kind"] in ("message", "note") and e.get("text")
             and e.get("source", "").startswith(("llm", "ollama"))]
    if not texts:
        return {}
    toks = [t.lower().split() for t in texts]
    uni = [w for t in toks for w in t]
    bi = [tuple(t[i:i + 2]) for t in toks for i in range(len(t) - 1)]
    openers = Counter(" ".join(t[:2]) for t in toks if len(t) >= 2)
    return {"llm_texts": len(texts), "distinct_1": round(len(set(uni)) / len(uni), 3),
            "distinct_2": round(len(set(bi)) / len(bi), 3),
            "top_opener_share": round(openers.most_common(1)[0][1] / len(toks), 3)}


def shortcuts(v: View) -> dict[str, Any]:
    prefixes = Counter(prefix(c) for c in v.canaries)
    by_kind = defaultdict(set)
    train_tmpl, test_tmpl = defaultdict(set), []
    for t in v.tasks:
        tmpl = _template(t["text"], t.get("names", []) + FACT_WORDS)
        by_kind[t["kind"]].add(tmpl)
        if t["window"] == "test" and t["persona_split"] in ("test", "ood_test", "all"):
            test_tmpl.append((t["kind"], tmpl))
        elif t["persona_split"] in ("train", "all"):
            train_tmpl[t["kind"]].add(tmpl)
    overlap = np.mean([tmpl in train_tmpl[k] for k, tmpl in test_tmpl]) if test_tmpl else float("nan")
    return {"canary_top_prefix_share": round(prefixes.most_common(1)[0][1] / len(v.canaries), 3),
            "canary_prefixes": len(prefixes),
            "canary_caught_by_v1_regex": round(float(np.mean([bool(V1_CANARY.fullmatch(c)) for c in v.canaries])), 3),
            "wordings_per_task_kind": {k: len(s) for k, s in sorted(by_kind.items())},
            "test_wording_seen_in_train": round(float(overlap), 3)}


def keyword_agent(v: View) -> dict[str, Any]:
    """No LLM. Recall/update/deletion: answer with the newest earlier text that shares the most
    words with the question. Scheduling: book 'day N' at the start of the most-accepted window.
    High scores mean the data can be passed by string matching."""
    hist = defaultdict(list)
    for e in v.events:
        if e.get("text") and e["kind"] not in ("task",):
            hist[e["persona_id"]].append(e)
    for h in hist.values():
        h.sort(key=lambda e: (e["day"], e["minute"]))
    stop = {"what", "is", "my", "the", "me", "can", "you", "a", "i", "do", "have", "was", "it", "find", "for",
            "please", "remind", "look", "up", "of", "to", "did", "which", "end", "with", "need", "quick", "one"}
    hits = defaultdict(list)
    for t in v.tasks:
        if t["kind"] not in ("recall", "update", "deletion", "scheduling"):
            continue
        before = [e for e in hist[t["persona_id"]] if (e["day"], e["minute"]) < (t["day"], t["minute"])]
        args = t["checker"]["args"]
        if t["kind"] == "scheduling":
            m = re.search(r"day (\d+)", t["text"])
            ok = bool(m and int(m.group(1)) == args["day"])
            hits["scheduling_day_parsed"].append(ok)
            continue
        q = {w.strip("?.,!").lower() for w in t["text"].split()} - stop
        best, score = None, 0
        for e in reversed(before):
            s = len(q & {w.strip("?.,!:").lower() for w in e["text"].split()})
            if s > score:
                best, score = e, s
        ans = (best["text"] if best else "").lower()
        if t["kind"] == "recall":
            ok = args["expected"].lower() in ans and not any(d.lower() in ans for d in args.get("distractors", []))
        elif t["kind"] == "update":
            stale = args["stale"] if isinstance(args["stale"], list) else [args["stale"]]
            ok = args["expected"].lower() in ans and not any(s.lower() in ans for s in stale)
        else:
            continue  # answering nothing passes deletion; not informative here
        hits[t["kind"]].append(ok)
    return {k: round(float(np.mean(v_)), 3) for k, v_ in sorted(hits.items())}


def fact_detectability(v: View) -> dict[str, Any]:
    """Cross-validated AUC of a bag-of-words classifier that tells messages carrying a true fact
    from all other messages (filler and distractors). Near 1.0 means facts announce themselves."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    rows = [e for e in v.events if e["kind"] in ("message", "note") and e.get("text")]
    y = np.array([int(e.get("provenance", "").startswith(("fact:", "fact_repeat:", "update:"))) for e in rows])
    if y.sum() < 10:
        return {}
    groups = [e["persona_id"] for e in rows]
    X = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit_transform([e["text"] for e in rows])
    pred = cross_val_predict(LogisticRegression(max_iter=2000, class_weight="balanced"), X, y,
                             groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    return {"fact_message_auc": round(float(roc_auc_score(y, pred)), 3), "fact_messages": int(y.sum()),
            "messages": len(rows)}


def run(v: View) -> dict[str, Any]:
    return {"dataset": v.name, "representation": representation(v), "heterogeneity": heterogeneity(v),
            "realism": realism(v), "text_diversity": text_diversity(v), "shortcuts": shortcuts(v),
            "keyword_agent": keyword_agent(v), "fact_detectability": fact_detectability(v)}


# ------------------------------------------------------------------ loaders

def view_v1(data: Path, rendered: Path | None = None) -> View:
    """v1: the Workbook 1 cohort (data/v1). Text comes from the rendered events when present."""
    from p17sim.io import read_jsonl

    personas = read_jsonl(data / "personas.jsonl")
    splits = json.loads((data / "split_manifest.json").read_text())["splits"]
    evs = read_jsonl(rendered or data / "events.jsonl")
    kind = {"message": "message", "file": "note", "calendar_event": "calendar", "user_task": "task"}
    events, tasks, canaries, routines = [], [], [], defaultdict(list)
    names = {p["persona_id"]: [p["name"]] + [c["name"] for c in p["contacts"]] for p in personas}
    for e in evs:
        k = kind.get(e["type"], e["type"])
        row = {"persona_id": e["persona_id"], "day": e["day"], "minute": e["minute"], "weekday": (e["day"] - 1) % 7,
               "kind": k, "provenance": e["provenance"], "text": e.get("text"),
               "source": "llm" if e.get("render_status") == "ok" else "template"}
        events.append(row)
        if e.get("canary"):
            canaries.append(e["canary"])
        if k == "task":
            tasks.append({**row, "kind": e["checker"]["kind"], "checker": e["checker"], "names": names[e["persona_id"]],
                          "persona_split": "all", "window": "test" if e["day"] > 24 else "val"})
        if e["provenance"].startswith("routine:"):
            routines[(e["persona_id"], e["provenance"])].append(e["minute"])
    arch = {p["persona_id"]: p["archetype"] for p in personas}
    return View("v1", [{**p, "split": splits[p["persona_id"]]} for p in personas], events, tasks, canaries,
                [{"persona_id": pid, "archetype": arch[pid], "name": prov, "start_minutes": m}
                 for (pid, prov), m in routines.items()])


def view_v2(data: Path, events_file: str = "events.jsonl") -> View:
    from p17sim.gen2.build import load

    personas, extras, evs, splits = load(data, events_file)
    xb = {x.persona_id: x for x in extras}
    kind = {"message": "message", "file": "note", "calendar_event": "calendar", "user_task": "task"}
    events, tasks, routines = [], [], defaultdict(list)
    for e in evs:
        x = xb[e.persona_id]
        k = kind.get(e.type.value, e.type.value)
        row = {"persona_id": e.persona_id, "day": e.day, "minute": e.minute,
               "weekday": (x.start_weekday + e.day - 1) % 7, "kind": k, "provenance": e.provenance,
               "text": e.text, "source": x.renderer if e.render_status == "ok" else "template"}
        events.append(row)
        if k == "task":
            p = next(p for p in personas if p.persona_id == e.persona_id)
            tasks.append({**row, "kind": e.checker.kind, "checker": e.checker.model_dump(),
                          "names": [p.name] + [c.name for c in p.contacts], "persona_split": splits[e.persona_id],
                          "window": "test" if e.day > 24 else "val"})
        if e.provenance.startswith("routine:"):
            routines[(e.persona_id, e.provenance)].append(e.minute)
    pmeta = [{"persona_id": p.persona_id, "archetype": p.archetype, "region": xb[p.persona_id].region,
              "locale": xb[p.persona_id].locale, "name_list": xb[p.persona_id].name_list,
              "split": splits[p.persona_id]} for p in personas]
    arch = {p.persona_id: p.archetype for p in personas}
    canaries = [c for x in extras for c in x.sensitive_canaries.values()]
    return View("v2", pmeta, events, tasks, canaries,
                [{"persona_id": pid, "archetype": arch[pid], "name": prov, "start_minutes": m}
                 for (pid, prov), m in routines.items()])


def table(results: list[dict[str, Any]]) -> str:
    """Markdown table: one row per measurement, one column per dataset."""
    rows: dict[str, list[str]] = {}
    for col, res in enumerate(results):
        for section, vals in res.items():
            if not isinstance(vals, dict):
                continue
            for k, val in vals.items():
                rows.setdefault(f"{section}.{k}", [""] * len(results))[col] = json.dumps(val) if isinstance(
                    val, dict) else str(val)
    head = "| measurement | " + " | ".join(r["dataset"] for r in results) + " |\n|---|" + "---|" * len(results)
    return head + "\n" + "\n".join(f"| {k} | " + " | ".join(v) + " |" for k, v in rows.items())
