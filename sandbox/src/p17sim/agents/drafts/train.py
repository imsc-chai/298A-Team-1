"""Train Model A's three learned parts. Train personas only; every choice is made on val personas.

1. Reranker: does this record state the fact the question asks about (any version)?
   Training queries come from the train personas' facts at the times they are asked, plus extra
   times, worded with *seen* templates only. Grid over the regularization C; learning curve over the
   share of training queries; thresholds chosen on val (tau_relevant by F1, tau_delete for recall).
2. Intent model: request -> lookup | send | schedule | forget. TF-IDF word and character n-grams with
   logistic regression, trained on train-persona requests plus seen-template fills.
3. Preference estimator: grid search over half-life and signal weights for predicting the true
   meeting window on probe days.
Everything is written to models/model_a/, with a report of grids, curves, data sizes and compute time.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from p17sim.agents.drafts.memory import Memory, Reranker
from p17sim.agents.drafts.model_a import MODEL_DIR, preference_window
from p17sim.gen2 import phrasing as ph
from p17sim.gen2.personas import pref_value
from p17sim.system.llm import Profile, embed
from p17sim.system.runner import Dataset

INTENT_OF = {"recall": "lookup", "update": "lookup", "deletion": "lookup", "privacy": "send", "scheduling": "schedule"}


@dataclass
class Query:
    pid: str
    time: datetime
    text: str
    fact_id: str


def fact_queries(ds: Dataset, pids: list[str], seed: int, extra_per_fact: int = 2) -> list[Query]:
    rng = np.random.default_rng(seed)
    out = []
    for pid in pids:
        labs = [ds.labels[r["event_id"]] for r in ds.records[pid]]
        rec_time = {r["event_id"]: datetime.fromisoformat(r["timestamp"]) for r in ds.records[pid]}
        first: dict[str, datetime] = {}
        deleted_at: dict[str, datetime] = {}
        kind: dict[str, str] = {}
        asked: dict[str, list[datetime]] = {}
        for lab in labs:
            prov = lab.get("provenance", "")
            if prov.startswith("fact:"):
                first.setdefault(lab["fact_id"], rec_time[lab["event_id"]])
                kind[lab["fact_id"]] = prov.split(":", 1)[1]
            if prov.startswith("delete:"):
                deleted_at[lab["fact_id"]] = rec_time[lab["event_id"]]
            if lab.get("checker") and lab["checker"]["kind"] in ("recall", "update"):
                asked.setdefault(lab["fact_id"], []).append(rec_time[lab["event_id"]])
        end = max(rec_time.values())
        for fid, t0 in first.items():
            times = list(asked.get(fid, []))
            for _ in range(extra_per_fact):
                times.append(t0 + (end - t0) * float(rng.uniform(0.05, 1.0)))
            names = ph.FACT_NAMES[kind[fid]][0]
            for t in times:
                if fid in deleted_at and t > deleted_at[fid]:
                    continue
                tmpl = ph.ASK["seen"][int(rng.integers(len(ph.ASK["seen"])))]
                out.append(Query(pid, t, tmpl.format(k=str(rng.choice(names))), fid))
    return out


def candidate_sets(ds: Dataset, queries: list[Query], use_embeddings: bool, profile: Profile) -> list[tuple]:
    """For each query: the candidates retrieval returns from the records before the query time,
    and a 0/1 label per candidate (does it state this fact, in any version)."""
    out = []
    by_pid: dict[str, list[Query]] = {}
    for q in queries:
        by_pid.setdefault(q.pid, []).append(q)
    for pid, qs in by_pid.items():
        mem = Memory((lambda t: embed(t, profile=profile)) if use_embeddings else None, profile)
        recs = [r for r in ds.records[pid] if not ds.labels[r["event_id"]].get("checker")]
        i = 0
        for q in sorted(qs, key=lambda q: q.time):
            while i < len(recs) and datetime.fromisoformat(recs[i]["timestamp"]) < q.time:
                mem.add(recs[i])
                i += 1
            cands = mem.candidates(q.text, {"message", "note", "calendar"})
            if not cands:
                continue
            y = np.array([int(ds.labels[c.item.record["event_id"]].get("fact_id") == q.fact_id
                              and not ds.labels[c.item.record["event_id"]].get("provenance", "").startswith(
                                  "distractor")) for c in cands])
            out.append((q, cands, y))
    return out


def _matrices(rr: Reranker, sets: list[tuple]):
    from scipy.sparse import vstack

    X = vstack([rr.matrix(q.text, cands, q.time) for q, cands, _ in sets]).tocsr()
    y = np.concatenate([y for _, _, y in sets])
    groups = np.concatenate([[k] * len(y) for k, (_, _, y) in enumerate(sets)])
    return X, y, groups


def ranking_metrics(scores: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    mrr, p1, cand_recall = [], [], []
    for g in np.unique(groups):
        m = groups == g
        s, t = scores[m], y[m]
        cand_recall.append(float(t.any()))
        if not t.any():
            continue
        order = np.argsort(-s)
        rank = int(np.where(t[order] == 1)[0][0]) + 1
        mrr.append(1 / rank)
        p1.append(float(rank == 1))
    return {"mrr": round(float(np.mean(mrr)), 4), "precision_at_1": round(float(np.mean(p1)), 4),
            "candidate_recall": round(float(np.mean(cand_recall)), 4), "queries": int(len(np.unique(groups)))}


def train_reranker(train_sets, val_sets, report: dict) -> tuple[Reranker, float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from p17sim.agents.drafts.memory import features

    scaler = StandardScaler().fit(np.vstack([features(q.text, c, q.time) for q, c, _ in train_sets]))
    rr = Reranker(None, scaler)
    Xtr, ytr, gtr = _matrices(rr, train_sets)
    Xva, yva, gva = _matrices(rr, val_sets)
    # RRF without a reranker, as the reference line
    rrf = np.concatenate([[1 / (60 + c.bm25_rank) + 1 / (60 + c.dense_rank) for c in cands] for _, cands, _ in val_sets])
    report["reranker"] = {"train_rows": int(Xtr.shape[0]), "train_positives": int(ytr.sum()),
                          "val_rows": int(Xva.shape[0]), "features": int(Xtr.shape[1]),
                          "val_without_reranker_rrf": ranking_metrics(rrf, yva, gva), "grid": []}
    best = None
    for C in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        t = time.perf_counter()
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        fit_s = time.perf_counter() - t
        pv = m.predict_proba(Xva)[:, 1]
        row = {"C": C, "fit_s": round(fit_s, 2), "val_auc": round(float(roc_auc_score(yva, pv)), 4),
               **{f"val_{k}": v for k, v in ranking_metrics(pv, yva, gva).items()},
               **{f"train_{k}": v for k, v in ranking_metrics(m.predict_proba(Xtr)[:, 1], ytr, gtr).items()}}
        report["reranker"]["grid"].append(row)
        if best is None or (row["val_mrr"], row["val_auc"]) > (best[1]["val_mrr"], best[1]["val_auc"]):
            best = (m, row)
    model, row = best
    report["reranker"]["chosen_C"] = row["C"]
    # learning curve: share of training queries -> train and val MRR
    curve, rng = [], np.random.default_rng(0)
    order = rng.permutation(len(train_sets))
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        sub = [train_sets[i] for i in order[: max(2, int(frac * len(train_sets)))]]
        Xs, ys, gs = _matrices(rr, sub)
        if ys.sum() == 0:
            continue
        m = LogisticRegression(C=row["C"], max_iter=3000, class_weight="balanced").fit(Xs, ys)
        curve.append({"train_queries": len(sub),
                      "train_mrr": ranking_metrics(m.predict_proba(Xs)[:, 1], ys, gs)["mrr"],
                      "val_mrr": ranking_metrics(m.predict_proba(Xva)[:, 1], yva, gva)["mrr"]})
    report["reranker"]["learning_curve"] = curve
    # thresholds on val
    pv = model.predict_proba(Xva)[:, 1]
    taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
    f1 = []
    for tau in taus:
        pred = pv >= tau
        tp, fp, fn = int((pred & (yva == 1)).sum()), int((pred & (yva == 0)).sum()), int((~pred & (yva == 1)).sum())
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        f1.append({"tau": float(tau), "precision": round(prec, 3), "recall": round(rec, 3),
                   "f1": round(2 * prec * rec / max(1e-9, prec + rec), 3)})
    tau_rel = max(f1, key=lambda r: r["f1"])["tau"]
    tau_del = max([r["tau"] for r in f1 if r["recall"] >= 0.95] or [0.05])
    report["reranker"]["thresholds"] = {"table": f1, "tau_relevant": tau_rel, "tau_delete": tau_del}
    rr.model = model
    return rr, tau_rel, tau_del


def intent_data(ds: Dataset, pids: list[str]) -> tuple[list[str], list[str]]:
    X, y = [], []
    for pid in pids:
        for r in ds.records[pid]:
            lab = ds.labels[r["event_id"]]
            if lab.get("checker"):
                X.append(r["text"])
                y.append(INTENT_OF[lab["checker"]["kind"]])
            elif lab.get("provenance", "").startswith("delete:"):
                X.append(r["text"])
                y.append("forget")
    return X, y


def augment(ds: Dataset, pids: list[str], n: int, seed: int) -> tuple[list[str], list[str]]:
    """Fill the *seen* templates with train-persona names and fact names. Held-out wordings are never used."""
    rng = np.random.default_rng(seed)
    names = [c.name for pid in pids for c in ds.personas[pid].contacts]
    facts = [n_ for names_, _ in ph.FACT_NAMES.values() for n_ in names_]
    X, y = [], []
    for _ in range(n):
        k = int(rng.integers(4))
        if k == 0:
            X.append(str(rng.choice(ph.ASK["seen"])).format(k=rng.choice(facts))); y.append("lookup")
        elif k == 1:
            X.append(str(rng.choice(ph.FORGET)).format(k=rng.choice(facts))); y.append("forget")
        elif k == 2:
            X.append(str(rng.choice(ph.PRIVACY["seen"])).format(name=rng.choice(names))); y.append("send")
        else:
            when = str(rng.choice(["tomorrow", "this Friday", "on October 3rd", "on the 2nd", "today"]))
            X.append(str(rng.choice(ph.SCHEDULE["seen"])).format(d=int(rng.choice([15, 30, 45, 60])),
                                                                name=rng.choice(names), when=when)); y.append("schedule")
        X[-1] = ph.noisy(X[-1], [], rng)
    return X, y


def train_intent(ds: Dataset, train: list[str], val: list[str], report: dict):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    Xtr, ytr = intent_data(ds, train)
    Xa, ya = augment(ds, train, 400, seed=1)
    Xva, yva = intent_data(ds, val)
    grid, best = [], None
    for C in (0.3, 1.0, 3.0, 10.0):
        pipe = Pipeline([("f", FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                                             ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                                                   sublinear_tf=True))])),
                         ("lr", LogisticRegression(C=C, max_iter=3000))])
        t = time.perf_counter()
        pipe.fit(Xtr + Xa, ytr + ya)
        acc = float(np.mean(pipe.predict(Xva) == np.array(yva)))
        grid.append({"C": C, "fit_s": round(time.perf_counter() - t, 2), "val_accuracy": round(acc, 4)})
        if best is None or acc > best[1]:
            best = (pipe, acc)
    report["intent"] = {"train_requests": len(Xtr), "augmented": len(Xa), "val_requests": len(Xva),
                        "classes": sorted(set(ytr)), "grid": grid, "chosen_C": max(grid, key=lambda g: g["val_accuracy"])["C"]}
    return best[0]


def train_preference(ds: Dataset, train: list[str], val: list[str], report: dict) -> dict:
    def probes(pids):
        out = []
        for pid in pids:
            x = ds.extras[pid]
            recs = [r for r in ds.records[pid] if r["channel"] == "preference"]
            start = datetime.fromisoformat(ds.records[pid][0]["timestamp"]).replace(hour=0, minute=0)
            for day in range(21, 31):
                now = start + timedelta(days=day - 1, hours=9)
                out.append(([r for r in recs if datetime.fromisoformat(r["timestamp"]) < now], now,
                            pref_value(x, "meeting_window", day)))
        return out

    tr, va = probes(train), probes(val)
    grid = []
    for h in (0.5, 1, 2, 3, 5, 8, 13, 1000):
        for wf in (1, 2, 3, 5):
            for wd in (0, 0.25, 0.5, 1):
                acc = lambda P: float(np.mean([preference_window(r, now, h, wf, wd) == truth for r, now, truth in P]))  # noqa: E731
                grid.append({"half_life_days": h, "feedback_weight": wf, "decline_weight": wd,
                             "train_accuracy": round(acc(tr), 4), "val_accuracy": round(acc(va), 4)})
    best = max(grid, key=lambda g: (g["val_accuracy"], g["train_accuracy"]))
    report["preference"] = {"train_probes": len(tr), "val_probes": len(va), "chosen": best,
                            "no_decay_best_val": max(g["val_accuracy"] for g in grid if g["half_life_days"] == 1000),
                            "grid_size": len(grid), "top5": sorted(grid, key=lambda g: -g["val_accuracy"])[:5]}
    return {"pref_half_life_days": best["half_life_days"], "pref_feedback_weight": best["feedback_weight"],
            "pref_decline_weight": best["decline_weight"]}


def train_all(ds: Dataset, out: Path = MODEL_DIR, use_embeddings: bool = True, seed: int = 2026) -> dict:
    import joblib

    t0 = time.perf_counter()
    train = sorted(p for p, s in ds.splits.items() if s == "train")
    val = sorted(p for p, s in ds.splits.items() if s == "val")
    report: dict = {"train_personas": train, "val_personas": val, "embeddings": use_embeddings}
    profile = Profile()
    t = time.perf_counter()
    tr_sets = candidate_sets(ds, fact_queries(ds, train, seed), use_embeddings, profile)
    va_sets = candidate_sets(ds, fact_queries(ds, val, seed + 1), use_embeddings, profile)
    report["candidate_build_s"] = round(time.perf_counter() - t, 1)
    rr, tau_rel, tau_del = train_reranker(tr_sets, va_sets, report)
    intent = train_intent(ds, train, val, report)
    pref = train_preference(ds, train, val, report)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rr.model, "scaler": rr.scaler, "n_hash": rr.hasher.n_features}, out / "reranker.joblib")
    joblib.dump(intent, out / "intent.joblib")
    params = {"tau_relevant": tau_rel, "tau_delete": tau_del, **pref}
    (out / "params.json").write_text(json.dumps(params, indent=1) + "\n")
    report["params"] = params
    report["compute"] = {"total_s": round(time.perf_counter() - t0, 1), "embed_calls": len(profile.calls),
                         "embed_s": round(sum(c.wall_s for c in profile.calls), 1),
                         "artifact_kb": {f.name: round(f.stat().st_size / 1024, 1) for f in out.glob("*.joblib")}}
    (out / "training_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    return report
