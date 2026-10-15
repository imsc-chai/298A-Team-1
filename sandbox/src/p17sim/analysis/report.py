"""Build research/results/RESULTS.md and its figures from the run files.

Run files are named {system}_s{seed}_{persona splits}_{windows}.jsonl (see `p17sim eval`).
Every number in the report is computed here from those files; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from p17sim.agents.registry import ABLATIONS
from p17sim.analysis import stats
from p17sim.analysis.metrics import SCORED, load_run, summarize, tasks

# Reference palette slots 1-3 (validated: dataviz validate_palette.js, light surface).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAME = re.compile(r"(?P<system>.+)_s(?P<seed>\d+)_(?P<personas>[^_]+)_(?P<windows>[^_]+)\.jsonl$")


def discover(runs_dir: Path) -> dict[tuple[str, str, str], dict[int, list[dict]]]:
    """(system, personas, windows) -> seed -> rows."""
    out: dict = defaultdict(dict)
    for f in sorted(runs_dir.glob("*.jsonl")):
        m = NAME.match(f.name)
        if m:
            rows, _ = load_run(f)
            out[(m["system"], m["personas"], m["windows"])][int(m["seed"])] = rows
    return out


def _pm(values: list[float | None], pct: bool = True) -> str:
    v = [x for x in values if x is not None]
    if not v:
        return "–"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.3g}")
    return f(np.mean(v)) + (f" ± {100 * np.std(v, ddof=1):.1f}" if pct and len(v) > 1 else
                            f" ± {np.std(v, ddof=1):.2g}" if len(v) > 1 else "")


def _style(ax, title: str, ylabel: str = ""):
    import matplotlib.pyplot as plt  # noqa: F401

    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_success_by_kind(main: dict[str, dict[int, list[dict]]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = [s for s in ("baseline", "model_a") if s in main]
    kinds = list(SCORED) + ["all"]
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    width = 0.38
    for i, (s, color) in enumerate(zip(systems, (BLUE, ORANGE))):
        vals = []
        for k in kinds:
            per_seed = [summarize(rows)["task_success" if k == "all" else "success_by_kind"] for rows in main[s].values()]
            per_seed = [v if k == "all" else v.get(k) for v in per_seed]
            vals.append(np.mean([v for v in per_seed if v is not None]) if any(v is not None for v in per_seed) else 0)
        x = np.arange(len(kinds)) + (i - 0.5) * width
        bars = ax.bar(x, vals, width - 0.04, color=color, label={"baseline": "Baseline agent", "model_a": "Model A"}[s])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{100 * v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(kinds)), [k.capitalize() for k in kinds])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, "Task success by task type (test personas, days 25–30, mean of seeds)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_latency(main: dict[str, dict[int, list[dict]]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    for s, color, label in (("baseline", BLUE, "Baseline agent"), ("model_a", ORANGE, "Model A")):
        if s not in main:
            continue
        lat = np.sort([max(0.01, r["latency_s"]) for rows in main[s].values() for r in tasks(rows)])
        ax.plot(lat, np.arange(1, len(lat) + 1) / len(lat), color=color, linewidth=2, label=label)
        p95 = np.percentile(lat, 95)
        ax.axvline(p95, color=color, linewidth=1, linestyle=":")
        ax.text(p95, 0.05, f" p95 {p95:.0f}s", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Seconds per task (log scale)", color=INK2)
    _style(ax, "Latency per task: share of tasks finished within x seconds")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_learning_curve(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = report.get("reranker", {}).get("learning_curve", [])
    if not curve:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4), facecolor=SURFACE)
    x = [c["train_queries"] for c in curve]
    for key, color, label in (("train_mrr", BLUE, "Train personas"), ("val_mrr", ORANGE, "Val personas")):
        y = [c[key] for c in curve]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.text(x[-1], y[-1], f"  {y[-1]:.2f}", color=INK2, fontsize=8, va="center")
    rrf = report["reranker"]["val_without_reranker_rrf"]["mrr"]
    ax.axhline(rrf, color=INK2, linewidth=1, linestyle="--")
    ax.text(x[0], rrf, f"no reranker (val) {rrf:.2f}", color=INK2, fontsize=8, va="bottom")
    ax.set_xlabel("Training queries", color=INK2)
    _style(ax, "Reranker learning curve (MRR)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_gaps(gaps: list[tuple[str, dict]], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not gaps:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(gaps) + 1.2), facecolor=SURFACE)
    for i, (label, g) in enumerate(gaps):
        ax.plot([g["ci_low"], g["ci_high"]], [i, i], color=ORANGE, linewidth=2)
        ax.plot([g["gap"]], [i], "o", color=ORANGE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(g["ci_high"], i, f"  {100 * g['gap']:+.1f} pp", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.axvline(-0.10, color=INK2, linewidth=1, linestyle=":")
    ax.text(-0.10, len(gaps) - 0.4, "−10 pp limit (H5)", fontsize=8, color=INK2, ha="center")
    ax.set_yticks(range(len(gaps)), [g[0] for g in gaps])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(ax, title)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build(results: Path) -> str:
    runs = discover(results / "runs")
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    md: list[str] = ["# Results", "",
                     "Generated by `p17sim report` from the run files in `research/results/runs/`. "
                     "Hypotheses and tests are the ones in `research/PREREGISTRATION.md`.", ""]
    main = {s: seeds for (s, p, w), seeds in runs.items() if p == "test" and w == "test"}
    ood = {s: seeds for (s, p, w), seeds in runs.items() if p == "ood_test" and w == "test"}
    from p17sim.aios.generators import PENDING

    pending = [name for name in PENDING if name not in main]
    if pending:
        md += ["**Pending training:** " + ", ".join(f"`{n}` ({PENDING[n].description})" for n in pending)
               + ". Their columns, H1–H5 comparisons and ablations appear once they are trained and evaluated "
                 "with `p17sim eval`.", ""]

    # ---- main table
    md += ["## Main comparison (test personas, days 25–30)", "",
           "| Metric | " + " | ".join(main) + " |", "|---|" + "---|" * len(main)]
    sums = {s: [summarize(rows) for rows in seeds.values()] for s, seeds in main.items()}
    rows_spec = [("Seeds", lambda x: None, False), ("Task success", lambda x: x["task_success"], True)]
    rows_spec += [(f"  {k}", (lambda k: lambda x: x["success_by_kind"][k])(k), True) for k in SCORED]
    rows_spec += [("Intervention rate (per task)", lambda x: x["intervention_rate"], False),
                  ("Preference regret (per scheduling task)", lambda x: x["preference_regret"], False),
                  ("Deletion probes passed", lambda x: x["deletion_probe_pass"], True),
                  ("Deletion storage audit passed", lambda x: x["deletion_storage_audit_pass"], True),
                  ("Privacy leakage", lambda x: x["privacy_leakage"], True),
                  ("Raw-data export (bytes)", lambda x: x["raw_export_bytes"], False),
                  ("p95 latency (s)", lambda x: x["latency_p95_s"], False),
                  ("Cost per scored user-day (USD)", lambda x: x["cost_per_user_day_usd"], False),
                  ("Errors", lambda x: x["errors"], False)]
    for label, fn, pct in rows_spec:
        if label == "Seeds":
            md.append("| Seeds | " + " | ".join(", ".join(map(str, sorted(main[s]))) for s in main) + " |")
            continue
        md.append(f"| {label} | " + " | ".join(_pm([fn(x) for x in sums[s]], pct) for s in main) + " |")
    md += ["", f"Tasks per seed: " + ", ".join(f"{s} {sums[s][0]['tasks']}" for s in main), ""]
    if main:
        fig_success_by_kind(main, figs / "success_by_kind.png")
        fig_latency(main, figs / "latency.png")
        md += ["![Task success by kind](figures/success_by_kind.png)", "", "![Latency](figures/latency.png)", ""]

    # ---- hypotheses
    md += ["## Hypotheses", ""]
    findings: dict[str, Any] = {}
    if "model_a" in main and "baseline" in main:
        A = stats.per_task_success([tasks(r) for r in main["model_a"].values()])
        B = stats.per_task_success([tasks(r) for r in main["baseline"].values()])
        d = stats.cluster_bootstrap_diff(A, B)
        verdict = ("superior" if d["ci_low"] > 0 else "non-inferior" if d["ci_low"] > -0.05 else "not shown")
        findings["H1"] = {**d, "verdict": verdict}
        md += [f"**H1 (task success, Model A − baseline):** {100 * d['diff']:+.1f} pp, 95% CI "
               f"[{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}] over {d['tasks']} tasks and {d['personas']} "
               f"personas (cluster bootstrap). Verdict: **{verdict}** (rule: non-inferior if the lower bound > −5 pp).", ""]
        for kind in SCORED:
            Ak = {t: v for t, v in A.items() if t in {r["task_id"] for rows in main["model_a"].values()
                                                       for r in tasks(rows) if r["kind"] == kind}}
            Bk = {t: v for t, v in B.items() if t in Ak}
            if Ak:
                dk = stats.cluster_bootstrap_diff(Ak, Bk)
                md.append(f"- {kind}: {100 * dk['diff']:+.1f} pp [{100 * dk['ci_low']:+.1f}, {100 * dk['ci_high']:+.1f}] "
                          f"(n={dk['tasks']})")
        md.append("")
        sa = sums["model_a"]
        dc = [x["deletion_compliance"] for x in sa]
        findings["H2"] = {"deletion_compliance": dc}
        md += [f"**H2 (Model A deletion compliance = 100%):** probes {_pm([x['deletion_probe_pass'] for x in sa])}, "
               f"storage audit {_pm([x['deletion_storage_audit_pass'] for x in sa])}. "
               f"{'Held' if all(v == 1.0 for v in dc if v is not None) else 'Rejected'}.", ""]
        pa = stats.per_task_value([tasks(r) for r in main["model_a"].values()], "leak", {"privacy"})
        pb = stats.per_task_value([tasks(r) for r in main["baseline"].values()], "leak", {"privacy"})
        if pa:
            dl = stats.cluster_bootstrap_diff(pa, pb)
            findings["H3"] = dl
            md += [f"**H3 (privacy leakage, Model A − baseline, mean over seeds):** {100 * dl['diff']:+.1f} pp "
                   f"[{100 * dl['ci_low']:+.1f}, {100 * dl['ci_high']:+.1f}]. Raw-data export: "
                   + ", ".join(f"{s} {sum(x['raw_export_bytes'] for x in sums[s])} bytes" for s in main) + ".", ""]

    # ---- ablations
    abl_rows, pvals = [], {}
    if "model_a" in main and 11 in main["model_a"]:
        a_rows = {r["task_id"]: r for r in tasks(main["model_a"][11])}
        a_audit = [r for r in main["model_a"][11] if r["kind"] == "deletion_audit"]
        for name, (_, kinds) in ABLATIONS.items():
            if name not in main or 11 not in main[name]:
                continue
            b_rows = {r["task_id"]: r for r in tasks(main[name][11]) if r["kind"] in kinds}
            pa_ = {t: a_rows[t]["passed"] for t in b_rows if t in a_rows}
            pb_ = {t: r["passed"] for t, r in b_rows.items() if t in pa_}
            mc = stats.mcnemar(pa_, pb_)
            pvals[name] = mc["p"]
            b_audit = [r for r in main[name][11] if r["kind"] == "deletion_audit"]
            abl_rows.append((name, kinds, np.mean(list(pa_.values())) if pa_ else None,
                             np.mean(list(pb_.values())) if pb_ else None, mc,
                             np.mean([r["passed"] for r in a_audit]) if a_audit else None,
                             np.mean([r["passed"] for r in b_audit]) if b_audit else None))
    if abl_rows:
        adj = stats.holm(pvals)
        findings["H4"] = {n: {"p": pvals[n], "p_holm": adj[n]} for n in pvals}
        md += ["## Ablations (seed 11, target task kinds only)", "",
               "| Ablation | Target kinds | Model A | Ablated | A only / ablated only | p | p (Holm) | Storage audit A → ablated |",
               "|---|---|---|---|---|---|---|---|"]
        for name, kinds, a, b, mc, aa, ba in abl_rows:
            md.append(f"| {name} | {', '.join(sorted(kinds))} | {100 * a:.1f}% | {100 * b:.1f}% | {mc['a_only']} / "
                      f"{mc['b_only']} | {mc['p']} | {adj[name]} | "
                      f"{'–' if aa is None else f'{100 * aa:.0f}% → {100 * ba:.0f}%'} |")
        md.append("")

    # ---- generalization and fairness (Model A, and baseline for reference)
    gaps = []
    for s in ("model_a", "baseline"):
        if s not in main:
            continue
        T = stats.per_task_success([tasks(r) for r in main[s].values()])
        meta = {r["task_id"]: r for rows in main[s].values() for r in tasks(rows)}
        if s in ood:
            O = stats.per_task_success([tasks(r) for r in ood[s].values()])
            gaps.append((f"{s}: OOD archetypes vs in-distribution", stats.group_gap(O, T)))
        seen = {t: v for t, v in T.items() if meta[t].get("wording") == "seen"}
        held = {t: v for t, v in T.items() if meta[t].get("wording") == "heldout"}
        if seen and held:
            gaps.append((f"{s}: held-out vs seen wording", stats.group_gap(held, seen)))
        lla = {t: v for t, v in T.items() if "llama" in meta[t]["renderer"]}
        phi = {t: v for t, v in T.items() if "phi3" in meta[t]["renderer"]}
        if lla and phi:
            gaps.append((f"{s}: phi3-written vs llama-written", stats.group_gap(phi, lla)))
    if gaps:
        findings["H5"] = {k: v for k, v in gaps}
        fig_gaps([g for g in gaps if g[0].startswith("model_a")], figs / "generalization.png",
                 "Model A on held-out conditions (gap vs in-distribution, 95% CI)")
        md += ["## Generalization (H5)", "", "| Condition | Gap | 95% CI | n held-out / n reference |", "|---|---|---|---|"]
        md += [f"| {k} | {100 * g['gap']:+.1f} pp | [{100 * g['ci_low']:+.1f}, {100 * g['ci_high']:+.1f}] | "
               f"{g['n_a']} / {g['n_b']} |" for k, g in gaps]
        md += ["", "![Generalization](figures/generalization.png)", ""]

    for s in ("model_a", "baseline"):
        if s not in main:
            continue
        T = stats.per_task_success([tasks(r) for r in main[s].values()])
        meta = {r["task_id"]: r for rows in main[s].values() for r in tasks(rows)}
        md += [f"## Fairness slices: {s}", "", "| Group | Value | Success | 95% CI | Tasks |", "|---|---|---|---|---|"]
        for attr in ("region", "name_list", "archetype"):
            for val in sorted({meta[t][attr] for t in T}):
                sub = {t: v for t, v in T.items() if meta[t][attr] == val}
                b = stats.cluster_bootstrap_mean(sub)
                md.append(f"| {attr} | {val} | {100 * b['mean']:.1f}% | [{100 * b['ci_low']:.1f}, "
                          f"{100 * b['ci_high']:.1f}] | {b['tasks']} |")
        md.append("")

    # ---- profiling
    if main:
        md += ["## Profiling per system (mean per task)", "",
               "| | " + " | ".join(main) + " |", "|---|" + "---|" * len(main)]
        keys = ["llm_calls_per_task", "turns_per_task", "prompt_tokens_per_task", "output_tokens_per_task",
                "prefill_tok_per_s", "decode_tok_per_s", "ttft_median_s", "ttft_p95_s"]
        for k in keys:
            md.append(f"| {k} | " + " | ".join(_pm([x["profiling"].get(k) for x in sums[s]], False) for s in main) + " |")
        comps = sorted({c for s in main for x in sums[s] for c in x["profiling"].get("seconds_by_component_per_task", {})})
        for c in comps:
            md.append(f"| seconds in {c} | " + " | ".join(
                _pm([x["profiling"]["seconds_by_component_per_task"].get(c) for x in sums[s]], False) for s in main) + " |")
        md.append("")
    mp = results / "model_profile.json"
    if mp.exists():
        prof = json.loads(mp.read_text())
        md += ["## Model profiling (each model alone on this machine)", "",
               "| Model | Params | Quant | Disk GB | Resident GB | Cold load s | TTFT @1k s | Prefill tok/s @1k | "
               "Decode tok/s | Wh per 1k output tokens |", "|---|---|---|---|---|---|---|---|---|---|"]
        for p in prof:
            if p["kind"] == "chat":
                b = p["by_prompt_length"].get("1024", {})
                md.append(f"| {p['model']} | {p['parameters']} | {p['quantization']} | {p['disk_bytes'] / 1e9:.2f} | "
                          f"{(p['resident_bytes'] or 0) / 1e9:.2f} | {p['cold_load_s']} | {b.get('ttft_s')} | "
                          f"{b.get('prefill_tok_s')} | {p['decode_tok_s_median']} | {p['energy_wh_per_1k_output_tokens']} |")
            else:
                md.append(f"| {p['model']} (embeddings) | | | {p['disk_bytes'] / 1e9:.3f} | "
                          f"{(p['resident_bytes'] or 0) / 1e9:.2f} | {p['cold_first_call_s']} | | "
                          f"{p['texts_per_s_median']} texts/s | | |")
        md.append("")

    # ---- training
    tr = Path(__file__).resolve().parents[3] / "models" / "model_a" / "training_report.json"
    if tr.exists():
        rep = json.loads(tr.read_text())
        fig_learning_curve(rep, figs / "learning_curve.png")
        rr = rep["reranker"]
        md += ["## Model A training", "",
               f"Reranker: {rr['train_rows']} candidate rows ({rr['train_positives']} positive) from train personas; "
               f"{rr['features']} features; chosen C = {rr['chosen_C']}; thresholds τ_relevant = "
               f"{rr['thresholds']['tau_relevant']}, τ_delete = {rr['thresholds']['tau_delete']}.", "",
               "| C | val MRR | val P@1 | val AUC | train MRR | fit s |", "|---|---|---|---|---|---|"]
        md += [f"| {g['C']} | {g['val_mrr']} | {g['val_precision_at_1']} | {g['val_auc']} | {g['train_mrr']} | "
               f"{g['fit_s']} |" for g in rr["grid"]]
        md += ["", f"Without the reranker (reciprocal-rank fusion), val MRR = {rr['val_without_reranker_rrf']['mrr']}, "
               f"P@1 = {rr['val_without_reranker_rrf']['precision_at_1']}.", "",
               "![Learning curve](figures/learning_curve.png)", "",
               f"Intent model: {rep['intent']['train_requests']} train requests + {rep['intent']['augmented']} "
               f"seen-template fills; val accuracy by C: "
               + ", ".join(f"{g['C']}: {g['val_accuracy']}" for g in rep["intent"]["grid"]) + ".", "",
               f"Preference estimator: chosen half-life {rep['preference']['chosen']['half_life_days']} days, feedback "
               f"weight {rep['preference']['chosen']['feedback_weight']}, decline weight "
               f"{rep['preference']['chosen']['decline_weight']}; val accuracy {rep['preference']['chosen']['val_accuracy']} "
               f"(best without decay: {rep['preference']['no_decay_best_val']}).", "",
               f"Compute: {rep['compute']['total_s']} s in total on this laptop, {rep['compute']['embed_s']} s of it "
               f"embedding; artifacts {rep['compute']['artifact_kb']} KB.", ""]

    au = results / "audit.json"
    if au.exists():
        from p17sim.gen2.audit import table

        md += ["## Data audit (v1 vs v2)", "", table(json.loads(au.read_text())), ""]
    (results / "findings.json").write_text(json.dumps(findings, indent=1, default=str) + "\n")
    text = "\n".join(md) + "\n"
    (results / "RESULTS.md").write_text(text)
    return text
