"""EDA for the Workbook 1 trace: the plots and tables for the Team Meeting 1 ISA demo."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from p17sim.schema import HISTORY_LAST_DAY, VAL_LAST_DAY  # noqa: E402
from p17sim.trace import START  # noqa: E402

# Reference categorical palette, fixed order (dataviz skill, references/palette.md).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = {"message": "Messages", "calendar": "Calendar", "preference": "Preference signals", "file": "Files",
          "routine_task": "Tasks", "sensitive_task": "Tasks", "conflict": "Conflicts and deletions",
          "deletion_request": "Conflicts and deletions", "permission": "Permissions"}
ORDER = ["Messages", "Preference signals", "Calendar", "Files", "Tasks", "Conflicts and deletions"]


def _style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, out: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return name


def write_report(records: list[dict], labels: list[dict], log: dict, samples: list[dict], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records).merge(pd.DataFrame(labels)[["event_id", "split", "source", "checker"]],
                                     on="event_id", how="left")
    df["day"] = [(datetime.fromisoformat(t) - START).days + 1 for t in df.timestamp]
    df["split"] = df.split.fillna("train")  # permission records sit on day 1
    df["group"] = df.event_type.map(GROUPS)
    df["words"] = df.text.str.split().str.len()
    df["task_kind"] = df.checker.map(lambda c: c["kind"] if isinstance(c, dict) else None)
    figs = []

    per_day = df[df.group != "Permissions"].groupby(["day", "group"]).size().unstack(fill_value=0)[ORDER]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bottom = pd.Series(0, index=per_day.index)
    for color, g in zip(SERIES, ORDER):
        ax.bar(per_day.index, per_day[g], bottom=bottom, color=color, width=0.8, label=g,
               edgecolor="#fcfcfb", linewidth=0.6)
        bottom += per_day[g]
    for x in (HISTORY_LAST_DAY + 0.5, VAL_LAST_DAY + 0.5):
        ax.axvline(x, color=MUTED, linestyle="--", linewidth=1)
    top = bottom.max() * 1.08
    for x, name in ((10.5, "train (days 1-20)"), (22.5, "val"), (27.5, "test (25-30)")):
        ax.text(x, top, name, ha="center", fontsize=9, color=MUTED)
    ax.set_ylim(0, top * 1.08)
    ax.set_xlabel("Day of trace", color=MUTED, fontsize=9)
    ax.set_ylabel("Records (10 personas)", color=MUTED, fontsize=9)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18))
    _style(ax, "Records per day by type; tasks appear only in val and test")
    figs.append(_save(fig, out, "records_per_day.png"))

    share = df[df.group != "Permissions"].groupby(["split", "group"]).size().unstack(fill_value=0)[ORDER]
    share = share.loc[["train", "val", "test"]].div(share.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 2.6))
    left = pd.Series(0.0, index=share.index)
    for color, g in zip(SERIES, ORDER):
        ax.barh(share.index, share[g], left=left, color=color, label=g, edgecolor="#fcfcfb", linewidth=1)
        left += share[g]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of records in split (%)", color=MUTED, fontsize=9)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.3))
    _style(ax, "Event-type mix per split")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    figs.append(_save(fig, out, "type_share_by_split.png"))

    llm = df[df.source.isin(["llm", "imputed"])]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    bins = range(0, int(llm.words.max()) + 5, 5)
    for color, (t, name) in zip(SERIES, [("message", "Messages"), ("file", "Files"), ("conflict", "Conflict updates")]):
        ax.hist(llm[llm.event_type == t].words, bins=bins, histtype="step", linewidth=2, color=color, label=name)
    ax.set_xlabel("Words per LLM-written record", color=MUTED, fontsize=9)
    ax.set_ylabel("Records", color=MUTED, fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "Text length of LLM-written records")
    figs.append(_save(fig, out, "text_length.png"))

    tasks = df[df.task_kind.notna()].groupby(["task_kind", "split"]).size().unstack(fill_value=0)
    tasks = tasks.reindex(columns=["val", "test"], fill_value=0).sort_values("test")
    fig, ax = plt.subplots(figsize=(9, 2.8))
    y = range(len(tasks))
    ax.barh([i + 0.2 for i in y], tasks["val"], height=0.38, color=SERIES[0], label="val (days 21-24)")
    ax.barh([i - 0.2 for i in y], tasks["test"], height=0.38, color=SERIES[1], label="test (days 25-30)")
    ax.set_yticks(list(y), tasks.index)
    for i, (v, t) in enumerate(zip(tasks["val"], tasks["test"])):
        ax.text(v + 0.3, i + 0.2, str(v), va="center", fontsize=8, color=MUTED)
        ax.text(t + 0.3, i - 0.2, str(t), va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("Scored tasks", color=MUTED, fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, "Scored tasks by kind: every metric has test cases")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    figs.append(_save(fig, out, "tasks_by_split.png"))

    issues = {k: v for k, v in log.items() if k not in ("raw_records", "trace_records", "pii_remaining") and v}
    issues = dict(sorted(issues.items(), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.38 * max(len(issues), 1)))
    ax.barh(list(issues), list(issues.values()), color=SERIES[0], height=0.6)
    for i, v in enumerate(issues.values()):
        ax.text(v, i, f" {v}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel(f"Records affected (of {log['raw_records']} raw)", color=MUTED, fontsize=9)
    _style(ax, "Problems found and fixed during cleaning")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    figs.append(_save(fig, out, "cleaning_issues.png"))

    lines = ["# Trace v1: EDA and data-quality report", "",
             f"Personas: {df.persona_id.nunique()} · records: {len(df)} · window: {START.date()} + 30 days", "",
             "## Records by split", "", df.split.value_counts().reindex(["train", "val", "test"]).to_markdown(), "",
             "## Records by event type and split", "",
             pd.crosstab(df.event_type, df.split)[["train", "val", "test"]].to_markdown(), "",
             "## Scored tasks by kind and split", "", tasks.to_markdown(), "",
             "## Text source", "", df.source.value_counts().to_markdown(), "",
             "## Cleaning log", "", pd.Series(log, name="count").to_markdown(), "",
             "## Before and after cleaning (samples)", ""]
    for s in samples:
        lines += [f"**{s['event_id']}** ({', '.join(s['fixes'])})", "",
                  "Before: " + s["before"].replace("\n", " "), "", "After: " + s["after"].replace("\n", " "), ""]
    lines += ["## Known coverage gaps", "",
              "- English only; US-style weekly schedules; no shared or family accounts.",
              "- Archetypes chosen by the team; routines are more regular than real life.",
              "- LLM text is generic and short; real mail is longer and messier.",
              "- Would disadvantage: shift workers with irregular schedules, non-English users.", ""]
    lines += [f"![{f}]({f})" for f in figs]
    report = out / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report

