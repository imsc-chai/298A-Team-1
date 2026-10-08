"""EDA report for Checkpoint 2: distributions, imbalance, drift, coverage, render failures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from p17sim.schema import Event, Persona  # noqa: E402


def _save(fig, out: Path, name: str) -> str:
    fig.tight_layout()
    fig.savefig(out / name, dpi=120)
    plt.close(fig)
    return name


def write_report(personas: list[Persona], events: list[Event], splits: dict[str, str], out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    present = {e.persona_id for e in events}
    personas = [p for p in personas if p.persona_id in present]
    splits = {pid: s for pid, s in splits.items() if pid in present}
    arch = {p.persona_id: p.archetype for p in personas}
    df = pd.DataFrame([{"persona_id": e.persona_id, "day": e.day, "minute": e.minute, "type": e.type.value,
                        "provenance": e.provenance, "sensitivity": e.sensitivity.value,
                        "render_status": e.render_status, "checker": e.checker.kind if e.checker else None,
                        "words": len((e.text or "").split())} for e in events])
    df["archetype"] = df.persona_id.map(arch)
    df["split"] = df.persona_id.map(splits)
    figs = []

    fig, ax = plt.subplots(figsize=(7, 3.5))
    df.type.value_counts().plot.barh(ax=ax, title="Events by type")
    figs.append(_save(fig, out, "events_by_type.png"))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    df.groupby(["day", "archetype"]).size().unstack().div(
        df.groupby("archetype").persona_id.nunique()).plot(ax=ax, title="Events per persona-day by archetype")
    figs.append(_save(fig, out, "events_per_day.png"))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    df[df.sensitivity != "none"].sensitivity.value_counts().plot.bar(ax=ax, title="Sensitive events by category")
    figs.append(_save(fig, out, "sensitivity.png"))

    cal = df[df.type == "calendar_event"].assign(routine=lambda d: d.provenance.str.removeprefix("routine:"))
    fig, ax = plt.subplots(figsize=(7, 4))
    piv = cal.groupby(["archetype", "routine"]).minute.std().unstack()
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(piv.index)), piv.index)
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=60, ha="right", fontsize=7)
    ax.set_title("Routine start-time std (minutes); blank = routine not in archetype")
    fig.colorbar(im, ax=ax)
    figs.append(_save(fig, out, "routine_regularity.png"))

    drift = pd.DataFrame([{"day": d, "key": pr.key, "changed": pr.value_on(d) != pr.initial}
                          for p in personas for pr in p.preferences for d in range(1, 31)])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    drift.groupby(["day", "key"]).changed.mean().unstack().plot(ax=ax, title="Share of personas past preference drift")
    figs.append(_save(fig, out, "preference_drift.png"))

    tasks = df[df.checker.notna()]
    lines = ["# Sandbox EDA report", "",
             f"Personas: {len(personas)}; events: {len(df)}; tasks: {len(tasks)}", "",
             "## Split sizes (personas)", "", pd.Series(splits).value_counts().to_markdown(), "",
             "## Archetype by split (personas)", "",
             pd.crosstab(pd.Series(arch, name="archetype"), pd.Series(splits, name="split")).to_markdown(), "",
             "## Tasks by kind and split", "", pd.crosstab(tasks.checker, tasks.split).to_markdown(), "",
             "## Render status (LLM-rendered types only)", "",
             df[df.render_status != "not_needed"].render_status.value_counts().to_markdown(), "",
             "## Known coverage gaps", "",
             "- English only; US-style weekly schedules; no shared or family accounts.",
             "- Archetypes chosen by the team; routines more regular than real life (see regularity heatmap).",
             "- Would disadvantage: shift workers with irregular schedules, non-English users.", ""]
    lines += [f"![{f}]({f})" for f in figs]
    report = out / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report
