"""Draws docs/ARCHITECTURE.md's tier diagram with exact alignment: python scripts/draw_architecture.py"""
W = 112  # outer tier width
IN = W - 4


def box(title, lines, width):
    import textwrap
    out = ["┌" + "─" * (width - 2) + "┐", "│" + title[: width - 2].center(width - 2) + "│"]
    for l in lines:
        indent = "  " if l.startswith(("•", "-")) else ""
        for w in textwrap.wrap(l, width - 4, subsequent_indent=indent) or [""]:
            out.append("│ " + w.ljust(width - 4) + " │")
    out.append("└" + "─" * (width - 2) + "┘")
    for l in out:
        assert len(l) == width, (len(l), l)
    return out


def row(boxes, gap=2):
    n = len(boxes)
    bw = (IN - gap * (n - 1)) // n
    widths = [bw] * (n - 1) + [IN - gap * (n - 1) - bw * (n - 1)]
    rendered = [box(t, ls, w) for (t, ls), w in zip(boxes, widths)]
    h = max(len(r) for r in rendered)
    for r, w in zip(rendered, widths):  # equal heights: pad before the bottom border
        while len(r) < h:
            r.insert(-1, "│" + " " * (w - 2) + "│")
    return [(" " * gap).join(parts).ljust(IN) for parts in zip(*rendered)]


def tier(title, rows, style="solid"):
    tl, tr, bl, br, hz, vt = ("┌", "┐", "└", "┘", "─", "│") if style == "solid" else ("┌", "┐", "└", "┘", "┄", "┆")
    head = f" {title} "
    out = [tl + head.center(W - 2, hz) + tr]
    for i, r in enumerate(rows):
        lines = r if isinstance(r, list) else [r]
        out += [vt + " " + l.ljust(IN) + " " + vt for l in lines]
    out.append(bl + hz * (W - 2) + br)
    for l in out:
        assert len(l) == W, (len(l), l)
    return out


def arrow(label, down=True):
    mid = W // 2
    return [" " * mid + "│", " " * mid + "│  " + label, " " * mid + ("▼" if down else "▲")]


def text(t):
    return t


L = []
L.append("╔" + " DEVICE PLANE · the user's own computer · personal data lives only here ".center(W - 2, "═") + "╗")
L += tier("CLIENT TIER", [row([("Web app (browser) · localhost:8017 · vanilla JS + SVG, no build", [
    "• User view:   Assistant (goal → workflow → answer) · Approvals · Data (search, delete)",
    "• Owner view:  Workflows (n8n-style canvas) · Executions (node by node) · Privacy center · Benchmark"])])])
L += arrow("REST / JSON")
L += tier("API TIER", [row([
    ("FastAPI backend :8017", ["personas · load day/time · advance clock", "goals · workflows · executions · undo",
                               "approvals · policy · privacy · storage audit", "delete · generators · suggestions · results"]),
    ("CLI  p17sim  (same runtime, no UI)", ["generate2 · render2 · build-trace2 · audit", "export-training (Models A, B, C)",
                                             "eval (harness) · profile-models · report", "serve (starts the API + web app)"])])])
L += arrow("goal + the person's context (records visible at that moment)")
L += tier("INTELLIGENCE TIER · workflow generators (turn a goal into a workflow)", [row([
    ("Baseline agent [built]", ["OpenJarvis loop", "(native_openhands)", "qwen3.5:4b picks one", "tool at a time; each", "step = a node"]),
    ("Templates [built]", ["morning briefing", "weekly recap (asks)", "book a meeting", "forget something"]),
    ("Model A [pending]", ["retrieval-augmented", "hierarchical planner", "goal → full workflow"]),
    ("Model B / C [pending]", ["B: routines →", "   recurring triggers", "C: what to suggest,", "   and when"])])])
L += arrow("workflow = typed nodes + edges (JSON)")
L += tier("ORCHESTRATION & PRIVACY TIER · every step is checked before it runs", [
    row([
        ("Workflow engine", ["runs nodes in order", "if/else branches", "pauses for approval", "undoes bookings on error",
                             "logs every node run"]),
        ("Privacy layer", ["data class on every value", "cloud steps: public only", "tools: allow / ask / deny",
                           "egress: localhost only,", "  every byte logged", "deletion: all copies +", "  a receipt"]),
        ("Workspace runtime", ["one per persona", "simulated clock (a month)", "time + event triggers",
                               "Model D check (planned)", "audit log of decisions"])]),
    "",
    row([("Node library", ["trigger (chat · time · event)  ·  context (search · read · recent · calendar · contacts · preferences)",
                          "reason (local LLM)  ·  logic (if · merge)  ·  action (send email · create/cancel event · delete records)",
                          "human (approval)  ·  output (answer)  ·  cloud (blocked whenever the data is personal)"])])])
L += arrow("model calls (through the egress gateway)  ·  reads and writes")
L += tier("MODEL TIER · local only", [row([
    ("Ollama :11434", ["qwen3.5:4b  · the agent's model (4.2B, 3.3 GB)", "all-minilm · sentence embeddings (23M)",
                       "device builds of Models A–D (later)"]),
    ("Renderers (data creation only)", ["llama3.2 3B · writes synthetic messages", "phi3 mini  · held-out writing style",
                                        "never see a real person's data"])])])
L += arrow("SQL")
L += tier("DATA TIER · SQLite, one file per persona (in memory for the benchmark)", [row([
    ("Personal data", ["• records: mail, notes,", "  calendar, preferences,", "  instructions", "• full-text index (FTS5)", "• vectors"]),
    ("Activity", ["• outbox (sent mail)", "• events created", "• workflows", "• executions + node runs"]),
    ("Governance", ["• tool policies", "• approvals", "• audit log", "• deletion receipts", "• egress log"])])])
L.append("╚" + "═" * (W - 2) + "╝")
L.append("")
mid = W // 2
L.append(" " * (mid - 30) + "▲ signed model updates (registry release channel)")
L.append(" " * (mid - 30) + "▼ optional aggregate counts only, off by default — never text, ids or values")
L.append("━" * W)
L.append("  PRIVACY BOUNDARY: personal records, text and embeddings NEVER cross this line (enforced by the egress gateway)".ljust(W))
L.append("━" * W)
L.append("")
L.append("╔" + " CLOUD PLANE (planned) · synthetic data only · GCP ".center(W - 2, "═") + "╗")
L += tier("DATA PIPELINE TIER · Airflow :8080 (DAGs p17_build on demand, p17_daily on a schedule)", [
    "generate → render (GPU) → validate + leakage → bronze → Presidio PII → silver → dbt → tests → marts",
    "",
    row([("Cloud Storage lake", ["• bronze: raw JSONL", "  (write-once)", "• silver: cleaned Parquet"]),
         ("BigQuery warehouse", ["• p17_gold: records, tasks,", "  training marts", "• p17_labels (restricted)", "• p17_eval: runs, metrics"]),
         ("Quality & monitoring", ["• gates stop the DAG and", "  name the failing check", "• retries, email alerts", "• Looker Studio dashboards"])])],
    style="dashed")
L += arrow("versioned training marts (train + val personas only)")
L += tier("ML PLATFORM TIER", [row([
    ("GPU (team / Vertex AI)", ["• vLLM :8000: rendering", "  and evaluation", "• baseline 9B FP16", "  (the paper's setup)", "• training: Models A–D"]),
    ("MLflow :5000", ["• experiment tracking", "• model registry + gates:", "  candidate → validated", "  → released"]),
    ("Release channel", ["• quantized, signed builds", "• Ollama on the device", "  pulls them", "• checksum checked first"])])],
    style="dashed")
L += arrow("same app container")
L += tier("DEMO TIER", [row([("Cloud Run + IAP sign-in", [
    "the same web app and API, in 'demo' mode: synthetic personas only (refuses anything else)",
    "model calls go to a vLLM GPU endpoint · reachable by the team, instructor and ISA"])])], style="dashed")
L.append("╚" + "═" * (W - 2) + "╝")
L.append("")
L.append("Legend:  solid boxes = built and tested · dashed boxes = planned · [pending] = slot ready, model not trained")
print("\n".join(L))
