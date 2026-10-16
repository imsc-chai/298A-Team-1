// Personal AI OS front end: vanilla JS + SVG, no build step. Talks to the FastAPI backend in aios/api.py.
"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const S = { pid: null, personas: [], catalog: [], templates: [], tab: "assistant" };
const CATS = ["trigger", "context", "reason", "logic", "action", "human", "output", "cloud"];
const STATUS_MARK = { ok: "✓", failed: "✕", blocked: "⛔", denied: "⛔", rejected: "✕", waiting: "⏸", skipped: "–" };

function toast(msg, ms = 3500) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.add("hidden"), ms);
}

async function api(path, opts = {}) {
  const init = { headers: { "Content-Type": "application/json" }, ...opts };
  if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  const r = await fetch(path, init);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${r.status} ${r.statusText}`);
  return data;
}

function table(cols, rows, onClick) {
  if (!rows.length) return '<p class="muted">Nothing here yet.</p>';
  const head = cols.map((c) => `<th>${esc(c.label)}</th>`).join("");
  const body = rows.map((r, i) => `<tr data-i="${i}" class="${onClick ? "clickable" : ""}">` +
    cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.html ? c.html(r) : esc(c.get(r))}</td>`).join("") + "</tr>").join("");
  return `<div class="tablewrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
function bindRows(el, rows, onClick) {
  el.querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => onClick(rows[+tr.dataset.i])));
}
const cls = (c) => `<span class="badge cls-${esc(c)}">${esc(c)}</span>`;
const statusBadge = (s) => `<span class="badge ${["succeeded", "ok", "approved"].includes(s) ? "ok" : ["waiting_approval", "waiting", "pending", "skipped"].includes(s) ? "warn" : "bad"}">${esc(s)}</span>`;

// ------------------------------------------------------------------ workflow canvas
function drawCanvas(el, wf, runs = [], onSelect) {
  if (!wf || !wf.nodes?.length) { el.innerHTML = '<p class="muted" style="padding:12px">No workflow.</p>'; return; }
  const last = {}; runs.forEach((r) => { last[r.node_id] = r; });
  const depth = {}; wf.nodes.forEach((n) => { depth[n.id] = 0; });
  for (let k = 0; k < wf.nodes.length; k++) wf.edges.forEach((e) => { depth[e.target] = Math.max(depth[e.target], depth[e.source] + 1); });
  const cols = {}; wf.nodes.forEach((n) => { (cols[depth[n.id]] ||= []).push(n); });
  const W = 196, H = 66, GX = 46, GY = 22, P = 16, pos = {};
  let maxRows = 1;
  Object.entries(cols).forEach(([d, ns]) => { maxRows = Math.max(maxRows, ns.length); ns.forEach((n, i) => { pos[n.id] = { x: P + d * (W + GX), y: P + i * (H + GY) }; }); });
  const width = P * 2 + (Object.keys(cols).length) * (W + GX) - GX, height = P * 2 + maxRows * (H + GY) - GY;
  const edges = wf.edges.map((e) => {
    const a = pos[e.source], b = pos[e.target]; if (!a || !b) return "";
    const x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2, mx = (x1 + x2) / 2;
    const label = e.branch ? `<text class="branch" x="${mx}" y="${(y1 + y2) / 2 - 4}" text-anchor="middle">${esc(e.branch)}</text>` : "";
    return `<path class="edge" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}"/>${label}`;
  }).join("");
  const nodes = wf.nodes.map((n) => {
    const p = pos[n.id], cat = n.type.split(".")[0], run = last[n.id];
    const mark = run ? (STATUS_MARK[run.status] || "?") : "";
    const sub = run ? `${run.data_class} · ${run.location} · ${run.wall_s ?? 0}s` : "not run";
    const title = n.label && n.label !== n.type ? `${n.type} (${n.label})` : n.type;
    return `<g class="node" data-id="${esc(n.id)}" transform="translate(${p.x},${p.y})">
      <rect class="box" width="${W}" height="${H}" rx="8"/>
      <rect width="6" height="${H}" rx="3" fill="var(--c-${CATS.includes(cat) ? cat : "logic"})"/>
      <text x="16" y="22">${esc(title.length > 26 ? title.slice(0, 25) + "…" : title)}</text>
      <text class="sub" x="16" y="40">${esc(n.id)}${run ? " · " + esc(run.status) : ""}</text>
      <text class="sub" x="16" y="56">${esc(sub)}</text>
      <text x="${W - 16}" y="22" text-anchor="end">${mark}</text></g>`;
  }).join("");
  const used = [...new Set(wf.nodes.map((n) => n.type.split(".")[0]))];
  el.innerHTML = `<svg width="${width}" height="${height}" role="img" aria-label="Workflow with ${wf.nodes.length} nodes">${edges}${nodes}</svg>` +
    `<div class="legend" style="padding:0 12px 10px">${used.map((c) => `<span><i style="background:var(--c-${CATS.includes(c) ? c : "logic"})"></i>${esc(c)}</span>`).join("")}</div>`;
  el.querySelectorAll(".node").forEach((g) => g.addEventListener("click", () => {
    el.querySelectorAll(".node").forEach((x) => x.classList.remove("sel")); g.classList.add("sel");
    if (onSelect) onSelect(wf.nodes.find((n) => n.id === g.dataset.id), last[g.dataset.id]);
  }));
}

function nodeRunList(runs) {
  return runs.map((r) => `<details><summary>${esc(r.node_id)} · <b>${esc(r.type)}</b> ${statusBadge(r.status)} ${cls(r.data_class)}
    <span class="badge">${esc(r.location)}</span> <span class="muted small">${r.wall_s ?? 0}s</span></summary>
    <h3>Input</h3><pre>${esc(pretty(r.input))}</pre>
    ${r.output ? `<h3>Output</h3><pre>${esc(pretty(r.output))}</pre>` : ""}
    ${r.error ? `<h3>Error</h3><pre>${esc(r.error)}</pre>` : ""}</details>`).join("");
}
function pretty(s) { try { return JSON.stringify(JSON.parse(s), null, 1); } catch { return s ?? ""; } }

// ------------------------------------------------------------------ header and loading
async function init() {
  S.personas = await api("/api/personas");
  $("#persona").innerHTML = S.personas.map((p) => `<option value="${esc(p.id)}">${esc(p.name)} · ${esc(p.archetype)} · ${esc(p.split)}</option>`).join("");
  S.catalog = await api("/api/generators");
  S.templates = await api("/api/templates");
  $("#generator").innerHTML = S.catalog.map((g) => `<option value="${esc(g.name)}" ${g.status !== "ready" ? "disabled" : ""}>${esc(g.name)}${g.status !== "ready" ? " (" + esc(g.status) + ")" : ""}</option>`).join("");
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("#load").addEventListener("click", load);
  $("#plus1h").addEventListener("click", () => advance(60));
  $("#plus1d").addEventListener("click", () => advance(1440));
  $("#generator").addEventListener("change", genChanged);
  $("#run").addEventListener("click", runGoal);
  $("#dataSearch").addEventListener("click", refreshData);
  $("#dataChannel").addEventListener("change", refreshData);
  $("#dataDelete").addEventListener("click", deleteSelected);
  $("#calShow").addEventListener("click", showCalendar);
  $("#auditGo").addEventListener("click", storageAudit);
  // Deep links (demos, screenshots): ?persona=&day=&time=&tab=&generator=&goal=&run=1&load=0&exec=
  const q = new URLSearchParams(location.search);
  if (q.get("persona")) $("#persona").value = q.get("persona");
  if (q.get("day")) $("#day").value = q.get("day");
  if (q.get("time")) $("#time").value = q.get("time");
  if (q.get("generator")) $("#generator").value = q.get("generator");
  if (q.get("goal")) $("#goal").value = q.get("goal");
  genChanged();
  if (q.get("load") === "0") { S.pid = $("#persona").value; setClock(await api(`/api/ws/${S.pid}/state`)); }
  else await load();
  if (q.get("tab")) showTab(q.get("tab"));
  if (q.get("exec")) { showTab("executions"); await showExecution({ id: q.get("exec") }); }
  if (q.get("audit")) { showTab("privacy"); $("#auditValue").value = q.get("audit"); await storageAudit(); }
  if (q.get("run") === "1") await runGoal();
}

async function load() {
  S.pid = $("#persona").value;
  const btn = $("#load"); btn.disabled = true; btn.textContent = "Loading…";
  try {
    const st = await api(`/api/ws/${S.pid}/load`, { method: "POST", body: { day: +$("#day").value, time: $("#time").value, handle_instructions: $("#handle").checked, generator: "baseline" } });
    setClock(st); toast(`Replayed ${st.records_seen} of ${st.records_total} records up to ${st.now.replace("T", " ")}`);
    await refreshTab();
  } catch (e) { toast(e.message); } finally { btn.disabled = false; btn.textContent = "Load"; }
}
function setClock(st) {
  $("#clock").dataset.date = st.now.slice(0, 10);
  const d = new Date(st.now); $("#clock").textContent = `${st.name} · ${d.toLocaleString([], { weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`;
  const n = st.pending_approvals || 0, pill = $("#approvalCount"); pill.textContent = n; pill.classList.toggle("hidden", !n);
}
async function advance(minutes) {
  try {
    const r = await api(`/api/ws/${S.pid}/advance`, { method: "POST", body: { minutes } });
    setClock(r.state); toast(r.fired.length ? `${r.fired.length} triggered workflow(s) ran` : "Clock advanced");
    await refreshTab();
  } catch (e) { toast(e.message); }
}
function showTab(name) {
  S.tab = name;
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  refreshTab();
}
async function refreshTab() {
  if (!S.pid) return;
  const f = { workflows: refreshWorkflows, executions: refreshExecutions, data: refreshData, privacy: refreshPrivacy, approvals: refreshApprovals, benchmark: refreshBenchmark }[S.tab];
  try { if (f) await f(); } catch (e) { toast(e.message); }
}

// ------------------------------------------------------------------ assistant
function genChanged() {
  const g = S.catalog.find((x) => x.name === $("#generator").value) || {};
  $("#genDesc").textContent = g.description || "";
  const isTemplate = (g.name || "").startsWith("template:");
  $("#goalLabel").classList.toggle("hidden", isTemplate);
  $("#genParams").innerHTML = isTemplate ? Object.entries(g.params || {}).map(([k, hint]) => {
    const contacts = (S.personas.find((p) => p.id === S.pid) || {}).contacts || [];
    const input = ["to", "contact"].includes(k)
      ? `<select data-param="${esc(k)}">${contacts.map((c) => `<option>${esc(c)}</option>`).join("")}</select>`
      : `<input data-param="${esc(k)}" placeholder="${esc(hint)}" ${k === "date" ? 'type="date"' : ""}>`;
    return `<label>${esc(k)} ${input}</label>`;
  }).join("") : "";
}
async function runGoal() {
  const gen = $("#generator").value, params = {};
  document.querySelectorAll("#genParams [data-param]").forEach((el) => { if (el.value) params[el.dataset.param] = el.dataset.param === "duration_minutes" ? +el.value : el.value; });
  const btn = $("#run"); btn.disabled = true; $("#runStatus").textContent = "Running on this laptop…"; const t0 = performance.now();
  try {
    const res = await api(`/api/ws/${S.pid}/goal`, { method: "POST", body: { text: $("#goal").value, generator: gen, params } });
    renderResult(res, (performance.now() - t0) / 1000);
  } catch (e) { $("#answerCard").innerHTML = `<h2>Result</h2><p class="badge bad">${esc(e.message)}</p>`; }
  finally { btn.disabled = false; $("#runStatus").textContent = ""; refreshApprovalCount(); }
}
// Plain-language description of each step, for the person using the assistant.
function describe(r) {
  const inp = parse(r.input) || {}, out = parse(r.output) || {};
  const n = (x) => (Array.isArray(x) ? x.length : 0);
  const priv = (items) => (items || []).filter((i) => i.permission === "deny").length;
  if (r.status === "blocked") return `Stopped a step that would have sent ${r.data_class} data off this device`;
  if (r.status === "denied") return `Did not ${r.type.split(".")[1].replace(/_/g, " ")}: your settings do not allow it`;
  if (r.status === "waiting") return "Waiting for your OK before going on (see Approvals)";
  if (r.status === "rejected") return "You said no, so this step did not run";
  if (r.status === "failed") return `A step could not finish: ${r.error || ""}`;
  switch (r.type) {
    case "trigger.chat": return "Received your request";
    case "trigger.time": return "Started on schedule";
    case "trigger.event": return "Started because a new message arrived";
    case "reason.agent_step": {
      const calls = out.tool_calls || [];
      return calls.length ? `Decided to ${calls.map((c) => c.name.replace(/_/g, " ")).join(", then ")}` : "Wrote the answer";
    }
    case "reason.llm": return "Wrote a draft with the on-device model";
    case "context.search": {
      const where = !inp.kind || inp.kind === "all" ? "records" : `${inp.kind === "note" ? "notes" : inp.kind === "message" ? "messages" : inp.kind + " entries"}`;
      return `Searched your ${where} for "${inp.query}" — found ${n(out.items)}${priv(out.items) ? ` (${priv(out.items)} marked private)` : ""}`;
    }
    case "context.read": return "Opened one record";
    case "context.recent": return `Looked at your last ${inp.days || 7} days — ${n(out.items)} records${inp.exclude_private ? ", private ones left out" : ""}`;
    case "context.calendar": return `Checked your calendar for ${out.date || inp.date}`;
    case "context.contacts": return "Looked up your contacts";
    case "context.preferences": return "Checked how you like your meetings";
    case "human.approval": return "You approved it";
    case "action.send_email": return `Emailed ${out.to}`;
    case "action.create_event": return `Booked "${out.title}" on ${out.date} at ${out.start}`;
    case "action.cancel_event": return "Cancelled an event";
    case "action.delete_records": return `Deleted ${n(out.deleted)} record(s) and every copy of what they said`;
    case "output.respond": return "Gave you the answer";
    default: return r.type;
  }
}
function parse(s) { try { return JSON.parse(s); } catch { return null; } }
const inlineMd = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");

function renderResult(res, secs) {
  const llmCalls = res.node_runs.filter((r) => r.type === "reason.agent_step" || r.type === "reason.llm").length;
  const blocked = res.node_runs.filter((r) => ["blocked", "denied"].includes(r.status)).length;
  const local = res.node_runs.every((r) => r.location === "local");
  const steps = res.node_runs.filter((r) => r.status !== "skipped").map((r) => `<li>${esc(describe(r))}</li>`).join("");
  $("#answerCard").innerHTML = `<h2>Result</h2><div class="answer">${inlineMd(res.answer || "(no answer)")}</div>
    <h3>What I did</h3><ol class="steps">${steps}</ol>
    <p class="small ${local ? "okText" : "warnText"}">${local ? "🔒 Everything ran on this device. Nothing was sent to the internet." : "Some steps were blocked from leaving this device."}</p>
    <div class="badges">${statusBadge(res.status)}<span class="badge">${(res.latency_s ?? secs).toFixed(1)} s</span><span class="badge">${llmCalls} model step(s)</span>
    <span class="badge">${res.node_runs.length} node runs</span><span class="badge">${res.interventions} confirmation(s)</span>
    ${blocked ? `<span class="badge bad">${blocked} blocked by policy</span>` : ""}
    ${res.approval_id ? `<span class="badge warn">waiting in Approvals</span>` : ""}</div>
    ${res.error ? `<p class="small muted">${esc(res.error)}</p>` : ""}`;
  drawCanvas($("#assistantCanvas"), res.workflow, res.node_runs);
  $("#assistantNodes").innerHTML = nodeRunList(res.node_runs);
}

// ------------------------------------------------------------------ workflows
async function refreshWorkflows() {
  const wfs = await api(`/api/ws/${S.pid}/workflows`);
  $("#wfList").innerHTML = table([{ label: "Name", get: (w) => w.name }, { label: "Source", get: (w) => w.source }, { label: "Saved", get: (w) => (w.ts || "").replace("T", " ").slice(0, 16) }], wfs, true);
  bindRows($("#wfList"), wfs, async (w) => {
    const wf = await api(`/api/ws/${S.pid}/workflows/${w.id}`);
    $("#wfTitle").textContent = wf.name;
    drawCanvas($("#wfCanvas"), wf, [], (n) => { $("#wfInspector").innerHTML = `<h3>${esc(n.id)} · ${esc(n.type)}</h3><pre>${esc(JSON.stringify(n.params, null, 1))}</pre>`; });
    $("#wfInspector").innerHTML = '<p class="muted small">Click a node to see its parameters.</p>';
  });
  $("#tplList").innerHTML = S.templates.map((t) => `<p><b>${esc(t.id)}</b> <span class="muted">${esc(t.description)}</span>
    <button data-tpl="${esc(t.id)}">Use</button></p>`).join("");
  $("#tplList").querySelectorAll("[data-tpl]").forEach((b) => b.addEventListener("click", () => {
    $("#generator").value = `template:${b.dataset.tpl}`; genChanged(); showTab("assistant");
  }));
}

// ------------------------------------------------------------------ executions
async function refreshExecutions() {
  const xs = await api(`/api/ws/${S.pid}/executions`);
  $("#exList").innerHTML = table([{ label: "When", get: (x) => (x.started || "").replace("T", " ").slice(0, 16) }, { label: "Generator", get: (x) => x.generator },
    { label: "Goal", get: (x) => (x.goal || "").slice(0, 70) }, { label: "Status", html: (x) => statusBadge(x.status) }, { label: "Answer", get: (x) => (x.answer || "").slice(0, 60) }], xs, true);
  bindRows($("#exList"), xs, showExecution);
}
async function showExecution(x) {
  const e = await api(`/api/ws/${S.pid}/executions/${x.id}`);
  $("#exTitle").textContent = `${e.generator}: ${e.goal || e.id}`;
  drawCanvas($("#exCanvas"), e.workflow, e.node_runs);
  const prof = e.profile || {};
  $("#exDetail").innerHTML = `<div class="badges">${statusBadge(e.status)}<span class="badge">${prof.llm_calls ?? 0} model calls</span>
    <span class="badge">${prof.prompt_tokens ?? 0} in / ${prof.output_tokens ?? 0} out tokens</span>
    <span class="badge ${prof.raw_export_bytes ? "bad" : "ok"}">${prof.raw_export_bytes ?? 0} bytes exported</span></div>
    ${e.sent.length ? `<h3>Sent</h3>${table([{ label: "To", get: (m) => m.to_name }, { label: "Body", get: (m) => m.body }], e.sent)}` : ""}
    ${e.created.length ? `<h3>Booked</h3>${table([{ label: "Date", get: (c) => c.date }, { label: "Start", get: (c) => `${String(Math.floor(c.start_min / 60)).padStart(2, "0")}:${String(c.start_min % 60).padStart(2, "0")}` },
      { label: "With", get: (c) => c.with_name }, { label: "Cancelled", get: (c) => (c.cancelled ? "yes" : "") }], e.created)}
      <button id="undo">Undo this execution (cancel its events)</button>` : ""}
    <h3>Node runs</h3>${nodeRunList(e.node_runs)}`;
  const u = $("#undo"); if (u) u.addEventListener("click", async () => { const r = await api(`/api/ws/${S.pid}/executions/${x.id}/undo`, { method: "POST" }); toast(`Cancelled ${r.cancelled_events.length} event(s). ${r.note}`); showExecution(x); });
}

// ------------------------------------------------------------------ data
async function refreshData() {
  const ch = $("#dataChannel").value, q = $("#dataQuery").value;
  const rows = await api(`/api/ws/${S.pid}/records?channel=${encodeURIComponent(ch)}&q=${encodeURIComponent(q)}&limit=150`);
  $("#dataRows").innerHTML = table([{ label: "", html: (r) => `<input type="checkbox" data-del="${esc(r.id)}">` },
    { label: "Time", get: (r) => r.ts.replace("T", " ").slice(0, 16) }, { label: "From", get: (r) => r.sender }, { label: "Text", get: (r) => r.text },
    { label: "Class", html: (r) => cls(r.data_class) }, { label: "Permission", get: (r) => r.permission }], rows);
  if (!$("#calDate").value) $("#calDate").value = ($("#clock").dataset.date || "");
}
async function deleteSelected() {
  const ids = [...document.querySelectorAll("[data-del]:checked")].map((c) => c.dataset.del);
  if (!ids.length) return toast("Select records to delete first");
  if (!confirm(`Delete ${ids.length} record(s) and every copy of what they said? This cannot be undone.`)) return;
  const r = await api(`/api/ws/${S.pid}/delete`, { method: "POST", body: { record_ids: ids, request: "Deleted from the Data tab" } });
  toast(`Deleted ${r.record_ids.length} record(s); ${r.redactions} copies redacted. Receipt ${r.id}`); refreshData();
}
async function showCalendar() {
  const d = $("#calDate").value; const c = await api(`/api/ws/${S.pid}/calendar?date=${encodeURIComponent(d)}`);
  $("#calendar").innerHTML = `<p class="muted small">${esc(c.date)}</p>` + table([{ label: "Start", get: (e) => e.start }, { label: "End", get: (e) => e.end }, { label: "Title", get: (e) => e.title }, { label: "Source", get: (e) => e.source }], c.entries);
}

// ------------------------------------------------------------------ privacy
async function refreshPrivacy() {
  const [pol, pv] = await Promise.all([api(`/api/ws/${S.pid}/policy`), api(`/api/ws/${S.pid}/privacy`)]);
  $("#toolPolicy").innerHTML = table([{ label: "Tool", get: (r) => r[0] }, { label: "Policy", html: (r) => `<select data-tool="${esc(r[0])}">${["allow", "ask", "deny"].map((v) => `<option ${v === r[1] ? "selected" : ""}>${v}</option>`).join("")}</select>` }], Object.entries(pol.tools));
  $("#toolPolicy").querySelectorAll("[data-tool]").forEach((s) => s.addEventListener("change", async () => {
    await api(`/api/ws/${S.pid}/policy`, { method: "PUT", body: { tool: s.dataset.tool, value: s.value } }); toast(`${s.dataset.tool}: ${s.value}`);
  }));
  $("#locPolicy").innerHTML = table([{ label: "Location", get: (r) => r[0] }, { label: "Rule", get: (r) => r[1] }], Object.entries(pol.locations)) + `<p class="small muted">Egress allowlist: ${esc(pol.egress_allowlist.join(", "))}</p>`;
  const s = pv.egress_summary;
  $("#egress").innerHTML = `<div class="badges"><span class="badge">${s.calls} outbound call(s) logged</span><span class="badge ${s.blocked ? "warn" : ""}">${s.blocked} blocked</span>
    <span class="badge ${s.nonlocal_bytes ? "bad" : "ok"}">${s.nonlocal_bytes} bytes to non-local hosts</span></div>` +
    table([{ label: "Time", get: (e) => e.ts.slice(11, 19) }, { label: "Host", get: (e) => e.host }, { label: "Purpose", get: (e) => e.purpose }, { label: "Bytes", num: true, get: (e) => e.bytes }, { label: "Persona bytes", num: true, get: (e) => e.persona_bytes }, { label: "Allowed", get: (e) => (e.allowed ? "yes" : "BLOCKED") }], pv.egress.slice(0, 25));
  $("#classes").innerHTML = Object.entries(pv.classes).map(([k, v]) => `${cls(k)} ${v}`).join(" ");
  $("#receipts").innerHTML = table([{ label: "When", get: (r) => r.ts.replace("T", " ").slice(0, 16) }, { label: "Request", get: (r) => r.request }, { label: "Records", get: (r) => r.record_ids.length }, { label: "Copies redacted", num: true, get: (r) => r.redactions }], pv.receipts);
  $("#auditLog").innerHTML = table([{ label: "Time", get: (a) => a.ts.replace("T", " ").slice(5, 16) }, { label: "Actor", get: (a) => a.actor }, { label: "Action", get: (a) => a.action }, { label: "Target", get: (a) => a.target }], pv.audit);
}
async function storageAudit() {
  const v = $("#auditValue").value; if (!v) return;
  const r = await api(`/api/ws/${S.pid}/storage-audit`, { method: "POST", body: { value: v } });
  $("#auditOut").innerHTML = r.found_in.length ? table([{ label: "Table", get: (h) => h.table }, { label: "Column", get: (h) => h.column }, { label: "Row", get: (h) => h.row }], r.found_in)
    : `<p class="badge ok">"${esc(v)}" is not stored anywhere</p>`;
}

// ------------------------------------------------------------------ approvals
function approvalBody(a) {
  const p = a.request.params || {};
  const field = (k, v) => `<p class="small"><b>${esc(k)}:</b> ${esc(v)}</p>`;
  const prose = (t) => `<div class="prose">${esc(String(t).replace(/\\n/g, "\n"))}</div>`;
  if (a.tool === "mail.send") return `<h3>Send this email?</h3>${field("To", p.to)}${field("Subject", p.subject || "(none)")}${prose(p.body || "")}`;
  if (a.tool === "calendar.create") return `<h3>Book this meeting?</h3>${field("With", p.with_contact)}${field("When", `${p.date} at ${p.start}, ${p.duration_minutes} min`)}`;
  if (a.tool === "memory.forget") return `<h3>Delete these records everywhere?</h3>${field("Records", (p.ids || []).map((i) => i.id || i).join(", "))}`;
  const msg = String(p.message || "Continue?");
  const [q, ...rest] = msg.split("? ");
  return `<h3>${esc(rest.length ? q + "?" : "Continue?")}</h3>${prose(rest.length ? rest.join("? ") : msg)}`;
}
async function refreshApprovalCount() { if (S.pid) setClock(await api(`/api/ws/${S.pid}/state`)); }
async function refreshApprovals() {
  const xs = await api(`/api/ws/${S.pid}/approvals`);
  $("#approvals").innerHTML = xs.length ? xs.map((a) => `<div class="card"><div class="badges">${statusBadge(a.status === "pending" ? "waiting" : a.status)}
    <span class="muted small">${esc(a.ts.replace("T", " ").slice(0, 16))}</span></div>
    ${approvalBody(a)}
    <details><summary class="small muted">Technical details</summary><pre>${esc(JSON.stringify(a.request, null, 1))}</pre></details>
    ${a.status === "pending" ? `<div class="row"><button class="primary" data-ok="${esc(a.id)}">Approve</button><button data-no="${esc(a.id)}">Reject</button></div>` : ""}</div>`).join("")
    : '<p class="muted">No approvals yet. Actions the policy marks "ask" wait here.</p>';
  const decide = async (id, decision) => {
    const r = await api(`/api/ws/${S.pid}/approvals/${id}`, { method: "POST", body: { decision } });
    toast(`${decision === "approve" ? "Approved" : "Rejected"}: execution ${r.status}`); refreshApprovals(); refreshApprovalCount();
  };
  $("#approvals").querySelectorAll("[data-ok]").forEach((b) => b.addEventListener("click", () => decide(b.dataset.ok, "approve")));
  $("#approvals").querySelectorAll("[data-no]").forEach((b) => b.addEventListener("click", () => decide(b.dataset.no, "reject")));
}

// ------------------------------------------------------------------ benchmark
function md(text) {
  const lines = text.split("\n"), out = []; let i = 0;
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>");
  while (i < lines.length) {
    const l = lines[i];
    if (/^\|/.test(l)) {
      const rows = []; while (i < lines.length && /^\|/.test(lines[i])) rows.push(lines[i++]);
      const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const body = rows.slice(2).map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("");
      out.push(`<div class="tablewrap"><table><thead><tr>${cells(rows[0]).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`); continue;
    }
    const img = l.match(/^!\[(.*?)\]\(figures\/(.+?)\)/);
    if (img) out.push(`<img alt="${esc(img[1])}" src="/api/results/figures/${encodeURIComponent(img[2])}">`);
    else if (/^#+ /.test(l)) { const n = l.match(/^#+/)[0].length; out.push(`<h${n}>${inline(l.replace(/^#+ /, ""))}</h${n}>`); }
    else if (/^- /.test(l)) out.push(`<p>• ${inline(l.slice(2))}</p>`);
    else if (l.trim()) out.push(`<p>${inline(l)}</p>`);
    i++;
  }
  return out.join("");
}
async function refreshBenchmark() {
  const [sug, bench, prof] = await Promise.all([api(`/api/ws/${S.pid}/suggestions`), api("/api/benchmark"), api("/api/profile")]);
  $("#generators").innerHTML = table([{ label: "Generator", get: (g) => g.name }, { label: "Kind", get: (g) => g.kind }, { label: "Status", html: (g) => `<span class="badge ${g.status === "ready" ? "ok" : "warn"}">${esc(g.status)}</span>` }], S.catalog);
  $("#suggestions").innerHTML = `<p class="small muted">Model B: ${esc(sug.models.model_b)} · Model C: ${esc(sug.models.model_c)}. Until they are trained, simple references fill the slots.</p>
    <h3>Weekly routines (counting reference)</h3>${table([{ label: "Routine", get: (r) => r.title }, { label: "Weekday", get: (r) => ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][r.weekday] }, { label: "Usual start", get: (r) => r.typical_start }, { label: "Seen", num: true, get: (r) => r.occurrences }, { label: "Proposed reminder", get: (r) => r.proposed_trigger.at }], sug.routines)}
    <h3>Suggestions (most-accepted reference)</h3>${table([{ label: "Preference", get: (s) => s.key }, { label: "Suggest", get: (s) => s.suggest }, { label: "Accepted / offered", get: (s) => `${s.accepted} / ${s.offered}` }], sug.suggestions)}`;
  const pct = (v) => (v == null ? "–" : `${(100 * v).toFixed(1)}%`);
  $("#runs").innerHTML = table([{ label: "Run", get: (r) => r.file }, { label: "Tasks", num: true, get: (r) => r.tasks }, { label: "Success", num: true, get: (r) => pct(r.task_success) },
    { label: "Deletion compliance", num: true, get: (r) => pct(r.deletion_compliance) }, { label: "Leakage", num: true, get: (r) => pct(r.privacy_leakage) }, { label: "p95 s", num: true, get: (r) => r.latency_p95_s }, { label: "Exported bytes", num: true, get: (r) => r.raw_export_bytes }], bench.runs);
  $("#profile").innerHTML = table([{ label: "Model", get: (p) => p.model }, { label: "Params", get: (p) => p.parameters || "" }, { label: "Disk GB", num: true, get: (p) => (p.disk_bytes / 1e9).toFixed(2) },
    { label: "Resident GB", num: true, get: (p) => ((p.resident_bytes || 0) / 1e9).toFixed(2) }, { label: "Decode tok/s", num: true, get: (p) => p.decode_tok_s_median ?? `${p.texts_per_s_median} texts/s` },
    { label: "TTFT @1k s", num: true, get: (p) => p.by_prompt_length?.["1024"]?.ttft_s ?? "" }], prof);
  $("#results").innerHTML = bench.results_md ? md(bench.results_md) : '<p class="muted">No results yet: run <code>p17sim eval</code> then <code>p17sim report</code>.</p>';
}

init().catch((e) => toast(e.message, 8000));
