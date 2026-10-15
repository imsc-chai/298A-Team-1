"""Prototype interface: one complete path through the system, for a live demo.

Pick a persona and a moment in their 30 days, type any request, and Model A (or the baseline) runs
it through the shell on this machine. The page shows the plan, the evidence with provenance, every
tool call, what was sent or booked, the timing and token profile, and the egress counter.

Run: p17sim serve   (then open http://localhost:8017)
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

from p17sim.agents.registry import factory
from p17sim.system.runner import Dataset
from p17sim.system.shell import Shell

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Local Assistant Demo</title>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#ffffff;--accent:#2a78d6;--ok:#1a7f37;--warn:#b35900}
@media (prefers-color-scheme: dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3936;--card:#232321;--accent:#3987e5;--ok:#4cb46a;--warn:#e08a3c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--ink2);margin:0 0 20px}
.grid{display:grid;grid-template-columns:320px 1fr;gap:16px}@media (max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px}
label{display:block;font-size:13px;color:var(--ink2);margin:10px 0 4px}select,input,textarea{width:100%;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);font:inherit}
button{margin-top:12px;width:100%;padding:10px;border:0;border-radius:6px;background:var(--accent);color:#fff;font-weight:600;cursor:pointer}button:disabled{opacity:.5}
h2{font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2);margin:0 0 8px}
.answer{font-size:17px}.badge{display:inline-block;padding:2px 8px;border-radius:99px;border:1px solid var(--line);font-size:12px;margin-right:6px}
.ok{color:var(--ok)}.warn{color:var(--warn)}pre{white-space:pre-wrap;word-break:break-word;font-size:12.5px;margin:0;color:var(--ink2)}
ul{margin:0;padding-left:18px}li{margin:2px 0}.muted{color:var(--ink2);font-size:13px}
</style></head><body><main>
<h1>Privacy-preserving personal assistant</h1>
<p class="sub">Everything runs on this laptop: the model (qwen3.5:4b), the memory and the tools. Data is synthetic.</p>
<div class="grid"><div>
<div class="card"><label for="persona">Persona</label><select id="persona"></select>
<div id="pinfo" class="muted" style="margin-top:8px"></div>
<label for="day">Day of their month (1–30)</label><input id="day" type="number" min="2" max="30" value="25">
<label for="hour">Time</label><input id="hour" type="time" value="09:00">
<label for="system">System</label><select id="system"><option value="model_a">Model A (planner)</option><option value="baseline">Baseline (OpenJarvis-style)</option></select>
<label for="req">Request</label><textarea id="req" rows="3">Email my friend a short summary of my week.</textarea>
<button id="go">Run</button><p class="muted" id="status"></p></div>
<div class="card"><h2>Try</h2><ul class="muted"><li>What is my gate code?</li><li>Book a 30-minute meeting with … tomorrow.</li><li>Please forget my parking spot.</li><li>Email … a recap of this week.</li></ul></div>
</div><div id="out"><div class="card muted">Pick a persona and run a request.</div></div></div></main>
<script>
const $=s=>document.querySelector(s);let personas=[];
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
fetch('/api/personas').then(r=>r.json()).then(ps=>{personas=ps;$('#persona').innerHTML=ps.map(p=>`<option value="${p.id}">${esc(p.name)} · ${esc(p.archetype)} · ${p.split}</option>`).join('');info();});
function info(){const p=personas.find(p=>p.id==$('#persona').value);if(p)$('#pinfo').innerHTML=`Contacts: ${p.contacts.map(esc).join(', ')}`;}
$('#persona').onchange=info;
$('#go').onclick=async()=>{$('#go').disabled=true;$('#status').textContent='Replaying the month up to that moment, then running…';
 const body={persona_id:$('#persona').value,day:+$('#day').value,time:$('#hour').value,system:$('#system').value,request:$('#req').value};
 try{const r=await fetch('/api/run',{method:'POST',body:JSON.stringify(body)});const d=await r.json();render(d);}catch(e){$('#out').innerHTML=`<div class="card warn">${esc(e)}</div>`}
 $('#go').disabled=false;$('#status').textContent='';};
function render(d){if(d.error){$('#out').innerHTML=`<div class="card warn">${esc(d.error)}</div>`;return;}
 const p=d.profile;const eg=p.raw_export_bytes===0&&p.nonlocal_calls===0;
 $('#out').innerHTML=`<div class="card"><h2>Answer</h2><div class="answer">${esc(d.answer)}</div>
 <p style="margin:10px 0 0"><span class="badge">${d.latency_s}s</span><span class="badge">${p.llm_calls} model calls</span><span class="badge">${p.prompt_tokens} in / ${p.output_tokens} out tokens</span><span class="badge ${eg?'ok':'warn'}">${eg?'0 bytes left the device':p.raw_export_bytes+' bytes left the device'}</span><span class="badge">${d.interventions} confirmations</span></p></div>
 <div class="card"><h2>Plan and steps</h2><pre>${esc(d.trace.map(t=>JSON.stringify(t)).join('\\n'))||'—'}</pre></div>
 <div class="card"><h2>Tool calls (provenance log)</h2><pre>${esc(d.tool_calls.map(t=>`${t.time.slice(11,16)} ${t.actor} ${t.tool} ${JSON.stringify(t.args)}\\n   → ${t.result}`).join('\\n'))||'—'}</pre></div>
 <div class="card"><h2>Sent and booked during this request</h2><pre>${esc(JSON.stringify({sent:d.sent,created:d.created},null,1))}</pre></div>
 <div class="card"><h2>Time by component (s)</h2><pre>${esc(JSON.stringify(p.time_by_component_s,null,1))}</pre>
 <p class="muted">State replayed: ${d.replayed} records, ${d.deletions_handled} deletion requests handled before this moment.</p></div>`;}
</script></body></html>"""


class Demo:
    def __init__(self, ds: Dataset, model: str):
        self.ds, self.model, self.lock = ds, model, Lock()

    def personas(self) -> list[dict]:
        order = {"test": 0, "ood_test": 1, "val": 2, "train": 3}
        out = [{"id": pid, "name": p.name, "archetype": p.archetype, "split": self.ds.splits[pid],
                "contacts": [c.name for c in p.contacts]} for pid, p in self.ds.personas.items()]
        return sorted(out, key=lambda x: (order[x["split"]], x["id"]))

    def run(self, body: dict) -> dict[str, Any]:
        pid, system = body["persona_id"], body.get("system", "model_a")
        records = self.ds.records[pid]
        start = datetime.fromisoformat(records[0]["timestamp"]).replace(hour=0, minute=0)
        hh, mm = (int(x) for x in body.get("time", "09:00").split(":"))
        now = start + timedelta(days=int(body.get("day", 25)) - 1, hours=hh, minutes=mm)
        perms = {r["text"].split(":")[0]: r["permission_label"] for r in records if r["channel"] == "permission"}
        shell = Shell(persona=self.ds.personas[pid], permissions=perms, now=start)
        agent = factory(system, self.ds, self.model, seed=11)(shell, self.ds.personas[pid], self.ds.extras[pid])
        replayed = deletions = 0
        with self.lock:  # one request at a time: one model on an 8 GB machine
            for r in records:
                t = datetime.fromisoformat(r["timestamp"])
                if t >= now:
                    break
                lab = self.ds.labels[r["event_id"]]
                if lab.get("checker"):
                    continue  # benchmark questions are not part of the user's history
                shell.now = t
                shell.observe(r)
                agent.observe(r)
                replayed += 1
                if lab.get("provenance", "").startswith("delete:"):
                    agent.run(r["text"])
                    deletions += 1
            shell.now = now
            n_log, n_sent, n_created, n_conf = len(shell.log), len(shell.sent), len(shell.created), shell.confirmations
            calls = len(agent.profile.calls)
            blocks = dict(agent.profile.blocks)
            t0 = time.perf_counter()
            res = agent.run(body["request"])
            wall = time.perf_counter() - t0
        from p17sim.system.llm import Profile

        prof = Profile(calls=agent.profile.calls[calls:], blocks={k: v - blocks.get(k, 0) for k, v in
                                                                  agent.profile.blocks.items()}).summary()
        return {"answer": res.answer, "trace": res.trace, "tool_calls": shell.log[n_log:], "sent": shell.sent[n_sent:],
                "created": shell.created[n_created:], "interventions": shell.confirmations - n_conf,
                "latency_s": round(wall, 2), "profile": prof, "replayed": replayed, "deletions_handled": deletions,
                "now": now.isoformat()}


def serve(trace: Path, data: Path, model: str = "qwen3.5:4b", port: int = 8017) -> None:
    demo = Demo(Dataset.load(trace, data), model)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/personas":
                self._send(200, json.dumps(demo.personas()).encode(), "application/json")
            else:
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/api/run":
                return self._send(404, b"{}", "application/json")
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                out = demo.run(body)
            except Exception as exc:  # shown in the page instead of a blank screen
                out = {"error": f"{type(exc).__name__}: {exc}"}
            self._send(200, json.dumps(out, default=str).encode(), "application/json")

        def log_message(self, *args):
            pass

    print(f"Serving on http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
