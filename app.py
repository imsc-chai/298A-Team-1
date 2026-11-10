"""
Multimodal Personal Agent Interactive Dashboard (Zero-Dependency Web UI)
========================================================================
Runs on any Mac using standard Python (no pip install or external packages needed).

Usage:
  python3 "/Users/chai/gemini /data/app.py"

Opens automatically in your browser at http://localhost:8501
Features:
  - Persona selector (Maya, Elena, David, Marcus, Sophia)
  - Custom query input box ("Type any question from your mind")
  - Quick benchmark task picker
  - Live 3-panel visualizer (User & Memory | Chat & Bounded Actions | Behind-the-Scenes Inspector)
"""

import os
import sys
import json
import http.server
import socketserver
import urllib.parse
import webbrowser
import threading
from typing import Dict, Any

DATA_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(DATA_ROOT, "models"))
from model_m1_react import ModelM1ReActAgent

PORT = 8501

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Multimodal Personal Agent Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background: #f8fafc; color: #0f172a; height: 100vh; display: flex; flex-direction: column; }
    header { background: #1e293b; color: white; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; }
    header h1 { font-size: 18px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
    .badge { background: #3b82f6; font-size: 11px; padding: 3px 8px; border-radius: 12px; font-weight: normal; }
    
    .container { display: flex; flex: 1; overflow: hidden; }
    
    /* Left: User & Memory */
    .panel-left { width: 300px; background: #ffffff; border-right: 1px solid #e2e8f0; padding: 16px; overflow-y: auto; }
    .section-title { font-size: 12px; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 8px; letter-spacing: 0.5px; }
    .select-box { width: 100%; padding: 8px; border-radius: 6px; border: 1px solid #cbd5e1; margin-bottom: 16px; font-size: 14px; background: white; }
    .profile-card { background: #f1f5f9; border-radius: 8px; padding: 12px; margin-bottom: 16px; font-size: 13px; line-height: 1.5; }
    .profile-card strong { color: #1e293b; }
    .tag { display: inline-block; background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin: 2px; }
    .tag-danger { background: #fee2e2; color: #991b1b; }
    
    /* Middle: Chat & Action */
    .panel-mid { flex: 1; display: flex; flex-direction: column; background: #ffffff; border-right: 1px solid #e2e8f0; }
    .chat-history { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 16px; }
    .msg { max-width: 85%; padding: 12px 16px; border-radius: 10px; font-size: 14px; line-height: 1.5; }
    .msg-user { align-self: flex-end; background: #2563eb; color: white; border-bottom-right-radius: 2px; }
    .msg-agent { align-self: flex-start; background: #f8fafc; border: 1px solid #e2e8f0; border-bottom-left-radius: 2px; }
    .draft-box { background: #fefce8; border: 1px solid #fef08a; border-radius: 6px; padding: 12px; margin-top: 8px; font-family: monospace; font-size: 12px; color: #854d0e; }
    
    .chat-input-area { padding: 16px; border-top: 1px solid #e2e8f0; background: #f8fafc; }
    .input-row { display: flex; gap: 8px; }
    .text-input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 14px; outline: none; }
    .text-input:focus { border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37,99,235,0.1); }
    .btn-send { background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.15s; }
    .btn-send:hover { background: #1d4ed8; }
    
    /* Right: Behind the Scenes Inspector */
    .panel-right { width: 360px; background: #fdfdfd; padding: 16px; overflow-y: auto; font-size: 13px; }
    .step-item { background: white; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; margin-bottom: 8px; border-left: 4px solid #3b82f6; }
    .step-title { font-weight: 600; color: #1e293b; margin-bottom: 4px; display: flex; justify-content: space-between; }
    .step-args { background: #f8fafc; padding: 6px; border-radius: 4px; font-family: monospace; font-size: 11px; color: #475569; word-break: break-all; }
    .guard-box { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; padding: 10px; border-radius: 6px; margin-bottom: 12px; font-size: 12px; display: flex; align-items: center; gap: 6px; }
    .loading { display: none; color: #64748b; font-size: 13px; padding: 8px; font-style: italic; }
  </style>
</head>
<body>
  <header>
    <h1>Personal Agent Dashboard <span class="badge">Model M1 (ReAct Baseline)</span></h1>
    <div style="font-size: 13px; color: #94a3b8;">Local Inference: <b>qwen3.5:9b</b></div>
  </header>

  <div class="container">
    <!-- LEFT PANEL: USER & MEMORY -->
    <div class="panel-left">
      <div class="section-title">Active Persona</div>
      <select id="personaSelect" class="select-box" onchange="loadPersona()">
        <option value="persona_01_maya">1. Maya Lin (CS Grad & TA)</option>
        <option value="persona_02_elena">2. Elena Gomez (UX Designer)</option>
        <option value="persona_03_david">3. David Kim (ER Nurse & Parent)</option>
        <option value="persona_04_marcus">4. Marcus Vance (Startup CTO)</option>
        <option value="persona_05_sophia">5. Sophia Martinez (AP Bio Teacher)</option>
      </select>

      <div class="section-title">Persistent User Profile</div>
      <div id="profileCard" class="profile-card">Loading profile...</div>

      <div class="section-title">Active Privacy Policy</div>
      <div id="privacyCard" class="profile-card">Loading rules...</div>
    </div>

    <!-- MIDDLE PANEL: CHAT & ACTIONS -->
    <div class="panel-mid">
      <div id="chatHistory" class="chat-history">
        <div class="msg msg-agent">
          Hello! I am your personal assistant. Ask me anything, or pick one of the benchmark evaluation tasks below.
        </div>
      </div>
      <div id="loadingIndicator" class="loading">⏳ Agent is reasoning and executing tools via local Qwen...</div>
      
      <div class="chat-input-area">
        <div style="margin-bottom: 8px; display: flex; gap: 8px; align-items: center;">
          <span style="font-size: 12px; color: #64748b; font-weight: 600;">Benchmark Presets:</span>
          <select id="taskPresetSelect" class="select-box" style="margin-bottom: 0; flex: 1;" onchange="usePreset()">
            <option value="">-- Choose an evaluation task or type custom below --</option>
          </select>
        </div>
        <div class="input-row">
          <input type="text" id="userInput" class="text-input" placeholder="Type any question from your mind (e.g. 'What is my apartment gate code?')..." onkeydown="if(event.key==='Enter') sendQuery()">
          <button class="btn-send" onclick="sendQuery()">Ask Agent</button>
        </div>
      </div>
    </div>

    <!-- RIGHT PANEL: INSPECTOR -->
    <div class="panel-right">
      <div class="section-title">Behind-the-Scenes Inspector</div>
      <div id="guardrailStatus" class="guard-box">
        ✔ Privacy Firewall: Active (Zero Raw PII Export)
      </div>

      <div class="section-title">ReAct Execution Steps</div>
      <div id="stepsContainer">
        <div style="color: #94a3b8; font-style: italic; font-size: 12px;">No steps executed yet. Ask a question to see the agent's thought and tool calls.</div>
      </div>
    </div>
  </div>

  <script>
    let currentPersona = "persona_01_maya";

    async function loadPersona() {
      currentPersona = document.getElementById("personaSelect").value;
      const res = await fetch(`/api/persona?id=${currentPersona}`);
      const data = await res.json();
      
      // Update profile
      let p = data.profile;
      document.getElementById("profileCard").innerHTML = `
        <strong>${p.name}</strong><br>
        <span style="color: #64748b;">${p.role}</span><br><br>
        <b>Preferences:</b><br>
        • Meeting Window: <span class="tag">${p.preferences.meeting_window}</span><br>
        • Buffer: ${p.preferences.buffer_between_meetings_minutes || 15} mins<br>
        • Focus Slots: ${p.preferences.focus_days || 'Friday Deep Work'}
      `;

      // Update privacy rules
      let priv = data.privacy;
      let tags = priv.sensitive_categories.map(c => `<span class="tag tag-danger">${c.category}</span>`).join(" ");
      document.getElementById("privacyCard").innerHTML = `
        <b>Restricted Categories:</b><br>${tags}<br><br>
        <b>Bound:</b> Outbound actions must be DRAFT status only.
      `;

      // Update tasks dropdown
      let taskSelect = document.getElementById("taskPresetSelect");
      taskSelect.innerHTML = `<option value="">-- Choose an evaluation task or type custom below --</option>`;
      data.tasks.forEach(t => {
        let opt = document.createElement("option");
        opt.value = t.user_query;
        opt.innerText = `[${t.task_id}] ${t.name}`;
        taskSelect.appendChild(opt);
      });
    }

    function usePreset() {
      const select = document.getElementById("taskPresetSelect");
      if (select.value) {
        document.getElementById("userInput").value = select.value;
      }
    }

    async function sendQuery() {
      const input = document.getElementById("userInput");
      const query = input.value.trim();
      if (!query) return;

      // Add user message to UI
      const chat = document.getElementById("chatHistory");
      chat.innerHTML += `<div class="msg msg-user">${query}</div>`;
      input.value = "";
      document.getElementById("loadingIndicator").style.display = "block";
      chat.scrollTop = chat.scrollHeight;

      try {
        const res = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: json_stringify({ persona: currentPersona, query: query })
        });
        const data = await res.json();
        document.getElementById("loadingIndicator").style.display = "none";

        // Display steps in inspector
        const stepsContainer = document.getElementById("stepsContainer");
        stepsContainer.innerHTML = "";
        data.steps.forEach(s => {
          stepsContainer.innerHTML += `
            <div class="step-item">
              <div class="step-title">
                <span>Step ${s.step}: ${s.tool}</span>
                <span class="badge" style="background:#64748b;">${s.status}</span>
              </div>
              <div class="step-args">Args: ${JSON.stringify(s.args)}</div>
              <div style="font-size: 11px; color:#475569; margin-top:4px;">Result: ${s.observation.substring(0, 100)}...</div>
            </div>
          `;
        });

        // Add agent response
        let finalAction = data.steps.find(s => s.tool.includes("draft"));
        let replyHtml = `<div class="msg msg-agent">I have processed your request across your local files and calendar.`;
        if (finalAction) {
          replyHtml += ` I prepared a reviewable <b>DRAFT</b> action:<div class="draft-box"><strong>${finalAction.tool.toUpperCase()}:</strong><br>To: ${finalAction.args.to || 'Recipient'}<br>Body: ${finalAction.args.body || ''}</div>`;
        } else if (data.steps.length > 0) {
          let lastStep = data.steps[data.steps.length - 1];
          replyHtml += `<div class="draft-box">Result: ${lastStep.observation}</div>`;
        }
        replyHtml += `</div>`;
        chat.innerHTML += replyHtml;
        chat.scrollTop = chat.scrollHeight;

      } catch (err) {
        document.getElementById("loadingIndicator").style.display = "none";
        chat.innerHTML += `<div class="msg msg-agent" style="color:red;">Error running agent: ${err}</div>`;
      }
    }

    function json_stringify(obj) { return JSON.stringify(obj); }

    // Init on load
    window.onload = loadPersona;
  </script>
</body>
</html>
"""

class AgentServerHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/" or url.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        elif url.path == "/api/persona":
            params = urllib.parse.parse_qs(url.query)
            persona_id = params.get("id", ["persona_01_maya"])[0]
            p_dir = os.path.join(DATA_ROOT, persona_id)
            
            with open(os.path.join(p_dir, "agent_memory", "user_profile.json")) as f:
                profile = json.load(f)
            with open(os.path.join(p_dir, "agent_memory", "privacy_policy.json")) as f:
                privacy = json.load(f)
            with open(os.path.join(p_dir, "test_suite", "eval_tasks.json")) as f:
                tasks = json.load(f)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "profile": profile,
                "privacy": privacy,
                "tasks": tasks
            }).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        if self.path == "/api/run":
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len)
            payload = json.loads(post_data.decode("utf-8"))
            
            persona_id = payload.get("persona", "persona_01_maya")
            query = payload.get("query", "")
            
            # Execute Model M1 on the custom query
            agent = ModelM1ReActAgent(persona_id=persona_id, model_name="qwen3.5:9b")
            custom_task = {
                "task_id": "dashboard_query",
                "name": "Live Dashboard Query",
                "user_query": query,
                "reference_time": "2026-11-09T20:00:00"
            }
            workflow = agent.run_task(custom_task)
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(workflow).encode("utf-8"))
            return

def start_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), AgentServerHandler) as httpd:
        print(f"\n=======================================================")
        print(f"Personal Agent Interactive Dashboard Running!")
        print(f"URL: http://localhost:{PORT}")
        print(f"=======================================================\n")
        httpd.serve_forever()

if __name__ == "__main__":
    # Start server in background thread and launch browser
    threading.Thread(target=start_server, daemon=True).start()
    webbrowser.open(f"http://localhost:{PORT}")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
