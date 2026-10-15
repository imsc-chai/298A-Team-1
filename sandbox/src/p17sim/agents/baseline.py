"""Baseline agent: OpenJarvis `native_openhands` (commit e97088f) re-implemented over our tool shell.

What is copied from src/openjarvis/agents/native_openhands.py:
- the system prompt (tool-use instructions and the two general rules);
- the loop: native tool calls first, then the text fallback ("Action:" / "Action Input:", or a
  ```python block routed to a code_interpreter tool), otherwise the reply is the final answer;
- tool output cut at 4,000 characters; `_truncate_if_needed` with a 3,000-token estimate
  (characters / 4) that trims the last user message;
- think tags stripped; "Maximum turns reached." when the turn limit runs out.
Settings follow the baseline run (baseline/SETUP.md): temperature 0.6, 8,192 max tokens, 10 turns.

What differs, and why:
- The tool set is our shell's (search, calendar, email, delete), not PinchBench's (web, shell,
  code). The rules about web_search, URLs, calculator and file_read are dropped because those
  tools do not exist here.
- A context line gives the user's name and the current date and time; every agent gets the same line.
- The model is qwen3.5:4b instead of 9B (8 GB laptop); Model A uses the same model.
- A sampling seed is passed for repeatability; OpenJarvis passes none.
"""
from __future__ import annotations

import json
import re
from typing import Any

from p17sim.schema import Persona
from p17sim.system.llm import Chat
from p17sim.system.runner import AgentResult
from p17sim.system.shell import Shell

OPENHANDS_SYSTEM_PROMPT = (
    "You are an AI assistant with access to tools. "
    "You MUST use tools when they would help answer "
    "the user's question.\n\n"
    "## How to use tools\n\n"
    "To call a tool, write on its own lines:\n\n"
    "Action: <tool_name>\n"
    "Action Input: <json_arguments>\n\n"
    "You will receive the result, then continue your "
    "response.\n\n"
    "## Available tools\n\n"
    "{tool_descriptions}\n\n"
    "## Important rules\n\n"
    "- If no tool or code is needed, respond directly "
    "with your answer.\n"
    "- Do NOT include <think> tags or internal reasoning "
    "in your response. Respond directly."
)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def context_line(shell: Shell, persona: Persona) -> str:
    return (f"\n\n## Context\n\nYou are {persona.name}'s personal assistant. "
            f"Current date and time: {shell.now.strftime('%A, %Y-%m-%d %H:%M')}.")


def strip_think(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def strip_tool_call_text(text: str) -> str:
    text = re.sub(r"Action:\s*.+?(?:Action Input:\s*.+?)?(?=\n\n|\Z)", "", text, flags=re.S | re.I)
    text = re.sub(r"<tool_call>.*?</\w+>", "", text, flags=re.S)
    return text.strip()


def extract_code(text: str) -> str | None:
    m = re.search(r"```python\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else None


def extract_tool_call(text: str) -> tuple[str, str] | None:
    action = re.search(r"Action:\s*(.+)", text, re.I)
    action_input = re.search(r"Action Input:\s*(.+?)(?=\n\n|\Z)", text, re.S | re.I)
    if action:
        return action.group(1).strip(), action_input.group(1).strip() if action_input else "{}"
    xml = re.search(r"<tool_call>\s*(\w+)\s*(.*?)</\w+>", text, re.S)
    if xml:
        params: dict[str, Any] = {}
        for m in re.finditer(r"\$(\w+)=(.+?)(?=\$|\n<|</|$)", xml.group(2).strip(), re.S):
            params[m.group(1)] = m.group(2).strip().rstrip("</>\n")
        for m in re.finditer(r"<(\w+)>(.*?)</\1>", xml.group(2), re.S):
            params[m.group(1)] = m.group(2).strip()
        return xml.group(1).strip(), json.dumps(params)
    return None


def truncate_if_needed(messages: list[dict], max_prompt_tokens: int = 3000) -> list[dict]:
    """OpenJarvis's rule, unchanged: if the estimated prompt is too long, trim the last user message."""
    total = sum(len(m.get("content") or "") for m in messages)
    if total // 4 <= max_prompt_tokens:
        return messages
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            excess = (total // 4 - max_prompt_tokens) * 4
            original = messages[i]["content"]
            if len(original) > excess + 200:
                messages[i] = {"role": "user",
                               "content": original[: len(original) - excess] + "\n\n[Input truncated to fit context window]"}
            break
    return messages


class BaselineAgent:
    name = "baseline"

    def __init__(self, shell: Shell, persona: Persona, llm: Chat, max_turns: int = 10):
        self.shell, self.persona, self.llm, self.max_turns = shell, persona, llm, max_turns
        self.profile = llm.profile

    def observe(self, record: dict) -> None:
        pass  # no memory of its own: it reads the shell's stores through tools

    def memory_holds(self, value: str) -> list[str]:
        return []

    def _execute(self, name: str, args: Any) -> str:
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        obs = self.shell.execute(name, args if isinstance(args, dict) else {})
        return obs[:4000] + "\n\n[Output truncated]" if len(obs) > 4000 else obs

    def run(self, request: str) -> AgentResult:
        system = OPENHANDS_SYSTEM_PROMPT.format(tool_descriptions=self.shell.tool_descriptions())
        system += context_line(self.shell, self.persona)
        messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": request}]
        tools = self.shell.tool_schemas()
        last, trace = "", []
        with self.profile.time("agent_loop"):
            for turn in range(1, self.max_turns + 1):
                messages = truncate_if_needed(messages)
                r = self.llm.chat(messages, tools=tools, component="act")
                content = strip_think(r["content"])
                last = content
                if r["tool_calls"]:
                    messages.append({"role": "assistant", "content": content, "tool_calls": [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in r["tool_calls"]]})
                    for tc in r["tool_calls"]:
                        with self.profile.time("tools"):
                            obs = self._execute(tc["name"], tc["arguments"])
                        trace.append({"turn": turn, "tool": tc["name"], "args": tc["arguments"], "result": obs[:300]})
                        messages.append({"role": "tool", "content": obs, "tool_name": tc["name"]})
                    continue
                code = extract_code(content)
                if code:
                    messages.append({"role": "assistant", "content": content})
                    obs = self._execute("code_interpreter", {"code": code})
                    messages.append({"role": "user", "content": f"Output:\n{obs}"})
                    trace.append({"turn": turn, "tool": "code_interpreter", "result": obs[:300]})
                    continue
                info = extract_tool_call(content)
                if info:
                    messages.append({"role": "assistant", "content": content})
                    with self.profile.time("tools"):
                        obs = self._execute(*info)
                    messages.append({"role": "user", "content": f"Result: {obs}"})
                    trace.append({"turn": turn, "tool": info[0], "args": info[1], "result": obs[:300]})
                    continue
                return AgentResult(answer=strip_tool_call_text(content), turns=turn, trace=trace)
        final = strip_tool_call_text(last) or "Maximum turns reached."
        return AgentResult(answer=final, turns=self.max_turns, trace=trace)
