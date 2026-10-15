"""Hand-built workflows: what a person would draw in n8n. Used for demos, triggers and as examples.

Each template is a function of a few parameters that returns a Workflow. Every action that needs the
user's say goes through the tool policy, and `weekly_recap` and `forget` add an explicit approval step.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable

from p17sim.aios.workflow import Edge, Node, Workflow


def _wf(name: str, tid: str, goal: str, nodes: list[tuple[str, str, dict]], trigger: dict | None = None) -> Workflow:
    return Workflow(id=f"{tid}-{uuid.uuid4().hex[:6]}", name=name, source=f"template:{tid}", goal=goal,
                    nodes=[Node(id=i, type=t, params=p) for i, t, p in nodes],
                    edges=[Edge(source=nodes[k][0], target=nodes[k + 1][0]) for k in range(len(nodes) - 1)],
                    trigger=trigger)


def morning_briefing(at: str = "08:00", **_) -> Workflow:
    return _wf("Morning briefing", "morning_briefing", "Summarize today's calendar and yesterday's messages", [
        ("trigger", "trigger.time", {"at": at}),
        ("calendar", "context.calendar", {"date": ""}),
        ("recent", "context.recent", {"days": 1, "kind": "message", "limit": 15}),
        ("summary", "reason.llm", {"prompt": "Write a short morning briefing (at most 5 bullet points) from today's "
                                             "calendar and yesterday's messages.\nCalendar: {{calendar.entries}}\n"
                                             "Messages: {{recent.items}}", "max_tokens": 300}),
        ("answer", "output.respond", {"text": "{{summary.text}}"})], trigger={"kind": "time", "at": at})


def weekly_recap(to: str, **_) -> Workflow:
    return _wf(f"Weekly recap to {to}", "weekly_recap", f"Email {to} a recap of my week", [
        ("trigger", "trigger.chat", {"goal": f"Email {to} a recap of my week"}),
        ("week", "context.recent", {"days": 7, "kind": "all", "exclude_private": True, "limit": 15}),
        ("draft", "reason.llm", {"prompt": "Write a short, friendly email to " + to + " that recaps the user's week "
                                           "from these records. At most four sentences. Leave out codes and private "
                                           "details. Output only the body.\nRecords: {{week.items}}",
                                 "max_tokens": 300}),
        ("approve", "human.approval", {"message": "Send this recap? {{draft.text}}"}),
        ("send", "action.send_email", {"to": to, "subject": "My week", "body": "{{draft.text}}"}),
        ("answer", "output.respond", {"text": "Sent the recap to {{send.to}}."})])


def book_meeting(contact: str, date: str, duration_minutes: int = 30, **_) -> Workflow:
    return _wf(f"Book {duration_minutes} min with {contact}", "book_meeting",
               f"Book {duration_minutes} minutes with {contact} on {date}", [
        ("trigger", "trigger.chat", {"goal": f"Book {duration_minutes} minutes with {contact} on {date}"}),
        ("calendar", "context.calendar", {"date": date}),
        ("prefs", "context.preferences", {"key": "meeting_window", "days": 14}),
        ("slot", "reason.llm", {"prompt": "Pick a start time for a " + str(duration_minutes) + "-minute meeting on "
                                          + date + " that does not clash with the calendar and fits the user's recent "
                                          'preference signals. Reply as JSON {"start": "HH:MM"}.\n'
                                          "Calendar: {{calendar.entries}}\nPreference signals: {{prefs.signals}}",
                                "json": True, "max_tokens": 60}),
        ("book", "action.create_event", {"date": date, "start": "{{slot.json.start}}",
                                         "duration_minutes": duration_minutes, "with_contact": contact}),
        ("answer", "output.respond", {"text": "Booked {{book.title}} on {{book.date}} at {{book.start}}."})])


def forget(query: str, **_) -> Workflow:
    return _wf(f"Forget: {query}", "forget", f"Forget {query}", [
        ("trigger", "trigger.chat", {"goal": f"Forget {query}"}),
        ("find", "context.search", {"query": query, "limit": 5}),
        ("approve", "human.approval", {"message": "Delete these records? {{find.items}}"}),
        ("delete", "action.delete_records", {"ids": "{{find.items}}", "request": f"Forget {query}"}),
        ("answer", "output.respond", {"text": "Deleted {{delete.deleted}}."})])


TEMPLATES: dict[str, tuple[Callable[..., Workflow], str, dict[str, str]]] = {
    "morning_briefing": (morning_briefing, "Every morning: today's calendar and yesterday's messages, summarized.",
                         {"at": "HH:MM"}),
    "weekly_recap": (weekly_recap, "Draft a recap of the week, ask before sending, then email it.",
                     {"to": "contact name"}),
    "book_meeting": (book_meeting, "Check the calendar and preferences, pick a slot, book it.",
                     {"contact": "name", "date": "YYYY-MM-DD", "duration_minutes": "minutes"}),
    "forget": (forget, "Find records about something, ask, then delete them everywhere.", {"query": "what to forget"}),
}


def build(template_id: str, params: dict[str, Any]) -> Workflow:
    if template_id not in TEMPLATES:
        raise KeyError(f"unknown template {template_id}")
    return TEMPLATES[template_id][0](**params)
