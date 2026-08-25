"""Supervisor: the session event vocabulary and the observers that consume it."""

from gymrat_py.supervisor.event_log import create_event_log_writer
from gymrat_py.supervisor.events import (
    SUMMARY_MAX_CHARS,
    CapEvent,
    DirtyInfo,
    LaunchEvent,
    SessionEvent,
    SessionObserver,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    UsageUpdateEvent,
    combine_observers,
    summarize,
    summarize_input,
    to_json_line,
)
from gymrat_py.supervisor.kickoff import KickoffResult, compose_kickoff

__all__ = [
    "SUMMARY_MAX_CHARS",
    "CapEvent",
    "DirtyInfo",
    "KickoffResult",
    "LaunchEvent",
    "SessionEvent",
    "SessionObserver",
    "TextDeltaEvent",
    "ThinkingUpdateEvent",
    "ToolEndEvent",
    "ToolProgressEvent",
    "ToolStartEvent",
    "UsageUpdateEvent",
    "combine_observers",
    "compose_kickoff",
    "create_event_log_writer",
    "summarize",
    "summarize_input",
    "to_json_line",
]
