"""Supervisor: the session event vocabulary and the observers that consume it."""

from gymrat_py.supervisor.claude import ClientFactory, create_claude_driver
from gymrat_py.supervisor.driver import (
    Driver,
    DriverSession,
    SessionEndReason,
    SessionOutcome,
    SessionPrompt,
)
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
from gymrat_py.supervisor.supervise import SupervisionResult, supervise

__all__ = [
    "SUMMARY_MAX_CHARS",
    "CapEvent",
    "ClientFactory",
    "DirtyInfo",
    "Driver",
    "DriverSession",
    "KickoffResult",
    "LaunchEvent",
    "SessionEndReason",
    "SessionEvent",
    "SessionObserver",
    "SessionOutcome",
    "SessionPrompt",
    "SupervisionResult",
    "TextDeltaEvent",
    "ThinkingUpdateEvent",
    "ToolEndEvent",
    "ToolProgressEvent",
    "ToolStartEvent",
    "UsageUpdateEvent",
    "combine_observers",
    "compose_kickoff",
    "create_claude_driver",
    "create_event_log_writer",
    "summarize",
    "summarize_input",
    "supervise",
    "to_json_line",
]
