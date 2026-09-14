"""Supervisor: the session event vocabulary and the observers that consume it."""

from gymrat.supervisor.claude import ClientFactory, create_claude_driver
from gymrat.supervisor.context import SupervisedSession
from gymrat.supervisor.driver import (
    Driver,
    DriverSession,
    SessionEndReason,
    SessionOutcome,
    SessionPrompt,
)
from gymrat.supervisor.event_log import create_event_log_writer
from gymrat.supervisor.events import (
    SUMMARY_MAX_CHARS,
    CapAction,
    CapEvent,
    CapType,
    CompactionEvent,
    DirtyInfo,
    FollowUpEvent,
    LaunchEvent,
    ModelPhaseEvent,
    SessionEvent,
    SessionObserver,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    combine_observers,
    event_from_wire,
    summarize,
    summarize_input,
    to_json_line,
)
from gymrat.supervisor.kickoff import KickoffResult, compose_kickoff
from gymrat.supervisor.stdio import create_stdio_driver
from gymrat.supervisor.supervise import SupervisionResult, supervise
from gymrat.supervisor.tools import ToolsFactory, create_gymrat_tools, gymrat_tools_factory

__all__ = [
    "SUMMARY_MAX_CHARS",
    "CapAction",
    "CapEvent",
    "CapType",
    "ClientFactory",
    "CompactionEvent",
    "DirtyInfo",
    "Driver",
    "DriverSession",
    "FollowUpEvent",
    "KickoffResult",
    "LaunchEvent",
    "ModelPhaseEvent",
    "SessionEndReason",
    "SessionEvent",
    "SessionObserver",
    "SessionOutcome",
    "SessionPrompt",
    "SupervisedSession",
    "SupervisionResult",
    "TextDeltaEvent",
    "ThinkingUpdateEvent",
    "ToolEndEvent",
    "ToolProgressEvent",
    "ToolStartEvent",
    "ToolsFactory",
    "TurnEndEvent",
    "UsageUpdateEvent",
    "combine_observers",
    "compose_kickoff",
    "create_claude_driver",
    "create_event_log_writer",
    "create_gymrat_tools",
    "create_stdio_driver",
    "event_from_wire",
    "gymrat_tools_factory",
    "summarize",
    "summarize_input",
    "supervise",
    "to_json_line",
]
