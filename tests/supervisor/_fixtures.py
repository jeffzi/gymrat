"""Shared builders and probes for the supervisor event tests.

These helpers are reused by later supervisor suites (the event log and the
stdio driver), so they live in one module rather than being duplicated per
test file. ``collecting_observer`` hands back an appending observer paired with
the list it fills; ``make_launch`` builds a fully-populated ``LaunchEvent`` from
overridable defaults; ``read_log_lines`` parses a JSONL log into dicts.
"""

import json
from pathlib import Path
from typing import NamedTuple

from gymrat_py.supervisor.driver import SessionPrompt
from gymrat_py.supervisor.events import LaunchEvent, SessionEvent, SessionObserver


class ObserverProbe(NamedTuple):
    """An observer paired with the list it appends every received event to."""

    events: list[SessionEvent]
    observer: SessionObserver


def collecting_observer() -> ObserverProbe:
    """Return an observer that records each event it receives, and its list."""
    events: list[SessionEvent] = []
    return ObserverProbe(events, events.append)


def make_launch(**overrides: object) -> LaunchEvent:
    """Build a ``LaunchEvent`` from shared defaults, overridden per keyword."""
    params: dict[str, object] = {
        "timestamp": 1000,
        "head_sha": "abc123def",
        "dirty": False,
        "max_minutes": 5,
        "max_usd": None,
        "model": None,
        "runbook_path": "/path/to/runbook.md",
        "kickoff_summary": "test kickoff",
    }
    params.update(overrides)
    return LaunchEvent(**params)  # type: ignore[arg-type]


def read_log_lines(log_path: str | Path) -> list[dict[str, object]]:
    """Parse a JSONL log file into a list of decoded JSON objects."""
    text = Path(log_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def make_prompt(**overrides: object) -> SessionPrompt:
    """Build a ``SessionPrompt`` from shared defaults, overridden per keyword."""
    params: dict[str, object] = {"kickoff": "do the thing", "cwd": "/tmp/test"}
    params.update(overrides)
    return SessionPrompt(**params)  # type: ignore[arg-type]


def noop_observer() -> SessionObserver:
    """Return an observer that discards every event it receives."""

    def _observer(_event: SessionEvent) -> None:
        return None

    return _observer
