"""Shared builders and probes for the supervisor event tests.

These helpers are reused by later supervisor suites (the event log and the
stdio driver), so they live in one module rather than being duplicated per
test file. ``collecting_observer`` hands back an appending observer paired with
the list it fills; ``make_launch`` builds a fully-populated ``LaunchEvent`` from
overridable defaults; ``read_log_lines`` parses a JSONL log into dicts.
"""

import json
from pathlib import Path
from typing import Literal, NamedTuple

from gymrat_py.supervisor.driver import SessionPrompt
from gymrat_py.supervisor.events import DirtyInfo, LaunchEvent, SessionEvent, SessionObserver


class ObserverProbe(NamedTuple):
    """An observer paired with the list it appends every received event to."""

    events: list[SessionEvent]
    observer: SessionObserver


def collecting_observer() -> ObserverProbe:
    """Return an observer that records each event it receives, and its list."""
    events: list[SessionEvent] = []
    return ObserverProbe(events, events.append)


def make_launch(
    *,
    timestamp: int = 1000,
    head_sha: str = "abc123def",
    dirty: Literal[False] | DirtyInfo = False,
    max_minutes: float = 5,
    max_usd: float | None = None,
    model: str | None = None,
    runbook_path: str = "/path/to/runbook.md",
    kickoff_summary: str = "test kickoff",
) -> LaunchEvent:
    """Build a ``LaunchEvent`` from shared defaults, overridden per keyword."""
    return LaunchEvent(
        timestamp=timestamp,
        head_sha=head_sha,
        dirty=dirty,
        max_minutes=max_minutes,
        max_usd=max_usd,
        model=model,
        runbook_path=runbook_path,
        kickoff_summary=kickoff_summary,
    )


def read_log_lines(log_path: str | Path) -> list[dict[str, object]]:
    """Parse a JSONL log file into a list of decoded JSON objects."""
    text = Path(log_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def make_prompt(
    *,
    kickoff: str = "do the thing",
    cwd: str = "/tmp/test",
    system_prompt_append: str | None = None,
    model: str | None = None,
) -> SessionPrompt:
    """Build a ``SessionPrompt`` from shared defaults, overridden per keyword."""
    return SessionPrompt(
        kickoff=kickoff,
        cwd=cwd,
        system_prompt_append=system_prompt_append,
        model=model,
    )


def noop_observer() -> SessionObserver:
    """Return an observer that discards every event it receives."""

    def _observer(_event: SessionEvent) -> None:
        return None

    return _observer
