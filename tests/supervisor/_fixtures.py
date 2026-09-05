"""Shared builders and probes for the supervisor event tests.

These helpers are reused by later supervisor suites (the event log and the
stdio driver), so they live in one module rather than being duplicated per
test file. ``collecting_observer`` hands back an appending observer paired with
the list it fills; ``make_launch`` builds a fully-populated ``LaunchEvent`` from
overridable defaults; ``read_log_lines`` parses a JSONL log into dicts.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, NamedTuple

from gymrat.config import BenchlessConfig, Effort
from gymrat.session.clock import now_ms
from gymrat.supervisor.context import SupervisedSession
from gymrat.supervisor.driver import SessionPrompt
from gymrat.supervisor.events import DirtyInfo, LaunchEvent, SessionEvent, SessionObserver


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
    effort: Effort | None = None,
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
        effort=effort,
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
    effort: Effort | None = None,
    command_timeout_ms: int | None = None,
) -> SessionPrompt:
    """Build a ``SessionPrompt`` from shared defaults, overridden per keyword."""
    return SessionPrompt(
        kickoff=kickoff,
        cwd=cwd,
        system_prompt_append=system_prompt_append,
        model=model,
        effort=effort,
        command_timeout_ms=command_timeout_ms,
    )


def noop_observer() -> SessionObserver:
    """Return an observer that discards every event it receives."""

    def _observer(_event: SessionEvent) -> None:
        return None

    return _observer


def result_message(
    *,
    subtype: str = "success",
    is_error: bool = False,
    num_turns: int = 1,
    total_cost_usd: float | None = None,
    result: str | None = None,
) -> SimpleNamespace:
    """Build a result message shaped like the SDK's ``ResultMessage``.

    A result message is identified by having both ``subtype`` and ``num_turns``
    attributes; a system message has ``subtype`` alone.
    """
    return SimpleNamespace(
        subtype=subtype,
        is_error=is_error,
        num_turns=num_turns,
        total_cost_usd=total_cost_usd,
        result=result,
    )


def system_message(*, subtype: str = "init") -> SimpleNamespace:
    """Build a system message (has ``subtype`` but lacks ``num_turns``)."""
    return SimpleNamespace(subtype=subtype)


def _default_benchless_config() -> BenchlessConfig:
    """A minimal ``BenchlessConfig`` for tests that need a context but not a real config."""
    return BenchlessConfig(
        adapter="mitata",
        samples=1,
        timeout_seconds=60,
        unstable_noise_pct=5.0,
        primary="geomean",
        runbook=None,
        stop=None,
    )


def make_context(
    *,
    root: str = "/tmp/test-repo",
    log_path: str = "/tmp/events.jsonl",
    lock_path: str = "/tmp/test-repo/.gymrat/lockfile",
    config: BenchlessConfig | None = None,
    deadline_ms: float | None = None,
    max_minutes: float = 10,
    max_usd: float | None = None,
) -> SupervisedSession:
    """Build a ``SupervisedSession`` from shared defaults, overridden per keyword.

    ``deadline_ms`` defaults to ``now_ms() + max_minutes * 60_000`` when not
    supplied, matching the computation ``_run_session`` performs.
    """
    if deadline_ms is None:
        deadline_ms = now_ms() + max_minutes * 60_000
    return SupervisedSession(
        root=root,
        log_path=log_path,
        lock_path=lock_path,
        config=config if config is not None else _default_benchless_config(),
        deadline_ms=deadline_ms,
        max_minutes=max_minutes,
        max_usd=max_usd,
    )
