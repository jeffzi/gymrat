"""Session and supervisor log writers shared by the export and replay tests.

Both suites replay a pair of JSONL logs into spans, so both need the same
ordered timestamps, the same launch and turn-end events, and the same writers
that put records and events on disk in their wire form.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gymrat.session.records import CommandRecord, record_to_wire
from gymrat.supervisor.events import LaunchEvent, TurnEndEvent, to_json_line
from tests.session.records._fixtures import (
    AT,
    BASELINE_SHA,
    SESSION_ID,
    command_record,
    session_record,
)
from tests.supervisor._fixtures import make_launch, make_turn_end

_ONE_SECOND_NS = 1_000_000_000

# Nanosecond offsets for deterministic ordering.
T0 = AT
T1 = T0 + _ONE_SECOND_NS
T2 = T1 + _ONE_SECOND_NS
T3 = T2 + _ONE_SECOND_NS
T4 = T3 + _ONE_SECOND_NS
T5 = T4 + _ONE_SECOND_NS


def write_lines(path: str, lines: list[str]) -> None:
    """Write ``lines`` to ``path``, one per line."""
    Path(path).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def write_records_log(path: str, records: list[Any]) -> None:
    """Write ``records`` in wire form to the session log at ``path``, creating its directory."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_lines(path, [json.dumps(record_to_wire(rec)) for rec in records])


def write_supervisor_log(path: str, events: list[Any]) -> None:
    """Write ``events`` as JSON lines to the supervisor log at ``path``."""
    write_lines(path, [to_json_line(ev) for ev in events])


def replay_launch_event(
    session_id: str = SESSION_ID,
    at: int = T0,
    head_sha: str = BASELINE_SHA,
    **kwargs: Any,
) -> LaunchEvent:
    """Build the launch event that opens a supervisor log."""
    return make_launch(
        at=at,
        session_id=session_id,
        head_sha=head_sha,
        max_minutes=60.0,
        runbook_path="/dev/null",
        kickoff_summary="test",
        **kwargs,
    )


def replay_turn_end(at: int = T3, cost_usd: float = 0.42) -> TurnEndEvent:
    """Build an agent turn-end event carrying ``cost_usd``."""
    return make_turn_end(at=at, text="done", cost_usd=cost_usd)


def replay_command(
    name: str,
    *,
    at: int = T2,
    duration_ms: int = 500,
    exit_code: int = 0,
    reason: Any = None,
    seq: int | None = None,
    **kwargs: Any,
) -> CommandRecord:
    """Build a successful ``name`` command record ending at ``T2``, every field overridable."""
    return command_record(
        name=name,
        at=at,
        duration_ms=duration_ms,
        exit_code=exit_code,
        reason=reason,
        seq=seq,
        **kwargs,
    )


def write_standard_run(sup_log: str, *, session_id: str = SESSION_ID) -> None:
    """Write a launch/turn-end pair spanning the standard run window (``T1`` to ``T3``)."""
    write_supervisor_log(
        sup_log, [replay_launch_event(session_id=session_id, at=T1), replay_turn_end(at=T3)]
    )


def write_measure_command_run(
    session_log: str, sup_log: str, *, session_id: str = SESSION_ID
) -> None:
    """Write a session log with a single ``measure`` command, under the standard run window.

    Args:
        session_log: The session log path; its directory is created.
        sup_log: The supervisor log path.
        session_id: The session both logs belong to.
    """
    header = session_record(session_id=session_id, at=T0)
    write_records_log(session_log, [header, replay_command("measure")])
    write_standard_run(sup_log, session_id=session_id)
