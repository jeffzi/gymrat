"""Tests for log-to-spans replay: session, run, and command spans from JSONL logs."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import create_autospec

import pytest

from gymrat.session.records import BaselineRecord, HookRecord, record_to_wire
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    LaunchEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    to_json_line,
)
from gymrat.telemetry.ids import span_id_of, trace_id_of
from gymrat.telemetry.replay import replay_session
from tests.session.records._fixtures import (
    AT,
    SESSION_ID,
    command_record,
    iteration_record,
    session_record,
    write_session_log,
)
from tests.telemetry._fixtures import memory_tracing

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _isolate_tracing_provider() -> Iterator[None]:
    """Reset the telemetry provider singleton between tests."""
    yield
    from gymrat.telemetry.provider import _reset_for_tests

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _reset_for_tests()


# Nanosecond offsets for deterministic ordering.
_T0 = AT
_T1 = _T0 + 1_000_000_000
_T2 = _T1 + 1_000_000_000
_T3 = _T2 + 1_000_000_000
_T4 = _T3 + 1_000_000_000
_T5 = _T4 + 1_000_000_000

_HEAD_SHA = "a" * 40


def _write_session_log(path: str, records: list[Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(record_to_wire(rec)) + "\n" for rec in records)


def _write_supervisor_log(path: str, events: list[Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        fh.writelines(to_json_line(ev) + "\n" for ev in events)


def _launch_event(
    session_id: str = SESSION_ID,
    at: int = _T0,
    head_sha: str = _HEAD_SHA,
    **kwargs: Any,
) -> LaunchEvent:
    return LaunchEvent(
        at=at,
        schema_version=1,
        session_id=session_id,
        head_sha=head_sha,
        dirty=False,
        max_minutes=60.0,
        runbook_path="/dev/null",
        kickoff_summary="test",
        **kwargs,
    )


def _turn_end(at: int = _T3, cost_usd: float = 0.42) -> TurnEndEvent:
    return TurnEndEvent(
        at=at,
        text="done",
        cost_usd=cost_usd,
        origin="agent",
        budget_exhausted=False,
    )


def _write_standard_run(sup_log: str) -> None:
    """Write a launch/turn_end pair spanning the standard run window (``_T1`` to ``_T3``)."""
    _write_supervisor_log(sup_log, [_launch_event(at=_T1), _turn_end(at=_T3)])


def _command(
    name: str,
    *,
    at: int = _T2,
    duration_ms: int = 500,
    exit_code: int = 0,
    reason: Any = None,
    seq: int | None = None,
    **kwargs: Any,
) -> Any:
    return command_record(
        name=name,
        at=at,
        duration_ms=duration_ms,
        exit_code=exit_code,
        reason=reason,
        seq=seq,
        **kwargs,
    )


def _write_basic_run(session_log: str, sup_log: str) -> None:
    """Write a session log with only the header, under the standard run window."""
    header = session_record(at=_T0)
    _write_session_log(session_log, [header])
    _write_standard_run(sup_log)


def _write_measure_command_run(session_log: str, sup_log: str) -> None:
    """Write a session log with a single ``measure`` command, under the standard run window."""
    header = session_record(at=_T0)
    cmd = _command("measure")
    _write_session_log(session_log, [header, cmd])
    _write_standard_run(sup_log)


def _replay(session_log: str, sup_log: str) -> tuple[Any, ...]:
    with memory_tracing(SESSION_ID) as exporter:
        replay_session(session_log, [sup_log])
    return exporter.get_finished_spans()


@pytest.fixture
def log_paths(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """Return (session_log, sup_log) paths under a fresh temp directory."""
    tmp_dir = tmp_path_factory.mktemp("replay")
    return str(tmp_dir / "session.jsonl"), str(tmp_dir / "run.jsonl")


def _span_by_name(spans: tuple[Any, ...] | list[Any], name: str) -> Any:
    matches = [s for s in spans if s.name == name]
    assert len(matches) == 1, f"expected 1 span named {name!r}, got {len(matches)}"
    return matches[0]


def _spans_by_prefix(spans: tuple[Any, ...] | list[Any], prefix: str) -> list[Any]:
    return [s for s in spans if s.name.startswith(prefix)]


def _span_signature(span: Any) -> tuple[str, int, int, int | None, str, tuple[str, ...]]:
    parent_span_id = span.parent.span_id if span.parent else None
    event_names = tuple(sorted(ev.name for ev in span.events))
    status_name = span.status.status_code.name
    return (
        span.name,
        span.context.trace_id,
        span.context.span_id,
        parent_span_id,
        status_name,
        event_names,
    )


# ---------------------------------------------------------------------------
# session span
# ---------------------------------------------------------------------------


def test_replay_session_when_called_does_create_session_span(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    _write_session_log(session_log, [header])
    _write_supervisor_log(sup_log, [_launch_event(at=_T1), _turn_end(at=_T2)])

    spans = _replay(session_log, sup_log)
    session_span = _span_by_name(spans, "gymrat.session")
    assert session_span.context.trace_id == trace_id_of(SESSION_ID)
    assert session_span.context.span_id == span_id_of(SESSION_ID, "session")


def test_replay_session_when_called_does_set_session_span_timing(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    _write_basic_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    session_span = _span_by_name(spans, "gymrat.session")
    assert session_span.start_time == _T0
    assert session_span.end_time == _T3


# ---------------------------------------------------------------------------
# run spans
# ---------------------------------------------------------------------------


def test_replay_session_when_supervisor_log_exists_does_create_run_span(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    _write_basic_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    assert run_span.context.span_id == span_id_of(SESSION_ID, f"run:{_T1}")
    assert run_span.start_time == _T1
    assert run_span.end_time == _T3


def test_replay_session_when_run_span_created_does_set_run_attributes(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    launch = _launch_event(at=_T1, max_usd=5.0, effort="high", model="claude-sonnet-4-20250514")
    _write_session_log(session_log, [header])
    _write_supervisor_log(sup_log, [launch, _turn_end(at=_T3)])

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    attrs = dict(run_span.attributes)
    assert attrs["gymrat.session.id"] == SESSION_ID
    assert attrs["gymrat.run.head_sha"] == _HEAD_SHA
    assert attrs["gymrat.run.max_minutes"] == 60.0
    assert attrs["gymrat.run.max_usd"] == 5.0
    assert attrs["gymrat.run.effort"] == "high"
    assert attrs["gen_ai.request.model"] == "claude-sonnet-4-20250514"
    assert attrs["gen_ai.provider.name"] == "anthropic"


def test_replay_session_when_optional_run_fields_none_does_omit_attributes(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    _write_basic_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    attrs = dict(run_span.attributes)
    assert "gymrat.run.max_usd" not in attrs
    assert "gymrat.run.effort" not in attrs
    assert "gen_ai.request.model" not in attrs


def test_replay_session_when_turn_end_in_supervisor_does_set_cost_usd(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    _write_session_log(session_log, [header])
    _write_supervisor_log(
        sup_log,
        [_launch_event(at=_T1), _turn_end(at=_T2, cost_usd=0.10), _turn_end(at=_T3, cost_usd=0.42)],
    )

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    assert run_span.attributes["gymrat.run.cost_usd"] == pytest.approx(0.42)


def test_replay_session_when_usage_update_in_supervisor_does_set_cost_usd(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    usage = UsageUpdateEvent(at=_T3, cost_usd=1.23, settled=True)
    _write_session_log(session_log, [header])
    _write_supervisor_log(sup_log, [_launch_event(at=_T1), usage])

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    assert run_span.attributes["gymrat.run.cost_usd"] == pytest.approx(1.23)


def test_replay_session_when_supervisor_events_present_does_mirror_onto_run_span(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    _write_session_log(session_log, [header])
    _write_supervisor_log(
        sup_log,
        [
            _launch_event(at=_T1),
            _turn_end(at=_T2),
            FollowUpEvent(at=_T3, action="replied", reason="continue"),
            CapEvent(at=_T4, cap="wall-clock"),
            CompactionEvent(at=_T5),
        ],
    )

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    event_names = [ev.name for ev in run_span.events]
    assert "gymrat.turn_end" in event_names
    assert "gymrat.follow_up" in event_names
    assert "gymrat.cap" in event_names
    assert "gymrat.compaction" in event_names


# ---------------------------------------------------------------------------
# command spans
# ---------------------------------------------------------------------------


def test_replay_session_when_command_record_present_does_create_command_span(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    _write_measure_command_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.context.trace_id == trace_id_of(SESSION_ID)


def test_replay_session_when_command_span_created_does_set_deterministic_id(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    _write_measure_command_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.context.span_id == span_id_of(SESSION_ID, "command:2")


def test_replay_session_when_command_span_created_does_set_timing_from_duration(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    _write_measure_command_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.end_time == _T2
    assert cmd_span.start_time == _T2 - 500 * 1_000_000


def test_replay_session_when_command_span_created_does_use_command_attributes(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    _write_measure_command_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.attributes["gymrat.session.id"] == SESSION_ID
    assert cmd_span.attributes["gymrat.command.name"] == "measure"


@pytest.mark.parametrize(
    ("exit_code", "expected_status"),
    [
        pytest.param(0, "OK", id="exit-0-ok"),
        pytest.param(2, "ERROR", id="exit-2-error"),
        pytest.param(1, "UNSET", id="exit-1-unset"),
    ],
)
def test_replay_session_when_command_exit_code_does_set_span_status(
    log_paths: tuple[str, str],
    exit_code: int,
    expected_status: str,
):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    cmd = _command("iterate", duration_ms=100, exit_code=exit_code)
    _write_session_log(session_log, [header, cmd])
    _write_standard_run(sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.iterate")
    status_code = cmd_span.status.status_code
    from opentelemetry.trace import StatusCode

    expected = getattr(StatusCode, expected_status)
    assert status_code == expected


def test_replay_session_when_command_has_traceparent_does_add_link(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    cmd = _command("measure", traceparent=traceparent)
    _write_session_log(session_log, [header, cmd])
    _write_standard_run(sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    assert len(cmd_span.links) == 1


def test_replay_session_when_command_in_run_range_does_parent_under_run(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    cmd = _command("measure", duration_ms=100)
    _write_session_log(session_log, [header, cmd])
    _write_supervisor_log(sup_log, [_launch_event(at=_T1), _turn_end(at=_T4)])

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    run_span = _span_by_name(spans, "gymrat.run")
    assert cmd_span.parent is not None
    assert cmd_span.parent.span_id == run_span.context.span_id


def test_replay_session_when_command_outside_run_range_does_parent_under_session(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    # Command at is after run end time
    cmd = _command("measure", at=_T5, duration_ms=100)
    _write_session_log(session_log, [header, cmd])
    _write_supervisor_log(sup_log, [_launch_event(at=_T1), _turn_end(at=_T2)])

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    session_span = _span_by_name(spans, "gymrat.session")
    assert cmd_span.parent is not None
    assert cmd_span.parent.span_id == session_span.context.span_id


def test_replay_session_when_command_present_does_delegate_to_command_span_inputs(
    log_paths: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
):
    from gymrat.telemetry import replay as replay_mod
    from gymrat.telemetry.attributes import CommandSpanInputs, command_span_inputs

    session_log, sup_log = log_paths
    _write_measure_command_run(session_log, sup_log)

    marker_attrs: dict[str, str | int | float | bool] = {
        "gymrat.session.id": SESSION_ID,
        "gymrat.command.name": "measure",
        "test.injected": "via-helper",
    }
    fake_inputs = CommandSpanInputs(
        name="gymrat.command.measure",
        key="command:2",
        attributes=marker_attrs,
        link=None,
    )
    mock_helper = create_autospec(command_span_inputs, return_value=fake_inputs)
    monkeypatch.setattr(replay_mod, "command_span_inputs", mock_helper, raising=False)

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")

    assert cmd_span.attributes["test.injected"] == "via-helper"

    mock_helper.assert_called_once()
    _, call_kwargs = mock_helper.call_args
    assert call_kwargs["session_id"] == SESSION_ID
    assert call_kwargs["line_number"] == 2


# ---------------------------------------------------------------------------
# records between commands become events on the correct span
# ---------------------------------------------------------------------------


def test_replay_session_when_records_between_commands_does_add_events_on_command_span(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    cmd1 = _command("measure", duration_ms=100, seq=1)
    iter_rec = iteration_record(at=_T2)
    cmd2 = _command("iterate", duration_ms=100, seq=2)
    _write_session_log(session_log, [header, cmd1, iter_rec, cmd2])
    _write_supervisor_log(sup_log, [_launch_event(at=_T0), _turn_end(at=_T4)])

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.iterate")
    event_names = [ev.name for ev in cmd_span.events]
    assert "gymrat.iteration" in event_names


def test_replay_session_when_records_before_first_command_does_add_events_on_session_span(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    from gymrat.session.records import BaselineRecord

    baseline = BaselineRecord(
        type="baseline",
        at=_T1,
        label="initial",
        samples=({"total_ms": 100},),
    )
    _write_session_log(session_log, [header, baseline])
    _write_supervisor_log(sup_log, [_launch_event(at=_T0), _turn_end(at=_T3)])

    spans = _replay(session_log, sup_log)
    session_span = _span_by_name(spans, "gymrat.session")
    event_names = [ev.name for ev in session_span.events]
    assert "gymrat.baseline" in event_names


def test_replay_session_when_pre_command_records_followed_by_command_does_attach_to_session_not_command(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)

    baseline = BaselineRecord(
        type="baseline", at=_T1, label="initial", samples=({"total_ms": 100},)
    )
    hook = HookRecord(
        type="hook",
        at=_T1,
        stage="before",
        seq=1,
        exit_code=0,
        duration_ms=50,
        stdout_bytes=10,
        timed_out=False,
    )
    cmd = _command("measure", at=_T2, duration_ms=100)
    _write_session_log(session_log, [header, baseline, hook, cmd])
    _write_supervisor_log(sup_log, [_launch_event(at=_T0), _turn_end(at=_T4)])

    spans = _replay(session_log, sup_log)
    session_span = _span_by_name(spans, "gymrat.session")
    cmd_span = _span_by_name(spans, "gymrat.command.measure")

    session_event_names = [ev.name for ev in session_span.events]
    cmd_event_names = [ev.name for ev in cmd_span.events]

    assert "gymrat.baseline" in session_event_names
    assert "gymrat.hook" in session_event_names
    assert "gymrat.baseline" not in cmd_event_names
    assert "gymrat.hook" not in cmd_event_names


# ---------------------------------------------------------------------------
# idempotent replay
# ---------------------------------------------------------------------------


def test_replay_session_when_replayed_twice_does_produce_identical_ids(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    cmd = _command("measure", duration_ms=100)
    _write_session_log(session_log, [header, cmd])
    _write_standard_run(sup_log)

    def run_replay() -> dict[str, tuple[int, int]]:
        with memory_tracing(SESSION_ID) as exporter:
            replay_session(session_log, [sup_log])
            finished = exporter.get_finished_spans()
            return {s.name: (s.context.trace_id, s.context.span_id) for s in finished}  # pyrefly: ignore[missing-attribute]

    first = run_replay()
    second = run_replay()

    assert first == second


# ---------------------------------------------------------------------------
# span count
# ---------------------------------------------------------------------------


def test_replay_session_when_called_does_return_span_count(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    cmd = _command("measure", duration_ms=100)
    _write_session_log(session_log, [header, cmd])
    _write_standard_run(sup_log)

    with memory_tracing(SESSION_ID) as exporter:
        count = replay_session(session_log, [sup_log])

    spans = exporter.get_finished_spans()
    assert count == len(spans)


# ---------------------------------------------------------------------------
# robustness: unrecognized supervisor lines
# ---------------------------------------------------------------------------


def test_replay_session_when_supervisor_line_unrecognized_does_skip_it(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    _write_session_log(session_log, [header])
    with Path(sup_log).open("w", encoding="utf-8") as fh:
        fh.write(to_json_line(_launch_event(at=_T1)) + "\n")
        fh.write('{"type": "unknown_future_event"}\n')
        fh.write(to_json_line(_turn_end(at=_T3)) + "\n")

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    assert run_span is not None


# ---------------------------------------------------------------------------
# supervisor log without matching session_id is skipped
# ---------------------------------------------------------------------------


def test_replay_session_when_launch_session_id_differs_does_skip_log(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    _write_session_log(session_log, [header])
    _write_supervisor_log(sup_log, [_launch_event(at=_T1, session_id="other-session")])

    spans = _replay(session_log, sup_log)
    run_spans = _spans_by_prefix(spans, "gymrat.run")
    assert len(run_spans) == 0


# ---------------------------------------------------------------------------
# observability: warning logs for silent failures
# ---------------------------------------------------------------------------

_REPLAY_LOGGER = "gymrat.telemetry.replay"


def _assert_warning_mentions(
    records: list[logging.LogRecord], substring: str, description: str
) -> None:
    matches = [r for r in records if r.name == _REPLAY_LOGGER and r.levelno == logging.WARNING]
    assert any(substring in r.message for r in matches), (
        f"expected a warning mentioning {description}; got {[r.message for r in matches]}"
    )


def test_replay_session_when_session_line_unparseable_does_log_warning_with_line_number(
    log_paths: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_log, sup_log = log_paths
    header = session_record(at=_T0)
    cmd = _command("measure")

    with Path(session_log).open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(record_to_wire(header)) + "\n")
        fh.write("not valid json\n")
        fh.write(json.dumps(record_to_wire(cmd)) + "\n")
    _write_standard_run(sup_log)

    with (
        memory_tracing(SESSION_ID) as exporter,
        caplog.at_level(logging.WARNING, logger=_REPLAY_LOGGER),
    ):
        replay_session(session_log, [sup_log])

    _assert_warning_mentions(caplog.records, "2", "line 2")

    spans = exporter.get_finished_spans()
    _span_by_name(spans, "gymrat.command.measure")


def test_replay_session_when_command_uses_model_construct_fallback_does_log_warning_with_name(
    log_paths: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_log, sup_log = log_paths
    header = session_record(at=_T0)

    wire_cmd = {
        "type": "command",
        "at": _T2,
        "name": "measure",
        "args": dict[str, object](),
        "exit_code": 0,
        "duration_ms": 500,
        "seq": None,
        "unknown_future_field": "triggers-extra-forbid",
    }

    with Path(session_log).open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(record_to_wire(header)) + "\n")
        fh.write(json.dumps(wire_cmd) + "\n")
    _write_standard_run(sup_log)

    with (
        memory_tracing(SESSION_ID),
        caplog.at_level(logging.WARNING, logger=_REPLAY_LOGGER),
    ):
        replay_session(session_log, [sup_log])

    _assert_warning_mentions(caplog.records, "measure", "'measure'")


# ---------------------------------------------------------------------------
# replay keying by physical line number
# ---------------------------------------------------------------------------


def test_replay_session_when_skipped_line_before_command_does_key_span_id_on_physical_line_number(
    tmp_path_factory: pytest.TempPathFactory,
):
    tmp_dir = tmp_path_factory.mktemp("replay-key")
    session_clean = str(tmp_dir / "clean.jsonl")
    session_dirty = str(tmp_dir / "dirty.jsonl")
    sup_log = str(tmp_dir / "run.jsonl")

    header = session_record(at=_T0)
    cmd = _command("measure", at=_T2, duration_ms=100)

    _write_session_log(session_clean, [header, cmd])

    with Path(session_dirty).open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(record_to_wire(header)) + "\n")
        fh.write("not valid json\n")
        fh.write(json.dumps(record_to_wire(cmd)) + "\n")

    _write_supervisor_log(sup_log, [_launch_event(at=_T1), _turn_end(at=_T4)])

    spans_clean = _replay(session_clean, sup_log)
    cmd_clean = _span_by_name(spans_clean, "gymrat.command.measure")

    spans_dirty = _replay(session_dirty, sup_log)
    cmd_dirty = _span_by_name(spans_dirty, "gymrat.command.measure")

    # The dirty log has the command on physical line 3, clean on line 2.
    # The span ids differ because the physical line number differs.
    assert cmd_clean.context.span_id == span_id_of(SESSION_ID, "command:2")
    assert cmd_dirty.context.span_id == span_id_of(SESSION_ID, "command:3")


# ---------------------------------------------------------------------------
# live-vs-replay parity
# ---------------------------------------------------------------------------


async def test_replay_session_when_two_commands_run_live_does_match_replayed_spans(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from gymrat.cli.shared import CommandTrace, with_repo_lock
    from gymrat.session.paths import session_jsonl_path

    header = session_record()
    write_session_log(repo, header)
    monkeypatch.delenv("TRACEPARENT", raising=False)
    monkeypatch.delenv("GYMRAT_TRACEPARENT", raising=False)

    jsonl_path = session_jsonl_path(repo)

    with memory_tracing(header.session_id) as live_exporter:

        async def body_measure(trace: CommandTrace) -> str:
            return "ok"

        async def body_compare(trace: CommandTrace) -> str:
            return "ok"

        await with_repo_lock("measure", body_measure)
        await with_repo_lock("compare", body_compare)

    live_spans = live_exporter.get_finished_spans()
    live_cmd_spans = _spans_by_prefix(live_spans, "gymrat.command.")

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(jsonl_path, [])

    replay_spans = replay_exporter.get_finished_spans()
    replay_cmd_spans = _spans_by_prefix(replay_spans, "gymrat.command.")

    assert len(live_cmd_spans) == len(replay_cmd_spans), (
        f"live={[s.name for s in live_cmd_spans]} vs replay={[s.name for s in replay_cmd_spans]}"
    )

    live_sigs = {s.name: _span_signature(s) for s in live_cmd_spans}
    replay_sigs = {s.name: _span_signature(s) for s in replay_cmd_spans}

    assert live_sigs == replay_sigs


async def test_replay_session_when_gymrat_traceparent_set_live_does_match_replayed_spans(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
):
    from gymrat.cli.shared import CommandTrace, with_repo_lock
    from gymrat.session.paths import session_jsonl_path

    header = session_record()
    write_session_log(repo, header)
    monkeypatch.delenv("TRACEPARENT", raising=False)

    # Build a GYMRAT_TRACEPARENT from deterministic IDs so live and replay agree
    run_trace = trace_id_of(header.session_id)
    run_span = span_id_of(header.session_id, f"run:{_T0}")
    run_traceparent = f"00-{run_trace:032x}-{run_span:016x}-01"
    monkeypatch.setenv("GYMRAT_TRACEPARENT", run_traceparent)

    jsonl_path = session_jsonl_path(repo)

    # Write a supervisor log that covers the command's time
    sup_dir = tmp_path_factory.mktemp("parity-sup")
    sup_log = str(sup_dir / "run.jsonl")
    _write_supervisor_log(
        sup_log,
        [_launch_event(session_id=header.session_id, at=_T0), _turn_end(at=_T5)],
    )

    with memory_tracing(header.session_id) as live_exporter:

        async def body(trace: CommandTrace) -> str:
            return "ok"

        await with_repo_lock("measure", body)

    live_spans = live_exporter.get_finished_spans()
    live_cmd = next(s for s in live_spans if s.name.startswith("gymrat.command."))

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(jsonl_path, [sup_log])

    replay_spans = replay_exporter.get_finished_spans()
    replay_cmd = next(s for s in replay_spans if s.name.startswith("gymrat.command."))

    assert live_cmd.name == replay_cmd.name
    assert live_cmd.context is not None
    assert replay_cmd.context is not None
    assert live_cmd.context.trace_id == replay_cmd.context.trace_id
    assert live_cmd.context.span_id == replay_cmd.context.span_id
    assert live_cmd.parent is not None
    assert replay_cmd.parent is not None
    assert live_cmd.parent.span_id == replay_cmd.parent.span_id
