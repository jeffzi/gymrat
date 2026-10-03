"""Tests for log-to-spans replay: session, run, and command spans from JSONL logs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest

from gymrat.session.records import HookRecord, record_to_wire
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    UsageUpdateEvent,
    to_json_line,
)
from gymrat.telemetry.ids import span_id_of, trace_id_of
from gymrat.telemetry.replay import replay_session
from tests.session.records._fixtures import (
    SESSION_ID,
    baseline_record,
    command_record,
    iteration_record,
    session_record,
    write_session_log,
)
from tests.session.records._wire import with_raw_number
from tests.telemetry._fixtures import (
    isolate_tracing_provider as _isolate_tracing_provider,  # noqa: F401 -- registers the autouse fixture
)
from tests.telemetry._fixtures import memory_tracing
from tests.telemetry._replay_logs import (
    HEAD_SHA,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    launch_event,
    turn_end,
    write_lines,
    write_records_log,
    write_supervisor_log,
)


def _write_standard_run(sup_log: str) -> None:
    """Write a launch/turn_end pair spanning the standard run window (``T1`` to ``T3``)."""
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T3)])


def _command(
    name: str,
    *,
    at: int = T2,
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
    header = session_record(at=T0)
    write_records_log(session_log, [header])
    _write_standard_run(sup_log)


def _write_measure_command_run(session_log: str, sup_log: str) -> None:
    """Write a session log with a single ``measure`` command, under the standard run window."""
    header = session_record(at=T0)
    cmd = _command("measure")
    write_records_log(session_log, [header, cmd])
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
    header = session_record(at=T0)
    write_records_log(session_log, [header])
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T2)])

    spans = _replay(session_log, sup_log)
    session_span = _span_by_name(spans, "gymrat.session")
    assert session_span.context.trace_id == trace_id_of(SESSION_ID)
    assert session_span.context.span_id == span_id_of(SESSION_ID, "session")


def test_replay_session_when_called_does_set_session_span_timing(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    _write_basic_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    session_span = _span_by_name(spans, "gymrat.session")
    assert session_span.start_time == T0
    assert session_span.end_time == T3


# ---------------------------------------------------------------------------
# run spans
# ---------------------------------------------------------------------------


def test_replay_session_when_supervisor_log_exists_does_create_run_span(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    _write_basic_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    assert run_span.context.span_id == span_id_of(SESSION_ID, f"run:{T1}")
    assert run_span.start_time == T1
    assert run_span.end_time == T3


def test_replay_session_when_run_span_created_does_set_run_attributes(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    launch = launch_event(at=T1, max_usd=5.0, effort="high", model="claude-sonnet-4-20250514")
    write_records_log(session_log, [header])
    write_supervisor_log(sup_log, [launch, turn_end(at=T3)])

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    attrs = dict(run_span.attributes)
    assert attrs["gymrat.session.id"] == SESSION_ID
    assert attrs["gymrat.run.head_sha"] == HEAD_SHA
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
    header = session_record(at=T0)
    write_records_log(session_log, [header])
    write_supervisor_log(
        sup_log,
        [launch_event(at=T1), turn_end(at=T2, cost_usd=0.10), turn_end(at=T3, cost_usd=0.42)],
    )

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    assert run_span.attributes["gymrat.run.cost_usd"] == pytest.approx(0.42)


def test_replay_session_when_turn_end_in_supervisor_does_mirror_session_cost_onto_span_event(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T3, cost_usd=0.42)])

    spans = _replay(session_log, sup_log)

    run_span = _span_by_name(spans, "gymrat.run")
    turn_ends = [ev for ev in run_span.events if ev.name == "gymrat.turn_end"]
    assert [dict(ev.attributes) for ev in turn_ends] == [
        {
            "gymrat.turn.session_cost_usd": pytest.approx(0.42),
            "gymrat.turn.origin": "agent",
            "gymrat.turn.budget_exhausted": False,
        }
    ]


def test_replay_session_when_usage_update_in_supervisor_does_set_cost_usd(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    usage = UsageUpdateEvent(at=T3, cost_usd=1.23, settled=True)
    write_records_log(session_log, [header])
    write_supervisor_log(sup_log, [launch_event(at=T1), usage])

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    assert run_span.attributes["gymrat.run.cost_usd"] == pytest.approx(1.23)


def test_replay_session_when_supervisor_events_present_does_mirror_onto_run_span(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    write_records_log(session_log, [header])
    write_supervisor_log(
        sup_log,
        [
            launch_event(at=T1),
            turn_end(at=T2),
            FollowUpEvent(at=T3, action="replied", reason="continue"),
            CapEvent(at=T4, cap="wall-clock", action="interrupting"),
            CompactionEvent(at=T5),
        ],
    )

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    mirrored = {ev.name: dict(ev.attributes) for ev in run_span.events}
    assert "gymrat.turn_end" in mirrored
    assert mirrored["gymrat.follow_up"] == {
        "gymrat.follow_up.action": "replied",
        "gymrat.follow_up.reason": "continue",
    }
    assert mirrored["gymrat.cap"] == {"gymrat.cap.name": "wall-clock"}
    assert mirrored["gymrat.compaction"] == {}


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
    assert cmd_span.end_time == T2
    assert cmd_span.start_time == T2 - 500 * 1_000_000


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
    ("exit_code", "reason", "expected_status", "expected_description"),
    [
        pytest.param(0, None, "OK", None, id="exit-0-ok"),
        pytest.param(2, "error", "ERROR", "error", id="exit-2-error-with-its-reason"),
        pytest.param(1, "gating-block", "UNSET", None, id="exit-1-unset"),
    ],
)
def test_replay_session_when_command_exit_code_does_set_span_status(
    log_paths: tuple[str, str],
    exit_code: int,
    reason: str | None,
    expected_status: str,
    expected_description: str | None,
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    cmd = _command("iterate", duration_ms=100, exit_code=exit_code, reason=reason)
    write_records_log(session_log, [header, cmd])
    _write_standard_run(sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.iterate")
    from opentelemetry.trace import StatusCode

    assert (cmd_span.status.status_code, cmd_span.status.description) == (
        getattr(StatusCode, expected_status),
        expected_description,
    )


def test_replay_session_when_command_has_traceparent_does_add_link(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    cmd = _command("measure", traceparent=traceparent)
    write_records_log(session_log, [header, cmd])
    _write_standard_run(sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    assert len(cmd_span.links) == 1


def test_replay_session_when_command_in_run_range_does_parent_under_run(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    cmd = _command("measure", duration_ms=100)
    write_records_log(session_log, [header, cmd])
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T4)])

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    run_span = _span_by_name(spans, "gymrat.run")
    assert cmd_span.parent is not None
    assert cmd_span.parent.span_id == run_span.context.span_id


def test_replay_session_when_command_outside_run_range_does_parent_under_session(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    # Command at is after run end time
    cmd = _command("measure", at=T5, duration_ms=100)
    write_records_log(session_log, [header, cmd])
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T2)])

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    session_span = _span_by_name(spans, "gymrat.session")
    assert cmd_span.parent is not None
    assert cmd_span.parent.span_id == session_span.context.span_id


def test_replay_session_when_a_record_outlasts_every_run_does_end_the_session_span_at_it(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    cmd = _command("measure", at=T5, duration_ms=100)
    write_records_log(session_log, [session_record(at=T0), cmd])
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T2)])

    spans = _replay(session_log, sup_log)

    assert _span_by_name(spans, "gymrat.session").end_time == T5


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
        status="OK",
        status_description=None,
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
    header = session_record(at=T0)
    cmd1 = _command("measure", duration_ms=100, seq=1)
    iter_rec = iteration_record(at=T2)
    cmd2 = _command("iterate", duration_ms=100, seq=2)
    write_records_log(session_log, [header, cmd1, iter_rec, cmd2])
    write_supervisor_log(sup_log, [launch_event(at=T0), turn_end(at=T4)])

    spans = _replay(session_log, sup_log)
    cmd_span = _span_by_name(spans, "gymrat.command.iterate")
    event_names = [ev.name for ev in cmd_span.events]
    assert "gymrat.iteration" in event_names


def test_replay_session_when_records_before_first_command_does_add_events_on_session_span(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    baseline = baseline_record(at=T1)
    write_records_log(session_log, [header, baseline])
    write_supervisor_log(sup_log, [launch_event(at=T0), turn_end(at=T3)])

    spans = _replay(session_log, sup_log)
    session_span = _span_by_name(spans, "gymrat.session")
    event_names = [ev.name for ev in session_span.events]
    assert "gymrat.baseline" in event_names


def test_replay_session_when_pre_command_records_followed_by_command_does_attach_to_session_not_command(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)

    baseline = baseline_record(at=T1)
    hook = HookRecord(
        type="hook",
        at=T1,
        stage="before",
        seq=1,
        exit_code=0,
        duration_ms=50,
        stdout_bytes=10,
        timed_out=False,
    )
    cmd = _command("measure", at=T2, duration_ms=100)
    write_records_log(session_log, [header, baseline, hook, cmd])
    write_supervisor_log(sup_log, [launch_event(at=T0), turn_end(at=T4)])

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
    header = session_record(at=T0)
    cmd = _command("measure", duration_ms=100)
    write_records_log(session_log, [header, cmd])
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
    header = session_record(at=T0)
    cmd = _command("measure", duration_ms=100)
    write_records_log(session_log, [header, cmd])
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
    header = session_record(at=T0)
    write_records_log(session_log, [header])
    with Path(sup_log).open("w", encoding="utf-8") as fh:
        fh.write(to_json_line(launch_event(at=T1)) + "\n")
        fh.write('{"type": "unknown_future_event"}\n')
        fh.write(to_json_line(turn_end(at=T3)) + "\n")

    spans = _replay(session_log, sup_log)
    run_span = _span_by_name(spans, "gymrat.run")
    assert run_span is not None


def test_replay_session_when_supervisor_line_holds_non_finite_number_does_skip_it_silently(
    log_paths: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    write_lines(
        sup_log,
        [
            to_json_line(launch_event(at=T1)),
            to_json_line(turn_end(at=T2, cost_usd=0.10)),
            with_raw_number(to_json_line(turn_end(at=T3)), ("cost_usd",), "NaN"),
        ],
    )

    with caplog.at_level(logging.WARNING, logger=_REPLAY_LOGGER):
        spans = _replay(session_log, sup_log)

    run_span = _span_by_name(spans, "gymrat.run")
    assert run_span.attributes["gymrat.run.cost_usd"] == pytest.approx(0.10)
    assert [r for r in caplog.records if r.name == _REPLAY_LOGGER] == []


def test_replay_session_when_supervisor_first_line_malformed_does_ignore_the_log(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    write_lines(sup_log, ["{not json", to_json_line(turn_end(at=T3))])

    spans = _replay(session_log, sup_log)

    assert _spans_by_prefix(spans, "gymrat.run") == []


# ---------------------------------------------------------------------------
# supervisor log without matching session_id is skipped
# ---------------------------------------------------------------------------


def test_replay_session_when_launch_session_id_differs_does_skip_log(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    write_records_log(session_log, [header])
    write_supervisor_log(sup_log, [launch_event(at=T1, session_id="other-session")])

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


_ITERATION_LINE = json.dumps(record_to_wire(iteration_record(at=T1)))


@pytest.mark.parametrize(
    "bad_line",
    [
        pytest.param(_ITERATION_LINE[:-1], id="malformed-json"),
        pytest.param(
            with_raw_number(_ITERATION_LINE, ("primary", "delta_pct"), "NaN"), id="nan-literal"
        ),
    ],
)
def test_replay_session_when_session_line_unparseable_does_log_warning_with_line_number(
    log_paths: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
    bad_line: str,
) -> None:
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    cmd = _command("measure")
    write_lines(
        session_log,
        [json.dumps(record_to_wire(header)), bad_line, json.dumps(record_to_wire(cmd))],
    )
    _write_standard_run(sup_log)

    with (
        memory_tracing(SESSION_ID) as exporter,
        caplog.at_level(logging.WARNING, logger=_REPLAY_LOGGER),
    ):
        replay_session(session_log, [sup_log])

    _assert_warning_mentions(caplog.records, "skipping line 2 (invalid JSON)", "line 2")
    spans = exporter.get_finished_spans()
    _span_by_name(spans, "gymrat.command.measure")
    assert all(ev.name != "gymrat.iteration" for span in spans for ev in span.events)


def _wire_command_with_unknown_field(**overrides: object) -> dict[str, object]:
    """A command wire dict carrying an unrecognized field, forcing the model_construct fallback."""
    wire_cmd: dict[str, object] = {
        "type": "command",
        "at": T2,
        "name": "measure",
        "args": {},
        "exit_code": 0,
        "duration_ms": 500,
        "unknown_future_field": "triggers-extra-forbid",
    }
    wire_cmd.update(overrides)
    return wire_cmd


@pytest.mark.parametrize(
    ("optional_fields", "expected"),
    [
        pytest.param({}, (None, None, 0), id="absent-default-to-none"),
        pytest.param(
            {
                "reason": "budget-exceeded",
                "seq": 3,
                "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            },
            ("budget-exceeded", 3, 1),
            id="present-kept",
        ),
    ],
)
def test_replay_session_when_command_has_unknown_field_does_keep_optional_fields(
    log_paths: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
    optional_fields: dict[str, object],
    expected: tuple[str | None, int | None, int],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    wire_cmd = _wire_command_with_unknown_field(**optional_fields)
    with Path(session_log).open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(record_to_wire(header)) + "\n")
        fh.write(json.dumps(wire_cmd) + "\n")
    _write_standard_run(sup_log)

    with caplog.at_level(logging.WARNING, logger=_REPLAY_LOGGER):
        spans = _replay(session_log, sup_log)

    cmd_span = _span_by_name(spans, "gymrat.command.measure")
    observed = (
        cmd_span.attributes.get("gymrat.command.reason"),
        cmd_span.attributes.get("gymrat.iteration.seq"),
        len(cmd_span.links),
    )
    assert observed == expected
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

    header = session_record(at=T0)
    cmd = _command("measure", at=T2, duration_ms=100)

    write_records_log(session_clean, [header, cmd])

    with Path(session_dirty).open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(record_to_wire(header)) + "\n")
        fh.write("not valid json\n")
        fh.write(json.dumps(record_to_wire(cmd)) + "\n")

    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T4)])

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
):
    from gymrat.command_run import CommandTrace, with_repo_lock
    from gymrat.session.paths import session_jsonl_path

    header = session_record()
    write_session_log(repo, header)

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
    from gymrat.command_run import CommandTrace, with_repo_lock
    from gymrat.session.paths import session_jsonl_path

    header = session_record()
    write_session_log(repo, header)

    # Build a GYMRAT_TRACEPARENT from deterministic IDs so live and replay agree
    run_trace = trace_id_of(header.session_id)
    run_span = span_id_of(header.session_id, f"run:{T0}")
    run_traceparent = f"00-{run_trace:032x}-{run_span:016x}-01"
    monkeypatch.setenv("GYMRAT_TRACEPARENT", run_traceparent)

    jsonl_path = session_jsonl_path(repo)

    # Write a supervisor log that covers the command's time
    sup_dir = tmp_path_factory.mktemp("parity-sup")
    sup_log = str(sup_dir / "run.jsonl")
    write_supervisor_log(
        sup_log,
        [launch_event(session_id=header.session_id, at=T0), turn_end(at=T5)],
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
