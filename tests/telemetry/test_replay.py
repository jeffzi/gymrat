"""Tests for log-to-spans replay: session, run, and command spans from JSONL logs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from gymrat.session.records import HookRecord, record_to_wire
from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    UsageUpdateEvent,
    to_json_line,
)
from gymrat.telemetry.provider import span_id_of, trace_id_of
from gymrat.telemetry.replay import replay_session
from tests.session.records._fixtures import (
    SESSION_ID,
    baseline_record,
    command_record,
    iteration_record,
    session_record,
)
from tests.session.records._wire import with_raw_number
from tests.telemetry._fixtures import (
    isolate_tracing_provider as _isolate_tracing_provider,  # noqa: F401 -- registers the autouse fixture
)
from tests.telemetry._fixtures import memory_tracing, span_by_name, spans_by_prefix
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


# ---------------------------------------------------------------------------
# session span
# ---------------------------------------------------------------------------


def test_replay_session_when_called_does_create_session_span(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    write_records_log(session_log, [header])
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T2)])

    spans = _replay(session_log, sup_log)
    session_span = span_by_name(spans, "gymrat.session")
    assert session_span.context.trace_id == trace_id_of(SESSION_ID)
    assert session_span.context.span_id == span_id_of(SESSION_ID, "session")


def test_replay_session_when_called_does_set_session_span_timing(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    _write_basic_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    session_span = span_by_name(spans, "gymrat.session")
    assert session_span.start_time == T0
    assert session_span.end_time == T3


# ---------------------------------------------------------------------------
# run spans
# ---------------------------------------------------------------------------


def test_replay_session_when_supervisor_log_exists_does_create_run_span(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    _write_basic_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    run_span = span_by_name(spans, "gymrat.run")
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
    run_span = span_by_name(spans, "gymrat.run")
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
    run_span = span_by_name(spans, "gymrat.run")
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
    run_span = span_by_name(spans, "gymrat.run")
    assert run_span.attributes["gymrat.run.cost_usd"] == pytest.approx(0.42)


def test_replay_session_when_turn_end_in_supervisor_does_mirror_session_cost_onto_span_event(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T3, cost_usd=0.42)])

    spans = _replay(session_log, sup_log)

    run_span = span_by_name(spans, "gymrat.run")
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
    usage = UsageUpdateEvent(at=T3, cost_usd=1.23)
    write_records_log(session_log, [header])
    write_supervisor_log(sup_log, [launch_event(at=T1), usage])

    spans = _replay(session_log, sup_log)
    run_span = span_by_name(spans, "gymrat.run")
    assert run_span.attributes["gymrat.run.cost_usd"] == pytest.approx(1.23)


def test_replay_session_when_usage_update_line_has_settled_does_set_cost_usd(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    usage = {"type": "usage_update", "at": T3, "cost_usd": 1.23, "settled": True}
    write_records_log(session_log, [session_record(at=T0)])
    write_lines(sup_log, [to_json_line(launch_event(at=T1)), json.dumps(usage)])

    spans = _replay(session_log, sup_log)

    run_span = span_by_name(spans, "gymrat.run")
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
    run_span = span_by_name(spans, "gymrat.run")
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
    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.context.trace_id == trace_id_of(SESSION_ID)


def test_replay_session_when_command_span_created_does_set_deterministic_id(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    _write_measure_command_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.context.span_id == span_id_of(SESSION_ID, "command:2")


def test_replay_session_when_command_span_created_does_set_timing_from_duration(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    _write_measure_command_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.end_time == T2
    assert cmd_span.start_time == T2 - 500 * 1_000_000


def test_replay_session_when_command_span_created_does_use_command_attributes(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    _write_measure_command_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = span_by_name(spans, "gymrat.command.measure")
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
    cmd_span = span_by_name(spans, "gymrat.command.iterate")
    from opentelemetry.trace import StatusCode

    assert (cmd_span.status.status_code, cmd_span.status.description) == (
        getattr(StatusCode, expected_status),
        expected_description,
    )


@pytest.mark.parametrize(
    ("traceparent", "expected_links"),
    [
        pytest.param(
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            [0xB7AD6B7169203331],
            id="valid",
        ),
        pytest.param("not-valid", [], id="malformed"),
    ],
)
def test_replay_session_when_command_has_traceparent_does_link_only_a_valid_one(
    log_paths: tuple[str, str], traceparent: str, expected_links: list[int]
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    cmd = _command("measure", traceparent=traceparent)
    write_records_log(session_log, [header, cmd])
    _write_standard_run(sup_log)

    spans = _replay(session_log, sup_log)
    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert [link.context.span_id for link in cmd_span.links] == expected_links


def test_replay_session_when_command_in_run_range_does_parent_under_run(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    cmd = _command("measure", duration_ms=100)
    write_records_log(session_log, [header, cmd])
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T4)])

    spans = _replay(session_log, sup_log)
    cmd_span = span_by_name(spans, "gymrat.command.measure")
    run_span = span_by_name(spans, "gymrat.run")
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
    cmd_span = span_by_name(spans, "gymrat.command.measure")
    session_span = span_by_name(spans, "gymrat.session")
    assert cmd_span.parent is not None
    assert cmd_span.parent.span_id == session_span.context.span_id


def _traceparent_of(span_key: str) -> str:
    """The traceparent a command records when it ran under the span keyed ``span_key``."""
    return f"00-{trace_id_of(SESSION_ID):032x}-{span_id_of(SESSION_ID, span_key):016x}-01"


@pytest.mark.parametrize(
    ("linked_span_key", "expected_parent"),
    [
        pytest.param(f"run:{T1}", "gymrat.run", id="links-to-the-run"),
        pytest.param("run:0", "gymrat.session", id="links-to-no-run"),
    ],
)
def test_replay_session_when_command_outside_run_range_has_traceparent_does_parent_by_its_link(
    log_paths: tuple[str, str], linked_span_key: str, expected_parent: str
):
    session_log, sup_log = log_paths
    cmd = _command("measure", at=T5, duration_ms=100, traceparent=_traceparent_of(linked_span_key))
    write_records_log(session_log, [session_record(at=T0), cmd])
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T2)])

    spans = _replay(session_log, sup_log)

    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.parent.span_id == span_by_name(spans, expected_parent).context.span_id


def test_replay_session_when_command_in_one_run_links_to_another_does_parent_under_its_time_range(
    log_paths: tuple[str, str], tmp_path: Path
):
    session_log, first_log = log_paths
    second_log = str(tmp_path / "second.jsonl")
    cmd = _command("measure", at=T2, duration_ms=100, traceparent=_traceparent_of(f"run:{T4}"))
    write_records_log(session_log, [session_record(at=T0), cmd])
    write_supervisor_log(first_log, [launch_event(at=T1), turn_end(at=T3)])
    write_supervisor_log(second_log, [launch_event(at=T4), turn_end(at=T5)])

    with memory_tracing(SESSION_ID) as exporter:
        replay_session(session_log, [first_log, second_log])

    cmd_span = span_by_name(exporter.get_finished_spans(), "gymrat.command.measure")
    assert cmd_span.parent.span_id == span_id_of(SESSION_ID, f"run:{T1}")


def test_replay_session_when_a_record_outlasts_every_run_does_end_the_session_span_at_it(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    cmd = _command("measure", at=T5, duration_ms=100)
    write_records_log(session_log, [session_record(at=T0), cmd])
    write_supervisor_log(sup_log, [launch_event(at=T1), turn_end(at=T2)])

    spans = _replay(session_log, sup_log)

    assert span_by_name(spans, "gymrat.session").end_time == T5


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
    cmd_span = span_by_name(spans, "gymrat.command.iterate")
    event_names = [ev.name for ev in cmd_span.events]
    assert "gymrat.iteration" in event_names


def test_replay_session_when_log_has_no_command_does_add_events_on_session_span(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    baseline = baseline_record(at=T1)
    write_records_log(session_log, [header, baseline])
    write_supervisor_log(sup_log, [launch_event(at=T0), turn_end(at=T3)])

    spans = _replay(session_log, sup_log)
    session_span = span_by_name(spans, "gymrat.session")
    event_names = [ev.name for ev in session_span.events]
    assert "gymrat.baseline" in event_names
    assert "gymrat.session" not in event_names


def test_replay_session_when_records_follow_last_command_does_add_events_on_session_span(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    cmd = _command("measure", at=T1, duration_ms=100)
    baseline = baseline_record(at=T2)
    write_records_log(session_log, [header, cmd, baseline])
    write_supervisor_log(sup_log, [launch_event(at=T0), turn_end(at=T3)])

    spans = _replay(session_log, sup_log)
    session_span = span_by_name(spans, "gymrat.session")
    cmd_span = span_by_name(spans, "gymrat.command.measure")

    session_event_names = [ev.name for ev in session_span.events]
    assert "gymrat.baseline" in session_event_names
    assert "gymrat.session" not in session_event_names
    assert "gymrat.baseline" not in [ev.name for ev in cmd_span.events]


def test_replay_session_when_records_precede_first_command_does_attach_them_to_command_span(
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
    session_span = span_by_name(spans, "gymrat.session")
    cmd_span = span_by_name(spans, "gymrat.command.measure")

    session_event_names = [ev.name for ev in session_span.events]
    cmd_event_names = [ev.name for ev in cmd_span.events]

    assert "gymrat.baseline" in cmd_event_names
    assert "gymrat.hook" in cmd_event_names
    assert "gymrat.baseline" not in session_event_names
    assert "gymrat.hook" not in session_event_names


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
    run_span = span_by_name(spans, "gymrat.run")
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

    run_span = span_by_name(spans, "gymrat.run")
    assert run_span.attributes["gymrat.run.cost_usd"] == pytest.approx(0.10)
    assert [r for r in caplog.records if r.name == _REPLAY_LOGGER] == []


def test_replay_session_when_supervisor_first_line_malformed_does_ignore_the_log(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    write_lines(sup_log, ["{not json", to_json_line(turn_end(at=T3))])

    spans = _replay(session_log, sup_log)

    assert spans_by_prefix(spans, "gymrat.run") == []


# ---------------------------------------------------------------------------
# supervisor log without matching session_id is skipped
# ---------------------------------------------------------------------------


def test_replay_session_when_launch_session_id_differs_does_skip_log(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    write_records_log(session_log, [header])
    write_supervisor_log(sup_log, [launch_event(at=T1, session_id="other-session")])

    spans = _replay(session_log, sup_log)
    run_spans = spans_by_prefix(spans, "gymrat.run")
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


def _assert_one_warning_naming(records: list[logging.LogRecord], *fragments: str) -> None:
    messages = [
        r.message for r in records if r.name == _REPLAY_LOGGER and r.levelno == logging.WARNING
    ]
    assert len(messages) == 1, messages
    assert all(fragment in messages[0] for fragment in fragments), (fragments, messages[0])


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
    span_by_name(spans, "gymrat.command.measure")
    assert all(ev.name != "gymrat.iteration" for span in spans for ev in span.events)


_ITERATE_WIRE = record_to_wire(_command("iterate"))


@pytest.mark.parametrize(
    "invalid_cmd",
    [
        pytest.param({**_ITERATE_WIRE, "unknown_future_field": "banana"}, id="unknown-key"),
        pytest.param({**_ITERATE_WIRE, "exit_code": "banana"}, id="value-check-failure"),
        pytest.param(
            {**_ITERATE_WIRE, "exit_code": 0, "reason": "error"}, id="exit-code-reason-mismatch"
        ),
        pytest.param(
            {key: value for key, value in _ITERATE_WIRE.items() if key != "at"}, id="missing-at"
        ),
        pytest.param({**_ITERATE_WIRE, "duration_ms": "banana"}, id="duration-not-a-number"),
        pytest.param({**_ITERATE_WIRE, "args": "banana"}, id="args-not-a-mapping"),
        pytest.param({**_ITERATE_WIRE, "traceparent": 42}, id="traceparent-not-a-string"),
    ],
)
def test_replay_session_when_command_line_fails_validation_does_skip_it_with_warning(
    log_paths: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
    invalid_cmd: dict[str, object],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    valid_cmd = _command("measure")
    write_lines(
        session_log,
        [
            json.dumps(record_to_wire(header)),
            json.dumps(invalid_cmd),
            json.dumps(record_to_wire(valid_cmd)),
        ],
    )
    _write_standard_run(sup_log)

    with caplog.at_level(logging.WARNING, logger=_REPLAY_LOGGER):
        spans = _replay(session_log, sup_log)

    _assert_one_warning_naming(
        caplog.records, session_log, "skipping line 2 (invalid command record)"
    )
    assert [s.name for s in spans_by_prefix(spans, "gymrat.command.")] == ["gymrat.command.measure"]


_TORN_UTF8_LINE = b'{"type": "iteration", "note": "caf\xc3'


@pytest.mark.parametrize(
    ("line_order", "torn_line_number"),
    [
        pytest.param(("header", "command", "torn"), 3, id="torn-final-line"),
        pytest.param(("header", "torn", "command"), 2, id="torn-middle-line"),
    ],
)
def test_replay_session_when_session_line_is_torn_utf8_does_skip_it_with_warning(
    log_paths: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
    line_order: tuple[str, ...],
    torn_line_number: int,
):
    session_log, sup_log = log_paths
    lines = {
        "header": json.dumps(record_to_wire(session_record(at=T0))).encode() + b"\n",
        "command": json.dumps(record_to_wire(_command("measure"))).encode() + b"\n",
        "torn": _TORN_UTF8_LINE + (b"" if line_order[-1] == "torn" else b"\n"),
    }
    Path(session_log).write_bytes(b"".join(lines[name] for name in line_order))
    _write_standard_run(sup_log)

    with caplog.at_level(logging.WARNING, logger=_REPLAY_LOGGER):
        spans = _replay(session_log, sup_log)

    _assert_one_warning_naming(
        caplog.records, session_log, f"skipping line {torn_line_number} (invalid JSON)"
    )
    assert [s.name for s in spans_by_prefix(spans, "gymrat.command.")] == ["gymrat.command.measure"]


@pytest.mark.parametrize(
    "torn_index",
    [
        pytest.param(1, id="torn-middle-line"),
        pytest.param(3, id="torn-final-line"),
    ],
)
def test_replay_session_when_supervisor_line_is_torn_utf8_does_replay_its_other_lines(
    log_paths: tuple[str, str],
    torn_index: int,
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    lines = [
        to_json_line(launch_event(at=T1)).encode(),
        to_json_line(turn_end(at=T2, cost_usd=0.10)).encode(),
        to_json_line(turn_end(at=T3, cost_usd=0.42)).encode(),
    ]
    lines.insert(torn_index, _TORN_UTF8_LINE)
    Path(sup_log).write_bytes(b"\n".join(lines))

    spans = _replay(session_log, sup_log)

    run_span = span_by_name(spans, "gymrat.run")
    turn_end_costs = [
        ev.attributes["gymrat.turn.session_cost_usd"]
        for ev in run_span.events
        if ev.name == "gymrat.turn_end"
    ]
    assert run_span.attributes["gymrat.run.cost_usd"] == pytest.approx(0.42)
    assert turn_end_costs == [pytest.approx(0.10), pytest.approx(0.42)]


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
    cmd_clean = span_by_name(spans_clean, "gymrat.command.measure")

    spans_dirty = _replay(session_dirty, sup_log)
    cmd_dirty = span_by_name(spans_dirty, "gymrat.command.measure")

    # The dirty log has the command on physical line 3, clean on line 2.
    # The span ids differ because the physical line number differs.
    assert cmd_clean.context.span_id == span_id_of(SESSION_ID, "command:2")
    assert cmd_dirty.context.span_id == span_id_of(SESSION_ID, "command:3")
