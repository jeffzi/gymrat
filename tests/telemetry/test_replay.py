"""Tests for log-to-spans replay: session, run, and command spans from JSONL logs.

Replaying a session log must also yield the spans live tracing emitted for it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from gymrat.command_run import CommandTrace, with_repo_lock
from gymrat.session.paths import session_jsonl_path
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
    BASELINE_SHA,
    SESSION_ID,
    append_records,
    archive_and_reopen_session_log,
    baseline_record,
    hook_record,
    iteration_record,
    session_record,
    write_session_log,
)
from tests.session.records._wire import with_raw_number
from tests.telemetry._fixtures import memory_tracing, span_by_name, spans_by_prefix
from tests.telemetry._replay_logs import (
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    replay_command,
    replay_launch_event,
    replay_turn_end,
    write_lines,
    write_measure_command_run,
    write_records_log,
    write_standard_run,
    write_supervisor_log,
)


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


@pytest.mark.parametrize(
    ("records", "run_end", "session_end"),
    [
        pytest.param([], T3, T3, id="ends-with-the-run"),
        pytest.param(
            [replay_command("measure", at=T5, duration_ms=100)],
            T2,
            T5,
            id="ends-with-a-record-after-the-run",
        ),
    ],
)
def test_replay_session_when_called_does_span_the_session_from_header_to_latest_run_end_or_record(
    log_paths: tuple[str, str], records: list[Any], run_end: int, session_end: int
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0), *records])
    write_supervisor_log(sup_log, [replay_launch_event(at=T1), replay_turn_end(at=run_end)])

    spans = _replay(session_log, sup_log)

    session_span = span_by_name(spans, "gymrat.session")
    assert session_span.context.trace_id == trace_id_of(SESSION_ID)
    assert session_span.context.span_id == span_id_of(SESSION_ID, "session")
    assert (session_span.start_time, session_span.end_time) == (T0, session_end)


# ---------------------------------------------------------------------------
# run spans
# ---------------------------------------------------------------------------


_FIXED_RUN_ATTRIBUTES = {
    "gymrat.session.id": SESSION_ID,
    "gymrat.run.head_sha": BASELINE_SHA,
    "gymrat.run.max_minutes": 60.0,
    "gen_ai.provider.name": "anthropic",
}
_LAUNCH_OPTION_ATTRIBUTES = ("gymrat.run.max_usd", "gymrat.run.effort", "gen_ai.request.model")


@pytest.mark.parametrize(
    ("launch_options", "expected_option_attrs"),
    [
        pytest.param({}, dict.fromkeys(_LAUNCH_OPTION_ATTRIBUTES), id="options-unset-are-left-off"),
        pytest.param(
            {"max_usd": 5.0, "effort": "high", "model": "claude-sonnet-4-20250514"},
            {
                "gymrat.run.max_usd": 5.0,
                "gymrat.run.effort": "high",
                "gen_ai.request.model": "claude-sonnet-4-20250514",
            },
            id="options-set-are-recorded",
        ),
    ],
)
def test_replay_session_when_supervisor_log_exists_does_create_run_span_with_its_launch_attributes(
    log_paths: tuple[str, str],
    launch_options: dict[str, Any],
    expected_option_attrs: dict[str, Any],
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    launch = replay_launch_event(at=T1, **launch_options)
    write_supervisor_log(sup_log, [launch, replay_turn_end(at=T3)])

    spans = _replay(session_log, sup_log)

    run_span = span_by_name(spans, "gymrat.run")
    attrs = dict(run_span.attributes)
    assert run_span.context.span_id == span_id_of(SESSION_ID, f"run:{T1}")
    assert (run_span.start_time, run_span.end_time) == (T1, T3)
    assert {key: attrs[key] for key in _FIXED_RUN_ATTRIBUTES} == _FIXED_RUN_ATTRIBUTES
    assert {key: attrs.get(key) for key in _LAUNCH_OPTION_ATTRIBUTES} == expected_option_attrs


@pytest.mark.parametrize(
    "usage_line",
    [
        pytest.param(to_json_line(UsageUpdateEvent(at=T3, cost_usd=1.23)), id="usage-update"),
        pytest.param(
            json.dumps({"type": "usage_update", "at": T3, "cost_usd": 1.23, "settled": True}),
            id="usage-update-with-settled",
        ),
    ],
)
def test_replay_session_when_usage_update_in_supervisor_does_set_cost_usd(
    log_paths: tuple[str, str], usage_line: str
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    write_lines(sup_log, [to_json_line(replay_launch_event(at=T1)), usage_line])

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
            replay_launch_event(at=T1),
            replay_turn_end(at=T2),
            FollowUpEvent(at=T3, action="replied", reason="continue"),
            CapEvent(at=T4, cap="wall-clock", action="interrupting"),
            CompactionEvent(at=T5),
        ],
    )

    spans = _replay(session_log, sup_log)

    run_span = span_by_name(spans, "gymrat.run")
    assert [ev.name for ev in run_span.events] == [
        "gymrat.turn_end",
        "gymrat.follow_up",
        "gymrat.cap",
        "gymrat.compaction",
    ]


# ---------------------------------------------------------------------------
# command spans
# ---------------------------------------------------------------------------


def test_replay_session_when_command_record_present_does_create_command_span(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    write_measure_command_run(session_log, sup_log)

    spans = _replay(session_log, sup_log)

    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.context.trace_id == trace_id_of(SESSION_ID)
    assert cmd_span.context.span_id == span_id_of(SESSION_ID, "command:2")
    assert (cmd_span.start_time, cmd_span.end_time) == (T2 - 500 * 1_000_000, T2)
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
    cmd = replay_command("iterate", duration_ms=100, exit_code=exit_code, reason=reason)
    write_records_log(session_log, [header, cmd])
    write_standard_run(sup_log)

    spans = _replay(session_log, sup_log)

    cmd_span = span_by_name(spans, "gymrat.command.iterate")
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
    cmd = replay_command("measure", traceparent=traceparent)
    write_records_log(session_log, [header, cmd])
    write_standard_run(sup_log)

    spans = _replay(session_log, sup_log)

    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert [link.context.span_id for link in cmd_span.links] == expected_links


@pytest.mark.parametrize(
    ("command_at", "run_end", "parent"),
    [
        pytest.param(T2, T4, "gymrat.run", id="inside-the-run-parents-under-it"),
        pytest.param(T5, T2, "gymrat.session", id="after-every-run-parents-under-the-session"),
    ],
)
def test_replay_session_when_command_time_falls_in_or_out_of_a_run_does_parent_it_accordingly(
    log_paths: tuple[str, str], command_at: int, run_end: int, parent: str
):
    session_log, sup_log = log_paths
    cmd = replay_command("measure", at=command_at, duration_ms=100)
    write_records_log(session_log, [session_record(at=T0), cmd])
    write_supervisor_log(sup_log, [replay_launch_event(at=T1), replay_turn_end(at=run_end)])

    spans = _replay(session_log, sup_log)

    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.parent is not None
    assert cmd_span.parent.span_id == span_by_name(spans, parent).context.span_id


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
    cmd = replay_command(
        "measure", at=T5, duration_ms=100, traceparent=_traceparent_of(linked_span_key)
    )
    write_records_log(session_log, [session_record(at=T0), cmd])
    write_supervisor_log(sup_log, [replay_launch_event(at=T1), replay_turn_end(at=T2)])

    spans = _replay(session_log, sup_log)

    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.parent.span_id == span_by_name(spans, expected_parent).context.span_id


def test_replay_session_when_command_in_one_run_links_to_another_does_parent_under_its_time_range(
    log_paths: tuple[str, str], tmp_path: Path
):
    session_log, first_log = log_paths
    second_log = str(tmp_path / "second.jsonl")
    cmd = replay_command(
        "measure", at=T2, duration_ms=100, traceparent=_traceparent_of(f"run:{T4}")
    )
    write_records_log(session_log, [session_record(at=T0), cmd])
    write_supervisor_log(first_log, [replay_launch_event(at=T1), replay_turn_end(at=T3)])
    write_supervisor_log(second_log, [replay_launch_event(at=T4), replay_turn_end(at=T5)])

    with memory_tracing(SESSION_ID) as exporter:
        replay_session(session_log, [first_log, second_log])

    cmd_span = span_by_name(exporter.get_finished_spans(), "gymrat.command.measure")
    assert cmd_span.parent.span_id == span_id_of(SESSION_ID, f"run:{T1}")


# ---------------------------------------------------------------------------
# records between commands become events on the correct span
# ---------------------------------------------------------------------------


def _event_carriers(spans: tuple[Any, ...], event_names: tuple[str, ...]) -> dict[str, list[str]]:
    """Map each of ``event_names`` to the names of the spans that carry it."""
    return {
        event_name: [span.name for span in spans for ev in span.events if ev.name == event_name]
        for event_name in event_names
    }


_PRE_COMMAND_HOOK = HookRecord(
    type="hook",
    at=T1,
    stage="before",
    seq=1,
    exit_code=0,
    duration_ms=50,
    stdout_bytes=10,
    timed_out=False,
)


@pytest.mark.parametrize(
    ("records", "owner", "event_names"),
    [
        pytest.param(
            [
                replay_command("measure", duration_ms=100, seq=1),
                iteration_record(at=T2),
                replay_command("iterate", duration_ms=100, seq=2),
            ],
            "gymrat.command.iterate",
            ("gymrat.iteration",),
            id="between-commands-go-to-the-next",
        ),
        pytest.param(
            [baseline_record(at=T1), _PRE_COMMAND_HOOK, replay_command("measure", duration_ms=100)],
            "gymrat.command.measure",
            ("gymrat.baseline", "gymrat.hook"),
            id="before-the-first-command-go-to-it",
        ),
    ],
)
def test_replay_session_when_records_precede_a_command_does_add_them_as_events_on_that_command_only(
    log_paths: tuple[str, str],
    records: list[Any],
    owner: str,
    event_names: tuple[str, ...],
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0), *records])
    write_supervisor_log(sup_log, [replay_launch_event(at=T0), replay_turn_end(at=T4)])

    spans = _replay(session_log, sup_log)

    assert _event_carriers(spans, event_names) == {name: [owner] for name in event_names}


@pytest.mark.parametrize(
    "records",
    [
        pytest.param([baseline_record(at=T1)], id="log-without-a-command"),
        pytest.param(
            [replay_command("measure", at=T1, duration_ms=100), baseline_record(at=T2)],
            id="after-the-last-command",
        ),
    ],
)
def test_replay_session_when_records_follow_every_command_does_add_them_as_events_on_session_span_only(
    log_paths: tuple[str, str], records: list[Any]
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0), *records])
    write_supervisor_log(sup_log, [replay_launch_event(at=T0), replay_turn_end(at=T3)])

    spans = _replay(session_log, sup_log)

    assert _event_carriers(spans, ("gymrat.baseline", "gymrat.session")) == {
        "gymrat.baseline": ["gymrat.session"],
        "gymrat.session": [],
    }


# ---------------------------------------------------------------------------
# span count
# ---------------------------------------------------------------------------


def test_replay_session_when_called_does_return_span_count(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    cmd = replay_command("measure", duration_ms=100)
    write_records_log(session_log, [header, cmd])
    write_standard_run(sup_log)

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
    write_lines(
        sup_log,
        [
            to_json_line(replay_launch_event(at=T1)),
            '{"type": "unknown_future_event"}',
            to_json_line(replay_turn_end(at=T3)),
        ],
    )

    spans = _replay(session_log, sup_log)

    run_span = span_by_name(spans, "gymrat.run")
    assert (run_span.start_time, run_span.end_time) == (T1, T3)
    assert [ev.name for ev in run_span.events] == ["gymrat.turn_end"]


def test_replay_session_when_supervisor_line_holds_non_finite_number_does_skip_it_silently(
    log_paths: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    write_lines(
        sup_log,
        [
            to_json_line(replay_launch_event(at=T1)),
            to_json_line(replay_turn_end(at=T2, cost_usd=0.10)),
            with_raw_number(to_json_line(replay_turn_end(at=T3)), ("cost_usd",), "NaN"),
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
    write_lines(sup_log, ["{not json", to_json_line(replay_turn_end(at=T3))])

    spans = _replay(session_log, sup_log)

    assert spans_by_prefix(spans, "gymrat.run") == []


# ---------------------------------------------------------------------------
# supervisor log without matching session_id is skipped
# ---------------------------------------------------------------------------


def test_replay_session_when_launch_session_id_differs_does_skip_log(log_paths: tuple[str, str]):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    write_records_log(session_log, [header])
    write_supervisor_log(sup_log, [replay_launch_event(at=T1, session_id="other-session")])

    spans = _replay(session_log, sup_log)

    run_spans = spans_by_prefix(spans, "gymrat.run")
    assert len(run_spans) == 0


# ---------------------------------------------------------------------------
# observability: warning logs for silent failures
# ---------------------------------------------------------------------------

_REPLAY_LOGGER = "gymrat.telemetry.replay"


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
    cmd = replay_command("measure")
    write_lines(
        session_log,
        [json.dumps(record_to_wire(header)), bad_line, json.dumps(record_to_wire(cmd))],
    )
    write_standard_run(sup_log)

    with (
        memory_tracing(SESSION_ID) as exporter,
        caplog.at_level(logging.WARNING, logger=_REPLAY_LOGGER),
    ):
        replay_session(session_log, [sup_log])

    _assert_one_warning_naming(caplog.records, "skipping line 2 (invalid JSON)")
    spans = exporter.get_finished_spans()
    span_by_name(spans, "gymrat.command.measure")
    assert all(ev.name != "gymrat.iteration" for span in spans for ev in span.events)


_ITERATE_WIRE = record_to_wire(replay_command("iterate"))


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
    valid_cmd = replay_command("measure")
    write_lines(
        session_log,
        [
            json.dumps(record_to_wire(header)),
            json.dumps(invalid_cmd),
            json.dumps(record_to_wire(valid_cmd)),
        ],
    )
    write_standard_run(sup_log)

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
        "command": json.dumps(record_to_wire(replay_command("measure"))).encode() + b"\n",
        "torn": _TORN_UTF8_LINE + (b"" if line_order[-1] == "torn" else b"\n"),
    }
    Path(session_log).write_bytes(b"".join(lines[name] for name in line_order))
    write_standard_run(sup_log)

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
        to_json_line(replay_launch_event(at=T1)).encode(),
        to_json_line(replay_turn_end(at=T2, cost_usd=0.10)).encode(),
        to_json_line(replay_turn_end(at=T3, cost_usd=0.42)).encode(),
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
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    cmd = replay_command("measure", at=T2, duration_ms=100)
    write_lines(
        session_log,
        [json.dumps(record_to_wire(header)), "not valid json", json.dumps(record_to_wire(cmd))],
    )
    write_supervisor_log(sup_log, [replay_launch_event(at=T1), replay_turn_end(at=T4)])

    spans = _replay(session_log, sup_log)

    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert cmd_span.context.span_id == span_id_of(SESSION_ID, "command:3")


def _span_signature(span: Any) -> tuple[Any, ...]:
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
        dict(span.attributes),
        tuple(link.context.span_id for link in span.links),
    )


def _event_records(span: Any) -> list[tuple[str, dict[str, Any]]]:
    return [(ev.name, dict(ev.attributes)) for ev in span.events]


async def test_replay_session_when_later_command_appends_record_live_does_match_replayed_events(
    repo: str,
):
    header = session_record()
    write_session_log(repo, header)

    async def body_measure(trace: CommandTrace) -> str:
        return "ok"

    async def body_compare(trace: CommandTrace) -> str:
        append_records(repo, hook_record())
        return "ok"

    with memory_tracing(header.session_id) as live_exporter:
        await with_repo_lock("measure", body_measure)
        await with_repo_lock("compare", body_compare)

    live_spans = live_exporter.get_finished_spans()

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(session_jsonl_path(repo), [])

    replay_spans = replay_exporter.get_finished_spans()
    live_compare = span_by_name(live_spans, "gymrat.command.compare")
    replay_compare = span_by_name(replay_spans, "gymrat.command.compare")
    live_measure = span_by_name(live_spans, "gymrat.command.measure")
    replay_measure = span_by_name(replay_spans, "gymrat.command.measure")
    assert "gymrat.hook" in [ev.name for ev in live_compare.events]
    assert _event_records(replay_compare) == _event_records(live_compare)
    assert _event_records(replay_measure) == _event_records(live_measure)


async def test_replay_session_when_command_opens_session_with_baseline_live_does_match_replayed_events(
    repo: str,
):
    previous = session_record()
    write_session_log(repo, previous)
    fresh = session_record(session_id="20260809-090000-b7e4")

    async def body_supervise(trace: CommandTrace) -> str:
        archive_and_reopen_session_log(previous, fresh)
        return "ok"

    with memory_tracing(fresh.session_id) as live_exporter:
        await with_repo_lock("supervise", body_supervise)

    live_cmd = span_by_name(live_exporter.get_finished_spans(), "gymrat.command.supervise")

    with memory_tracing(fresh.session_id) as replay_exporter:
        replay_session(session_jsonl_path(repo), [])

    replay_cmd = span_by_name(replay_exporter.get_finished_spans(), "gymrat.command.supervise")
    live_events = [ev.name for ev in live_cmd.events]
    assert "gymrat.baseline" in live_events
    assert [ev.name for ev in replay_cmd.events] == live_events


async def test_replay_session_when_gymrat_traceparent_set_live_does_match_replayed_spans(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
):
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
        [replay_launch_event(session_id=header.session_id, at=T0), replay_turn_end(at=T5)],
    )

    with memory_tracing(header.session_id) as live_exporter:

        async def body(trace: CommandTrace) -> str:
            return "ok"

        await with_repo_lock("measure", body)

    live_spans = live_exporter.get_finished_spans()
    (live_cmd,) = spans_by_prefix(live_spans, "gymrat.command.")

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(jsonl_path, [sup_log])

    replay_spans = replay_exporter.get_finished_spans()
    (replay_cmd,) = spans_by_prefix(replay_spans, "gymrat.command.")

    assert live_cmd.parent is not None
    assert _span_signature(live_cmd) == _span_signature(replay_cmd)
