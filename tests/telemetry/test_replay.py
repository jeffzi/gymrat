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
from gymrat.session.records import record_to_wire
from gymrat.supervisor.events import UsageUpdateEvent, to_json_line
from gymrat.telemetry.command_span import command_attributes
from gymrat.telemetry.provider import run_attributes, span_id_of, trace_id_of
from gymrat.telemetry.replay import replay_session
from gymrat.telemetry.run_spans import finalize_tracing, setup_tracing
from tests.session.records._fixtures import (
    FRESH_SESSION_ID,
    SESSION_ID,
    append_records,
    archive_and_reopen_session_log,
    baseline_record,
    hook_record,
    iteration_record,
    seeded_session,
    session_record,
)
from tests.session.records._wire import with_raw_number
from tests.supervisor._fixtures import make_prompt
from tests.telemetry._fixtures import (
    memory_tracing,
    session_traceparent,
    span_by_name,
    spans_by_prefix,
)
from tests.telemetry._replay_logs import (
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    TORN_UTF8_LINE,
    record_line,
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


def test_replay_session_when_supervisor_log_exists_does_create_run_span_with_its_launch_attributes(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    # Every launch option set, so a round trip that dropped one would fail.
    launch = replay_launch_event(
        at=T1, max_usd=5.0, effort="high", model="claude-sonnet-4-20250514"
    )
    turn_end = replay_turn_end(at=T3)
    write_supervisor_log(sup_log, [launch, turn_end])

    spans = _replay(session_log, sup_log)

    run_span = span_by_name(spans, "gymrat.run")
    assert run_span.context.span_id == span_id_of(SESSION_ID, f"run:{T1}")
    assert (run_span.start_time, run_span.end_time) == (T1, T3)
    assert dict(run_span.attributes) == run_attributes(launch) | {
        "gymrat.run.cost_usd": turn_end.cost_usd
    }


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
    assert dict(cmd_span.attributes) == command_attributes(
        replay_command("measure"), session_id=SESSION_ID
    )


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
        "measure",
        at=T5,
        duration_ms=100,
        traceparent=session_traceparent(SESSION_ID, linked_span_key),
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
        "measure", at=T2, duration_ms=100, traceparent=session_traceparent(SESSION_ID, f"run:{T4}")
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
            [
                baseline_record(at=T1),
                hook_record(at=T1),
                replay_command("measure", duration_ms=100),
            ],
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
# observability: stderr warnings for skipped session-log lines
# ---------------------------------------------------------------------------

_REPLAY_LOGGER = "gymrat.telemetry.replay"


def _assert_one_stderr_warning_naming(
    stderr: str, records: list[logging.LogRecord], *fragments: str
) -> None:
    warnings = [line for line in stderr.splitlines() if fragments[0] in line]
    assert len(warnings) == 1, stderr
    assert all(fragment in warnings[0] for fragment in fragments), (fragments, warnings[0])
    assert [r.message for r in records if r.name == _REPLAY_LOGGER] == []


_ITERATION_LINE = record_line(iteration_record(at=T1))


@pytest.mark.parametrize(
    ("line_order", "bad_line", "bad_line_number"),
    [
        pytest.param(
            ("header", "bad", "command"),
            _ITERATION_LINE[:-1].encode(),
            2,
            id="malformed-json-middle",
        ),
        pytest.param(
            ("header", "bad", "command"),
            with_raw_number(_ITERATION_LINE, ("primary", "delta_pct"), "NaN").encode(),
            2,
            id="nan-literal-middle",
        ),
        pytest.param(("header", "bad", "command"), TORN_UTF8_LINE, 2, id="torn-utf8-middle"),
    ],
)
def test_replay_session_when_session_line_undecodable_does_skip_it_with_stderr_warning(
    *,
    log_paths: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    line_order: tuple[str, ...],
    bad_line: bytes,
    bad_line_number: int,
):
    session_log, sup_log = log_paths
    lines = {
        "header": record_line(session_record(at=T0)).encode(),
        "command": record_line(replay_command("measure")).encode(),
        "bad": bad_line,
    }
    Path(session_log).write_bytes(b"".join(lines[name] + b"\n" for name in line_order))
    write_standard_run(sup_log)

    with caplog.at_level(logging.DEBUG, logger=_REPLAY_LOGGER):
        spans = _replay(session_log, sup_log)

    _assert_one_stderr_warning_naming(
        capsys.readouterr().err,
        caplog.records,
        session_log,
        f"skipping line {bad_line_number} (invalid JSON)",
    )
    assert [s.name for s in spans_by_prefix(spans, "gymrat.command.")] == ["gymrat.command.measure"]
    assert all(ev.name != "gymrat.iteration" for span in spans for ev in span.events)


@pytest.mark.parametrize(
    "torn_tail",
    [
        pytest.param(record_line(replay_command("iterate")).encode(), id="a-complete-record"),
        pytest.param(TORN_UTF8_LINE, id="cut-inside-a-utf8-character"),
    ],
)
def test_replay_session_when_final_line_unterminated_does_ignore_it_silently(
    log_paths: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
    torn_tail: bytes,
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0), replay_command("measure")])
    with Path(session_log).open("ab") as log:
        log.write(torn_tail)
    write_standard_run(sup_log)

    spans = _replay(session_log, sup_log)

    assert [s.name for s in spans_by_prefix(spans, "gymrat.command.")] == ["gymrat.command.measure"]
    assert capsys.readouterr().err == ""


_ITERATE_WIRE = record_to_wire(replay_command("iterate"))


@pytest.mark.parametrize(
    ("bad_line", "reason"),
    [
        pytest.param(
            json.dumps({**_ITERATE_WIRE, "unknown_future_field": "banana"}),
            "invalid command record",
            id="command-fails-validation",
        ),
        pytest.param(
            json.dumps({**record_to_wire(iteration_record(at=T1)), "seq": "banana"}),
            "invalid iteration record",
            id="iteration-fails-validation",
        ),
        pytest.param("[1, 2]", "not a record", id="json-that-is-not-an-object"),
    ],
)
def test_replay_session_when_session_line_fails_validation_does_skip_it_with_stderr_warning(
    *,
    log_paths: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    bad_line: str,
    reason: str,
):
    session_log, sup_log = log_paths
    write_lines(
        session_log,
        [
            record_line(session_record(at=T0)),
            bad_line,
            record_line(replay_command("measure")),
        ],
    )
    write_standard_run(sup_log)

    with caplog.at_level(logging.DEBUG, logger=_REPLAY_LOGGER):
        spans = _replay(session_log, sup_log)

    _assert_one_stderr_warning_naming(
        capsys.readouterr().err,
        caplog.records,
        session_log,
        f"skipping line 2 ({reason})",
    )
    assert [s.name for s in spans_by_prefix(spans, "gymrat.command.")] == ["gymrat.command.measure"]


@pytest.mark.parametrize(
    ("bad_line", "bad_index"),
    [
        pytest.param(
            with_raw_number(to_json_line(replay_turn_end(at=T4)), ("cost_usd",), "NaN").encode(),
            3,
            id="nan-literal-final",
        ),
        pytest.param(TORN_UTF8_LINE, 1, id="torn-utf8-middle"),
        pytest.param(TORN_UTF8_LINE, 3, id="torn-utf8-final"),
    ],
)
def test_replay_session_when_supervisor_line_undecodable_does_skip_it_silently(
    log_paths: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
    bad_line: bytes,
    bad_index: int,
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    lines = [
        to_json_line(replay_launch_event(at=T1)).encode(),
        to_json_line(replay_turn_end(at=T2, cost_usd=0.10)).encode(),
        to_json_line(replay_turn_end(at=T3, cost_usd=0.42)).encode(),
    ]
    lines.insert(bad_index, bad_line)
    Path(sup_log).write_bytes(b"\n".join(lines))

    with caplog.at_level(logging.WARNING, logger=_REPLAY_LOGGER):
        spans = _replay(session_log, sup_log)

    run_span = span_by_name(spans, "gymrat.run")
    turn_end_costs = [
        ev.attributes["gymrat.turn.session_cost_usd"]
        for ev in run_span.events
        if ev.name == "gymrat.turn_end"
    ]
    assert run_span.attributes["gymrat.run.cost_usd"] == pytest.approx(0.42)
    assert turn_end_costs == [pytest.approx(0.10), pytest.approx(0.42)]
    assert [r for r in caplog.records if r.name == _REPLAY_LOGGER] == []


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


# ---------------------------------------------------------------------------
# live tracing and replay agree
# ---------------------------------------------------------------------------


def _event_records(span: Any) -> list[tuple[str, dict[str, Any]]]:
    return [(ev.name, dict(ev.attributes)) for ev in span.events]


async def test_replay_session_when_later_command_appends_record_live_does_match_replayed_events(
    repo: str,
):
    header = seeded_session(repo)

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
    previous = seeded_session(repo)
    fresh = session_record(session_id=FRESH_SESSION_ID)

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
    header = seeded_session(repo)

    # Deterministic IDs, so the live command and replay derive the same parent link.
    monkeypatch.setenv("GYMRAT_TRACEPARENT", session_traceparent(header.session_id, f"run:{T0}"))

    jsonl_path = session_jsonl_path(repo)

    # A run whose span the GYMRAT_TRACEPARENT above points at; the live command falls
    # after it, so replay parents it by that link.
    sup_dir = tmp_path_factory.mktemp("parity-sup")
    sup_log = str(sup_dir / "run.jsonl")
    write_supervisor_log(
        sup_log,
        [replay_launch_event(session_id=header.session_id, at=T0), replay_turn_end(at=T5)],
    )

    async def body(trace: CommandTrace) -> str:
        return "ok"

    with memory_tracing(header.session_id) as live_exporter:
        await with_repo_lock("measure", body)

    live_spans = live_exporter.get_finished_spans()
    (live_cmd,) = spans_by_prefix(live_spans, "gymrat.command.")

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(jsonl_path, [sup_log])

    replay_spans = replay_exporter.get_finished_spans()
    (replay_cmd,) = spans_by_prefix(replay_spans, "gymrat.command.")

    assert live_cmd.parent is not None
    assert _span_signature(live_cmd) == _span_signature(replay_cmd)


def _command_span_ids(spans: tuple[Any, ...]) -> dict[str, int]:
    return {span.name: span.context.span_id for span in spans_by_prefix(spans, "gymrat.command.")}


def _append_blank_line(jsonl_path: str) -> None:
    with Path(jsonl_path).open("a", encoding="utf-8") as log:
        log.write("\n")


async def test_replay_session_when_session_log_holds_blank_lines_live_does_match_replayed_span_ids(
    repo: str,
):
    header = seeded_session(repo)
    jsonl_path = session_jsonl_path(repo)

    async def body(trace: CommandTrace) -> str:
        return "ok"

    with memory_tracing(header.session_id) as live_exporter:
        _append_blank_line(jsonl_path)
        await with_repo_lock("measure", body)
        _append_blank_line(jsonl_path)
        await with_repo_lock("compare", body)

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(jsonl_path, [])

    live_ids = _command_span_ids(live_exporter.get_finished_spans())
    assert list(live_ids) == ["gymrat.command.measure", "gymrat.command.compare"]
    assert _command_span_ids(replay_exporter.get_finished_spans()) == live_ids


def _attributes_by_span(spans: tuple[Any, ...]) -> dict[str, dict[str, Any]]:
    return {span.name: dict(span.attributes or {}) for span in spans}


def test_replay_session_when_launch_traced_live_does_match_replayed_session_and_run_attributes(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    header = session_record(at=T0)
    launch = replay_launch_event(
        at=T1, max_usd=5.0, effort="high", model="claude-sonnet-4-20250514"
    )
    write_records_log(session_log, [header])
    write_supervisor_log(sup_log, [launch])
    with memory_tracing(SESSION_ID) as live_exporter:
        _, _, state = setup_tracing(launch, branch=header.branch, prompt=make_prompt())
        finalize_tracing(state, None)

    replay_spans = _replay(session_log, sup_log)

    live_attributes = _attributes_by_span(live_exporter.get_finished_spans())
    assert live_attributes["gymrat.session"] == {
        "gymrat.session.id": SESSION_ID,
        "gymrat.session.branch": header.branch,
    }
    assert _attributes_by_span(replay_spans) == live_attributes
