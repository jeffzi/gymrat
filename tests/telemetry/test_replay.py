"""Tests for log-to-spans replay: session, run, and command spans from JSONL logs.

Replaying a session log must also yield the spans live tracing emitted for it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from gymrat.command_run import CommandTrace, with_repo_lock
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import record_to_wire
from gymrat.supervisor.events import UsageUpdateEvent, to_json_line
from gymrat.telemetry.provider import span_id_of
from gymrat.telemetry.replay import replay_session
from gymrat.telemetry.run_spans import (
    EVENT_TURN_END,
    RUN_COST_USD,
    RUN_SPAN,
    SESSION_SPAN,
    TURN_SESSION_COST_USD,
    run_span_key,
)
from tests._command_bodies import ok_body
from tests.session.records._fixtures import (
    FRESH_SESSION_ID,
    SESSION_ID,
    append_raw,
    append_records,
    archive_and_reopen_session_log,
    baseline_record,
    hook_record,
    iteration_record,
    seeded_session,
    session_record,
)
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
    write_records_log,
    write_standard_run,
    write_supervisor_log,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.session.records import SessionLogRecord


def _replay(session_log: str, sup_log: str) -> tuple[Any, ...]:
    with memory_tracing(SESSION_ID) as exporter:
        replay_session(session_log, [sup_log])
    return exporter.get_finished_spans()


def _assert_one_stderr_warning_naming(
    stderr: str, path: str, *fragments: str, line: int | None = None
) -> None:
    warnings = [warning for warning in stderr.splitlines() if path in warning]
    assert len(warnings) == 1, stderr
    assert all(fragment in warnings[0] for fragment in fragments), (fragments, warnings[0])
    if line is not None:
        assert re.search(rf"\b{line}\b", warnings[0].replace(path, "")), (line, warnings[0])


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

    session_span = span_by_name(spans, SESSION_SPAN)
    assert (session_span.start_time, session_span.end_time) == (T0, session_end)


# ---------------------------------------------------------------------------
# run spans
# ---------------------------------------------------------------------------


def test_replay_session_when_usage_update_in_supervisor_does_set_cost_usd(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    usage_line = to_json_line(UsageUpdateEvent(at=T3, cost_usd=1.23))
    write_records_log(session_log, [session_record(at=T0)])
    write_lines(sup_log, [to_json_line(replay_launch_event(at=T1)), usage_line])

    spans = _replay(session_log, sup_log)

    run_span = span_by_name(spans, RUN_SPAN)
    assert run_span.attributes[RUN_COST_USD] == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# command spans
# ---------------------------------------------------------------------------


def test_replay_session_when_command_record_present_does_create_command_span(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    command = replay_command("measure", at=T2, duration_ms=500)
    write_records_log(session_log, [session_record(at=T0), command])
    write_standard_run(sup_log)

    spans = _replay(session_log, sup_log)

    cmd_span = span_by_name(spans, "gymrat.command.measure")
    assert (cmd_span.start_time, cmd_span.end_time) == (T2 - 500 * 1_000_000, T2)


@pytest.mark.parametrize(
    ("command_at", "run_end", "parent"),
    [
        pytest.param(T2, T4, RUN_SPAN, id="inside-the-run-parents-under-it"),
        pytest.param(T5, T2, SESSION_SPAN, id="after-every-run-parents-under-the-session"),
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
        pytest.param(run_span_key(T1), RUN_SPAN, id="links-to-the-run"),
        pytest.param(run_span_key(0), SESSION_SPAN, id="links-to-no-run"),
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
        "measure",
        at=T2,
        duration_ms=100,
        traceparent=session_traceparent(SESSION_ID, run_span_key(T4)),
    )
    write_records_log(session_log, [session_record(at=T0), cmd])
    write_supervisor_log(first_log, [replay_launch_event(at=T1), replay_turn_end(at=T3)])
    write_supervisor_log(second_log, [replay_launch_event(at=T4), replay_turn_end(at=T5)])

    with memory_tracing(SESSION_ID) as exporter:
        replay_session(session_log, [first_log, second_log])

    cmd_span = span_by_name(exporter.get_finished_spans(), "gymrat.command.measure")
    assert cmd_span.parent.span_id == span_id_of(SESSION_ID, run_span_key(T1))


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
    ("records", "turn_end_at", "carrier"),
    [
        pytest.param(
            [baseline_record(at=T1), replay_command("measure", duration_ms=100)],
            T4,
            "gymrat.command.measure",
            id="records-precede-a-command",
        ),
        pytest.param(
            [replay_command("measure", at=T1, duration_ms=100), baseline_record(at=T2)],
            T3,
            SESSION_SPAN,
            id="records-follow-every-command",
        ),
    ],
)
def test_replay_session_when_records_surround_commands_does_add_each_to_one_span(
    log_paths: tuple[str, str],
    records: list[SessionLogRecord],
    turn_end_at: int,
    carrier: str,
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0), *records])
    write_supervisor_log(sup_log, [replay_launch_event(at=T0), replay_turn_end(at=turn_end_at)])

    spans = _replay(session_log, sup_log)

    assert _event_carriers(spans, ("gymrat.baseline",)) == {"gymrat.baseline": [carrier]}


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
# supervisor logs replay skips: silently, or with a stderr warning
# ---------------------------------------------------------------------------


def _write_other_session_log(sup_log: str) -> None:
    write_supervisor_log(sup_log, [replay_launch_event(at=T1, session_id="other-session")])


def _first_line(raw: bytes) -> Callable[[str], None]:
    """Build an arrange step writing a supervisor log that opens on ``raw``."""

    def write(sup_log: str) -> None:
        Path(sup_log).write_bytes(raw + b"\n" + to_json_line(replay_turn_end(at=T3)).encode())

    return write


def _leave_absent(_sup_log: str) -> None:
    """Write nothing, as when a listed log vanished before it was read."""


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_write_other_session_log, id="a-log-of-another-session"),
        pytest.param(_first_line(b"{not json"), id="a-first-line-that-is-not-json"),
        pytest.param(_first_line(b"\xff\xff"), id="a-first-line-that-is-not-utf8"),
        pytest.param(_first_line(b"[1, 2]"), id="a-first-line-that-is-not-an-object"),
        pytest.param(_leave_absent, id="a-log-that-vanished-after-listing"),
    ],
)
def test_replay_session_when_supervisor_log_not_a_launch_of_this_session_does_skip_it_silently(
    log_paths: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
    arrange: Callable[[str], None],
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    arrange(sup_log)

    spans = _replay(session_log, sup_log)

    assert (spans_by_prefix(spans, RUN_SPAN), capsys.readouterr().err) == ([], "")


def _make_directory(sup_log: str) -> None:
    Path(sup_log).mkdir()


def _write_invalid_launch(sup_log: str) -> None:
    write_lines(sup_log, [json.dumps({"type": "launch", "session_id": SESSION_ID, "at": "banana"})])


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_make_directory, id="a-directory"),
        pytest.param(_write_invalid_launch, id="a-launch-of-this-session-that-fails-to-parse"),
    ],
)
def test_replay_session_when_supervisor_log_cannot_be_read_does_skip_it_with_a_warning(
    *,
    log_paths: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
    arrange: Callable[[str], None],
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    arrange(sup_log)

    spans = _replay(session_log, sup_log)

    _assert_one_stderr_warning_naming(capsys.readouterr().err, sup_log)
    assert spans_by_prefix(spans, RUN_SPAN) == []


# ---------------------------------------------------------------------------
# observability: stderr warnings for skipped session-log lines
# ---------------------------------------------------------------------------

_ITERATION_LINE = record_line(iteration_record(at=T1))


_ITERATE_WIRE = record_to_wire(replay_command("iterate"))


@pytest.mark.parametrize(
    ("bad_line", "inserted"),
    [
        pytest.param(_ITERATION_LINE[:-1].encode(), (), id="malformed-json-middle"),
        pytest.param(TORN_UTF8_LINE, (), id="torn-utf8-middle"),
        pytest.param(
            json.dumps({**_ITERATE_WIRE, "unknown_future_field": "banana"}).encode(),
            ("command",),
            id="command-fails-validation",
        ),
        pytest.param(b"[1, 2]", (), id="json-that-is-not-an-object"),
    ],
)
def test_replay_session_when_session_line_unreadable_does_skip_it_with_stderr_warning(
    *,
    log_paths: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
    bad_line: bytes,
    inserted: tuple[str, ...],
):
    session_log, sup_log = log_paths
    header = record_line(session_record(at=T0)).encode()
    command = record_line(replay_command("measure")).encode()
    Path(session_log).write_bytes(header + b"\n" + bad_line + b"\n" + command + b"\n")
    write_standard_run(sup_log)

    spans = _replay(session_log, sup_log)

    _assert_one_stderr_warning_naming(capsys.readouterr().err, session_log, *inserted, line=2)
    assert [s.name for s in spans_by_prefix(spans, "gymrat.command.")] == ["gymrat.command.measure"]


def _turn_end_costs(spans: tuple[Any, ...]) -> list[float]:
    run_span = span_by_name(spans, RUN_SPAN)
    return [
        ev.attributes[TURN_SESSION_COST_USD] for ev in run_span.events if ev.name == EVENT_TURN_END
    ]


# ---------------------------------------------------------------------------
# observability: stderr warnings for skipped supervisor-log lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_line",
    [
        pytest.param(b"{not json", id="not-json"),
        pytest.param(b'{"type": "unknown_future_event"}', id="matches-no-event"),
    ],
)
def test_replay_session_when_supervisor_line_after_launch_unreadable_does_skip_it_with_stderr_warning(
    *,
    log_paths: tuple[str, str],
    capsys: pytest.CaptureFixture[str],
    bad_line: bytes,
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    launch = to_json_line(replay_launch_event(at=T1)).encode()
    turn_end = to_json_line(replay_turn_end(at=T3)).encode()
    Path(sup_log).write_bytes(launch + b"\n\n" + bad_line + b"\n" + turn_end + b"\n")

    spans = _replay(session_log, sup_log)

    _assert_one_stderr_warning_naming(capsys.readouterr().err, sup_log, line=3)
    assert _turn_end_costs(spans) == [pytest.approx(0.42)]


def test_replay_session_when_supervisor_log_last_line_unterminated_does_drop_it(
    log_paths: tuple[str, str],
):
    session_log, sup_log = log_paths
    write_records_log(session_log, [session_record(at=T0)])
    launch = to_json_line(replay_launch_event(at=T1)).encode()
    first = to_json_line(replay_turn_end(at=T2, cost_usd=0.10)).encode()
    torn = to_json_line(replay_turn_end(at=T3, cost_usd=0.42)).encode()
    Path(sup_log).write_bytes(launch + b"\n" + first + b"\n" + torn)

    spans = _replay(session_log, sup_log)

    assert _turn_end_costs(spans) == [pytest.approx(0.10)]


# ---------------------------------------------------------------------------
# live tracing and replay agree
# ---------------------------------------------------------------------------


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
    header = seeded_session(repo)

    async def body_compare(trace: CommandTrace) -> str:
        append_records(repo, hook_record())
        return "ok"

    with memory_tracing(header.session_id) as live_exporter:
        await with_repo_lock("measure", ok_body)
        await with_repo_lock("compare", body_compare)

    live_spans = live_exporter.get_finished_spans()

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(session_jsonl_path(repo), [])

    replay_spans = replay_exporter.get_finished_spans()
    live_compare = span_by_name(live_spans, "gymrat.command.compare")
    replay_compare = span_by_name(replay_spans, "gymrat.command.compare")
    assert "gymrat.hook" in [ev.name for ev in live_compare.events]
    assert _event_records(replay_compare) == _event_records(live_compare)


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
    monkeypatch.setenv(
        "GYMRAT_TRACEPARENT", session_traceparent(header.session_id, run_span_key(T0))
    )

    jsonl_path = session_jsonl_path(repo)

    # A run whose span the GYMRAT_TRACEPARENT above points at; the live command falls
    # after it, so replay parents it by that link.
    sup_dir = tmp_path_factory.mktemp("parity-sup")
    sup_log = str(sup_dir / "run.jsonl")
    write_supervisor_log(
        sup_log,
        [replay_launch_event(session_id=header.session_id, at=T0), replay_turn_end(at=T5)],
    )

    with memory_tracing(header.session_id) as live_exporter:
        await with_repo_lock("measure", ok_body)

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


async def test_replay_session_when_session_log_holds_blank_lines_live_does_match_replayed_span_ids(
    repo: str,
):
    header = seeded_session(repo)
    jsonl_path = session_jsonl_path(repo)

    with memory_tracing(header.session_id) as live_exporter:
        append_raw(jsonl_path, b"\n")
        await with_repo_lock("measure", ok_body)
        append_raw(jsonl_path, b"\n")
        await with_repo_lock("compare", ok_body)

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(jsonl_path, [])

    live_ids = _command_span_ids(live_exporter.get_finished_spans())
    assert list(live_ids) == ["gymrat.command.measure", "gymrat.command.compare"]
    assert _command_span_ids(replay_exporter.get_finished_spans()) == live_ids
