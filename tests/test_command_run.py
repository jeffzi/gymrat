"""Tests for :mod:`gymrat.command_run`.

They exercise ``with_repo_lock``'s locking and command-record behavior, the
edge cases around missing session logs and failed appends, the command spans it
emits when tracing is enabled, and the layering guard that keeps the seam free
of the CLI package.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest
import typer
from opentelemetry.trace import StatusCode

from gymrat.command_run import CommandTrace, command_origin, with_repo_lock
from gymrat.errors import GymratError
from gymrat.loop.iterate.run import LoopStopError
from gymrat.session.lock import LockContentionError, is_held
from gymrat.session.paths import lockfile_path, repo_root, session_jsonl_path
from gymrat.session.records import CommandRecord, SessionRecord
from gymrat.session.schema import CommandReason
from gymrat.session.store import session_header
from gymrat.telemetry.provider import export_failed, record_event, span_id_of, trace_id_of
from tests._imports import loaded_under, modules_imported_by
from tests._lock import hold_lock, remove_lock_files
from tests.session.records._fixtures import (
    append_records,
    archive_and_reopen_session_log,
    baseline_record,
    iteration_record,
    log_records,
    session_record,
    tear_final_line,
    write_session_log,
)
from tests.telemetry._collector import otlp_collector
from tests.telemetry._fixtures import (
    hide_otlp_exporter,
    memory_tracing,
    span_by_name,
    spans_by_prefix,
)

#: The id of the session a ``start`` opens over the previous one.
_FRESH_SESSION_ID = "20260809-090000-b7e4"


def _seeded_session(repo: str) -> SessionRecord:
    """Write a session header to ``repo``'s session log."""
    header = session_record()
    write_session_log(repo, header)
    return header


async def _ok_body(trace: CommandTrace) -> str:
    """Trivial command body for tests that only inspect what the command leaves behind."""
    return "ok"


def _broken_append(path: str, record: object) -> None:
    msg = "disk full"
    raise GymratError(msg)


def _last_command_record() -> CommandRecord:
    """Read the current session log and return its last record, asserted to be a command."""
    cmd = log_records(repo_root())[-1]
    assert isinstance(cmd, CommandRecord)
    return cmd


# ---------------------------------------------------------------------------
# command_origin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        pytest.param("tool", "tool", id="env-tool"),
        pytest.param("banana", "cli", id="env-other-value"),
        pytest.param("TOOL", "cli", id="env-uppercase-tool"),
        pytest.param("", "cli", id="env-empty"),
        pytest.param(None, "cli", id="env-absent"),
    ],
)
def test_command_origin_when_env_varies_does_answer_tool_only_for_exact_tool(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, expected: str
):
    monkeypatch.delenv("GYMRAT_COMMAND_ORIGIN", raising=False)
    if env_value is not None:
        monkeypatch.setenv("GYMRAT_COMMAND_ORIGIN", env_value)

    origin = command_origin()

    assert origin == expected


# ---------------------------------------------------------------------------
# with_repo_lock — locking behavior
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_inside_repo_does_scope_the_lock_to_the_body(
    repo: str,
):
    lock_path = lockfile_path(repo_root())
    held_during: list[bool] = []

    async def body(trace: CommandTrace) -> str:
        held_during.append(is_held(lock_path))
        return "measured"

    result = await with_repo_lock("compare", body)

    assert result == "measured"
    assert held_during == [True]
    assert not is_held(lock_path)
    assert not await asyncio.to_thread(Path(session_jsonl_path(repo_root())).exists)


async def test_with_repo_lock_when_outside_repo_does_run_body_without_a_lock_or_a_record(
    plain_directory: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(plain_directory)

    async def body(trace: CommandTrace) -> str:
        return "ran"

    result = await with_repo_lock("compare", body)

    assert result == "ran"
    assert not await asyncio.to_thread(Path(lockfile_path(plain_directory)).exists)
    assert not await asyncio.to_thread(Path(session_jsonl_path(plain_directory)).exists)


async def test_with_repo_lock_when_session_log_torn_does_repair_it_before_the_body(repo: str):
    header = session_record()
    write_session_log(repo, header)
    jsonl_path = Path(session_jsonl_path(repo_root()))
    intact_log = await asyncio.to_thread(jsonl_path.read_bytes)
    tear_final_line(jsonl_path)
    iteration = iteration_record()
    seen: dict[str, bytes] = {}

    async def body(trace: CommandTrace) -> str:
        seen["log"] = await asyncio.to_thread(jsonl_path.read_bytes)
        append_records(repo, iteration)
        return "ran"

    result = await with_repo_lock("compare", body)

    assert result == "ran"
    assert seen["log"] == intact_log
    records = log_records(repo_root())
    assert records[0] == header
    assert records[1] == iteration
    assert isinstance(records[-1], CommandRecord)


async def test_with_repo_lock_when_session_log_torn_and_lock_held_elsewhere_does_leave_it_torn(
    repo: str,
):
    write_session_log(repo, session_record())
    jsonl_path = Path(session_jsonl_path(repo_root()))
    tear_final_line(jsonl_path)
    torn_log = await asyncio.to_thread(jsonl_path.read_bytes)
    blocker = hold_lock(lockfile_path(repo_root()))

    try:
        with pytest.raises(LockContentionError):
            await with_repo_lock("compare", _ok_body)
    finally:
        blocker.release()

    assert await asyncio.to_thread(jsonl_path.read_bytes) == torn_log


async def test_with_repo_lock_when_git_fails_otherwise_does_raise_without_running_body_locking_or_recording(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    header = _seeded_session(repo)
    git_error = GymratError("detected dubious ownership", hint="banana hint")

    def broken_git(*_args: object, **_kwargs: object) -> str:
        raise git_error

    monkeypatch.setattr("gymrat.command_run.repo_root", broken_git)
    called: list[bool] = []

    async def body(trace: CommandTrace) -> str:
        called.append(True)
        return "should-not-run"

    with pytest.raises(GymratError) as exc:
        await with_repo_lock("compare", body)

    assert exc.value is git_error
    assert called == []
    assert not await asyncio.to_thread(Path(lockfile_path(repo_root())).exists)
    assert log_records(repo_root()) == [header]


# ---------------------------------------------------------------------------
# with_repo_lock — command record appending
# ---------------------------------------------------------------------------


#: The traceparent an outer caller hands down through ``TRACEPARENT``.
_OUTER_TRACEPARENT = "00-abc-def-01"


@pytest.mark.parametrize(
    ("args", "env", "expected"),
    [
        pytest.param(None, {}, ({}, "cli", None), id="args-none"),
        pytest.param({"samples": 5}, {}, ({"samples": 5}, "cli", None), id="args-given"),
        pytest.param(None, {"GYMRAT_COMMAND_ORIGIN": "tool"}, ({}, "tool", None), id="origin-tool"),
        pytest.param(
            None,
            {"TRACEPARENT": _OUTER_TRACEPARENT},
            ({}, "cli", _OUTER_TRACEPARENT),
            id="traceparent-only",
        ),
        pytest.param(
            None,
            {"TRACEPARENT": _OUTER_TRACEPARENT, "GYMRAT_TRACEPARENT": "00-123-456-01"},
            ({}, "cli", "00-123-456-01"),
            id="gymrat-traceparent-wins",
        ),
        pytest.param(
            None,
            {"TRACEPARENT": _OUTER_TRACEPARENT, "GYMRAT_TRACEPARENT": ""},
            ({}, "cli", _OUTER_TRACEPARENT),
            id="empty-gymrat-traceparent-falls-back",
        ),
    ],
)
async def test_with_repo_lock_when_body_succeeds_does_append_its_command_record(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    args: dict[str, object] | None,
    env: dict[str, str],
    expected: tuple[dict[str, object], str, str | None],
):
    _seeded_session(repo)
    frozen_ns = 1_000_000_000
    monotonic_readings = iter([100.0, 350.0])
    for name in ("GYMRAT_COMMAND_ORIGIN", "TRACEPARENT", "GYMRAT_TRACEPARENT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("gymrat.clock.now_ns", lambda: frozen_ns)
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: next(monotonic_readings))

    await with_repo_lock("measure", _ok_body, args=args)

    cmd = _last_command_record()
    assert (cmd.name, cmd.exit_code, cmd.reason, cmd.at, cmd.duration_ms) == (
        "measure",
        0,
        None,
        frozen_ns,
        250,
    )
    assert (cmd.args, cmd.origin, cmd.traceparent) == expected


async def _gate_body(trace: CommandTrace) -> str:
    trace.gate = True
    trace.reason = "gating-regression"
    return "gated"


async def _loop_stop_body(trace: CommandTrace) -> str:
    msg = "stopping"
    raise LoopStopError(msg)


async def _loop_stop_with_reason_body(trace: CommandTrace) -> str:
    msg = "out of time"
    raise LoopStopError(msg, reason="budget-exceeded")


async def _exit_with_reason_body(trace: CommandTrace) -> str:
    trace.reason = "fail-on"
    raise typer.Exit(code=1)


async def _exit_two_body(trace: CommandTrace) -> str:
    raise typer.Exit(code=2)


async def _exit_zero_body(trace: CommandTrace) -> str:
    raise typer.Exit(code=0)


async def _exit_three_with_reason_body(trace: CommandTrace) -> str:
    trace.reason = "fail-on"
    raise typer.Exit(code=3)


async def _exit_three_body(trace: CommandTrace) -> str:
    raise typer.Exit(code=3)


async def _gymrat_error_with_reason_body(trace: CommandTrace) -> str:
    msg = "boom"
    raise GymratError(msg, reason="no-session")


async def _gymrat_error_body(trace: CommandTrace) -> str:
    msg = "boom"
    raise GymratError(msg)


async def _unexpected_error_body(trace: CommandTrace) -> str:
    msg = "unexpected"
    raise RuntimeError(msg)


async def _seq_body(trace: CommandTrace) -> str:
    trace.seq = 7
    return "ok"


async def _seq_then_error_body(trace: CommandTrace) -> str:
    trace.seq = 7
    msg = "boom"
    raise GymratError(msg)


@pytest.mark.parametrize(
    ("body", "raised", "recorded"),
    [
        pytest.param(_gate_body, None, (1, "gating-regression"), id="gate-set"),
        pytest.param(_loop_stop_body, LoopStopError, (1, "stop-condition"), id="loop-stop"),
        pytest.param(
            _loop_stop_with_reason_body,
            LoopStopError,
            (1, "budget-exceeded"),
            id="loop-stop-with-reason",
        ),
        pytest.param(_exit_with_reason_body, typer.Exit, (1, "fail-on"), id="exit-1"),
        pytest.param(_exit_two_body, typer.Exit, (2, "error"), id="exit-2-no-reason"),
        pytest.param(_exit_zero_body, typer.Exit, (0, None), id="exit-0"),
        pytest.param(
            _exit_three_with_reason_body, typer.Exit, (2, "fail-on"), id="exit-3-with-reason"
        ),
        pytest.param(_exit_three_body, typer.Exit, (2, "error"), id="exit-3-no-reason"),
        pytest.param(
            _gymrat_error_with_reason_body,
            GymratError,
            (2, "no-session"),
            id="gymrat-error-with-reason",
        ),
        pytest.param(_gymrat_error_body, GymratError, (2, "error"), id="gymrat-error"),
        pytest.param(_unexpected_error_body, RuntimeError, (2, "error"), id="unexpected"),
    ],
)
async def test_with_repo_lock_when_body_ends_does_record_its_exit_code_and_reason(
    repo: str,
    body: Callable[[CommandTrace], Awaitable[str]],
    raised: type[BaseException] | None,
    recorded: tuple[int, CommandReason | None],
):
    _seeded_session(repo)

    with pytest.raises(raised) if raised is not None else contextlib.nullcontext():
        await with_repo_lock("iterate", body)

    cmd = _last_command_record()
    assert (cmd.exit_code, cmd.reason) == recorded


@pytest.mark.parametrize(
    ("body", "raised"),
    [
        pytest.param(_seq_body, None, id="seq-set"),
        pytest.param(_seq_then_error_body, GymratError, id="seq-set-then-raise"),
    ],
)
async def test_with_repo_lock_when_body_sets_seq_does_record_it(
    repo: str,
    body: Callable[[CommandTrace], Awaitable[str]],
    raised: type[BaseException] | None,
):
    _seeded_session(repo)

    with pytest.raises(raised) if raised is not None else contextlib.nullcontext():
        await with_repo_lock("iterate", body)

    assert _last_command_record().seq == 7


# ---------------------------------------------------------------------------
# with_repo_lock — failed append warning
# ---------------------------------------------------------------------------


async def _result_body(trace: CommandTrace) -> str:
    return "result"


async def _failing_body(trace: CommandTrace) -> str:
    msg = "body failed"
    raise RuntimeError(msg)


@pytest.mark.parametrize(
    ("command", "append", "warning", "body", "raised", "returned"),
    [
        pytest.param(
            "measure",
            _broken_append,
            "failed to append command record: disk full",
            _result_body,
            None,
            ["result"],
            id="append-fails-body-returns",
        ),
        pytest.param(
            "",
            None,
            "failed to append command record: 1 validation error for CommandRecord",
            _failing_body,
            RuntimeError,
            [],
            id="record-construction-fails-body-raises",
        ),
    ],
)
async def test_with_repo_lock_when_recording_fails_does_keep_the_body_outcome_with_a_warning(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    command: str,
    append: Callable[[str, object], None] | None,
    warning: str,
    body: Callable[[CommandTrace], Awaitable[str]],
    raised: type[BaseException] | None,
    returned: list[str],
):
    _seeded_session(repo)
    if append is not None:
        monkeypatch.setattr("gymrat.command_run.append_record", append)
    outcome: list[str] = []

    with pytest.raises(raised) if raised is not None else contextlib.nullcontext():
        outcome.append(await with_repo_lock(command, body))

    assert outcome == returned
    assert capsys.readouterr().err.splitlines()[0] == warning
    assert not is_held(lockfile_path(repo_root()))


# ---------------------------------------------------------------------------
# with_repo_lock — explicit root
# ---------------------------------------------------------------------------


@pytest.fixture
def plain_directory(tmp_path: Path) -> Iterator[str]:
    """A directory that is not a git repository, with its lock files removed on teardown."""
    directory = tmp_path / "plain"
    directory.mkdir()
    yield str(directory)
    remove_lock_files(str(directory))


async def test_with_repo_lock_when_root_given_does_operate_on_that_repo_not_the_cwd(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    cwd_repo = create_scratch_repo()
    target_repo = create_scratch_repo()
    monkeypatch.chdir(cwd_repo)
    cwd_header = session_record()
    write_session_log(cwd_repo, cwd_header)
    write_session_log(target_repo, session_record())
    jsonl_path = Path(session_jsonl_path(target_repo))
    intact_log = await asyncio.to_thread(jsonl_path.read_bytes)
    tear_final_line(jsonl_path)
    seen: dict[str, object] = {}

    async def body(trace: CommandTrace) -> str:
        seen["log"] = await asyncio.to_thread(jsonl_path.read_bytes)
        seen["held"] = (
            is_held(lockfile_path(target_repo)),
            is_held(lockfile_path(cwd_repo)),
        )
        return "ran"

    result = await with_repo_lock("measure", body, args={"samples": 5}, root=target_repo)

    cmd = log_records(target_repo)[-1]
    assert result == "ran"
    assert seen == {"log": intact_log, "held": (True, False)}
    assert isinstance(cmd, CommandRecord)
    assert (cmd.name, cmd.args) == ("measure", {"samples": 5})
    assert log_records(cwd_repo) == [cwd_header]


async def test_with_repo_lock_when_root_is_not_a_repository_does_hold_its_lock_without_consulting_git(
    plain_directory: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def broken_git(*_args: object, **_kwargs: object) -> str:
        message = "detected dubious ownership"
        raise GymratError(message)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("gymrat.command_run.repo_root", broken_git)
    held: dict[str, bool] = {}

    async def body(trace: CommandTrace) -> str:
        held["target"] = is_held(lockfile_path(plain_directory))
        return "ran"

    result = await with_repo_lock("compare", body, root=plain_directory)

    assert result == "ran"
    assert held["target"] is True


# ---------------------------------------------------------------------------
# with_repo_lock — command span export (tracing enabled)
# ---------------------------------------------------------------------------

#: A W3C traceparent whose parent span id the command span must pick up.
_VALID_TRACEPARENT = "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01"


async def test_with_repo_lock_when_tracing_configured_for_another_session_does_degrade_to_a_warning(
    repo: str,
    capsys: pytest.CaptureFixture[str],
):
    _seeded_session(repo)

    with memory_tracing(_FRESH_SESSION_ID):
        result = await with_repo_lock("measure", _ok_body)

    warnings = capsys.readouterr().err.splitlines()
    assert result == "ok"
    assert len(warnings) == 1
    assert warnings[0].startswith("failed to emit command span: ")
    assert not is_held(lockfile_path(repo_root()))


async def test_with_repo_lock_when_tracing_enabled_does_export_one_keyed_command_span_under_the_session(
    repo: str,
):
    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body, args={"samples": 5})

    span = span_by_name(exporter.get_finished_spans(), "gymrat.command.measure")
    context = span.context
    attrs = dict(span.attributes or {})
    assert context is not None
    assert context.trace_id == trace_id_of(header.session_id)
    assert context.span_id == span_id_of(header.session_id, f"command:{len(log_records(repo))}")
    assert span.parent is not None
    assert span.parent.span_id == span_id_of(header.session_id, "session")
    assert span.status.status_code == StatusCode.OK
    assert span.links == ()
    assert attrs["gymrat.session.id"] == header.session_id
    assert attrs["gymrat.command.name"] == "measure"
    assert attrs["gymrat.command.exit_code"] == 0
    assert attrs["gymrat.command.args.samples"] == 5


async def test_with_repo_lock_when_body_appends_records_does_add_span_events(repo: str):
    header = _seeded_session(repo)

    async def body(trace: CommandTrace) -> str:
        append_records(repo_root(), iteration_record())
        return "ok"

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("iterate", body)

    span = span_by_name(exporter.get_finished_spans(), "gymrat.command.iterate")
    assert "gymrat.iteration" in [e.name for e in span.events]


@pytest.mark.parametrize(
    ("gymrat_traceparent", "parent_key"),
    [
        pytest.param(_VALID_TRACEPARENT, None, id="valid-is-the-parent"),
        pytest.param("not-a-valid-traceparent", "session", id="malformed-falls-back-to-session"),
    ],
)
async def test_with_repo_lock_when_gymrat_traceparent_set_does_parent_on_it_only_when_valid(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    gymrat_traceparent: str,
    parent_key: str | None,
):
    header = _seeded_session(repo)
    monkeypatch.setenv("GYMRAT_TRACEPARENT", gymrat_traceparent)
    expected_parent = (
        0x1112131415161718 if parent_key is None else span_id_of(header.session_id, parent_key)
    )

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    span = span_by_name(exporter.get_finished_spans(), "gymrat.command.measure")
    assert span.parent is not None
    assert span.parent.span_id == expected_parent


async def test_with_repo_lock_when_otlp_exporter_missing_does_run_body_without_tracing(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _seeded_session(repo)
    hide_otlp_exporter(monkeypatch)

    with otlp_collector() as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint)
        result = await with_repo_lock("measure", _ok_body)

    records = log_records(repo_root())
    assert result == "ok"
    assert isinstance(records[-1], CommandRecord)
    assert records[-1].name == "measure"
    assert collector.received == []
    assert capsys.readouterr().err == ""


async def test_with_repo_lock_when_tracing_enabled_does_flush_before_return(repo: str):
    header = _seeded_session(repo)

    with memory_tracing(header.session_id, buffered=True) as exporter:
        await with_repo_lock("measure", _ok_body)

        spans_before_exit = exporter.get_finished_spans()

    assert [s.name for s in spans_by_prefix(spans_before_exit, "gymrat.command.")] == [
        "gymrat.command.measure"
    ]


# ---------------------------------------------------------------------------
# with_repo_lock — a start that opens a new session log
# ---------------------------------------------------------------------------


def _start_body(previous: SessionRecord) -> Callable[[CommandTrace], Awaitable[str]]:
    """A ``start`` body that replaces ``previous``'s log with a fresh session's."""

    async def body(trace: CommandTrace) -> str:
        archive_and_reopen_session_log(previous, session_record(session_id=_FRESH_SESSION_ID))
        return "started"

    return body


def _seeded_long_session(repo: str) -> SessionRecord:
    """Write a session header followed by three iterations, longer than a fresh session's log."""
    header = session_record()
    write_session_log(repo, header, tuple(iteration_record(seq=seq) for seq in (1, 2, 3)))
    return header


async def test_with_repo_lock_when_start_opens_a_new_session_does_trace_the_log_holding_the_record(
    repo: str,
):
    previous = _seeded_long_session(repo)
    baseline_event, _attrs = record_event(baseline_record())
    iteration_event, _attrs = record_event(iteration_record())

    with memory_tracing(_FRESH_SESSION_ID) as exporter:
        await with_repo_lock("start", _start_body(previous))

    records = log_records(repo_root())
    assert isinstance(records[-1], CommandRecord)
    holder = session_header(repo_root())
    assert holder is not None
    span = span_by_name(exporter.get_finished_spans(), "gymrat.command.start")
    assert dict(span.attributes or {})["gymrat.session.id"] == holder.session_id
    context = span.context
    assert context is not None
    assert (context.trace_id, context.span_id) == (
        trace_id_of(holder.session_id),
        span_id_of(holder.session_id, f"command:{len(records)}"),
    )
    event_names = [e.name for e in span.events]
    assert baseline_event in event_names
    assert iteration_event not in event_names


async def test_with_repo_lock_when_start_opens_the_first_session_does_key_span_on_it(
    repo: str,
):
    async def first_start(trace: CommandTrace) -> str:
        write_session_log(repo_root(), session_record(session_id=_FRESH_SESSION_ID))
        return "started"

    with memory_tracing(_FRESH_SESSION_ID) as exporter:
        await with_repo_lock("start", first_start)

    records = log_records(repo_root())
    span = span_by_name(exporter.get_finished_spans(), "gymrat.command.start")
    assert dict(span.attributes or {})["gymrat.session.id"] == _FRESH_SESSION_ID
    context = span.context
    assert context is not None
    assert (context.trace_id, context.span_id) == (
        trace_id_of(_FRESH_SESSION_ID),
        span_id_of(_FRESH_SESSION_ID, f"command:{len(records)}"),
    )


async def test_with_repo_lock_when_start_opens_a_new_session_under_gymrat_traceparent_does_parent_on_it(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    previous = _seeded_long_session(repo)
    env_trace_id = 0x0102030405060708090A0B0C0D0E0F10
    env_span_id = 0x1112131415161718
    monkeypatch.setenv("GYMRAT_TRACEPARENT", f"00-{env_trace_id:032x}-{env_span_id:016x}-01")

    with memory_tracing(_FRESH_SESSION_ID) as exporter:
        await with_repo_lock("start", _start_body(previous))

    records = log_records(repo_root())
    span = span_by_name(exporter.get_finished_spans(), "gymrat.command.start")
    assert span.parent is not None
    assert (span.parent.trace_id, span.parent.span_id) == (env_trace_id, env_span_id)
    assert dict(span.attributes or {})["gymrat.session.id"] == _FRESH_SESSION_ID
    context = span.context
    assert context is not None
    assert (context.trace_id, context.span_id) == (
        env_trace_id,
        span_id_of(_FRESH_SESSION_ID, f"command:{len(records)}"),
    )


async def test_with_repo_lock_when_start_cannot_append_its_record_does_emit_no_command_span(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    previous = _seeded_long_session(repo)

    def broken_append(path: str, record: object) -> None:
        msg = "disk full"
        raise GymratError(msg)

    monkeypatch.setattr("gymrat.command_run.append_record", broken_append)

    with memory_tracing(_FRESH_SESSION_ID) as exporter:
        await with_repo_lock("start", _start_body(previous))

    assert spans_by_prefix(exporter.get_finished_spans(), "gymrat.command.") == []


# ---------------------------------------------------------------------------
# with_repo_lock — command span export to a real collector
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_collector_rejects_the_export_does_return_body_result(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):

    _seeded_session(repo)

    with otlp_collector(statuses=[400]) as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint)
        result = await with_repo_lock("measure", _ok_body)

    assert (result, export_failed(), "gymrat.command.measure" in collector.span_names) == (
        "ok",
        True,
        True,
    )


async def test_with_repo_lock_when_first_start_traced_only_by_endpoint_does_export_its_span_keyed_on_the_new_session(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    async def first_start(trace: CommandTrace) -> str:
        write_session_log(repo_root(), session_record(session_id=_FRESH_SESSION_ID))
        return "started"

    with otlp_collector() as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint)
        await with_repo_lock("start", first_start)

    start_sessions = [
        span.attributes.get("gymrat.session.id")
        for span in collector.spans
        if span.name == "gymrat.command.start"
    ]
    assert start_sessions == [_FRESH_SESSION_ID]


async def test_with_repo_lock_when_endpoint_whitespace_only_does_export_nothing(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):

    _seeded_session(repo)

    with otlp_collector() as collector:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", " \t ")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"{collector.endpoint}/v1/traces")
        await with_repo_lock("measure", _ok_body)

    assert collector.received == []


# ---------------------------------------------------------------------------
# layering — the seam stays free of the CLI package
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [
        pytest.param("gymrat.command_run", id="command-run"),
        pytest.param("gymrat.supervisor.exit_sequence", id="exit-sequence"),
    ],
)
def test_importing_module_when_fresh_interpreter_does_not_load_the_cli_package(module: str):
    loaded = modules_imported_by(module)

    assert loaded_under(loaded, "gymrat.cli") == []
