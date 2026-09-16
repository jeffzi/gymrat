"""Tests for the repository lock, the command trace, and the command-record seam.

The seam lives in :mod:`gymrat.cli.lock`; these tests exercise the
``CommandTrace`` dataclass, the ``with_repo_lock`` wrapper's recording behavior,
and the edge cases around missing session logs and failed appends.
"""

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import typer
from filelock import FileLock, Timeout

from gymrat.cli.shared import (
    TOOL_FAILURE_EXIT_CODE,
    CommandTrace,
    with_repo_lock,
)
from gymrat.errors import GymratError
from gymrat.git import NotAGitRepositoryError
from gymrat.loop.iterate import LoopStopError
from gymrat.session import (
    CommandRecord,
    append_record,
    read_records,
    session_jsonl_path,
)
from gymrat.session.lock import _os_lock_file
from gymrat.session.paths import lockfile_path, repo_root
from tests.cli._lock_fixtures import (
    isolate_tracing_provider as _isolate_tracing_provider,  # noqa: F401 -- registers the autouse fixture
)
from tests.cli._lock_fixtures import (
    seeded_session as _seeded_session,
)
from tests.cli._lock_fixtures import (
    seeded_session_no_trace_context as _seeded_session_no_trace_context,
)
from tests.session.records._fixtures import (
    iteration_record,
    session_record,
    tear_final_line,
    write_session_log,
)


def _read_bytes(path: Path) -> bytes:
    """Filesystem read kept out of the async body so it is not flagged as blocking I/O."""
    return path.read_bytes()


def _not_a_repo(*_args: object, **_kwargs: object) -> str:
    message = "nope"
    raise NotAGitRepositoryError(message)


def _broken_append(path: str, record: object) -> None:
    msg = "disk full"
    raise GymratError(msg)


def _broken_record(*_args: object, **_kwargs: object) -> None:
    msg = "record construction failed"
    raise ValueError(msg)


async def _ok_body(trace: CommandTrace) -> str:
    """Trivial command body for tests that only inspect recorded or exported side effects."""
    return "ok"


def _last_command_record() -> CommandRecord:
    """Read the current session log and return its last record, asserted to be a command."""
    jsonl_path = session_jsonl_path(repo_root())
    records = read_records(jsonl_path)
    cmd = records[-1]
    assert isinstance(cmd, CommandRecord)
    return cmd


def _tracking_acquire(released: list[bool]) -> Callable[[str, str], Callable[[], None]]:
    """Wrap the real lock acquire so each ``release()`` call is recorded in ``released``."""
    from gymrat.session.lock import acquire_lock as real_acquire

    def spy_acquire(lock_path: str, command: str) -> Callable[[], None]:
        release = real_acquire(lock_path, command)

        def tracked_release() -> None:
            released.append(True)
            release()

        return tracked_release

    return spy_acquire


# ---------------------------------------------------------------------------
# CommandTrace dataclass
# ---------------------------------------------------------------------------


def test_command_trace_when_default_does_carry_empty_args_and_none_fields():
    trace = CommandTrace()

    assert trace.args == {}
    assert trace.seq is None
    assert trace.gate is False
    assert trace.reason is None


def test_command_trace_when_constructed_with_args_does_carry_them():
    trace = CommandTrace(args={"samples": 10}, seq=3, gate=True, reason="stop-condition")

    assert trace.args == {"samples": 10}
    assert trace.seq == 3
    assert trace.gate is True
    assert trace.reason == "stop-condition"


def test_command_trace_when_mutated_does_reflect_new_values():
    trace = CommandTrace()

    trace.seq = 5
    trace.gate = True
    trace.reason = "budget-exceeded"

    assert trace.seq == 5
    assert trace.gate is True  # pyrefly: ignore[unnecessary-comparison] -- verifying mutation
    assert trace.reason == "budget-exceeded"


# ---------------------------------------------------------------------------
# with_repo_lock — locking behavior
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_inside_repo_holds_lock_during_body_and_releases_before_return(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    repo = create_scratch_repo()
    monkeypatch.chdir(repo)
    lock_path = lockfile_path(repo_root())
    os_lock_path = _os_lock_file(lock_path)
    probed: dict[str, bool] = {}

    async def body(trace: CommandTrace) -> str:
        try:
            FileLock(os_lock_path, timeout=0).acquire()
            probed["held_during"] = False
        except Timeout:
            probed["held_during"] = True
        return "measured"

    result = await with_repo_lock("compare", body)

    assert result == "measured"
    assert probed["held_during"] is True
    try:
        probe = FileLock(os_lock_path, timeout=0)
        probe.acquire()
        probe.release()
    except Timeout:
        pytest.fail("lock was still held after with_repo_lock returned")


async def test_with_repo_lock_when_outside_repo_runs_body_without_a_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    acquired: list[object] = []

    def spy_acquire(*args: object, **_kwargs: object):
        acquired.append(args)
        return lambda: None

    monkeypatch.setattr("gymrat.cli.lock.repo_root", _not_a_repo)
    monkeypatch.setattr("gymrat.cli.lock.acquire_lock", spy_acquire)

    async def body(trace: CommandTrace) -> str:
        return "ran"

    result = await with_repo_lock("compare", body)

    assert result == "ran"
    assert acquired == []


async def test_with_repo_lock_when_session_log_torn_does_repair_it_before_running_the_body(
    repo: str,
):
    header = session_record()
    write_session_log(repo, header)
    jsonl_path = Path(session_jsonl_path(repo_root()))
    intact_log = _read_bytes(jsonl_path)
    tear_final_line(jsonl_path)
    iteration = iteration_record()
    seen: dict[str, bytes] = {}

    async def body(trace: CommandTrace) -> str:
        seen["log"] = _read_bytes(jsonl_path)
        append_record(str(jsonl_path), iteration)
        return "ran"

    result = await with_repo_lock("compare", body)

    assert result == "ran"
    assert seen["log"] == intact_log
    records = read_records(str(jsonl_path))
    assert records[0] == header
    assert records[1] == iteration
    assert isinstance(records[-1], CommandRecord)


async def test_with_repo_lock_when_git_fails_otherwise_exits_two_without_running_body(
    monkeypatch: pytest.MonkeyPatch,
):
    def broken_git(*_args: object, **_kwargs: object) -> str:
        message = "detected dubious ownership"
        raise GymratError(message)

    monkeypatch.setattr("gymrat.cli.lock.repo_root", broken_git)
    called: list[bool] = []

    async def body(trace: CommandTrace) -> str:
        called.append(True)
        return "should-not-run"

    with pytest.raises(typer.Exit) as exc:
        await with_repo_lock("compare", body)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE
    assert called == []


# ---------------------------------------------------------------------------
# with_repo_lock — command trace passing
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_body_runs_does_pass_command_trace_with_args(
    repo: str,
):
    received: list[CommandTrace] = []

    async def body(trace: CommandTrace) -> str:
        received.append(trace)
        return "ok"

    await with_repo_lock("measure", body, args={"samples": 5})

    assert len(received) == 1
    assert received[0].args == {"samples": 5}


async def test_with_repo_lock_when_args_is_none_does_pass_empty_dict(
    repo: str,
):
    received: list[CommandTrace] = []

    async def body(trace: CommandTrace) -> str:
        received.append(trace)
        return "ok"

    await with_repo_lock("measure", body)

    assert received[0].args == {}


# ---------------------------------------------------------------------------
# with_repo_lock — command record appending
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_body_succeeds_does_append_command_record_with_exit_zero(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)
    frozen_ns = 1_000_000_000
    monkeypatch.setattr("gymrat.cli.lock._clock.now_ns", lambda: frozen_ns)

    await with_repo_lock("measure", _ok_body, args={"samples": 5})

    cmd = _last_command_record()
    assert cmd.name == "measure"
    assert cmd.args == {"samples": 5}
    assert cmd.exit_code == 0
    assert cmd.reason is None
    assert cmd.duration_ms >= 0
    assert cmd.at == frozen_ns
    assert cmd.traceparent is None


async def test_with_repo_lock_when_body_sets_gate_does_record_exit_one_with_trace_reason(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)

    async def body(trace: CommandTrace) -> str:
        trace.gate = True
        trace.reason = "gating-regression"
        return "gated"

    await with_repo_lock("compare", body, args={})

    cmd = _last_command_record()
    assert cmd.exit_code == 1
    assert cmd.reason == "gating-regression"


async def test_with_repo_lock_when_body_raises_loop_stop_error_does_record_exit_one_with_error_reason(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)

    async def body(trace: CommandTrace) -> str:
        msg = "stopping"
        raise LoopStopError(msg)

    with pytest.raises(LoopStopError):
        await with_repo_lock("iterate", body)

    cmd = _last_command_record()
    assert cmd.exit_code == 1
    assert cmd.reason == "stop-condition"


async def test_with_repo_lock_when_body_raises_typer_exit_does_record_its_exit_code(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)

    async def body(trace: CommandTrace) -> str:
        trace.reason = "fail-on"
        raise typer.Exit(code=1)

    with pytest.raises(typer.Exit):
        await with_repo_lock("compare", body)

    cmd = _last_command_record()
    assert cmd.exit_code == 1
    assert cmd.reason == "fail-on"


async def test_with_repo_lock_when_typer_exit_code_two_and_no_reason_does_default_to_error(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)

    async def body(trace: CommandTrace) -> str:
        raise typer.Exit(code=2)

    with pytest.raises(typer.Exit):
        await with_repo_lock("compare", body)

    cmd = _last_command_record()
    assert cmd.exit_code == 2
    assert cmd.reason == "error"


async def test_with_repo_lock_when_body_raises_gymrat_error_does_record_exit_two(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)

    async def body(trace: CommandTrace) -> str:
        msg = "boom"
        raise GymratError(msg, reason="no-session")

    with pytest.raises(GymratError):
        await with_repo_lock("measure", body)

    cmd = _last_command_record()
    assert cmd.exit_code == 2
    assert cmd.reason == "no-session"


async def test_with_repo_lock_when_gymrat_error_has_no_reason_does_default_to_error(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)

    async def body(trace: CommandTrace) -> str:
        msg = "boom"
        raise GymratError(msg)

    with pytest.raises(GymratError):
        await with_repo_lock("measure", body)

    cmd = _last_command_record()
    assert cmd.exit_code == 2
    assert cmd.reason == "error"


async def test_with_repo_lock_when_body_raises_unexpected_error_does_record_exit_two_error(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)

    async def body(trace: CommandTrace) -> str:
        msg = "unexpected"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="unexpected"):
        await with_repo_lock("measure", body)

    cmd = _last_command_record()
    assert cmd.exit_code == 2
    assert cmd.reason == "error"


async def test_with_repo_lock_when_body_sets_seq_does_record_it(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)

    async def body(trace: CommandTrace) -> str:
        trace.seq = 7
        return "ok"

    await with_repo_lock("iterate", body)

    cmd = _last_command_record()
    assert cmd.seq == 7


async def test_with_repo_lock_when_traceparent_env_set_does_record_it(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)
    monkeypatch.setenv("TRACEPARENT", "00-abc-def-01")

    await with_repo_lock("measure", _ok_body)

    cmd = _last_command_record()
    assert cmd.traceparent == "00-abc-def-01"


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        pytest.param("tool", "tool", id="env-tool"),
        pytest.param("banana", "cli", id="env-other-value"),
        pytest.param(None, "cli", id="env-absent"),
    ],
)
async def test_with_repo_lock_when_command_origin_env_varies_does_record_matching_origin(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
    expected: str,
):
    _seeded_session(repo, monkeypatch)
    if env_value is None:
        monkeypatch.delenv("GYMRAT_COMMAND_ORIGIN", raising=False)
    else:
        monkeypatch.setenv("GYMRAT_COMMAND_ORIGIN", env_value)

    await with_repo_lock("measure", _ok_body)

    cmd = _last_command_record()
    assert cmd.origin == expected


async def test_with_repo_lock_when_duration_recorded_does_reflect_wall_time(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)
    call_count = 0

    def fake_monotonic_ms() -> float:
        nonlocal call_count
        call_count += 1
        return 100.0 if call_count == 1 else 350.0

    monkeypatch.setattr("gymrat.cli.lock._clock.monotonic_ms", fake_monotonic_ms)

    await with_repo_lock("measure", _ok_body)

    cmd = _last_command_record()
    assert cmd.duration_ms == 250


# ---------------------------------------------------------------------------
# with_repo_lock — no append without session log
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_no_session_log_does_not_append(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    appended: list[object] = []

    def _fake_append(path: str, record: object) -> None:
        appended.append(record)

    monkeypatch.setattr("gymrat.cli.lock.append_record", _fake_append)

    result = await with_repo_lock("measure", _ok_body)

    assert result == "ok"
    assert appended == []


async def test_with_repo_lock_when_outside_repo_does_not_append(
    monkeypatch: pytest.MonkeyPatch,
):
    def _fake_acquire(*_a: object, **_kw: object) -> Callable[[], None]:
        return lambda: None

    monkeypatch.setattr("gymrat.cli.lock.repo_root", _not_a_repo)
    monkeypatch.setattr("gymrat.cli.lock.acquire_lock", _fake_acquire)
    appended: list[object] = []

    def _fake_append(path: str, record: object) -> None:
        appended.append(record)

    monkeypatch.setattr("gymrat.cli.lock.append_record", _fake_append)

    async def body(trace: CommandTrace) -> str:
        return "ran"

    await with_repo_lock("compare", body)

    assert appended == []


# ---------------------------------------------------------------------------
# with_repo_lock — failed append warning
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_append_fails_does_warn_and_still_return_body_result(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _seeded_session(repo, monkeypatch)
    monkeypatch.setattr("gymrat.cli.lock.append_record", _broken_append)

    async def body(trace: CommandTrace) -> str:
        return "result"

    result = await with_repo_lock("measure", body)

    assert result == "result"
    captured = capsys.readouterr()
    assert "warn" in captured.err.lower() or "disk full" in captured.err.lower()


async def test_with_repo_lock_when_append_fails_on_exception_does_warn_and_reraise(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _seeded_session(repo, monkeypatch)
    monkeypatch.setattr("gymrat.cli.lock.append_record", _broken_append)

    async def body(trace: CommandTrace) -> str:
        msg = "body failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="body failed"):
        await with_repo_lock("measure", body)

    captured = capsys.readouterr()
    assert "disk full" in captured.err


async def test_with_repo_lock_when_record_construction_raises_does_warn_and_return_body_result(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _seeded_session(repo, monkeypatch)
    monkeypatch.setattr("gymrat.cli.lock.CommandRecord", _broken_record)

    async def body(trace: CommandTrace) -> str:
        return "result"

    result = await with_repo_lock("measure", body)

    assert result == "result"
    captured = capsys.readouterr()
    assert "record construction failed" in captured.err


async def test_with_repo_lock_when_record_construction_raises_on_body_exception_does_warn_and_reraise(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _seeded_session(repo, monkeypatch)
    monkeypatch.setattr("gymrat.cli.lock.CommandRecord", _broken_record)

    async def body(trace: CommandTrace) -> str:
        msg = "body failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="body failed"):
        await with_repo_lock("measure", body)

    captured = capsys.readouterr()
    assert "record construction failed" in captured.err


# ---------------------------------------------------------------------------
# session_header
# ---------------------------------------------------------------------------


def test_session_header_when_log_absent_does_return_none(repo: str):
    from gymrat.session.store import session_header

    result = session_header(repo_root())

    assert result is None


def test_session_header_when_first_line_is_session_record_does_return_it(repo: str):
    from gymrat.session.store import session_header

    header = session_record()
    write_session_log(repo, header, (iteration_record(),))

    result = session_header(repo_root())

    assert result == header


def test_session_header_when_first_line_is_not_session_record_does_return_none(
    repo: str,
):
    from gymrat.session.store import session_header

    jsonl_path = session_jsonl_path(repo_root())
    append_record(jsonl_path, iteration_record())

    result = session_header(repo_root())

    assert result is None


def test_session_header_when_called_does_not_read_entire_log(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    from gymrat.session.store import session_header

    header = session_record()
    write_session_log(repo, header, (iteration_record(),) * 50)

    reads: list[int] = []
    original_path_open = Path.open

    def counting_open(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        f = original_path_open(self, *args, **kwargs)  # type: ignore[arg-type]
        original_readline = f.readline

        def tracked_readline(*a: object, **kw: object) -> str:
            line = original_readline(*a, **kw)
            reads.append(len(line))
            return line  # type: ignore[return-value]

        f.readline = tracked_readline
        return f

    monkeypatch.setattr("pathlib.Path.open", counting_open)

    session_header(repo_root())

    assert len(reads) == 1


# ---------------------------------------------------------------------------
# with_repo_lock — lock release guarantee
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_record_construction_raises_does_release_lock(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _seeded_session(repo, monkeypatch)
    released: list[bool] = []
    monkeypatch.setattr("gymrat.cli.lock.acquire_lock", _tracking_acquire(released))
    monkeypatch.setattr("gymrat.cli.lock.CommandRecord", _broken_record)

    with contextlib.suppress(Exception):
        await with_repo_lock("measure", _ok_body)

    assert released, "release() was never called"


async def test_with_repo_lock_when_span_emission_raises_does_release_lock(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    header = _seeded_session_no_trace_context(repo, monkeypatch)
    released: list[bool] = []
    monkeypatch.setattr("gymrat.cli.lock.acquire_lock", _tracking_acquire(released))

    span_called: list[bool] = []

    def broken_emit(*_args: object, **_kwargs: object) -> None:
        span_called.append(True)
        msg = "span export failed"
        raise RuntimeError(msg)

    monkeypatch.setattr("gymrat.cli.lock._emit_command_span", broken_emit)

    def stub_configure(_root: str, _jsonl: str) -> tuple[str, bool]:
        return (header.session_id, True)

    monkeypatch.setattr("gymrat.cli.lock._maybe_configure_tracing", stub_configure)

    await with_repo_lock("measure", _ok_body)

    assert span_called, "_emit_command_span was not reached"
    assert released, "release() was never called"


# ---------------------------------------------------------------------------
# with_repo_lock — explicit root
# ---------------------------------------------------------------------------


def _lock_is_held(lock_path: str) -> bool:
    """Probe the OS lock behind ``lock_path`` without blocking, releasing it if free."""
    probe = FileLock(_os_lock_file(lock_path), timeout=0)
    try:
        probe.acquire()
    except Timeout:
        return True
    probe.release()
    return False


@pytest.fixture
def plain_directory(tmp_path: Path) -> Iterator[str]:
    """A directory that is not a git repository, with its lock files removed on teardown."""
    directory = tmp_path / "plain"
    directory.mkdir()
    yield str(directory)
    lock_path = lockfile_path(str(directory))
    Path(lock_path).unlink(missing_ok=True)
    Path(_os_lock_file(lock_path)).unlink(missing_ok=True)


async def test_with_repo_lock_when_root_given_does_lock_that_repo_and_not_the_cwd_repo(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    cwd_repo = create_scratch_repo()
    target_repo = create_scratch_repo()
    monkeypatch.chdir(cwd_repo)
    held: dict[str, bool] = {}

    async def body(trace: CommandTrace) -> str:
        held["target"] = _lock_is_held(lockfile_path(target_repo))
        held["cwd"] = _lock_is_held(lockfile_path(cwd_repo))
        return "ran"

    result = await with_repo_lock("compare", body, root=target_repo)

    assert result == "ran"
    assert held == {"target": True, "cwd": False}


async def test_with_repo_lock_when_root_given_does_repair_that_repos_torn_session_log(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    cwd_repo = create_scratch_repo()
    target_repo = create_scratch_repo()
    monkeypatch.chdir(cwd_repo)
    write_session_log(target_repo, session_record())
    jsonl_path = Path(session_jsonl_path(target_repo))
    intact_log = _read_bytes(jsonl_path)
    tear_final_line(jsonl_path)
    seen: dict[str, bytes] = {}

    async def body(trace: CommandTrace) -> str:
        seen["log"] = _read_bytes(jsonl_path)
        return "ran"

    result = await with_repo_lock("compare", body, root=target_repo)

    assert result == "ran"
    assert seen["log"] == intact_log


async def test_with_repo_lock_when_root_given_does_append_command_record_to_that_repos_log(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    cwd_repo = create_scratch_repo()
    target_repo = create_scratch_repo()
    monkeypatch.chdir(cwd_repo)
    monkeypatch.delenv("TRACEPARENT", raising=False)
    cwd_header = session_record()
    write_session_log(cwd_repo, cwd_header)
    write_session_log(target_repo, session_record())

    await with_repo_lock("measure", _ok_body, args={"samples": 5}, root=target_repo)

    cmd = read_records(session_jsonl_path(target_repo))[-1]
    assert isinstance(cmd, CommandRecord)
    assert cmd.name == "measure"
    assert cmd.args == {"samples": 5}
    assert read_records(session_jsonl_path(cwd_repo)) == [cwd_header]


async def test_with_repo_lock_when_root_is_not_a_repository_does_still_hold_its_lock(
    plain_directory: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    held: dict[str, bool] = {}

    async def body(trace: CommandTrace) -> str:
        held["target"] = _lock_is_held(lockfile_path(plain_directory))
        return "ran"

    result = await with_repo_lock("compare", body, root=plain_directory)

    assert result == "ran"
    assert held["target"] is True
