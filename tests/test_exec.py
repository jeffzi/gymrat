"""Behavioral tests for the asyncio subprocess ``exec`` layer.

Real-subprocess tests are parallel-safe under ``pytest-xdist``: every run gets
its own directory via the ``tmp_path`` fixture, and POSIX-only shell constructs
(``$$``, ``>&2``, ``for``/``do``/``done``) mean the module is skipped on win32.
The win32 kill path is exercised by patching a platform seam *after* the
POSIX spawn and stubbing the ``taskkill`` subprocess call — no real Windows.
"""

import asyncio
import contextlib
import dataclasses
import errno
import json
import os
import signal
import subprocess
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import (
    ExecOptions,
    ExecResult,
    ExecTimeoutError,
    OutputBuffer,
)
from gymrat.exec import exec as run_exec
from gymrat.signals import TERMINATION_SIGNALS
from tests._process_helpers import (
    capture_spawns,
    is_alive,
    killpg_warnings,
    wait_for_pid_file,
    wait_until_dead,
)

# exec drives POSIX process groups (killpg) and sh-only shell syntax; neither
# works under cmd.exe, so the whole module is POSIX-only.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only shell and process groups"
)

# Runs repeated back to back to hit the window where the timeout fires while the
# shell is already exiting; a single run slips past it some of the time.
_RACE_RUNS = 40

# How long asyncio's child reaper is held back once it starts waiting on a
# shell. It outlasts exec's teardown of an exited shell, so the shell is still
# a zombie nobody has reaped while that teardown runs.
_REAP_HOLD_S = 1.0

# Timeout for teardown tests: fires well after the shell has exited, well
# before the held reap is released.
_TEARDOWN_TIMEOUT_MS = 250

# Upper bound for a cancelled exec to settle when its shell is never reaped.
_CANCEL_SETTLE_S = 5.0

# asyncio logs this when a child it waits on was reaped by someone else.
_UNKNOWN_CHILD = "Unknown child process"

_os_waitid: Callable[..., object] | None = getattr(os, "waitid", None)


async def wait_for_spawned(
    processes: list[asyncio.subprocess.Process],
    timeout_s: float = 3.0,
) -> asyncio.subprocess.Process:
    """Return the most recent child ``exec`` spawned, once the spawn has happened."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not processes:
        if loop.time() > deadline:
            msg = "exec() has not spawned a child yet"
            raise TimeoutError(msg)
        await asyncio.sleep(0.01)
    return processes[-1]


async def wait_for_shell_exit(proc: asyncio.subprocess.Process, timeout_s: float = 3.0) -> None:
    """Poll until the shell's stdout reaches EOF, which the shell exiting closes.

    The command must not hand stdout to anything that outlives the shell, and
    the check never reaps the shell itself.
    """
    assert proc.stdout is not None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not proc.stdout.at_eof():
        if loop.time() > deadline:
            msg = f"shell {proc.pid} never closed its stdout"
            raise TimeoutError(msg)
        await asyncio.sleep(0.01)


def pidfd_available() -> bool:
    """True when asyncio reaps children through a pidfd instead of a waitpid thread."""
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is None:
        return False
    try:
        os.close(pidfd_open(os.getpid()))
    except OSError:
        return False
    return True


@dataclasses.dataclass
class HeldReaper:
    """Stand-in for the ``os`` module asyncio's child watcher uses, holding back its blocking reap.

    Every attribute but ``waitpid`` and ``waitid`` forwards to the real ``os``
    module.  A blocking ``waitpid(pid, 0)`` or ``waitid(P_PID, …)`` first waits
    until ``release`` is set or ``hold_s`` passes.

    CPython 3.14.7+ changed ``_ThreadedChildWatcher._do_waitpid`` to call
    ``os.waitid(P_PID, pid, WEXITED | WNOWAIT)`` before scheduling the actual
    reap on the event-loop thread.  Without intercepting ``waitid``, the hold
    never fires and the reap completes before the test can cancel the task.
    """

    release: threading.Event
    hold_s: float

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)

    def waitpid(self, pid: int, options: int, /) -> tuple[int, int]:
        if options == 0:
            self.release.wait(self.hold_s)
        return os.waitpid(pid, options)

    if _os_waitid is not None:
        _P_PID: int = getattr(os, "P_PID", 0)

        def waitid(
            self,
            id_type: int,
            id_val: int,
            options: int,
            /,
        ) -> object:
            """Hold a blocking ``waitid`` the same way ``waitpid`` is held."""
            if id_type == self._P_PID:
                self.release.wait(self.hold_s)
            assert _os_waitid is not None
            return _os_waitid(id_type, id_val, options)


type ExecTask = asyncio.Task[ExecResult | ExecTimeoutError]


def leave_to_timeout(
    task: ExecTask, proc: asyncio.subprocess.Process, abort: asyncio.Event
) -> None:
    """Leave the run alone: its timeout tears it down."""


def set_abort(task: ExecTask, proc: asyncio.subprocess.Process, abort: asyncio.Event) -> None:
    """Tear the run down through its abort event."""
    abort.set()


def fail_stderr_read(
    task: ExecTask, proc: asyncio.subprocess.Process, abort: asyncio.Event
) -> None:
    """Tear the run down by failing the pending stderr read."""
    assert proc.stderr is not None
    proc.stderr.set_exception(RuntimeError("stream exploded"))


def cancel_task(task: ExecTask, proc: asyncio.subprocess.Process, abort: asyncio.Event) -> None:
    """Tear the run down by cancelling the exec task."""
    task.cancel()


@dataclasses.dataclass(frozen=True, slots=True)
class Teardown:
    """How a teardown test ends a run: an optional timeout plus an action on the running exec."""

    timeout_ms: int | None
    trigger: Callable[[ExecTask, asyncio.subprocess.Process, asyncio.Event], None]


def physical_path(path: Path) -> str:
    """Resolve symlinks so a directory compares equal to ``pwd -P`` output.

    Wrapped in a sync helper so the resolution stays out of the async test body,
    where a blocking filesystem call would trip the async-blocking-call lint.
    """
    return str(path.resolve())


def stub_taskkill(monkeypatch: pytest.MonkeyPatch, returncode: int) -> list[list[str]]:
    """Stub ``subprocess.run`` so a taskkill call records its args and fails with ``returncode``.

    Returns the list the stub appends each call's ``args`` to.
    """
    calls: list[list[str]] = []

    def fake_run(
        args: list[str],
        *_args: object,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        raise subprocess.CalledProcessError(returncode=returncode, cmd=args)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def assert_taskkill_invoked(calls: list[list[str]]) -> None:
    """Assert a forceful, tree-wide ``taskkill`` was invoked."""
    assert calls
    assert calls[0][0] == "taskkill"
    assert "/F" in calls[0]
    assert "/T" in calls[0]


async def start_abortable_run(
    make_opts: Callable[..., ExecOptions],
    spawned_processes: list[asyncio.subprocess.Process],
) -> tuple[ExecTask, asyncio.Event]:
    """Start ``sleep 0.5`` under a fresh abort event and wait for its child to spawn."""
    abort = asyncio.Event()
    task = asyncio.create_task(run_exec("sleep 0.5", make_opts(abort=abort)))
    await wait_for_spawned(spawned_processes)
    return task, abort


def expected_result(stdout: str, stderr: str, exit_code: int) -> ExecResult:
    """Build an expected ``ExecResult`` with byte counts derived from the strings."""
    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


@pytest.fixture
def make_opts(tmp_path: Path) -> Callable[..., ExecOptions]:
    """Build ``ExecOptions`` rooted at the test's ``tmp_path``, with any override."""

    def _make(
        *,
        timeout_ms: int | None = None,
        abort: asyncio.Event | None = None,
        stdin: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecOptions:
        return ExecOptions(
            cwd=str(tmp_path),
            timeout_ms=timeout_ms,
            abort=abort,
            stdin=stdin,
            env=env,
        )

    return _make


@pytest.fixture
def spawned_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[asyncio.subprocess.Process]]:
    """Record every child ``exec`` spawns, so a test can reach into its stdio pipes."""
    # exec calls asyncio.create_subprocess_shell (module-qualified), so
    # wrapping that attribute captures the real Process while leaving the spawn
    # itself real.
    processes = capture_spawns(monkeypatch, "create_subprocess_shell")
    yield processes

    # Safety net: reap any group a test deliberately stopped exec from killing.
    for proc in processes:
        if proc.returncode is not None or not proc.pid:
            continue
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)


@pytest.fixture(autouse=True)
def _isolate_live_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the module-level live-group registry from bleeding across tests."""
    monkeypatch.setattr(exec_mod, "_live_process_groups", set())


@pytest.fixture
def held_reaper(monkeypatch: pytest.MonkeyPatch) -> Iterator[HeldReaper]:
    """Hold asyncio's child reap back so an exited shell stays a zombie through exec's teardown."""
    if pidfd_available():
        pytest.skip("asyncio reaps through a pidfd here, so there is no waitpid thread to hold")
    reaper = HeldReaper(release=threading.Event(), hold_s=_REAP_HOLD_S)
    monkeypatch.setattr("asyncio.unix_events.os", reaper)
    yield reaper
    reaper.release.set()


@pytest.fixture
def kill_attempts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace ``kill_process_group`` with a spy that records each pid."""
    attempted: list[int] = []

    def record(pid: int, *_a: object, **_k: object) -> None:
        attempted.append(pid)

    monkeypatch.setattr(exec_mod, "kill_process_group", record)
    return attempted


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param("echo hello", expected_result("hello\n", "", 0), id="stdout"),
        pytest.param(
            "echo stdout && echo stderr >&2",
            expected_result("stdout\n", "stderr\n", 0),
            id="stdout-and-stderr-separated",
        ),
        pytest.param("exit 42", expected_result("", "", 42), id="non-zero-exit"),
        pytest.param(
            'echo "line1" && echo "line2" && echo "line3"',
            expected_result("line1\nline2\nline3\n", "", 0),
            id="multi-line",
        ),
        pytest.param(
            "sleep 0.1 && echo done",
            expected_result("done\n", "", 0),
            id="slow-no-timeout",
        ),
    ],
)
async def test_exec_when_command_runs_to_completion_does_capture_output(
    make_opts: Callable[..., ExecOptions],
    command: str,
    expected: ExecResult,
) -> None:
    result = await run_exec(command, make_opts())

    assert result == expected


async def test_exec_when_cwd_specified_does_run_in_that_directory(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await run_exec("pwd -P", make_opts())

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == physical_path(tmp_path)


async def test_exec_when_multi_byte_char_split_across_reads_does_decode_single_char(
    make_opts: Callable[..., ExecOptions],
) -> None:
    # The two bytes of U+00B5 are flushed by separate printf processes, so they
    # land in separate pipe reads; one incremental decoder joins them.
    result = await run_exec(
        "printf '\\302'; sleep 0.2; printf '\\265'",
        make_opts(),
    )

    assert result == expected_result("µ", "", 0)


async def test_exec_when_descendant_writes_after_shell_exits_does_capture_output(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await run_exec("(sleep 0.2; echo METRIC) &", make_opts())

    assert result == expected_result("METRIC\n", "", 0)


async def test_exec_when_stdin_provided_does_deliver_to_command(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await run_exec("cat", make_opts(stdin="piped input\n"))

    assert result == expected_result("piped input\n", "", 0)


async def test_exec_when_stdin_omitted_does_give_closed_input(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await run_exec("cat", make_opts())

    assert result == expected_result("", "", 0)


async def test_exec_when_stdin_unread_and_large_does_settle_with_command_result(
    make_opts: Callable[..., ExecOptions],
) -> None:
    # Larger than any OS pipe buffer, so the write cannot complete on its own:
    # the child exits first and the pending write breaks with a broken pipe,
    # which is swallowed rather than surfaced as an error.
    payload = "x" * (1024 * 1024)

    result = await run_exec("exit 3", make_opts(stdin=payload))

    assert result == expected_result("", "", 3)


async def test_exec_when_cwd_missing_does_resolve_with_failure_on_stderr(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    result = await run_exec("echo hello", ExecOptions(cwd=str(missing)))

    assert isinstance(result, ExecResult)
    assert result.stdout == ""
    assert result.exit_code == 1
    assert "No such file or directory" in result.stderr
    assert result.stderr.endswith("\n")
    assert result.stderr_bytes == len(result.stderr.encode())


async def test_exec_when_cwd_is_a_file_does_resolve_with_failure_on_stderr(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "regular-file"
    not_a_dir.write_text("")

    result = await run_exec("echo hello", ExecOptions(cwd=str(not_a_dir)))

    assert isinstance(result, ExecResult)
    assert result.stdout == ""
    assert result.exit_code == 1
    assert "Not a directory" in result.stderr
    assert result.stderr.endswith("\n")
    assert result.stderr_bytes == len(result.stderr.encode())
    assert exec_mod._live_process_groups == set()


@pytest.mark.parametrize(
    ("command", "cwd", "env"),
    [
        pytest.param("echo a\x00b", None, None, id="nul-in-command"),
        pytest.param("echo hello", "nu\x00l", None, id="nul-in-cwd"),
        pytest.param("echo hello", None, {"VAR": "a\x00b"}, id="nul-in-env-value"),
    ],
)
async def test_exec_when_spawn_argument_holds_nul_does_resolve_with_error_on_stderr(
    tmp_path: Path,
    command: str,
    cwd: str | None,
    env: dict[str, str] | None,
) -> None:
    opts = ExecOptions(cwd=str(tmp_path) if cwd is None else cwd, env=env)

    result = await run_exec(command, opts)

    assert isinstance(result, ExecResult)
    assert result.stdout == ""
    assert result.exit_code == 1
    assert "embedded null byte" in result.stderr
    assert result.stderr.endswith("\n")
    assert result.stderr_bytes == len(result.stderr.encode())
    assert exec_mod._live_process_groups == set()


async def test_exec_when_timeout_exceeded_does_return_timeout_error(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await run_exec("sleep 10", make_opts(timeout_ms=500))

    assert isinstance(result, ExecTimeoutError)
    assert result == ExecTimeoutError(
        stdout="",
        stderr="",
        timeout_ms=500,
        stdout_bytes=0,
        stderr_bytes=0,
    )


async def test_exec_when_timeout_exceeded_does_capture_partial_output(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await run_exec(
        'for i in 1 2 3; do echo "line $i"; sleep 1; done',
        make_opts(timeout_ms=1500),
    )

    assert isinstance(result, ExecTimeoutError)
    assert result.timeout_ms == 1500
    assert result.stderr == ""
    assert "line 1" in result.stdout


async def test_exec_when_aborted_mid_run_does_kill_whole_group(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    command = "sleep 30 & echo $! > grandchild.pid; echo $$ > shell.pid; wait"
    task = asyncio.create_task(run_exec(command, make_opts(abort=abort)))
    grandchild = await wait_for_pid_file(tmp_path / "grandchild.pid")

    abort.set()

    await wait_until_dead(grandchild, timeout_s=3.0)
    await task
    assert not is_alive(grandchild)


async def test_exec_when_aborted_mid_run_does_settle_as_failed_result(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    task = asyncio.create_task(
        run_exec("echo $$ > shell.pid; sleep 30", make_opts(abort=abort)),
    )
    await wait_for_pid_file(tmp_path / "shell.pid")

    abort.set()

    result = await asyncio.wait_for(task, 5)
    # A SIGKILLed child reports a negative returncode that exec maps to 1; the
    # exact match also rules out the timeout shape.
    assert result == expected_result("", "", 1)


async def test_exec_when_event_preset_does_not_spawn_and_settles_failed(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    abort.set()
    marker = tmp_path / "completed.marker"

    result = await asyncio.wait_for(
        run_exec(
            "echo $$ > shell.pid; sleep 30; echo done > completed.marker",
            make_opts(abort=abort),
        ),
        3,
    )

    assert spawned_processes == []
    assert result == expected_result("", "", 1)
    assert not marker.exists()


async def test_exec_result_when_field_assigned_does_raise_frozen(
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    abort.set()
    result = await run_exec("echo hello", make_opts(abort=abort))
    assert isinstance(result, ExecResult)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.exit_code = 999  # type: ignore[misc]


async def test_exec_timeout_error_when_field_assigned_does_raise_frozen(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await run_exec("sleep 10", make_opts(timeout_ms=300))
    assert isinstance(result, ExecTimeoutError)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.timeout_ms = 999  # type: ignore[misc]


async def test_exec_when_timeout_settles_does_close_stdio_pipes(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    task = asyncio.create_task(run_exec("sleep 30", make_opts(timeout_ms=500)))
    proc = await wait_for_spawned(spawned_processes)

    await task

    assert proc.stdout is not None
    assert proc.stderr is not None
    assert proc.stdout.at_eof()
    assert proc.stderr.at_eof()


async def test_exec_when_abort_settles_does_close_stdio_pipes(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    task = asyncio.create_task(run_exec("sleep 30", make_opts(abort=abort)))
    proc = await wait_for_spawned(spawned_processes)

    abort.set()
    await task

    assert proc.stdout is not None
    assert proc.stderr is not None
    assert proc.stdout.at_eof()
    assert proc.stderr.at_eof()


async def test_exec_when_child_killed_by_signal_does_report_exit_one(
    make_opts: Callable[..., ExecOptions],
) -> None:
    # The shell SIGKILLs itself; asyncio reports a negative returncode that exec
    # maps to the failure exit code rather than surfacing the raw signal value.
    result = await run_exec("kill -9 $$", make_opts())

    assert isinstance(result, ExecResult)
    assert result.exit_code == 1
    assert result.stdout == ""


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
async def test_exec_when_stream_read_fails_does_settle_as_failure(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    stream: str,
) -> None:
    task = asyncio.create_task(run_exec("sleep 0.5", make_opts()))
    proc = await wait_for_spawned(spawned_processes)
    pipe = proc.stdout if stream == "stdout" else proc.stderr
    assert pipe is not None

    pipe.set_exception(RuntimeError("stream exploded"))

    result = await asyncio.wait_for(task, 3)
    assert isinstance(result, ExecResult)
    assert result.exit_code == 1
    assert result.stderr == "stream exploded\n"
    # The diagnostic text is internal, not command output — byte count stays zero.
    assert result.stderr_bytes == 0


async def test_exec_when_stream_read_fails_does_kill_whole_group(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    task = asyncio.create_task(
        run_exec("sleep 30 & echo $! > grandchild.pid; wait", make_opts()),
    )
    proc = await wait_for_spawned(spawned_processes)
    grandchild = await wait_for_pid_file(tmp_path / "grandchild.pid")
    assert proc.stdout is not None

    proc.stdout.set_exception(RuntimeError("stream exploded"))

    await wait_until_dead(grandchild, timeout_s=3.0)
    await task
    assert not is_alive(grandchild)


def test_output_buffer_when_prior_total_below_cap_does_append_and_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", 10)
    buf = OutputBuffer()

    buf.append("aaaaa", 5)
    buf.append("bbbbbbbb", 8)

    # The chunk that crosses the cap is still appended whole (prior total < cap).
    assert buf.text == "aaaaabbbbbbbb"  # cspell:disable-line
    assert buf.byte_count == 13


def test_output_buffer_when_prior_total_reaches_cap_does_drop_chunk_but_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", 10)
    buf = OutputBuffer()

    buf.append("a" * 12, 12)
    buf.append("c", 1)

    # Prior total (12) already reached the cap, so the next chunk is dropped
    # from the text while its bytes are still counted.
    assert buf.text == "a" * 12
    assert buf.byte_count == 13


async def test_exec_when_output_exceeds_cap_does_stop_appending_but_keep_counting(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
) -> None:
    # Cap of zero drops every chunk (prior total 0 already reaches the cap),
    # while byte counting continues and the child is never signalled.
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", 0)

    result = await run_exec("echo hello", make_opts())

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stdout_bytes == len(b"hello\n")


async def test_exec_when_win32_taskkill_reports_gone_does_stay_silent(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_taskkill(monkeypatch, returncode=128)
    task, abort = await start_abortable_run(make_opts, spawned_processes)

    # Redirect only the kill-time platform read, after the POSIX spawn.
    monkeypatch.setattr(sys, "platform", "win32")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        abort.set()
        await asyncio.wait_for(task, 5)

    assert_taskkill_invoked(calls)
    assert not any(issubclass(w.category, RuntimeWarning) for w in caught)


async def test_exec_when_win32_taskkill_fails_otherwise_does_warn(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_taskkill(monkeypatch, returncode=5)
    task, abort = await start_abortable_run(make_opts, spawned_processes)

    monkeypatch.setattr(sys, "platform", "win32")
    # exec runs as a separate task, so it cannot process the abort until the
    # awaited wait_for yields control inside the warns block.
    abort.set()
    with pytest.warns(RuntimeWarning):
        await asyncio.wait_for(task, 5)

    assert_taskkill_invoked(calls)


async def test_exec_when_win32_taskkill_launch_raises_oserror_does_warn_not_raise(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_oserror(
        args: list[str],
        *_args: object,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(subprocess, "run", raise_oserror)
    task, abort = await start_abortable_run(make_opts, spawned_processes)

    monkeypatch.setattr(sys, "platform", "win32")
    abort.set()
    with pytest.warns(RuntimeWarning):
        await asyncio.wait_for(task, 5)


async def test_exec_when_killpg_fails_otherwise_does_warn_not_raise(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_eperm(group_pid: int, sig: int) -> None:
        raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))

    task, abort = await start_abortable_run(make_opts, spawned_processes)

    monkeypatch.setattr(os, "killpg", raise_eperm)
    # exec runs as a separate task, so it cannot process the abort until the
    # awaited wait_for yields control inside the warns block.
    abort.set()
    with pytest.warns(RuntimeWarning):
        await asyncio.wait_for(task, 5)


async def test_exec_when_timeout_lands_as_child_exits_does_not_warn_about_killpg(
    make_opts: Callable[..., ExecOptions],
    recwarn: pytest.WarningsRecorder,
) -> None:
    # Repeating the run is the scenario: a 2ms timeout on a command that exits at
    # once keeps landing while the shell is exiting, when its group refuses the kill.
    for _ in range(_RACE_RUNS):
        await run_exec("exit 0", make_opts(timeout_ms=2))

    assert killpg_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# teardown after the shell has exited but before asyncio reaps it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "teardown",
    [
        pytest.param(Teardown(_TEARDOWN_TIMEOUT_MS, leave_to_timeout), id="timeout"),
        pytest.param(Teardown(None, set_abort), id="abort"),
        pytest.param(Teardown(None, fail_stderr_read), id="stream-error"),
        pytest.param(Teardown(None, cancel_task), id="cancelled"),
    ],
)
async def test_exec_when_torn_down_after_shell_exits_does_leave_reaping_to_asyncio(
    held_reaper: HeldReaper,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    caplog: pytest.LogCaptureFixture,
    teardown: Teardown,
) -> None:
    # The background sleep holds stderr open, so exec is still running when the
    # shell exits and the teardown lands on a shell nobody has reaped yet.
    abort = asyncio.Event()
    task = asyncio.create_task(
        run_exec(
            "sleep 30 >/dev/null & exit 0",
            make_opts(timeout_ms=teardown.timeout_ms, abort=abort),
        ),
    )
    proc = await wait_for_spawned(spawned_processes)
    await wait_for_shell_exit(proc)

    teardown.trigger(task, proc, abort)
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    await asyncio.wait_for(proc.wait(), 5)

    assert [
        r.getMessage()
        for r in caplog.records
        if r.name == "asyncio" and _UNKNOWN_CHILD in r.getMessage()
    ] == []


async def test_exec_when_cancelled_after_shell_exits_does_not_warn_about_killpg(
    held_reaper: HeldReaper,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    recwarn: pytest.WarningsRecorder,
) -> None:
    task = asyncio.create_task(run_exec("exit 0", make_opts()))
    proc = await wait_for_spawned(spawned_processes)
    await wait_for_shell_exit(proc)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    await asyncio.wait_for(proc.wait(), 5)

    assert killpg_warnings(recwarn) == []


async def test_exec_when_cancelled_after_shell_exits_does_kill_live_descendants(
    tmp_path: Path,
    held_reaper: HeldReaper,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    task = asyncio.create_task(
        run_exec("sleep 30 >/dev/null & echo $! > grandchild.pid; exit 0", make_opts()),
    )
    proc = await wait_for_spawned(spawned_processes)
    grandchild = await wait_for_pid_file(tmp_path / "grandchild.pid")
    await wait_for_shell_exit(proc)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)

    await wait_until_dead(grandchild, timeout_s=3.0)
    assert not is_alive(grandchild)


async def test_exec_when_cancelled_and_shell_never_reaped_does_settle_promptly_without_warning(
    held_reaper: HeldReaper,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    recwarn: pytest.WarningsRecorder,
) -> None:
    held_reaper.hold_s = 60.0
    task = asyncio.create_task(run_exec("exit 0", make_opts()))
    proc = await wait_for_spawned(spawned_processes)
    await wait_for_shell_exit(proc)

    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=_CANCEL_SETTLE_S)
    # Let the held reap finish inside the test, while its loop is still open.
    held_reaper.release.set()
    await asyncio.wait_for(proc.wait(), 5)

    assert done == {task}
    assert task.cancelled()
    # The shell not yet reaped is the group's only member, a zombie with nothing
    # left to stop, so the group refusing the kill is not worth a warning.
    assert killpg_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# live process-group registry
# ---------------------------------------------------------------------------


async def test_exec_when_timed_out_does_deregister_group(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = asyncio.create_task(run_exec("sleep 30", make_opts(timeout_ms=300)))
    proc = await wait_for_spawned(spawned_processes)

    await task

    attempted: list[int] = []

    def record(pid: int, *_a: object, **_k: object) -> None:
        attempted.append(pid)

    monkeypatch.setattr(exec_mod, "kill_process_group", record)
    exec_mod.kill_live_process_groups()

    assert proc.pid not in attempted


async def test_exec_when_stream_read_fails_does_deregister_group(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = asyncio.create_task(run_exec("sleep 30", make_opts()))
    proc = await wait_for_spawned(spawned_processes)
    assert proc.stdout is not None

    proc.stdout.set_exception(RuntimeError("stream exploded"))
    await asyncio.wait_for(task, 3)

    attempted: list[int] = []

    def record(pid: int, *_a: object, **_k: object) -> None:
        attempted.append(pid)

    monkeypatch.setattr(exec_mod, "kill_process_group", record)
    exec_mod.kill_live_process_groups()

    assert proc.pid not in attempted


async def test_exec_when_completes_normally_does_deregister_group(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    kill_attempts: list[int],
) -> None:
    await run_exec("echo hello", make_opts())
    proc = spawned_processes[-1]

    exec_mod.kill_live_process_groups()

    assert proc.pid not in kill_attempts


async def test_kill_live_process_groups_when_child_alive_does_kill_group_and_descendants(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    task = asyncio.create_task(
        run_exec("sleep 30 & echo $! > grandchild.pid; wait", make_opts()),
    )
    await wait_for_spawned(spawned_processes)
    grandchild = await wait_for_pid_file(tmp_path / "grandchild.pid")

    exec_mod.kill_live_process_groups()

    await wait_until_dead(grandchild, timeout_s=3.0)
    await task
    assert not is_alive(grandchild)


def test_kill_live_process_groups_when_registry_empty_does_not_kill_anything(
    kill_attempts: list[int],
) -> None:
    exec_mod.kill_live_process_groups()

    assert kill_attempts == []


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(OSError("no such process"), id="oserror"),
        pytest.param(RuntimeError("unexpected"), id="runtime-error"),
    ],
)
def test_kill_live_process_groups_when_kill_raises_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch,
    exc: Exception,
) -> None:
    monkeypatch.setattr(exec_mod, "_live_process_groups", {4242})
    attempted: list[int] = []

    def boom(pid: int, *_a: object, **_k: object) -> None:
        attempted.append(pid)
        raise exc

    monkeypatch.setattr(exec_mod, "kill_process_group", boom)

    exec_mod.kill_live_process_groups()

    assert attempted == [4242]


# ---------------------------------------------------------------------------
# child signal mask
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="Signal masking requires POSIX pthread_sigmask",
)
async def test_exec_when_spawned_does_unblock_termination_signals_in_child(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await run_exec(
        'python3 -c "import signal, json; '
        'print(json.dumps(list(signal.pthread_sigmask(signal.SIG_BLOCK, []))))"',
        make_opts(),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    blocked: list[int] = json.loads(result.stdout.strip())
    for sig in TERMINATION_SIGNALS:
        assert sig not in blocked, f"signal {sig} should be unblocked in child"


# ---------------------------------------------------------------------------
# cancelling exec kills the child before dropping it from the registry
# ---------------------------------------------------------------------------


async def test_exec_when_cancelled_does_kill_child_before_dropping_from_registry(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    task = asyncio.create_task(
        run_exec("echo $$ > shell.pid; sleep 30", make_opts()),
    )
    proc = await wait_for_spawned(spawned_processes)
    await wait_for_pid_file(tmp_path / "shell.pid")

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    await wait_until_dead(proc.pid, timeout_s=3.0)
    assert not is_alive(proc.pid)
    assert proc.pid not in exec_mod._live_process_groups


# ---------------------------------------------------------------------------
# OutputBuffer list-join accumulation
# ---------------------------------------------------------------------------


def test_output_buffer_when_many_small_appends_does_accumulate_all_chunks() -> None:
    buf = OutputBuffer()

    for i in range(1000):
        chunk = f"chunk-{i}\n"
        buf.append(chunk, len(chunk.encode()))

    expected = "".join(f"chunk-{i}\n" for i in range(1000))
    assert buf.text == expected


# ---------------------------------------------------------------------------
# env option (shell form)
# ---------------------------------------------------------------------------


async def test_exec_when_env_set_does_use_exact_mapping(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await run_exec("echo $CUSTOM_VAR", make_opts(env={"CUSTOM_VAR": "custom_value"}))

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "custom_value"


async def test_exec_when_env_set_does_exclude_parent_vars(
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GYMRAT_TEST_MARKER", "should_not_appear")

    result = await run_exec(
        'echo "${GYMRAT_TEST_MARKER:-absent}"',
        make_opts(env={"PATH": os.environ.get("PATH", "/usr/bin")}),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "absent"


async def test_exec_when_env_none_does_inherit_parent_env(
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GYMRAT_TEST_MARKER", "inherited")

    result = await run_exec("echo $GYMRAT_TEST_MARKER", make_opts(env=None))

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "inherited"
