"""Behavioral tests for the asyncio subprocess ``exec`` layer.

Real-subprocess tests are parallel-safe under ``pytest-xdist``: every run gets
its own directory via the ``tmp_path`` fixture, and POSIX-only shell constructs
(``$$``, ``>&2``, ``for``/``do``/``done``) mean the directory is not collected
on win32.
The win32 ``taskkill`` fallback is tested in ``tests/test_process_group.py``.
"""

import asyncio
import contextlib
import dataclasses
import json
import os
import signal
import sys
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError, OutputBuffer
from gymrat.exec import exec as run_exec
from gymrat.signals import TERMINATION_SIGNALS
from tests._exec_fixtures import (
    RUNNERS,
    ExecRun,
    ExecTask,
    Runner,
    Teardown,
    cancel_and_settle,
    cancel_task,
    expected_result,
    fail_stderr_read,
    leave_to_timeout,
    run_argv,
    run_shell,
    set_abort,
    settle,
    shell_grandchild,
    wait_for_spawned,
)
from tests._process_helpers import (
    KillpgRefusal,
    is_alive,
    killpg_warnings,
    refuse_killpg,
    wait_for_pid_file,
    wait_until_dead,
)

# Runs repeated back to back to hit the window where the timeout fires while the
# shell is already exiting; a single run slips past it some of the time. Used
# only where asyncio reaps through a pidfd, so ``held_reaper`` cannot pin that
# state deterministically.
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


async def wait_for_shell_exit(proc: asyncio.subprocess.Process, timeout_s: float = 3.0) -> None:
    """Poll until the shell's stdout reaches EOF, which the shell exiting closes.

    The command must not hand stdout to anything that outlives the shell, and
    the check never reaps the shell itself.

    Args:
        proc: The shell whose stdout is watched.
        timeout_s: How long to wait for EOF before giving up.

    Raises:
        TimeoutError: stdout did not reach EOF within ``timeout_s``.
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


@pytest.fixture
def held_reaper(monkeypatch: pytest.MonkeyPatch) -> Iterator[HeldReaper]:
    """Hold asyncio's child reap back so an exited shell stays a zombie through exec's teardown."""
    if pidfd_available():
        pytest.skip("asyncio reaps through a pidfd here, so there is no waitpid thread to hold")
    reaper = HeldReaper(release=threading.Event(), hold_s=_REAP_HOLD_S)
    monkeypatch.setattr("asyncio.unix_events.os", reaper)
    yield reaper
    reaper.release.set()


# ---------------------------------------------------------------------------
# captured output and stdin delivery
# ---------------------------------------------------------------------------


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


async def test_exec_when_stdin_unread_and_large_does_settle_with_command_result(
    make_opts: Callable[..., ExecOptions],
) -> None:
    # Larger than any OS pipe buffer, so the write cannot complete on its own:
    # the child exits first and the pending write breaks with a broken pipe,
    # which is swallowed rather than surfaced as an error.
    payload = "x" * (1024 * 1024)

    result = await run_exec("exit 3", make_opts(stdin=payload))

    assert result == expected_result("", "", 3)


# ---------------------------------------------------------------------------
# the two entry points, for the behaviors they share
# ---------------------------------------------------------------------------


_ECHO_STDIN_ARGV = [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"]

# Larger than any OS pipe buffer: a child writing this much to stderr before
# stdout blocks unless both pipes are drained at once.
_MIB = 1024 * 1024


@pytest.mark.parametrize(
    ("run", "args", "expected"),
    [
        pytest.param(
            run_shell,
            "echo out; echo err >&2; exit 3",
            expected_result("out\n", "err\n", 3),
            id="exec",
        ),
        pytest.param(
            run_argv,
            [
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)",
            ],
            expected_result("out\n", "err\n", 3),
            id="exec_argv",
        ),
        pytest.param(
            run_argv,
            [
                sys.executable,
                "-c",
                f"import sys; sys.stderr.write('E' * {_MIB}); sys.stderr.flush(); print('OK')",
            ],
            expected_result("OK\n", "E" * _MIB, 0),
            id="exec_argv-stderr-exceeds-pipe-buffer",
        ),
    ],
)
async def test_exec_when_run_to_completion_does_capture_the_full_result(
    make_opts: Callable[..., ExecOptions],
    run: ExecRun,
    args: Any,
    expected: ExecResult,
) -> None:
    result = await run(args, make_opts())

    assert result == expected


@pytest.mark.parametrize(
    ("run", "args"),
    [
        pytest.param(run_shell, "cat", id="exec"),
        pytest.param(run_argv, _ECHO_STDIN_ARGV, id="exec_argv"),
    ],
)
@pytest.mark.parametrize(
    ("stdin", "expected_stdout"),
    [
        pytest.param("piped input\n", "piped input\n", id="stdin-delivered"),
        pytest.param(None, "", id="stdin-omitted-is-closed"),
    ],
)
@pytest.mark.usefixtures("stdin_holding_text")
async def test_exec_when_stdin_given_or_omitted_does_feed_it_or_close_it(
    make_opts: Callable[..., ExecOptions],
    *,
    run: ExecRun,
    args: Any,
    stdin: str | None,
    expected_stdout: str,
) -> None:
    result = await run(args, make_opts(stdin=stdin))

    assert result == expected_result(expected_stdout, "", 0)


@pytest.mark.parametrize(
    ("run", "args"),
    [
        pytest.param(run_shell, "pwd -P", id="exec"),
        pytest.param(
            run_argv, [sys.executable, "-c", "import os; print(os.getcwd())"], id="exec_argv"
        ),
    ],
)
async def test_exec_when_cwd_specified_does_run_in_that_directory(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    run: ExecRun,
    args: Any,
) -> None:
    result = await run(args, make_opts())

    assert result == expected_result(f"{await asyncio.to_thread(tmp_path.resolve)}\n", "", 0)


# ---------------------------------------------------------------------------
# spawn failure — never raises, always resolves to a failed ExecResult
# ---------------------------------------------------------------------------


def _in_tmp(tmp_path: Path) -> str:
    return str(tmp_path)


def _missing_dir(tmp_path: Path) -> str:
    return str(tmp_path / "does-not-exist")


def _regular_file(tmp_path: Path) -> str:
    not_a_dir = tmp_path / "regular-file"
    not_a_dir.write_text("")
    return str(not_a_dir)


def _nul_in_cwd(_tmp_path: Path) -> str:
    return "nu\x00l"


@pytest.mark.parametrize(
    ("run", "args", "cwd", "env", "message"),
    [
        pytest.param(
            run_shell,
            "echo hello",
            _missing_dir,
            None,
            "[Errno 2] No such file or directory: '{cwd}'",
            id="exec-cwd-missing",
        ),
        pytest.param(
            run_shell,
            "echo hello",
            _regular_file,
            None,
            "[Errno 20] Not a directory: '{cwd}'",
            id="exec-cwd-file",
        ),
        pytest.param(
            run_shell, "echo a\x00b", _in_tmp, None, "embedded null byte", id="exec-nul-in-command"
        ),
        pytest.param(
            run_shell, "echo hello", _nul_in_cwd, None, "embedded null byte", id="exec-nul-in-cwd"
        ),
        pytest.param(
            run_shell,
            "echo hello",
            _in_tmp,
            {"VAR": "a\x00b"},
            "embedded null byte",
            id="exec-nul-in-env",
        ),
        pytest.param(
            run_argv,
            ["no-such-binary-exists-anywhere"],
            _in_tmp,
            None,
            "[Errno 2] No such file or directory: 'no-such-binary-exists-anywhere'",
            id="argv-binary-not-found",
        ),
        pytest.param(
            run_argv,
            ["echo", "hello"],
            _missing_dir,
            None,
            "[Errno 2] No such file or directory: '{cwd}'",
            id="argv-cwd-missing",
        ),
        pytest.param(
            run_argv,
            ["echo", "a\x00b"],
            _in_tmp,
            None,
            "embedded null byte",
            id="argv-nul-in-argument",
        ),
        pytest.param(
            run_argv,
            ["echo", "hello"],
            _nul_in_cwd,
            None,
            "embedded null byte",
            id="argv-nul-in-cwd",
        ),
        pytest.param(
            run_argv,
            ["echo", "hello"],
            _in_tmp,
            {"VAR": "a\x00b"},
            "embedded null byte",
            id="argv-nul-in-env",
        ),
        pytest.param(
            run_argv, [], _in_tmp, None, "argv is empty: no program to run", id="argv-empty"
        ),
    ],
)
async def test_exec_when_spawn_fails_does_resolve_with_the_failure_on_stderr(
    tmp_path: Path,
    *,
    run: ExecRun,
    args: Any,
    cwd: Callable[[Path], str],
    env: dict[str, str] | None,
    message: str,
) -> None:
    run_dir = cwd(tmp_path)

    result = await run(args, ExecOptions(cwd=run_dir, env=env))

    assert result == expected_result("", message.format(cwd=run_dir) + "\n", exit_code=1)


# ---------------------------------------------------------------------------
# timeout, abort, and teardown of a running child
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("runner", RUNNERS)
async def test_exec_when_timeout_exceeded_does_return_timeout_error_with_the_partial_output(
    make_opts: Callable[..., ExecOptions], runner: Runner
) -> None:
    result = await runner.run(runner.partial_output, make_opts(timeout_ms=1500))

    assert result == ExecTimeoutError(
        stdout="line 1\n",
        stderr="",
        timeout_ms=1500,
        stdout_bytes=len(b"line 1\n"),
        stderr_bytes=0,
    )


#: A timeout long enough that the grandchild is up before it lands, short enough to stay quick.
_GRANDCHILD_TIMEOUT_MS = 3000


@pytest.mark.parametrize("runner", RUNNERS)
@pytest.mark.parametrize(
    "teardown",
    [
        pytest.param(Teardown(_GRANDCHILD_TIMEOUT_MS, leave_to_timeout), id="timeout"),
        pytest.param(Teardown(None, set_abort), id="abort"),
        pytest.param(Teardown(None, fail_stderr_read), id="stream-error"),
        pytest.param(Teardown(None, cancel_task), id="cancelled"),
    ],
)
async def test_exec_when_torn_down_does_kill_the_whole_group(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    runner: Runner,
    teardown: Teardown,
) -> None:
    abort = asyncio.Event()
    task = asyncio.create_task(
        runner.run(
            runner.grandchild(tmp_path / "grandchild.pid"),
            make_opts(timeout_ms=teardown.timeout_ms, abort=abort),
        )
    )
    proc = await wait_for_spawned(spawned_processes)
    grandchild = await wait_for_pid_file(tmp_path / "grandchild.pid")

    teardown.trigger(task, proc, abort)

    await settle(task)
    await wait_until_dead(grandchild, timeout_s=3.0)
    assert not is_alive(grandchild)


async def test_exec_when_abort_preset_does_settle_failed_without_spawning(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    abort.set()

    result = await asyncio.wait_for(run_exec("sleep 30", make_opts(abort=abort)), 3)

    assert spawned_processes == []
    assert result == expected_result("", "", 1)


@pytest.mark.parametrize(
    "teardown",
    [
        pytest.param(Teardown(500, leave_to_timeout), id="timeout"),
        pytest.param(Teardown(None, set_abort), id="abort"),
    ],
)
async def test_exec_when_run_settles_does_close_stdio_pipes(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    teardown: Teardown,
) -> None:
    abort = asyncio.Event()
    task = asyncio.create_task(
        run_exec("sleep 30", make_opts(timeout_ms=teardown.timeout_ms, abort=abort))
    )
    proc = await wait_for_spawned(spawned_processes)

    teardown.trigger(task, proc, abort)
    await task

    assert proc.stdout is not None
    assert proc.stderr is not None
    assert proc.stdout.at_eof()
    assert proc.stderr.at_eof()


# ---------------------------------------------------------------------------
# failures that settle as exit code 1
# ---------------------------------------------------------------------------


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
    pipe: asyncio.StreamReader = getattr(proc, stream)

    pipe.set_exception(RuntimeError("stream exploded"))

    result = await asyncio.wait_for(task, 3)
    assert isinstance(result, ExecResult)
    assert result.exit_code == 1
    assert result.stderr == "stream exploded\n"
    # The diagnostic text is internal, not command output — byte count stays zero.
    assert result.stderr_bytes == 0


# ---------------------------------------------------------------------------
# output cap
# ---------------------------------------------------------------------------


def test_output_buffer_when_chunk_crosses_the_cap_does_keep_it_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The prior total (5) is still under the cap, so the chunk that crosses it is
    # kept whole rather than cut at the cap. A chunk arriving once the cap is
    # reached is dropped; the exec-level cap test covers that branch.
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", 10)
    buf = OutputBuffer()
    buf.append("plums", 5)

    buf.append("cherries", 8)

    assert (buf.text, buf.byte_count) == ("plums" + "cherries", 13)


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


# ---------------------------------------------------------------------------
# a timeout racing the shell's exit
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not pidfd_available(),
    reason="held_reaper holds back the exited shell's reap, pinning this state deterministically",
)
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
    await settle(task)
    await asyncio.wait_for(proc.wait(), 5)

    assert [
        r.getMessage()
        for r in caplog.records
        if r.name == "asyncio" and _UNKNOWN_CHILD in r.getMessage()
    ] == []


@pytest.mark.parametrize(
    "teardown",
    [
        pytest.param(Teardown(_TEARDOWN_TIMEOUT_MS, leave_to_timeout), id="timeout"),
        pytest.param(Teardown(None, set_abort), id="abort"),
        pytest.param(Teardown(None, cancel_task), id="cancelled"),
    ],
)
async def test_exec_when_torn_down_after_shell_exits_does_not_warn_about_killpg(
    held_reaper: HeldReaper,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    recwarn: pytest.WarningsRecorder,
    teardown: Teardown,
) -> None:
    # The exited shell is the group's only member, a zombie nobody has reaped,
    # which is when a group refuses the kill.
    abort = asyncio.Event()
    task = asyncio.create_task(
        run_exec("exit 0", make_opts(timeout_ms=teardown.timeout_ms, abort=abort)),
    )
    proc = await wait_for_spawned(spawned_processes)
    await wait_for_shell_exit(proc)

    teardown.trigger(task, proc, abort)
    await settle(task)
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

    await cancel_and_settle(task)

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

_real_killpg = os.killpg

# Upper bound for a run left going by a test to finish once its group is killed.
_RUN_SETTLE_TIMEOUT_S = 5.0


@pytest.fixture
def killpg_refusal(
    monkeypatch: pytest.MonkeyPatch,
    spawned_processes: list[asyncio.subprocess.Process],
) -> Iterator[KillpgRefusal]:
    """Install a killpg stand-in a test can switch to refusing right before its act.

    Depends on ``spawned_processes``, so its own teardown runs first: clearing
    ``refusing`` here happens before ``spawned_processes`` kills any survivor's
    group, so that cleanup kill goes through the real ``killpg``.
    """
    refusal = refuse_killpg(monkeypatch, refusing=False)
    yield refusal
    refusal.refusing = False


@pytest.fixture
async def background_runs(
    spawned_processes: list[asyncio.subprocess.Process],
) -> AsyncIterator[list[ExecTask]]:
    """Collect the runs a test leaves going, then kill every spawned group and let each run settle."""
    runs: list[ExecTask] = []
    yield runs
    # An abandoned run leaves its child never reaped and its pipes open once the event
    # loop closes, surfacing as a ResourceWarning in whatever test the garbage collector
    # runs next. Killing through the real ``_real_killpg`` sidesteps any refusal the test
    # installed and ends every member holding the run's pipes, so each run completes on
    # its own. A group the test already ended is gone, or refuses while it finishes
    # exiting.
    for proc in spawned_processes:
        with contextlib.suppress(OSError):
            _real_killpg(proc.pid, signal.SIGKILL)
    await asyncio.wait_for(asyncio.gather(*runs), _RUN_SETTLE_TIMEOUT_S)


@pytest.mark.parametrize(
    ("run", "args", "timeout_ms"),
    [
        pytest.param(run_shell, "echo hello", None, id="exec-completes"),
        pytest.param(run_argv, [sys.executable, "-c", "pass"], None, id="exec_argv-completes"),
        # Every teardown (timeout, abort, stream error, cancel) settles through the
        # same release, so the timeout stands in for all of them.
        pytest.param(run_shell, "sleep 30", 200, id="exec-torn-down"),
        pytest.param(
            run_argv, ["no-such-binary-exists-anywhere"], None, id="exec_argv-spawn-fails"
        ),
    ],
)
@pytest.mark.usefixtures("spawned_processes")
async def test_kill_live_process_groups_when_run_has_settled_does_not_target_its_group(
    make_opts: Callable[..., ExecOptions],
    record_killpg: Callable[[], list[int]],
    *,
    run: ExecRun,
    args: Any,
    timeout_ms: int | None,
) -> None:
    await run(args, make_opts(timeout_ms=timeout_ms))
    attempted = record_killpg()

    exec_mod.kill_live_process_groups()

    assert attempted == []


async def test_kill_live_process_groups_when_registry_reset_does_spare_every_earlier_child(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    background_runs: list[ExecTask],
) -> None:
    background_runs.extend(
        asyncio.create_task(run_exec(shell_grandchild(tmp_path / f"earlier{n}.pid"), make_opts()))
        for n in range(2)
    )
    await wait_for_spawned(spawned_processes, count=2)
    earlier_grandchildren = [
        await wait_for_pid_file(tmp_path / f"earlier{n}.pid") for n in range(2)
    ]
    exec_mod.reset()
    background_runs.append(
        asyncio.create_task(run_exec(shell_grandchild(tmp_path / "later.pid"), make_opts())),
    )
    await wait_for_spawned(spawned_processes, count=3)
    later_grandchild = await wait_for_pid_file(tmp_path / "later.pid")

    exec_mod.kill_live_process_groups()

    await asyncio.wait_for(background_runs[-1], _RUN_SETTLE_TIMEOUT_S)
    alive = [is_alive(pid) for pid in (*earlier_grandchildren, later_grandchild)]
    assert alive == [True, True, False]


async def test_kill_live_process_groups_when_killpg_raises_does_signal_every_group_without_propagating(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    background_runs: list[ExecTask],
) -> None:
    # No grace, so the sweep reaches its kills without waiting on groups that
    # every signal fails to reach.
    background_runs.extend(asyncio.create_task(run_exec("sleep 30", make_opts())) for _ in range(2))
    await wait_for_spawned(spawned_processes, count=2)
    monkeypatch.setattr(exec_mod, "TERMINATE_GRACE_S", 0)
    killed: list[int] = []

    def raise_after_recording(group_pid: int, signal_number: int) -> None:
        if signal_number == signal.SIGKILL:
            killed.append(group_pid)
        raise RuntimeError(group_pid)

    monkeypatch.setattr(os, "killpg", raise_after_recording)

    exec_mod.kill_live_process_groups()

    assert sorted(killed) == sorted(proc.pid for proc in spawned_processes)


async def test_kill_live_process_groups_when_running_group_refuses_signals_does_warn(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    killpg_refusal: KillpgRefusal,
    background_runs: list[ExecTask],
) -> None:
    background_runs.append(asyncio.create_task(run_exec("sleep 30", make_opts())))
    await wait_for_spawned(spawned_processes)
    killpg_refusal.refusing = True

    with pytest.warns(RuntimeWarning, match="killpg failed"):
        exec_mod.kill_live_process_groups()


async def test_kill_live_process_groups_when_exited_leader_group_with_live_member_refuses_does_warn(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    stray_process_ids: list[int],
    make_opts: Callable[..., ExecOptions],
    *,
    killpg_refusal: KillpgRefusal,
    background_runs: list[ExecTask],
) -> None:
    # The grandchild holds the run's stdout, so the run keeps going after the
    # shell leading the group has exited.
    background_runs.append(
        asyncio.create_task(run_exec("sleep 30 & echo $! > grandchild.pid", make_opts())),
    )
    leader = await wait_for_spawned(spawned_processes)
    grandchild = await wait_for_pid_file(tmp_path / "grandchild.pid")
    stray_process_ids.append(grandchild)
    await wait_until_dead(leader.pid)
    killpg_refusal.refusing = True

    with pytest.warns(RuntimeWarning, match="killpg failed"):
        exec_mod.kill_live_process_groups()


# ---------------------------------------------------------------------------
# child signal mask
# ---------------------------------------------------------------------------


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
    assert sorted(set(TERMINATION_SIGNALS) & set(blocked)) == []
