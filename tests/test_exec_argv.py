"""Behavioral tests for the ``exec_argv`` subprocess layer.

``exec_argv`` runs ``argv[0]`` with the remaining items as its arguments and
no shell interpretation.  These tests mirror the ``exec`` (shell) tests in
``test_exec.py`` but exercise the direct-exec path: every argument is a
separate ``sys.argv`` entry, and no shell metacharacter is ever expanded.

Real-subprocess tests are POSIX-only for the same reasons as the shell form:
process groups, session leaders, and ``os.killpg`` do not exist on win32.
"""

import asyncio
import contextlib
import dataclasses
import errno
import json
import os
import signal
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import (
    ExecOptions,
    ExecResult,
    ExecTimeoutError,
    exec_argv,
)
from gymrat.signals import TERMINATION_SIGNALS
from tests._process_helpers import (
    ZOMBIE_ONLY_GROUP_SCRIPT,
    capture_spawns,
    is_alive,
    kill_surviving_groups,
    killpg_warnings,
    wait_for_pid_file,
    wait_until_dead,
)

if sys.platform == "win32":
    pytest.skip("POSIX-only process groups", allow_module_level=True)


def expected_result(stdout: str, stderr: str, exit_code: int) -> ExecResult:
    """Build an expected ``ExecResult`` with byte counts derived from the strings."""
    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


async def wait_for_spawned(
    processes: list[asyncio.subprocess.Process],
    timeout_s: float = 3.0,
) -> asyncio.subprocess.Process:
    """Return the most recent child ``exec_argv`` spawned, once the spawn has happened."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not processes:
        if loop.time() > deadline:
            msg = "exec_argv() has not spawned a child yet"
            raise TimeoutError(msg)
        await asyncio.sleep(0.01)
    return processes[-1]


def physical_path(path: Path) -> str:
    """Resolve symlinks so a directory compares equal to real-path output."""
    return str(path.resolve())


def script_writing_pid_and_sleeping(pid_file: Path) -> str:
    """Build a script that writes its own pid to ``pid_file``, then sleeps 30s."""
    return (
        "import os, time; "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()) + '\\n'); "
        "time.sleep(30)"
    )


def script_answering_graceful_signal(pid_file: Path) -> str:
    """Build a script that writes a last line to stderr when asked to stop, then exits.

    Used to verify that the stop request reaches the child and that the pipes
    outlive it: the farewell is written after the request lands, so it is only
    captured when the pipes are closed after the wait rather than before it.

    Args:
        pid_file: Where the script writes its own pid.

    Returns:
        The script's source, ready for ``python -c``.
    """
    return (
        "import os, signal, sys, time\n"
        "def bye(_signal_number, _frame):\n"
        "    sys.stderr.write('BYE\\n')\n"
        "    sys.stderr.flush()\n"
        "    sys.exit(7)\n"
        "signal.signal(signal.SIGTERM, bye)\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()) + '\\n')\n"
        "time.sleep(30)\n"
    )


def script_spawning_grandchild_and_sleeping(grandchild_pid_file: Path) -> str:
    """Build a script that spawns a sleeping grandchild, writes its pid, then sleeps itself.

    Used to verify that killing the child's process group also kills any
    descendants forked from a POSIX session leader.

    Args:
        grandchild_pid_file: Where the script writes the grandchild's pid.

    Returns:
        The script's source, ready for ``python -c``.
    """
    return (
        "import os, subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"open({str(grandchild_pid_file)!r}, 'w').write(str(p.pid) + '\\n')\n"
        "time.sleep(30)\n"
    )


def script_leaving_grandchild_and_exiting(grandchild_pid_file: Path) -> str:
    """Build a script that spawns a sleeping grandchild, writes its pid, then exits at once.

    The grandchild stays in the child's process group and inherits its stdout,
    so the run keeps waiting on the open pipe after the child itself is gone.

    Args:
        grandchild_pid_file: Where the script writes the grandchild's pid.

    Returns:
        The script's source, ready for ``python -c``.
    """
    return (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"open({str(grandchild_pid_file)!r}, 'w').write(str(p.pid) + '\\n')\n"
    )


_real_killpg = os.killpg

# Upper bound for a run left going by a test to finish once its group is killed.
_RUN_SETTLE_TIMEOUT_S = 5.0


@dataclasses.dataclass
class LiftableRefusal:
    """Stand-in ``os.killpg`` that refuses every signal with ``EPERM`` while ``refusing`` holds.

    Clearing ``refusing`` lets signals through for real, so a test can still
    tear its run down after the refusal it asserted on.
    """

    refusing: bool = True

    def __call__(self, group_pid: int, signal_number: int) -> None:
        if self.refusing:
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
        _real_killpg(group_pid, signal_number)


@pytest.fixture
def killpg_refusal(
    monkeypatch: pytest.MonkeyPatch,
    spawned_processes: list[asyncio.subprocess.Process],
) -> Iterator[LiftableRefusal]:
    """Install a killpg stand-in a test can switch to refusing right before its act.

    Depends on ``spawned_processes``, so its own teardown runs first: clearing
    ``refusing`` here happens before ``spawned_processes`` kills any survivor's
    group, so that cleanup kill goes through the real ``killpg``.
    """
    refusal = LiftableRefusal(refusing=False)
    monkeypatch.setattr(os, "killpg", refusal)
    yield refusal
    refusal.refusing = False


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
    """Record every child ``exec_argv`` spawns."""
    processes = capture_spawns(monkeypatch, "create_subprocess_exec")
    yield processes
    kill_surviving_groups(processes)


@pytest.fixture
async def background_runs(
    spawned_processes: list[asyncio.subprocess.Process],
) -> AsyncIterator[list["asyncio.Task[ExecResult | ExecTimeoutError]"]]:
    """Collect the runs a test leaves going, then kill every spawned group and let each run settle."""
    runs: list[asyncio.Task[ExecResult | ExecTimeoutError]] = []
    yield runs
    # An abandoned run leaves its child never reaped and its pipes open once the event
    # loop closes, surfacing as a ResourceWarning in whatever test the garbage collector
    # runs next. Killing through the real ``_real_killpg`` sidesteps any refusal the test
    # installed and ends every member holding the run's pipes, so each run completes on
    # its own.
    for proc in spawned_processes:
        with contextlib.suppress(ProcessLookupError):
            _real_killpg(proc.pid, signal.SIGKILL)
    await asyncio.wait_for(asyncio.gather(*runs), _RUN_SETTLE_TIMEOUT_S)


async def cancel_and_settle(task: "asyncio.Task[object]", timeout_s: float = 5) -> None:
    """Cancel ``task`` and wait for it to finish unwinding."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout_s)


@pytest.fixture(autouse=True)
def _isolate_live_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the module-level live-group registry from bleeding across tests."""
    monkeypatch.setattr(exec_mod, "_live_process_groups", set())


# ---------------------------------------------------------------------------
# basic output capture and exit code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param(
            ["echo", "hello"],
            expected_result("hello\n", "", 0),
            id="stdout",
        ),
        pytest.param(
            [sys.executable, "-c", "import sys; sys.exit(42)"],
            expected_result("", "", 42),
            id="non-zero-exit",
        ),
        pytest.param(
            [sys.executable, "-c", "print('line1'); print('line2'); print('line3')"],
            expected_result("line1\nline2\nline3\n", "", 0),
            id="multi-line",
        ),
        pytest.param(
            [
                sys.executable,
                "-c",
                "import sys; print('out'); print('err', file=sys.stderr)",
            ],
            expected_result("out\n", "err\n", 0),
            id="stdout-and-stderr-separated",
        ),
    ],
)
async def test_exec_argv_when_command_runs_to_completion_does_capture_output(
    make_opts: Callable[..., ExecOptions],
    argv: list[str],
    expected: ExecResult,
) -> None:
    result = await exec_argv(argv, make_opts())

    assert result == expected


# ---------------------------------------------------------------------------
# no shell interpretation — metacharacters pass through verbatim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param("hello world", id="spaces"),
        pytest.param("$HOME", id="dollar-variable"),
        pytest.param("foo|bar", id="pipe"),
        pytest.param('say "hi"', id="double-quotes"),
        pytest.param("it's", id="single-quote"),
        pytest.param("a;b", id="semicolon"),
        pytest.param("a && b", id="double-ampersand"),
    ],
)
async def test_exec_argv_when_arg_contains_metacharacter_does_pass_verbatim(
    make_opts: Callable[..., ExecOptions],
    arg: str,
) -> None:
    result = await exec_argv(
        [sys.executable, "-c", "import sys, json; print(json.dumps(sys.argv[1:]))", arg],
        make_opts(),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    received = json.loads(result.stdout.strip())
    assert received == [arg]


# ---------------------------------------------------------------------------
# cwd
# ---------------------------------------------------------------------------


async def test_exec_argv_when_cwd_specified_does_run_in_that_directory(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await exec_argv(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        make_opts(),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == physical_path(tmp_path)


# ---------------------------------------------------------------------------
# stdin
# ---------------------------------------------------------------------------


async def test_exec_argv_when_stdin_provided_does_deliver_to_child(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await exec_argv(
        [sys.executable, "-c", "import sys; print(sys.stdin.read(), end='')"],
        make_opts(stdin="piped input\n"),
    )

    assert result == expected_result("piped input\n", "", 0)


async def test_exec_argv_when_stdin_omitted_does_give_closed_input(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await exec_argv(
        [sys.executable, "-c", "import sys; print(sys.stdin.read(), end='')"],
        make_opts(),
    )

    assert result == expected_result("", "", 0)


# ---------------------------------------------------------------------------
# timeout
# ---------------------------------------------------------------------------


async def test_exec_argv_when_timeout_exceeded_does_return_timeout_error(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await exec_argv(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        make_opts(timeout_ms=500),
    )

    assert isinstance(result, ExecTimeoutError)
    assert result.timeout_ms == 500


async def test_exec_argv_when_timeout_exceeded_does_capture_partial_output(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await exec_argv(
        [
            sys.executable,
            "-c",
            "import time, sys; print('line 1', flush=True); time.sleep(10)",
        ],
        make_opts(timeout_ms=1500),
    )

    assert isinstance(result, ExecTimeoutError)
    assert "line 1" in result.stdout


# ---------------------------------------------------------------------------
# abort
# ---------------------------------------------------------------------------


async def test_exec_argv_when_aborted_mid_run_does_settle_as_failed_result(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    pid_file = tmp_path / "child.pid"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", script_writing_pid_and_sleeping(pid_file)],
            make_opts(abort=abort),
        ),
    )
    await wait_for_pid_file(pid_file)

    abort.set()

    result = await asyncio.wait_for(task, 5)
    assert result == expected_result("", "", 1)


async def test_exec_argv_when_event_preset_does_not_spawn_and_settles_failed(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    abort.set()
    marker = tmp_path / "completed.marker"

    result = await asyncio.wait_for(
        exec_argv(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('done')",
            ],
            make_opts(abort=abort),
        ),
        3,
    )

    assert spawned_processes == []
    assert result == expected_result("", "", 1)
    assert not marker.exists()


# ---------------------------------------------------------------------------
# the graceful request that precedes the kill
# ---------------------------------------------------------------------------


async def test_exec_argv_when_child_writes_stderr_on_graceful_request_does_capture_it(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    pid_file = tmp_path / "child.pid"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", script_answering_graceful_signal(pid_file)],
            make_opts(abort=abort),
        ),
    )
    await wait_for_pid_file(pid_file)

    abort.set()

    result = await asyncio.wait_for(task, 5)
    assert isinstance(result, ExecResult)
    assert result.stderr == "BYE\n"


async def test_exec_argv_when_graceful_signal_refused_does_not_warn_and_still_kills_group(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    # macOS answers EPERM for a group that is on its way out, which cannot be
    # told apart from a genuine refusal until the leader is reaped; the stop
    # request earns the same deferral the kill already gets.
    signalled: list[int] = []
    real_killpg = os.killpg

    def refuse_terminate(group_pid: int, sig: int) -> None:
        signalled.append(sig)
        if sig == signal.SIGTERM:
            raise PermissionError(errno.EPERM, os.strerror(errno.EPERM))
        real_killpg(group_pid, sig)

    monkeypatch.setattr(os, "killpg", refuse_terminate)
    abort = asyncio.Event()
    pid_file = tmp_path / "child.pid"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", script_writing_pid_and_sleeping(pid_file)],
            make_opts(abort=abort),
        ),
    )
    await wait_for_pid_file(pid_file)

    abort.set()

    result = await asyncio.wait_for(task, 5)
    assert signal.SIGTERM in signalled
    assert result == expected_result("", "", 1)
    assert killpg_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# spawn failure — never raises, always resolves to a failed ExecResult
# ---------------------------------------------------------------------------


async def test_exec_argv_when_binary_not_found_does_resolve_with_failure_on_stderr(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await exec_argv(
        ["no-such-binary-exists-anywhere"],
        make_opts(),
    )

    assert isinstance(result, ExecResult)
    assert result.stdout == ""
    assert result.exit_code == 1
    assert "No such file or directory" in result.stderr
    assert exec_mod._live_process_groups == set()


async def test_exec_argv_when_cwd_missing_does_resolve_with_failure_on_stderr(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist"

    result = await exec_argv(
        ["echo", "hello"],
        ExecOptions(cwd=str(missing)),
    )

    assert isinstance(result, ExecResult)
    assert result.stdout == ""
    assert result.exit_code == 1
    assert "No such file or directory" in result.stderr
    assert exec_mod._live_process_groups == set()


@pytest.mark.parametrize(
    ("argv", "cwd", "env"),
    [
        pytest.param(["echo", "a\x00b"], None, None, id="nul-in-argument"),
        pytest.param(["echo", "hello"], "nu\x00l", None, id="nul-in-cwd"),
        pytest.param(["echo", "hello"], None, {"VAR": "a\x00b"}, id="nul-in-env-value"),
    ],
)
async def test_exec_argv_when_spawn_argument_holds_nul_does_resolve_with_error_on_stderr(
    tmp_path: Path,
    argv: list[str],
    cwd: str | None,
    env: dict[str, str] | None,
) -> None:
    opts = ExecOptions(cwd=str(tmp_path) if cwd is None else cwd, env=env)

    result = await exec_argv(argv, opts)

    assert isinstance(result, ExecResult)
    assert result.stdout == ""
    assert result.exit_code == 1
    assert "embedded null byte" in result.stderr
    assert result.stderr.endswith("\n")
    assert result.stderr_bytes == len(result.stderr.encode())
    assert exec_mod._live_process_groups == set()


async def test_exec_argv_when_argv_empty_does_resolve_with_error_on_stderr(
    tmp_path: Path,
) -> None:
    result = await exec_argv([], ExecOptions(cwd=str(tmp_path)))

    assert isinstance(result, ExecResult)
    assert result.stdout == ""
    assert result.exit_code == 1
    assert result.stderr.strip() != ""
    assert result.stderr.endswith("\n")
    assert result.stderr_bytes == len(result.stderr.encode())
    assert exec_mod._live_process_groups == set()


# ---------------------------------------------------------------------------
# large stderr — concurrent pipe drain, no deadlock
# ---------------------------------------------------------------------------


async def test_exec_argv_when_stderr_exceeds_pipe_buffer_does_capture_both_streams(
    make_opts: Callable[..., ExecOptions],
) -> None:
    mib = 1024 * 1024
    result = await exec_argv(
        [
            sys.executable,
            "-c",
            (f"import sys; sys.stderr.write('E' * {mib}); sys.stderr.flush(); print('OK')"),
        ],
        make_opts(),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "OK"
    assert result.stderr_bytes >= mib


# ---------------------------------------------------------------------------
# live process-group registry
# ---------------------------------------------------------------------------


async def test_exec_argv_when_child_alive_does_register_in_live_groups(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            make_opts(timeout_ms=300),
        ),
    )
    proc = await wait_for_spawned(spawned_processes)

    assert proc.pid in exec_mod._live_process_groups

    await task


async def test_exec_argv_when_completed_does_deregister_from_live_groups(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await exec_argv([sys.executable, "-c", "pass"], make_opts())
    proc = spawned_processes[-1]

    attempted: list[int] = []

    def record(pid: int, *_a: object, **_k: object) -> None:
        attempted.append(pid)

    monkeypatch.setattr(exec_mod, "kill_process_group", record)
    exec_mod.kill_live_process_groups()

    assert proc.pid not in attempted


async def test_kill_live_process_groups_when_exec_argv_child_alive_does_kill_it(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    pid_file = tmp_path / "child.pid"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", script_writing_pid_and_sleeping(pid_file)],
            make_opts(),
        ),
    )
    child_pid = await wait_for_pid_file(pid_file)

    exec_mod.kill_live_process_groups()

    await wait_until_dead(child_pid, timeout_s=3.0)
    await task
    assert not is_alive(child_pid)


async def test_kill_live_process_groups_when_running_group_refuses_signals_does_warn(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    killpg_refusal: LiftableRefusal,
    background_runs: list["asyncio.Task[ExecResult | ExecTimeoutError]"],
) -> None:
    pid_file = tmp_path / "child.pid"
    background_runs.append(
        asyncio.create_task(
            exec_argv(
                [sys.executable, "-c", script_writing_pid_and_sleeping(pid_file)],
                make_opts(),
            ),
        ),
    )
    await wait_for_pid_file(pid_file)
    killpg_refusal.refusing = True

    with pytest.warns(RuntimeWarning, match="killpg failed"):
        exec_mod.kill_live_process_groups()


async def test_kill_live_process_groups_when_exited_leader_group_with_live_member_refuses_does_warn(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    stray_process_ids: list[int],
    make_opts: Callable[..., ExecOptions],
    *,
    killpg_refusal: LiftableRefusal,
    background_runs: list["asyncio.Task[ExecResult | ExecTimeoutError]"],
) -> None:
    grandchild_pid_file = tmp_path / "grandchild.pid"
    background_runs.append(
        asyncio.create_task(
            exec_argv(
                [sys.executable, "-c", script_leaving_grandchild_and_exiting(grandchild_pid_file)],
                make_opts(),
            ),
        ),
    )
    leader = await wait_for_spawned(spawned_processes)
    grandchild = await wait_for_pid_file(grandchild_pid_file)
    stray_process_ids.append(grandchild)
    await wait_until_dead(leader.pid)
    killpg_refusal.refusing = True

    with pytest.warns(RuntimeWarning, match="killpg failed"):
        exec_mod.kill_live_process_groups()


# ---------------------------------------------------------------------------
# cancellation kills the child
# ---------------------------------------------------------------------------


async def test_exec_argv_when_cancelled_does_kill_child_and_deregister(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    pid_file = tmp_path / "child.pid"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", script_writing_pid_and_sleeping(pid_file)],
            make_opts(),
        ),
    )
    proc = await wait_for_spawned(spawned_processes)
    await wait_for_pid_file(pid_file)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    await wait_until_dead(proc.pid, timeout_s=3.0)
    assert not is_alive(proc.pid)
    assert proc.pid not in exec_mod._live_process_groups


async def test_exec_argv_when_cancelled_does_keep_pid_registered_until_the_kill_lands(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A pid dropped from the registry before its group is killed is a pid the
    # interpreter-exit sweep can no longer reach, so the end state both steps
    # share says nothing about the order they ran in.
    registered_at_kill: list[bool] = []
    real_kill = exec_mod.kill_process_group

    def spy(pid: int, *, defer_refusal: bool = False) -> bool:
        registered_at_kill.append(pid in exec_mod._live_process_groups)
        return real_kill(pid, defer_refusal=defer_refusal)

    monkeypatch.setattr(exec_mod, "kill_process_group", spy)
    pid_file = tmp_path / "child.pid"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", script_writing_pid_and_sleeping(pid_file)],
            make_opts(),
        ),
    )
    proc = await wait_for_spawned(spawned_processes)
    await wait_for_pid_file(pid_file)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    await wait_until_dead(proc.pid, timeout_s=3.0)
    assert registered_at_kill, "the cancelled run never killed the child's group"
    assert all(registered_at_kill), "the pid left the registry before the kill reached its group"


# ---------------------------------------------------------------------------
# POSIX session leader — grandchild dies with child on abort / cancel
# ---------------------------------------------------------------------------


async def test_exec_argv_when_aborted_does_kill_grandchild(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    grandchild_pid_file = tmp_path / "grandchild.pid"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", script_spawning_grandchild_and_sleeping(grandchild_pid_file)],
            make_opts(abort=abort),
        ),
    )
    grandchild = await wait_for_pid_file(grandchild_pid_file)

    abort.set()

    await wait_until_dead(grandchild, timeout_s=3.0)
    await task
    assert not is_alive(grandchild)


async def test_exec_argv_when_cancelled_does_kill_grandchild(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    grandchild_pid_file = tmp_path / "grandchild.pid"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", script_spawning_grandchild_and_sleeping(grandchild_pid_file)],
            make_opts(abort=None),
        ),
    )
    await wait_for_spawned(spawned_processes)
    grandchild = await wait_for_pid_file(grandchild_pid_file)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    await wait_until_dead(grandchild, timeout_s=3.0)
    assert not is_alive(grandchild)


async def test_exec_argv_when_cancelled_leaving_only_a_zombie_in_group_does_not_warn_about_killpg(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    stray_process_ids: list[int],
    make_opts: Callable[..., ExecOptions],
    recwarn: pytest.WarningsRecorder,
) -> None:
    holder_pid_file = tmp_path / "holder.pid"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", ZOMBIE_ONLY_GROUP_SCRIPT, str(holder_pid_file)],
            make_opts(),
        ),
    )
    await wait_for_spawned(spawned_processes)
    stray_process_ids.append(await wait_for_pid_file(holder_pid_file))

    await cancel_and_settle(task)

    assert killpg_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# env option
# ---------------------------------------------------------------------------


async def test_exec_argv_when_env_set_does_use_exact_mapping(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await exec_argv(
        [sys.executable, "-c", "import os, json; print(json.dumps(dict(os.environ)))"],
        make_opts(env={"CUSTOM_VAR": "custom_value"}),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    child_env = json.loads(result.stdout.strip())
    assert child_env["CUSTOM_VAR"] == "custom_value"
    # A variable present in the parent but absent from the mapping must be
    # absent in the child.
    assert "HOME" not in child_env


async def test_exec_argv_when_env_none_does_inherit_parent_env(
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GYMRAT_TEST_MARKER", "inherited")

    result = await exec_argv(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('GYMRAT_TEST_MARKER', ''))",
        ],
        make_opts(env=None),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "inherited"


# ---------------------------------------------------------------------------
# child signal mask (POSIX)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="Signal masking requires POSIX pthread_sigmask",
)
async def test_exec_argv_when_spawned_does_unblock_termination_signals_in_child(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await exec_argv(
        [
            sys.executable,
            "-c",
            "import signal, json; print(json.dumps(list(signal.pthread_sigmask(signal.SIG_BLOCK, []))))",
        ],
        make_opts(),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    blocked: list[int] = json.loads(result.stdout.strip())
    for sig in TERMINATION_SIGNALS:
        assert sig not in blocked, f"signal {sig} should be unblocked in child"
