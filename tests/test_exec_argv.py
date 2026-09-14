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
import json
import os
import signal
import sys
from collections.abc import Callable, Iterator
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
from tests._process_helpers import capture_spawns, is_alive, wait_until_dead

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only process groups")


def expected_result(stdout: str, stderr: str, exit_code: int) -> ExecResult:
    """Build an expected ``ExecResult`` with byte counts derived from the strings."""
    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


def _read_pid(pid_path: Path) -> int | None:
    """Read a pid file; return ``None`` while the write is absent or incomplete."""
    try:
        raw = pid_path.read_text()
    except FileNotFoundError:
        return None
    return int(raw) if raw.endswith("\n") else None


async def wait_for_pid(pid_path: Path, timeout_s: float = 3.0) -> int:
    """Poll ``pid_path`` until it holds a complete, positive pid, then return it."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        pid = _read_pid(pid_path)
        if pid is not None and pid > 0:
            return pid
        if loop.time() > deadline:
            msg = f"pid never appeared at {pid_path}"
            raise TimeoutError(msg)
        await asyncio.sleep(0.025)


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


def script_spawning_grandchild_and_sleeping(grandchild_pid_file: Path) -> str:
    """Build a script that spawns a sleeping grandchild, writes its pid, then sleeps itself.

    Used to verify that killing the child's process group also kills any
    descendants forked from a POSIX session leader.
    """
    return (
        "import os, subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"open({str(grandchild_pid_file)!r}, 'w').write(str(p.pid) + '\\n')\n"
        "time.sleep(30)\n"
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
    """Record every child ``exec_argv`` spawns."""
    processes = capture_spawns(monkeypatch, "create_subprocess_exec")
    yield processes

    for proc in processes:
        if proc.returncode is not None or not proc.pid:
            continue
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)


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
    await wait_for_pid(pid_file)

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
# spawn failure (argv[0] not found, cwd missing)
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
    child_pid = await wait_for_pid(pid_file)

    exec_mod.kill_live_process_groups()

    await wait_until_dead(child_pid, timeout_s=3.0)
    await task
    assert not is_alive(child_pid)


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
    await wait_for_pid(pid_file)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    await wait_until_dead(proc.pid, timeout_s=3.0)
    assert not is_alive(proc.pid)


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
    grandchild = await wait_for_pid(grandchild_pid_file)

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
    grandchild = await wait_for_pid(grandchild_pid_file)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    await wait_until_dead(grandchild, timeout_s=3.0)
    assert not is_alive(grandchild)


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
