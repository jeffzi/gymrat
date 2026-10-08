"""Behavioral tests for the ``exec_argv`` subprocess layer.

``exec_argv`` runs ``argv[0]`` with the remaining items as its arguments and
no shell interpretation.  These tests mirror the ``exec`` (shell) tests in
``test_exec.py`` but exercise the direct-exec path: every argument is a
separate ``sys.argv`` entry, and no shell metacharacter is ever expanded.

Real-subprocess tests are POSIX-only for the same reasons as the shell form:
process groups, session leaders, and ``os.killpg`` do not exist on win32.
"""

import asyncio
import errno
import json
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecOptions, ExecResult, exec_argv
from tests._exec_fixtures import cancel_and_settle, expected_result, wait_for_spawned
from tests._process_helpers import (
    ZOMBIE_ONLY_GROUP_SCRIPT,
    killpg_warnings,
    wait_for_pid_file,
    wait_until_dead,
)

if sys.platform == "win32":
    pytest.skip("POSIX-only process groups", allow_module_level=True)


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
# abort
# ---------------------------------------------------------------------------


async def test_exec_argv_when_aborted_child_exits_zero_on_request_does_settle_as_failed_result(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    pid_file = tmp_path / "child.pid"
    script = (
        "import os, signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()) + '\\n'); "
        "time.sleep(30)"
    )
    task = asyncio.create_task(exec_argv([sys.executable, "-c", script], make_opts(abort=abort)))
    await wait_for_pid_file(pid_file)

    abort.set()

    result = await asyncio.wait_for(task, 5)
    assert result == expected_result("", "", 1)


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


async def test_exec_argv_when_graceful_signal_refused_does_kill_group_without_warning(
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
# cancellation kills the child
# ---------------------------------------------------------------------------


async def test_exec_argv_when_cancelled_does_keep_pid_registered_until_the_kill_lands(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A pid released before its group is killed is a pid the interpreter-exit
    # sweep can no longer reach, so the end state both steps share says nothing
    # about the order they ran in. Releasing the child's container is the step
    # that also drops its pid from the registry.
    steps: list[str] = []
    real_kill = exec_mod.kill_process_group
    real_release = exec_mod.release_process_group

    def record_kill(pid: int, *, defer_refusal: bool = False) -> bool:
        steps.append("kill")
        return real_kill(pid, defer_refusal=defer_refusal)

    def record_release(pid: int) -> None:
        steps.append("release")
        real_release(pid)

    monkeypatch.setattr(exec_mod, "kill_process_group", record_kill)
    monkeypatch.setattr(exec_mod, "release_process_group", record_release)
    pid_file = tmp_path / "child.pid"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", script_writing_pid_and_sleeping(pid_file)],
            make_opts(),
        ),
    )
    proc = await wait_for_spawned(spawned_processes, spawner="exec_argv")
    await wait_for_pid_file(pid_file)

    await cancel_and_settle(task)

    await wait_until_dead(proc.pid, timeout_s=3.0)
    assert "kill" in steps[: steps.index("release")], (
        "the pid left the registry before the kill reached its group"
    )


# ---------------------------------------------------------------------------
# POSIX session leader — grandchild dies with child on abort / cancel
# ---------------------------------------------------------------------------


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
    await wait_for_spawned(spawned_processes, spawner="exec_argv")
    stray_process_ids.append(await wait_for_pid_file(holder_pid_file))

    await cancel_and_settle(task)

    assert killpg_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# env option
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected_marker"),
    [
        pytest.param(None, "inherited", id="inherited-env"),
        pytest.param({"GYMRAT_TEST_MARKER": "mapped"}, "mapped", id="explicit-env"),
        pytest.param({}, None, id="empty-env"),
    ],
)
async def test_exec_argv_when_env_given_or_omitted_does_hand_child_that_env_one_nesting_level_deeper(
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str] | None,
    expected_marker: str | None,
) -> None:
    monkeypatch.setattr(exec_mod, "_NESTING_DEPTH", 2)
    monkeypatch.setenv("GYMRAT_TEST_MARKER", "inherited")

    result = await exec_argv(
        [sys.executable, "-c", "import os, json; print(json.dumps(dict(os.environ)))"],
        make_opts(env=env),
    )

    assert isinstance(result, ExecResult)
    child_env = json.loads(result.stdout.strip())
    assert {
        "GYMRAT_TEST_MARKER": child_env.get("GYMRAT_TEST_MARKER"),
        "GYMRAT_NESTING_DEPTH": child_env.get("GYMRAT_NESTING_DEPTH"),
    } == {"GYMRAT_TEST_MARKER": expected_marker, "GYMRAT_NESTING_DEPTH": "3"}


# ---------------------------------------------------------------------------
# child session (POSIX)
# ---------------------------------------------------------------------------


async def test_exec_argv_when_child_spawned_does_make_it_lead_its_own_session(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await exec_argv(
        [sys.executable, "-c", "import os; print(os.getsid(0) == os.getpid())"],
        make_opts(),
    )

    assert isinstance(result, ExecResult)
    assert result.stdout == "True\n"
