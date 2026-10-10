"""Behavioral tests for the asyncio subprocess ``exec`` layer.

The teardown of a run (timeout, abort, stream error, cancellation) is shared
with ``exec_argv`` and tested in ``test_teardown``.

Real-subprocess tests are parallel-safe under ``pytest-xdist``: every run gets
its own directory via the ``tmp_path`` fixture, and POSIX-only shell constructs
(``$$``, ``>&2``, ``for``/``do``/``done``) mean the directory is not collected
on win32.
The win32 ``taskkill`` fallback is tested in ``tests/test_process_group.py``.
"""

import asyncio
import contextlib
import json
import os
import shlex
import signal
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecOptions, ExecResult
from gymrat.exec import exec as run_exec
from gymrat.signals import TERMINATION_SIGNALS
from tests._exec_fixtures import (
    ExecRun,
    ExecTask,
    expected_result,
    fresh_live_process_groups,
    run_argv,
    run_shell,
    shell_grandchild,
    wait_for_spawned,
)
from tests._process_helpers import (
    REAL_KILLPG,
    KillpgRefusal,
    is_alive,
    refuse_killpg,
    wait_for_pid_file,
    wait_until_dead,
)

# ---------------------------------------------------------------------------
# captured output and stdin delivery
# ---------------------------------------------------------------------------

#: A process-group id no test spawns; recorded, never signaled.
_LEAKED_GROUP_PID = 999_999


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


# One OSError and one ValueError per entry point: every spawn the interpreter or
# host rejects takes the same path to a failed result, whatever check raised.
@pytest.mark.parametrize(
    ("run", "args", "cwd", "message"),
    [
        pytest.param(
            run_shell,
            "echo hello",
            _missing_dir,
            "[Errno 2] No such file or directory: '{cwd}'",
            id="exec-cwd-missing",
        ),
        pytest.param(
            run_shell, "echo a\x00b", _in_tmp, "embedded null byte", id="exec-nul-in-command"
        ),
        pytest.param(
            run_argv,
            ["no-such-binary-exists-anywhere"],
            _in_tmp,
            "[Errno 2] No such file or directory: 'no-such-binary-exists-anywhere'",
            id="argv-binary-not-found",
        ),
        pytest.param(
            run_argv, ["echo", "a\x00b"], _in_tmp, "embedded null byte", id="argv-nul-in-argument"
        ),
        pytest.param(run_argv, [], _in_tmp, "argv is empty: no program to run", id="argv-empty"),
    ],
)
async def test_exec_when_spawn_fails_does_resolve_with_the_failure_on_stderr(
    tmp_path: Path,
    *,
    run: ExecRun,
    args: Any,
    cwd: Callable[[Path], str],
    message: str,
) -> None:
    run_dir = cwd(tmp_path)

    result = await run(args, ExecOptions(cwd=run_dir))

    assert result == expected_result("", message.format(cwd=run_dir) + "\n", exit_code=1)


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


@pytest.mark.parametrize(
    ("cap", "command", "stdout", "stdout_bytes"),
    [
        # A cap of zero drops every chunk: the prior total of 0 already reaches it.
        pytest.param(0, "echo hello", "", len(b"hello\n"), id="cap-reached-drops-every-chunk"),
        # The prior total (5) is still under the cap, so the chunk that crosses
        # it is kept whole rather than cut at the cap.
        pytest.param(
            10,
            "printf plums; sleep 0.2; printf cherries",
            "plums" + "cherries",
            len(b"plums" + b"cherries"),
            id="crossing-chunk-kept-whole",
        ),
    ],
)
async def test_exec_when_output_exceeds_cap_does_stop_appending_but_keep_counting(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    *,
    cap: int,
    command: str,
    stdout: str,
    stdout_bytes: int,
) -> None:
    # Byte counting continues past the cap and the child is never signalled.
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", cap)

    result = await run_exec(command, make_opts())

    assert result == ExecResult(
        stdout=stdout, stderr="", exit_code=0, stdout_bytes=stdout_bytes, stderr_bytes=0
    )


# ---------------------------------------------------------------------------
# live process-group registry
# ---------------------------------------------------------------------------

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
    # runs next. Killing through ``REAL_KILLPG`` sidesteps any refusal the test
    # installed and ends every member holding the run's pipes, so each run completes on
    # its own. A group the test already ended is gone, or refuses while it finishes
    # exiting.
    for proc in spawned_processes:
        with contextlib.suppress(OSError):
            REAL_KILLPG(proc.pid, signal.SIGKILL)
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


def test_fresh_live_process_groups_when_a_group_leaked_in_does_hide_it_from_the_sweep_until_exit(
    record_killpg: Callable[[], list[int]],
) -> None:
    exec_mod._live_process_groups.add(_LEAKED_GROUP_PID)
    attempted = record_killpg()

    with fresh_live_process_groups():
        exec_mod.kill_live_process_groups()

    assert attempted == []
    assert _LEAKED_GROUP_PID in exec_mod._live_process_groups


async def test_kill_live_process_groups_when_registry_reset_does_spare_every_earlier_child(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    background_runs: list[ExecTask],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    background_runs.extend(
        asyncio.create_task(run_exec(shell_grandchild(tmp_path / f"earlier{n}.pid"), make_opts()))
        for n in range(2)
    )
    await wait_for_spawned(spawned_processes, count=2)
    earlier_grandchildren = [
        await wait_for_pid_file(tmp_path / f"earlier{n}.pid") for n in range(2)
    ]
    monkeypatch.setattr(exec_mod, "_live_process_groups", set())
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
        shlex.join([
            sys.executable,
            "-c",
            (
                "import signal, json; "
                "print(json.dumps(list(signal.pthread_sigmask(signal.SIG_BLOCK, []))))"
            ),
        ]),
        make_opts(),
    )

    assert isinstance(result, ExecResult)
    assert result.exit_code == 0
    blocked: list[int] = json.loads(result.stdout.strip())
    assert sorted(set(TERMINATION_SIGNALS) & set(blocked)) == []
