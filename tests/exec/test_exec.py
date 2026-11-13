"""Behavioral tests for the asyncio subprocess ``exec`` layer.

Both entry points are covered: ``exec`` runs a shell command, and ``exec_argv``
runs ``argv[0]`` with the remaining items as its arguments and no shell
interpretation. The teardown of a run (timeout, abort, stream error, cancellation) is shared
with ``exec_argv`` and tested in ``test_teardown``.

Real-subprocess tests are parallel-safe under ``pytest-xdist``: every run gets
its own directory via the ``tmp_path`` fixture, and POSIX-only shell constructs
(``$$``, ``>&2``, ``for``/``do``/``done``) mean the directory is not collected
on win32.
The win32 ``taskkill`` fallback is tested in ``tests/test_process_group.py``.
"""

import asyncio
import contextlib
import errno
import os
import re
import signal
import sys
import warnings
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecOptions, ExecResult, exec_argv
from gymrat.exec import exec as run_exec
from tests._exec_fixtures import (
    OVERRUN_KEPT_BYTES,
    SMALL_CAP,
    ExecRun,
    ExecTask,
    expected_result,
    open_gate_after_reading,
    shell_leaving_a_descendant,
    stderr_overruns,
    stdout_overruns,
    wait_for_gate,
    wait_for_spawned,
)
from tests._process_helpers import (
    REAL_KILLPG,
    killpg_warnings,
    wait_for_pid_file,
    wait_until_dead,
)

# ---------------------------------------------------------------------------
# captured output and stdin delivery
# ---------------------------------------------------------------------------


async def test_exec_when_multi_byte_char_split_across_reads_does_decode_single_char(
    make_opts: Callable[..., ExecOptions], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The second byte of U+00B5 is written only once the first has been read,
    # so they land in separate reads; one incremental decoder joins them.
    gate = tmp_path / "first-byte-read"
    open_gate_after_reading(monkeypatch, gate, b"\xc2")

    result = await run_exec(
        f"printf '\\302'; {wait_for_gate(gate)}; printf '\\265'",
        make_opts(),
    )

    assert result == expected_result("µ", "", 0)


async def test_exec_when_descendant_writes_after_shell_exits_does_capture_output(
    make_opts: Callable[..., ExecOptions],
) -> None:
    result = await run_exec("(sleep 0.2; echo METRIC) &", make_opts())

    assert result == expected_result("METRIC\n", "", 0)


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
            run_exec,
            "echo out; echo err >&2; exit 3",
            expected_result("out\n", "err\n", 3),
            id="exec",
        ),
        pytest.param(
            exec_argv,
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


# Both entry points feed stdin the same way, so each pairs with one case.
@pytest.mark.parametrize(
    ("run", "args", "stdin", "expected_stdout"),
    [
        pytest.param(run_exec, "cat", "piped input\n", "piped input\n", id="exec-stdin-delivered"),
        pytest.param(exec_argv, _ECHO_STDIN_ARGV, None, "", id="exec_argv-stdin-omitted-is-closed"),
    ],
)
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


def _fail_stdin_writes(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    # A run's stdin is the only stream writer it holds, so every write to it fails.
    def refuse(_writer: asyncio.StreamWriter, _data: object) -> None:
        raise error

    monkeypatch.setattr(asyncio.StreamWriter, "write", refuse)


async def test_exec_when_stdin_write_fails_does_settle_failed_with_the_error_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
) -> None:
    error = OSError(errno.EIO, "Input/output error")
    _fail_stdin_writes(monkeypatch, error)

    result = await asyncio.wait_for(run_exec("true", make_opts(stdin="payload\n")), 5)

    assert isinstance(result, ExecResult)
    assert result.exit_code == 1
    assert result.stderr.removesuffix("\n") == str(error)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(BrokenPipeError(errno.EPIPE, "Broken pipe"), id="broken-pipe"),
        pytest.param(ConnectionResetError("Connection reset"), id="reset"),
    ],
)
async def test_exec_when_stdin_closed_by_the_child_does_settle_normally(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    *,
    error: OSError,
) -> None:
    _fail_stdin_writes(monkeypatch, error)

    result = await asyncio.wait_for(run_exec("true", make_opts(stdin="payload\n")), 5)

    assert result == expected_result("", "", 0)


# ---------------------------------------------------------------------------
# spawn failure — never raises, always resolves to a failed ExecResult
# ---------------------------------------------------------------------------


def _in_tmp(tmp_path: Path) -> str:
    return str(tmp_path)


def _missing_dir(tmp_path: Path) -> str:
    return str(tmp_path / "does-not-exist")


# One OSError and one ValueError: every spawn the interpreter or host rejects
# takes the same path to a failed result, whatever check raised or entry point
# was used. An empty argv is the one rejection exec_argv makes itself.
@pytest.mark.parametrize(
    ("run", "args", "cwd"),
    [
        pytest.param(run_exec, "echo hello", _missing_dir, id="exec-cwd-missing"),
        pytest.param(run_exec, "echo a\x00b", _in_tmp, id="exec-nul-in-command"),
        pytest.param(exec_argv, [], _in_tmp, id="argv-empty"),
    ],
)
async def test_exec_when_spawn_fails_does_resolve_with_the_failure_on_stderr(
    tmp_path: Path,
    *,
    run: ExecRun,
    args: Any,
    cwd: Callable[[Path], str],
) -> None:
    result = await run(args, ExecOptions(cwd=cwd(tmp_path)))

    assert result == expected_result("", result.stderr, exit_code=1)
    assert result.stderr.strip() != ""


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


async def test_exec_when_stream_read_fails_does_settle_as_failure(
    monkeypatch: pytest.MonkeyPatch,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
) -> None:
    # A cap shorter than the diagnostic, so its length alone cannot pass for truncation.
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", 4)
    task = asyncio.create_task(run_exec("sleep 30", make_opts()))
    proc = await wait_for_spawned(spawned_processes)
    assert proc.stdout is not None

    proc.stdout.set_exception(RuntimeError("stream exploded"))

    result = await asyncio.wait_for(task, 3)
    assert isinstance(result, ExecResult)
    assert result.exit_code == 1
    assert result.stderr.removesuffix("\n") == "stream exploded"
    # The diagnostic text is internal, not command output — byte count stays zero.
    assert result.stderr_bytes == 0
    # Nor does the diagnostic count as output dropped past the cap.
    assert result.stderr_truncated is False


# ---------------------------------------------------------------------------
# output cap
# ---------------------------------------------------------------------------


async def test_exec_when_output_past_cap_is_dropped_does_still_count_its_bytes(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
) -> None:
    # A cap of zero drops every chunk: the prior total of 0 already reaches it.
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", 0)

    result = await run_exec("echo hello", make_opts())

    assert result == ExecResult(
        stdout="",
        stderr="",
        exit_code=0,
        stdout_bytes=len(b"hello\n"),
        stderr_bytes=0,
        stdout_truncated=True,
    )


async def test_exec_when_chunk_crosses_cap_does_keep_it_whole(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    tmp_path: Path,
) -> None:
    # The prior total (5) is still under the cap, so the chunk that crosses it
    # is kept whole rather than cut at the cap.
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", 10)
    gate = tmp_path / "plums-read"
    open_gate_after_reading(monkeypatch, gate, b"plums")

    result = await run_exec(f"printf plums; {wait_for_gate(gate)}; printf cherries", make_opts())

    assert result == expected_result("plums" + "cherries")


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param(
            stdout_overruns,
            (True, False),
            id="stdout-overran",
        ),
        pytest.param(
            stderr_overruns,
            (False, True),
            id="stderr-overran",
        ),
    ],
)
async def test_exec_when_run_finishes_does_flag_each_stream_that_overran_the_cap(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    *,
    tmp_path: Path,
    command: Callable[[Path], str],
    expected: tuple[bool, bool],
) -> None:
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", SMALL_CAP)
    gate = tmp_path / "kept-bytes-read"
    open_gate_after_reading(monkeypatch, gate, OVERRUN_KEPT_BYTES)

    result = await run_exec(command(gate), make_opts())

    assert (result.stdout_truncated, result.stderr_truncated) == expected


async def test_exec_when_stream_ends_at_cap_mid_character_does_keep_tail_untruncated(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
) -> None:
    # Three bytes land exactly on the cap; the last opens a character the stream
    # never finishes, so the decoder holds it back until EOF. Its replacement
    # belongs to a kept chunk, so nothing was dropped.
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", 3)

    result = await run_exec("printf 'ab\\342'", make_opts())

    assert (result.stdout, result.stdout_truncated) == ("ab�", False)


# ---------------------------------------------------------------------------
# live process-group registry
# ---------------------------------------------------------------------------

# Upper bound for a run left going by a test to finish once its group is killed.
_RUN_SETTLE_TIMEOUT_S = 5.0


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


@pytest.mark.usefixtures("spawned_processes")
async def test_kill_live_process_groups_when_run_has_settled_does_not_target_its_group(
    make_opts: Callable[..., ExecOptions],
    record_killpg: Callable[[], list[int]],
) -> None:
    await run_exec("echo hello", make_opts())
    attempted = record_killpg()

    exec_mod.kill_live_process_groups()

    assert attempted == []


async def test_kill_live_process_groups_when_group_leader_exited_does_still_signal_its_group(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    stray_process_ids: list[int],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    *,
    record_killpg: Callable[[], list[int]],
    background_runs: list[ExecTask],
) -> None:
    # The grandchild holds the run's stderr, so the run keeps going after the
    # shell leading the group has exited.
    background_runs.append(
        asyncio.create_task(
            run_exec(shell_leaving_a_descendant(tmp_path / "grandchild.pid"), make_opts())
        ),
    )
    leader = await wait_for_spawned(spawned_processes)
    grandchild = await wait_for_pid_file(tmp_path / "grandchild.pid")
    stray_process_ids.append(grandchild)
    await wait_until_dead(leader.pid)
    # No grace: the recorder signals nothing, so the sweep would otherwise wait it out.
    monkeypatch.setattr(exec_mod, "TERMINATE_GRACE_S", 0)
    attempted = record_killpg()

    exec_mod.kill_live_process_groups()

    assert set(attempted) == {leader.pid}


# ---------------------------------------------------------------------------
# a sweep step that raises
# ---------------------------------------------------------------------------

# A group that ignores the stop request, so only the kill step ends it.
_TERM_IGNORING_COMMAND = "trap '' TERM; sleep 30"

# Long enough that a sweep holding a wait step runs it, short enough to sit out.
_SWEEP_GRACE_S = 0.2

_STEP_ERROR = "sweep step exploded"


async def _start_term_ignoring_runs(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    background_runs: list[ExecTask],
) -> None:
    background_runs.extend(
        asyncio.create_task(run_exec(_TERM_IGNORING_COMMAND, make_opts())) for _ in range(2)
    )
    await wait_for_spawned(spawned_processes, count=2)


# The real ``os.kill``, captured before any test swaps it.
_REAL_KILL = os.kill


def _fail_sweep_step(monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    # Fail the OS call each sweep step makes: the group signal for terminate
    # (SIGTERM) and kill (SIGKILL), the first liveness probe (signal 0) for wait.
    # Every other call reaches the real OS, so the groups still end.
    monkeypatch.setattr(exec_mod, "TERMINATE_GRACE_S", _SWEEP_GRACE_S)
    if step == "wait":
        probes_failed: list[int] = []

        def kill_failing_first_probe(pid: int, signal_number: int) -> None:
            if signal_number == 0 and not probes_failed:
                probes_failed.append(pid)
                raise RuntimeError(_STEP_ERROR)
            _REAL_KILL(pid, signal_number)

        monkeypatch.setattr(os, "kill", kill_failing_first_probe)
        return
    failing_signal = signal.SIGTERM if step == "terminate" else signal.SIGKILL

    def killpg_failing_step(group_pid: int, signal_number: int) -> None:
        if signal_number == failing_signal:
            raise RuntimeError(_STEP_ERROR)
        REAL_KILLPG(group_pid, signal_number)

    monkeypatch.setattr(os, "killpg", killpg_failing_step)


@pytest.mark.parametrize("step", ["terminate", "kill"])
async def test_kill_live_process_groups_when_a_group_step_raises_does_warn_once_per_group_naming_step_and_error(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
    background_runs: list[ExecTask],
    *,
    step: str,
) -> None:
    await _start_term_ignoring_runs(spawned_processes, make_opts, background_runs)
    _fail_sweep_step(monkeypatch, step)

    exec_mod.kill_live_process_groups()

    for proc in spawned_processes:
        [message] = killpg_warnings(recwarn, proc.pid)
        assert re.search(rf"\b{step}\b", message)
        assert _STEP_ERROR in message


async def test_kill_live_process_groups_when_the_wait_step_raises_does_warn_once_naming_step_and_error(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
    background_runs: list[ExecTask],
) -> None:
    await _start_term_ignoring_runs(spawned_processes, make_opts, background_runs)
    _fail_sweep_step(monkeypatch, "wait")

    exec_mod.kill_live_process_groups()

    messages = [str(w.message) for w in recwarn if issubclass(w.category, RuntimeWarning)]
    [message] = [m for m in messages if _STEP_ERROR in m]
    assert re.search(r"\bwait\b", message)


@pytest.mark.parametrize("step", ["terminate", "wait"])
@pytest.mark.usefixtures("recwarn")
async def test_kill_live_process_groups_when_an_earlier_step_raises_does_still_kill_every_group(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    background_runs: list[ExecTask],
    *,
    step: str,
) -> None:
    await _start_term_ignoring_runs(spawned_processes, make_opts, background_runs)
    _fail_sweep_step(monkeypatch, step)

    exec_mod.kill_live_process_groups()

    for proc in spawned_processes:
        await wait_until_dead(proc.pid, timeout_s=3.0)


async def test_kill_live_process_groups_when_warnings_are_errors_and_a_step_raises_does_not_raise(
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    background_runs: list[ExecTask],
) -> None:
    await _start_term_ignoring_runs(spawned_processes, make_opts, background_runs)
    _fail_sweep_step(monkeypatch, "terminate")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exec_mod.kill_live_process_groups()
