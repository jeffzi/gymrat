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
import json
import os
import signal
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError, exec_argv
from gymrat.exec import exec as run_exec
from tests._exec_fixtures import (
    ExecRun,
    ExecTask,
    expected_result,
    run_argv,
    run_shell,
    shell_leaving_a_descendant,
    wait_for_spawned,
)
from tests._process_helpers import (
    REAL_KILLPG,
    wait_for_pid_file,
    wait_until_dead,
)

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
        pytest.param(run_shell, "cat", "piped input\n", "piped input\n", id="exec-stdin-delivered"),
        pytest.param(run_argv, _ECHO_STDIN_ARGV, None, "", id="exec_argv-stdin-omitted-is-closed"),
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
        pytest.param(run_shell, "echo hello", _missing_dir, id="exec-cwd-missing"),
        pytest.param(run_shell, "echo a\x00b", _in_tmp, id="exec-nul-in-command"),
        pytest.param(run_argv, [], _in_tmp, id="argv-empty"),
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
# exec_argv: no shell interpretation — metacharacters pass through verbatim
# ---------------------------------------------------------------------------


async def test_exec_argv_when_arg_contains_metacharacter_does_pass_verbatim(
    make_opts: Callable[..., ExecOptions],
) -> None:
    arg = "$HOME"

    result = await exec_argv(
        [sys.executable, "-c", "import sys, json; print(json.dumps(sys.argv[1:]))", arg],
        make_opts(),
    )

    received = json.loads(result.stdout.strip())
    assert received == [arg]


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
    task = asyncio.create_task(run_exec("sleep 0.5", make_opts()))
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


@pytest.mark.parametrize(
    ("cap", "command", "stdout", "stdout_bytes", "stdout_truncated"),
    [
        # A cap of zero drops every chunk: the prior total of 0 already reaches it.
        pytest.param(
            0, "echo hello", "", len(b"hello\n"), True, id="cap-reached-drops-every-chunk"
        ),
        # The prior total (5) is still under the cap, so the chunk that crosses
        # it is kept whole rather than cut at the cap.
        pytest.param(
            10,
            "printf plums; sleep 0.2; printf cherries",
            "plums" + "cherries",
            len(b"plums" + b"cherries"),
            False,
            id="crossing-chunk-kept-whole",
        ),
    ],
)
async def test_exec_when_output_reaches_cap_does_keep_chunks_started_under_it_and_count_every_byte(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    *,
    cap: int,
    command: str,
    stdout: str,
    stdout_bytes: int,
    stdout_truncated: bool,
) -> None:
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", cap)

    result = await run_exec(command, make_opts())

    assert result == ExecResult(
        stdout=stdout,
        stderr="",
        exit_code=0,
        stdout_bytes=stdout_bytes,
        stderr_bytes=0,
        stdout_truncated=stdout_truncated,
    )


_SMALL_CAP = 2

# Three invalid bytes, read before the cap is reached and so kept whole, decode
# to three U+FFFD that re-encode to nine bytes; the later "ab" is dropped. The
# kept text re-encodes longer than the five bytes written, yet the stream overran.
_OVERRUN_WITH_INVALID_KEPT = "printf '\\377\\377\\377'; sleep 0.2; printf ab"

# One invalid byte, short of the cap, decodes to a U+FFFD that re-encodes to
# three bytes: longer than the one byte written, yet nothing was dropped.
_SHORT_WITH_INVALID = "printf '\\377'"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param(
            f"{_SHORT_WITH_INVALID} >&2; {_OVERRUN_WITH_INVALID_KEPT}",
            (True, False),
            id="stdout-overran",
        ),
        pytest.param(
            f"{_SHORT_WITH_INVALID}; {{ {_OVERRUN_WITH_INVALID_KEPT}; }} >&2",
            (False, True),
            id="stderr-overran",
        ),
    ],
)
async def test_exec_when_run_finishes_does_flag_each_stream_that_overran_the_cap(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    *,
    command: str,
    expected: tuple[bool, bool],
) -> None:
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", _SMALL_CAP)

    result = await run_exec(command, make_opts())

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


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param(
            f"{_SHORT_WITH_INVALID} >&2; {_OVERRUN_WITH_INVALID_KEPT}; sleep 30",
            (True, False),
            id="stdout-overran",
        ),
        pytest.param(
            f"{_SHORT_WITH_INVALID}; {{ {_OVERRUN_WITH_INVALID_KEPT}; }} >&2; sleep 30",
            (False, True),
            id="stderr-overran",
        ),
    ],
)
async def test_exec_when_run_times_out_does_flag_each_stream_that_overran_the_cap(
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    *,
    command: str,
    expected: tuple[bool, bool],
) -> None:
    monkeypatch.setattr(exec_mod, "OUTPUT_CAP", _SMALL_CAP)

    result = await run_exec(command, make_opts(timeout_ms=1000))

    assert isinstance(result, ExecTimeoutError)
    assert (result.stdout_truncated, result.stderr_truncated) == expected


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


@pytest.mark.parametrize(
    ("run", "args", "timeout_ms"),
    [
        pytest.param(run_shell, "echo hello", None, id="exec-completes"),
        # Every teardown (timeout, abort, stream error, cancel) settles through the
        # same release, so the timeout stands in for all of them.
        pytest.param(run_shell, "sleep 30", 200, id="exec-torn-down"),
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
