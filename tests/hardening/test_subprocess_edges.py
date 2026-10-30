"""Hardening tests for the asyncio-subprocess edges of ``exec``.

Where :mod:`tests.exec.test_exec` pins the happy paths, these tests pin the ragged
edges that only surface with a real child process misbehaving:

- aborting a run mid-read leaks no "Task ... was never retrieved" / "Task was
  destroyed but it is pending" diagnostics from ``exec``,
- a termination signal landing between spawn and registration still kills the
  child's group.

The tests are POSIX-only: their abort paths rely on process-group tree-kill,
and the fixtures reap any group a child leaves behind so nothing is orphaned.
Every real-tree test roots its scratch files under ``tmp_path``, so the suite
stays order-independent under ``pytest-xdist`` and ``pytest-randomly``.
"""

import asyncio
import gc
import os
import signal
import threading
import warnings
from pathlib import Path
from typing import Any

import pytest

from gymrat import exec as exec_mod
from gymrat import signals
from gymrat.exec import FAILURE_EXIT_CODE, ExecOptions, ExecResult
from gymrat.exec import exec as run_exec
from tests._platform import needs_posix_process_groups
from tests._process_helpers import wait_for_file
from tests._task_leaks import install_task_leak_recorder, task_leak_messages

# ---------------------------------------------------------------------------
# aborting mid-read leaks no forgotten-task diagnostics
# ---------------------------------------------------------------------------


@needs_posix_process_groups
async def test_exec_when_aborted_mid_read_does_not_leak_task_diagnostics(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
) -> None:
    records = install_task_leak_recorder()
    abort = asyncio.Event()
    task = asyncio.create_task(
        run_exec("echo ready > ready.flag; sleep 30", ExecOptions(cwd=str(tmp_path), abort=abort))
    )
    await wait_for_file(tmp_path / "ready.flag")

    abort.set()
    await asyncio.wait_for(task, 10)
    await asyncio.wait_for(asyncio.gather(*(proc.wait() for proc in spawned_processes)), 10)
    # A forgotten task sits in a reference cycle with the future it waits on, so
    # only a collection finalizes it. The same collection finalizes the reaped
    # child's transport, whose ResourceWarning is not what this test is about.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        gc.collect()

    assert task_leak_messages(records) == []
    assert spawned_processes  # the abort ran through a real spawned child


# ---------------------------------------------------------------------------
# termination signal between spawn and registration kills the child
# ---------------------------------------------------------------------------


@needs_posix_process_groups
async def test_exec_when_termination_signal_during_spawn_does_still_kill_child_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_exits: list[tuple[int, float]],
    spawned_processes: list[asyncio.subprocess.Process],
) -> None:
    # Hold the spawn-to-register gap open until SIGTERM has been sent, so the
    # signal always lands inside it. The spawner read here is already the
    # recording wrapper, so every child lands in ``spawned_processes`` and is
    # reaped at teardown.
    real_spawn = asyncio.create_subprocess_shell
    spawn_barrier = threading.Event()
    sent = threading.Event()

    async def slow_spawn(command: str, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_spawn(command, **kwargs)
        spawn_barrier.set()
        await asyncio.to_thread(sent.wait, 5.0)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", slow_spawn)

    uninstall = signals.install_termination_cleanup(exec_mod.kill_live_process_groups)

    def send_signal() -> None:
        spawn_barrier.wait(5.0)
        os.kill(os.getpid(), signal.SIGTERM)
        sent.set()

    sender = threading.Thread(target=send_signal, daemon=True)
    try:
        sender.start()

        result = await asyncio.wait_for(
            run_exec(
                "sleep 30",
                ExecOptions(cwd=str(tmp_path), timeout_ms=8000),
            ),
            timeout=10,
        )
    finally:
        # Release the sender even when the spawn never happened, so its 5 s
        # barrier wait cannot outlast the join — a survivor would fire SIGTERM
        # after the exit-seam monkeypatch is undone and kill the worker.
        spawn_barrier.set()
        sender.join(timeout=6.0)
        assert not sender.is_alive(), "SIGTERM sender thread outlived its join window"
        uninstall()

    # The signal mask around spawn+register must cover the gap: a SIGTERM
    # landing before registration still finds the child in the live-groups
    # registry and kills it, so exec settles as a normal result, not a timeout.
    # An empty stderr rules out the spawn-failure result, which carries the
    # spawn error's message under the same failure code.
    assert isinstance(result, ExecResult), (
        "child was not killed by the signal handler — spawn-register window is unmasked"
    )
    assert (result.exit_code, result.stderr) == (FAILURE_EXIT_CODE, "")
    assert spawned_processes
    assert all(proc.returncode is not None and proc.returncode < 0 for proc in spawned_processes), (
        f"child was not killed by a signal: {[proc.returncode for proc in spawned_processes]}"
    )
    assert recorded_exits, "the termination handler never asked the process to exit"
