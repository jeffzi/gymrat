"""Hardening tests for the asyncio-subprocess edges of ``exec`` and the Claude driver.

Where :mod:`tests.exec.test_exec` pins the happy paths, these tests pin the ragged
edges that only surface with a real child process misbehaving:

- aborting a run mid-read leaks no "Task ... was never retrieved" / "Task was
  destroyed but it is pending" diagnostics from the driver or ``exec``,
- a termination signal landing between spawn and registration still kills the
  child's group.

The module is POSIX-only: the abort paths rely on process-group tree-kill and
the fixtures reap any group a child leaves behind so nothing is orphaned. Every
real-tree test roots its scratch files under ``tmp_path``, so the suite stays
order-independent under ``pytest-xdist`` and ``pytest-randomly``.
"""

import asyncio
import contextlib
import gc
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gymrat import exec as exec_mod
from gymrat import signals
from gymrat.exec import FAILURE_EXIT_CODE, ExecOptions, ExecResult
from gymrat.exec import exec as run_exec
from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.events import SessionEvent, SessionObserver, UsageUpdateEvent
from tests._process_helpers import wait_for_file
from tests.supervisor._fixtures import (
    FactoryProbe,
    FakeClient,
    FiniteClient,
    collecting_observer,
    make_prompt,
    result_message,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only process groups for tree-kill"
)

# The two asyncio diagnostics that mark a task the code forgot to await or
# retrieve; both route through the loop's exception handler.
_LEAK_MARKERS = ("was destroyed but it is pending", "exception was never retrieved")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def install_task_leak_recorder() -> list[dict[str, object]]:
    """Route the running loop's exception handler into a list.

    Both "Task was destroyed but it is pending!" and "Task exception was never
    retrieved" are reported through ``loop.call_exception_handler`` when the
    offending task is finalized, so recording every context the handler sees —
    then forcing a collection — captures a forgotten task deterministically.

    Returns:
        The list every context the handler sees is appended to.
    """
    loop = asyncio.get_running_loop()
    records: list[dict[str, object]] = []

    def handler(_loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        records.append(context)

    loop.set_exception_handler(handler)
    return records


def task_leak_messages(records: list[dict[str, object]]) -> list[str]:
    """The recorded handler messages that name a forgotten-task diagnostic."""
    messages = [str(context.get("message", "")) for context in records]
    return [msg for msg in messages if any(marker in msg for marker in _LEAK_MARKERS)]


# ---------------------------------------------------------------------------
# aborting mid-read leaks no forgotten-task diagnostics
# ---------------------------------------------------------------------------


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
    # No forced collection here: exec awaits its own reader/kill tasks on the
    # abort path, so there is no forgotten task exception to finalize, and forcing a
    # collection would only finalize the subprocess transport as loop noise.
    await asyncio.sleep(0)

    assert task_leak_messages(records) == []
    assert spawned_processes  # the abort ran through a real spawned child


def _never_abort(_abort: asyncio.Event) -> SessionObserver:
    """An observer that leaves the abort unfired, so its watch task must be cleaned up."""
    return collecting_observer().observer


def _abort_on_usage(abort: asyncio.Event) -> SessionObserver:
    """An observer that fires the abort on the first cost update, mid-read."""

    def observer(event: SessionEvent) -> None:
        if isinstance(event, UsageUpdateEvent):
            abort.set()

    return observer


@pytest.mark.parametrize(
    ("make_client", "observer_for", "reason"),
    [
        pytest.param(
            lambda: FiniteClient([result_message(total_cost_usd=0.05)]),
            _never_abort,
            "completed",
            id="abort-unused-and-settles",
        ),
        pytest.param(
            # The stream hangs after the cost update, so the abort is what unblocks it:
            # the watch task starts, fires, then teardown cancels it on the settle path.
            lambda: FakeClient([result_message(total_cost_usd=0.1)]),
            _abort_on_usage,
            "interrupted",
            id="abort-fires-mid-read",
        ),
    ],
)
async def test_claude_driver_when_session_settles_does_not_leak_task_diagnostics(
    make_client: Callable[[], FakeClient],
    observer_for: Callable[[asyncio.Event], SessionObserver],
    reason: str,
) -> None:
    records = install_task_leak_recorder()
    client = make_client()
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    abort = asyncio.Event()

    session = driver.start(make_prompt(), observer_for(abort), abort)
    outcome = await asyncio.wait_for(session.outcome, 10)
    del session, driver, client
    gc.collect()

    assert outcome.reason == reason
    assert task_leak_messages(records) == []


# ---------------------------------------------------------------------------
# termination signal between spawn and registration kills the child
# ---------------------------------------------------------------------------


async def test_exec_when_termination_signal_during_spawn_does_still_kill_child_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_exits: list[tuple[int, float]],
) -> None:
    # Widen the spawn-to-register gap so SIGTERM lands inside it.
    real_spawn = asyncio.create_subprocess_shell
    spawned: list[asyncio.subprocess.Process] = []
    spawn_barrier = threading.Event()

    async def slow_spawn(command: str, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_spawn(command, **kwargs)
        spawned.append(proc)
        spawn_barrier.set()
        await asyncio.sleep(1.0)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", slow_spawn)

    uninstall = signals.install_termination_cleanup(exec_mod.kill_live_process_groups)

    def send_signal() -> None:
        spawn_barrier.wait(5.0)
        time.sleep(0.1)
        os.kill(os.getpid(), signal.SIGTERM)

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
        # Recorded before the sweep below, which would otherwise hide a child
        # the termination handler left running.
        survivors = [proc.pid for proc in spawned if proc.returncode is None]
        for proc in spawned:
            if proc.returncode is not None or not proc.pid:
                continue
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGKILL)

    # The signal mask around spawn+register must cover the gap: a SIGTERM
    # landing before registration still finds the child in the live-groups
    # registry and kills it, so exec settles as a normal result, not a timeout.
    # An empty stderr rules out the spawn-failure result, which carries the
    # spawn error's message under the same failure code.
    assert isinstance(result, ExecResult), (
        "child was not killed by the signal handler — spawn-register window is unmasked"
    )
    assert (result.exit_code, result.stderr) == (FAILURE_EXIT_CODE, "")
    assert spawned
    assert survivors == []
    assert all(proc.returncode is not None and proc.returncode < 0 for proc in spawned), (
        f"child was not killed by a signal: {[proc.returncode for proc in spawned]}"
    )
    assert recorded_exits, "the termination handler never asked the process to exit"
