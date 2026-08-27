"""Hardening tests for the asyncio-subprocess edges of ``exec`` and the drivers.

Where :mod:`tests.test_exec` and :mod:`tests.supervisor.test_stdio` pin the happy
paths, these tests pin the ragged edges that only surface with a real child
process misbehaving:

- a subprocess-driver child that emits one oversized, unterminated stdout line
  settles a clean error outcome instead of letting a read-limit overrun escape,
- subprocess-driver teardown returns within a bound even when the child ignores
  the group kill, rather than blocking forever on ``proc.wait()``,
- aborting a run mid-read leaks no "Task ... was never retrieved" / "Task was
  destroyed but it is pending" diagnostics from either driver or ``exec``,
- a grandchild that holds stdio open makes ``exec`` wait indefinitely by design,
  with a timeout as the documented escape that still captures late output.

The module is POSIX-only: the abort paths rely on process-group tree-kill and
the fixtures reap any group a child leaves behind so nothing is orphaned. Every
real-tree test roots its scratch files under ``tmp_path``, so the suite stays
order-independent under ``pytest-xdist`` and ``pytest-randomly``.
"""

import asyncio
import contextlib
import gc
import json
import os
import signal
import sys
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gymrat_py.exec import ExecOptions, ExecResult, ExecTimeoutError
from gymrat_py.exec import exec as run_exec
from gymrat_py.supervisor import create_claude_driver, create_stdio_driver
from gymrat_py.supervisor.claude import ClaudeClient, ClientFactory
from gymrat_py.supervisor.events import SessionEvent, UsageUpdateEvent
from tests.supervisor._fixtures import collecting_observer, make_prompt

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only process groups for tree-kill"
)

_DOUBLE = str(Path(__file__).resolve().parents[1] / "supervisor" / "_stdio_double.py")

# The two asyncio diagnostics that mark a task the code forgot to await or
# retrieve; both route through the loop's exception handler.
_LEAK_MARKERS = ("was destroyed but it is pending", "exception was never retrieved")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def try_read_report(report_path: Path) -> dict[str, Any] | None:
    """Load the JSON report if it exists and is complete, else ``None``.

    Wrapped in a sync helper so the blocking filesystem read stays out of the
    async test body, where it would trip the async-blocking-call lint.
    """
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


async def read_report(report_path: Path, timeout_s: float = 5.0) -> dict[str, Any]:
    """Poll until ``report_path`` holds a complete JSON report, then return it."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        data = try_read_report(report_path)
        if data is not None:
            return data
        if loop.time() > deadline:
            message = f"report never appeared at {report_path}"
            raise TimeoutError(message)
        await asyncio.sleep(0.02)


def file_exists(path: Path) -> bool:
    """Whether ``path`` exists (sync helper to keep the stat out of async)."""
    return path.exists()


async def wait_for_file(path: Path, timeout_s: float = 5.0) -> None:
    """Poll until ``path`` exists."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not file_exists(path):
        if loop.time() > deadline:
            message = f"file never appeared at {path}"
            raise TimeoutError(message)
        await asyncio.sleep(0.02)


def make_fifo(path: Path) -> None:
    """Create a named pipe (sync helper to keep the blocking call out of async)."""
    os.mkfifo(path)


def release_fifo(path: Path) -> None:
    """Unblock a reader waiting on ``path`` by opening it for writing and sending a line.

    Opening the write end returns as soon as the child's read end is open, which
    it already is, so this never blocks. Sync so the async test body stays clean.
    """
    fd = os.open(str(path), os.O_WRONLY)
    try:
        os.write(fd, b"\n")
    finally:
        os.close(fd)


def install_task_leak_recorder() -> list[dict[str, object]]:
    """Route the running loop's exception handler into a list and return it.

    Both "Task was destroyed but it is pending!" and "Task exception was never
    retrieved" are reported through ``loop.call_exception_handler`` when the
    offending task is finalized, so recording every context the handler sees —
    then forcing a collection — captures a forgotten task deterministically.
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


class ScriptedClaudeClient:
    """A minimal stand-in for the SDK streaming client the Claude driver drives.

    It replays ``messages`` then, when ``hang`` is set, blocks until ``disconnect``
    releases it, so an abort has something to tear down mid-stream.
    """

    def __init__(self, messages: list[object], *, hang: bool = False) -> None:
        self.messages = messages
        self.hang = hang
        self.disconnect_count = 0
        self._released = asyncio.Event()

    async def connect(self) -> None:
        return None

    async def query(self, prompt: str) -> None:
        return None

    async def receive_messages(self) -> AsyncIterator[object]:
        for message in self.messages:
            await asyncio.sleep(0)
            yield message
        if self.hang:
            await self._released.wait()

    async def interrupt(self) -> None:
        return None

    async def disconnect(self) -> None:
        self.disconnect_count += 1
        self._released.set()


def claude_factory(client: ScriptedClaudeClient) -> ClientFactory:
    """A client factory that hands the Claude driver ``client`` for any options."""

    def factory(_options: Mapping[str, object]) -> ClaudeClient:
        return client

    return factory


def ignore_kill(*_args: object, **_kwargs: object) -> None:
    """Stand in for the group kill so a child outlives teardown."""


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


async def reap_children(processes: list[asyncio.subprocess.Process]) -> None:
    """Kill any survivor's group and reap it within the loop.

    Reaping with ``proc.wait()`` while the loop still runs lets asyncio finalize
    the child's transport in-loop, so no orphaned transport lingers for a later
    test's forced garbage collection to finalize against a closed loop (which
    would surface as a warning about an exception that cannot propagate). The kill uses the real
    ``os.killpg`` directly, so a test that neutralized the driver's own kill
    still gets its child torn down here.
    """
    for proc in processes:
        if proc.returncode is None and proc.pid:
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(TimeoutError, ProcessLookupError):
            await asyncio.wait_for(proc.wait(), 5)


@pytest.fixture
async def stdio_children(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[list[asyncio.subprocess.Process]]:
    """Record every child the stdio driver spawns and reap any survivor in-loop.

    The driver spawns through ``asyncio.create_subprocess_exec``; wrapping that
    attribute captures the real ``Process`` while leaving the spawn real, so a
    child left running by a deliberately hobbled kill never outlives the test.
    """
    processes: list[asyncio.subprocess.Process] = []
    real = asyncio.create_subprocess_exec

    async def wrapper(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        proc = await real(*args, **kwargs)  # type: ignore[arg-type]
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", wrapper)
    yield processes

    await reap_children(processes)


@pytest.fixture
async def exec_children(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[list[asyncio.subprocess.Process]]:
    """Record every child ``exec`` spawns and reap any survivor in-loop on teardown."""
    processes: list[asyncio.subprocess.Process] = []
    real = asyncio.create_subprocess_shell

    async def wrapper(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        proc = await real(*args, **kwargs)  # type: ignore[arg-type]
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", wrapper)
    yield processes

    await reap_children(processes)


# ---------------------------------------------------------------------------
# an oversized unterminated line settles error, not an escaping overrun
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_child_emits_oversized_unterminated_line_does_settle_error(
    tmp_path: Path,
    stdio_children: list[asyncio.subprocess.Process],
) -> None:
    # Well past any sane per-line read limit, with no newline, so the reader's
    # limit overruns instead of ever yielding a line.
    oversized_bytes = 12_000_000
    argv = [
        sys.executable,
        "-c",
        f"import sys; sys.stdout.write('x' * {oversized_bytes})",
    ]
    session = create_stdio_driver(argv).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )

    outcome = await asyncio.wait_for(session.outcome, 10)

    assert outcome.reason == "error"
    assert outcome.message


# ---------------------------------------------------------------------------
# teardown is bounded even when the child ignores the group kill
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_child_survives_kill_does_bound_teardown_and_settle(
    tmp_path: Path,
    stdio_children: list[asyncio.subprocess.Process],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Neutralize the group kill so the child outlives teardown, standing in for a
    # process stuck in an uninterruptible state that a real kill cannot reap.
    monkeypatch.setattr("gymrat_py.supervisor.stdio.kill_process_group", ignore_kill)
    # The child sends its terminal outcome line, then lingers instead of exiting.
    program = (
        "import json, sys, time\n"
        "sys.stdout.write(json.dumps("
        "{'type': 'outcome', 'reason': 'completed', 'costUsd': 0.0}) + '\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(120)\n"
    )
    argv = [sys.executable, "-c", program]
    session = create_stdio_driver(argv).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )

    with pytest.warns(RuntimeWarning, match="did not exit"):
        outcome = await asyncio.wait_for(session.outcome, 10)

    assert outcome.reason == "completed"


# ---------------------------------------------------------------------------
# aborting mid-read leaks no forgotten-task diagnostics
# ---------------------------------------------------------------------------


async def test_exec_when_aborted_mid_read_does_not_leak_task_diagnostics(
    tmp_path: Path,
    exec_children: list[asyncio.subprocess.Process],
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
    assert exec_children  # the abort ran through a real spawned child


async def test_claude_driver_when_abort_unused_and_settles_does_not_leak_task_diagnostics() -> None:
    records = install_task_leak_recorder()
    client = ScriptedClaudeClient([SimpleNamespace(total_cost_usd=0.05)])
    driver = create_claude_driver(client_factory=claude_factory(client))
    abort = asyncio.Event()  # provided but never fired: the watch task must be cleaned up

    session = driver.start(make_prompt(), collecting_observer().observer, abort)
    outcome = await asyncio.wait_for(session.outcome, 10)
    del session, driver, client
    gc.collect()

    assert outcome.reason == "completed"
    assert task_leak_messages(records) == []


async def test_claude_driver_when_abort_fires_mid_read_does_not_leak_task_diagnostics() -> None:
    records = install_task_leak_recorder()
    # The stream hangs after the cost update, so the abort is what unblocks it:
    # the watch task starts, fires, then teardown cancels it on the settle path.
    client = ScriptedClaudeClient([SimpleNamespace(total_cost_usd=0.1)], hang=True)
    driver = create_claude_driver(client_factory=claude_factory(client))
    abort = asyncio.Event()

    def observer(event: SessionEvent) -> None:
        if isinstance(event, UsageUpdateEvent):
            abort.set()

    session = driver.start(make_prompt(), observer, abort)
    outcome = await asyncio.wait_for(session.outcome, 10)
    del session, driver, client
    gc.collect()

    assert outcome.reason == "interrupted"
    assert task_leak_messages(records) == []


async def test_stdio_driver_when_aborted_mid_read_does_not_leak_task_diagnostics(
    tmp_path: Path,
    stdio_children: list[asyncio.subprocess.Process],
) -> None:
    records = install_task_leak_recorder()
    report = tmp_path / "child-processes.json"
    config = {
        "mode": "sleep_forever",
        "report_path": str(report),
        "lines": [{"json": {"type": "usage_update", "timestamp": 1, "costUsd": 0.4}}],
    }
    argv = [sys.executable, _DOUBLE, json.dumps(config)]
    abort = asyncio.Event()
    session = create_stdio_driver(argv).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer, abort
    )
    await read_report(report)

    abort.set()
    outcome = await asyncio.wait_for(session.outcome, 10)
    # No forced collection here: the driver awaits its cancelled reader and abort
    # tasks in teardown, so nothing is left waiting to be retrieved, and forcing a collection
    # would only finalize the child's subprocess transport as loop noise.
    await asyncio.sleep(0)

    assert outcome.reason == "interrupted"
    assert task_leak_messages(records) == []


# ---------------------------------------------------------------------------
# a grandchild holding stdio open makes exec wait indefinitely by design
# ---------------------------------------------------------------------------


async def test_exec_when_grandchild_holds_stdout_open_with_timeout_does_capture_late_output(
    tmp_path: Path,
    exec_children: list[asyncio.subprocess.Process],
) -> None:
    # The child shell writes its metric late, then holds the pipe open past the
    # timeout; the escape hatch fires while the late output is already captured.
    command = "( sleep 0.3; echo METRIC; sleep 30 ) &"

    result = await run_exec(command, ExecOptions(cwd=str(tmp_path), timeout_ms=2000))

    assert isinstance(result, ExecTimeoutError)
    assert result.timeout_ms == 2000
    assert "METRIC" in result.stdout


async def test_exec_when_grandchild_holds_stdout_open_without_timeout_does_wait_until_released(
    tmp_path: Path,
    exec_children: list[asyncio.subprocess.Process],
) -> None:
    fifo = tmp_path / "release.fifo"
    make_fifo(fifo)
    # The background child shell inherits the stdout pipe and blocks opening the
    # fifo, so exec never sees EOF until the fifo is released.
    command = f"( read line < '{fifo}'; echo METRIC ) &"
    task = asyncio.create_task(run_exec(command, ExecOptions(cwd=str(tmp_path))))
    await asyncio.sleep(0.5)

    open_ended = not task.done()
    release_fifo(fifo)
    result = await asyncio.wait_for(task, 10)

    assert open_ended, "exec returned before the grandchild released stdout"
    assert isinstance(result, ExecResult)
    assert "METRIC" in result.stdout


# ---------------------------------------------------------------------------
# termination signal between spawn and registration kills the child
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(signal, "pthread_sigmask"),
    reason="Signal masking requires POSIX pthread_sigmask",
)
async def test_exec_when_termination_signal_during_spawn_does_still_kill_child_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    import time

    from gymrat_py import exec as exec_mod
    from gymrat_py import signals

    # Widen the spawn-to-register gap so SIGTERM lands inside it.
    real_spawn = asyncio.create_subprocess_shell
    spawned: list[asyncio.subprocess.Process] = []
    spawn_barrier = threading.Event()

    async def slow_spawn(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        proc = await real_spawn(*args, **kwargs)  # type: ignore[arg-type]
        spawned.append(proc)
        spawn_barrier.set()
        await asyncio.sleep(1.0)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", slow_spawn)

    # Prevent the handler from actually terminating the process.
    exit_record: dict[str, object] = {}
    monkeypatch.setattr(signals, "_exit_process", lambda code: exit_record.update(code=code))  # pyrefly: ignore

    # Keep the live-groups registry clean for this test.
    saved = set(exec_mod._live_process_groups)
    exec_mod._live_process_groups.clear()

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
        sender.join(timeout=3)
        uninstall()
        exec_mod._live_process_groups.clear()
        exec_mod._live_process_groups.update(saved)
        for proc in spawned:
            if proc.returncode is None and proc.pid:
                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signal.SIGKILL)

    # With the fix (signal mask around spawn+register), the deferred handler
    # finds the child in the registry and kills it — exec settles quickly as
    # a normal result, not a timeout. Without the fix the handler fires before
    # registration and the child survives until the 8 s timeout.
    assert not isinstance(result, ExecTimeoutError), (
        "child was not killed by the signal handler — spawn-register window is unmasked"
    )
    assert "code" in exit_record
