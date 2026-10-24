"""Behavioral tests for the teardown of a run, shared by ``exec`` and ``exec_argv``.

A run that times out, is aborted, hits a stream error, or is cancelled is torn
down the same way whichever entry point started it: a graceful stop request,
then a kill of the whole process group, then the reap. These tests pin that
teardown both while the child is running and once the shell has exited but
asyncio has not reaped it yet, including the warnings it must not raise and
the registry entry it must keep until the group is killed.

Python children (``exec_argv`` running ``python -c``) stand in wherever a test
needs a child that installs its own signal handler as the group leader.

Real-subprocess tests are POSIX-only: process groups, session leaders, and
``os.killpg`` do not exist on win32.
"""

import asyncio
import contextlib
import dataclasses
import functools
import os
import shlex
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecOptions, ExecTimeoutError, exec_argv
from gymrat.exec import exec as run_exec
from tests._exec_fixtures import (
    RUNNERS,
    ExecRun,
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
    wait_for_spawned,
)
from tests._process_helpers import (
    SLEEPER_ARGV,
    ZOMBIE_ONLY_GROUP_SCRIPT,
    is_alive,
    is_zombie,
    killpg_warnings,
    poll_until,
    refuse_killpg,
    wait_for_file,
    wait_for_pid_file,
    wait_until_dead,
    wait_until_dead_blocking,
)

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
    stdout = proc.stdout
    assert stdout is not None
    await poll_until(
        stdout.at_eof,
        timeout_s,
        lambda: TimeoutError(f"shell {proc.pid} never closed its stdout"),
    )


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
    """Stand-in for the ``os`` module asyncio's child watcher uses, holding back its reap.

    Every attribute but ``waitpid``, ``waitid`` and ``pidfd_open`` forwards to the
    real ``os`` module. The reap waits until ``release`` is set or ``hold_s``
    passes, wherever the host's child watcher would start it:

    - **Through a pidfd:** the watcher reaps once the pidfd turns readable, so
      ``pidfd_open`` hands it the read end of a pipe that turns readable only
      when the hold ends. The reap that follows runs on the event loop and is
      never held there.
    - **Through a waitpid thread:** a blocking ``waitpid(pid, 0)`` or
      ``waitid(P_PID, …)`` on that thread waits out the hold. CPython 3.14.7+
      changed ``_ThreadedChildWatcher._do_waitpid`` to call
      ``os.waitid(P_PID, pid, WEXITED | WNOWAIT)`` before scheduling the actual
      reap on the event-loop thread, so ``waitid`` is held too.

    Each hold is recorded under the pid it holds back for as long as it runs, on
    either path; :meth:`is_holding` reads that record. Holds of different
    processes overlap, so each is tracked on its own.
    """

    release: threading.Event
    hold_s: float
    via_pidfd: bool

    def __post_init__(self) -> None:
        self._holding_lock = threading.Lock()
        self._holding: set[int] = set()

    def __getattr__(self, name: str) -> object:
        return getattr(os, name)

    def is_holding(self, pid: int) -> bool:
        """Whether the reap of ``pid`` is being held back right now."""
        with self._holding_lock:
            return pid in self._holding

    def _hold(self, pid: int) -> None:
        with self._holding_lock:
            self._holding.add(pid)
        try:
            self.release.wait(self.hold_s)
        finally:
            with self._holding_lock:
                self._holding.discard(pid)

    def pidfd_open(self, pid: int, flags: int = 0, /) -> int:
        """Return a stand-in pidfd that turns readable once the hold ends.

        On a host without pidfd support the call is refused, as the real one
        would be, so asyncio's pidfd probe run under the stand-in keeps choosing
        the waitpid thread.

        Args:
            pid: The child whose reap is held back.
            flags: Ignored, as the stand-in pidfd takes none.

        Returns:
            The read end of a pipe that turns readable once the hold ends.

        Raises:
            OSError: When the host reaps through a waitpid thread.
        """
        del flags
        if not self.via_pidfd:
            message = "pidfd_open is unavailable on this host"
            raise OSError(message)
        readable, writable = os.pipe()

        def signal_after_hold() -> None:
            self._hold(pid)
            with contextlib.suppress(OSError):
                os.write(writable, b"x")
            os.close(writable)

        threading.Thread(target=signal_after_hold, daemon=True).start()
        return readable

    def waitpid(self, pid: int, options: int, /) -> tuple[int, int]:
        """Hold a blocking ``waitpid(pid, 0)`` on the waitpid thread until the hold ends."""
        if options == 0 and not self.via_pidfd:
            self._hold(pid)
        return os.waitpid(pid, options)

    if _os_waitid is not None:
        _P_PID: ClassVar[int] = getattr(os, "P_PID", 0)

        def waitid(
            self,
            id_type: int,
            id_val: int,
            options: int,
            /,
        ) -> object:
            """Hold a blocking ``waitid`` the same way ``waitpid`` is held."""
            if id_type == self._P_PID and not self.via_pidfd:
                self._hold(id_val)
            assert _os_waitid is not None
            return _os_waitid(id_type, id_val, options)


@pytest.fixture
def held_reaper(monkeypatch: pytest.MonkeyPatch) -> Iterator[HeldReaper]:
    """Hold asyncio's child reap back so an exited shell stays a zombie through exec's teardown."""
    reaper = HeldReaper(release=threading.Event(), hold_s=_REAP_HOLD_S, via_pidfd=pidfd_available())
    monkeypatch.setattr("asyncio.unix_events.os", reaper)
    yield reaper
    reaper.release.set()


# Whether a shell's reap was still held back, and whether the shell had exited
# without anyone reaping it, at one signal exec sent the shell's group.
ReapSnapshot = tuple[bool, bool]


@pytest.fixture
def reap_at_group_signals(
    monkeypatch: pytest.MonkeyPatch, held_reaper: HeldReaper
) -> Callable[[asyncio.subprocess.Process], list[ReapSnapshot]]:
    """Return a switch that snapshots a shell's held reap at each signal exec sends its group."""

    def watch(proc: asyncio.subprocess.Process) -> list[ReapSnapshot]:
        snapshots: list[ReapSnapshot] = []
        real_killpg = os.killpg

        def snapshot_then_signal(group_pid: int, signal_number: int) -> None:
            if group_pid == proc.pid and signal_number != 0:
                snapshots.append((held_reaper.is_holding(proc.pid), is_zombie(proc.pid)))
            real_killpg(group_pid, signal_number)

        monkeypatch.setattr(os, "killpg", snapshot_then_signal)
        return snapshots

    return watch


def script_writing_pid_and_sleeping(pid_file: Path) -> str:
    """Build a script that writes its own pid to ``pid_file``, then sleeps 30s."""
    return (
        "import os, time; "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()) + '\\n'); "
        "time.sleep(30)"
    )


def script_escaping_the_group(pid_file: Path) -> str:
    """Build a script that leaves its process group, writes its pid to ``pid_file``, then sleeps 30s.

    Run in the background of the shell, it keeps the inherited stdio pipes open
    after the group is killed, as a descendant that escaped the group would.

    Args:
        pid_file: Where the script writes its own pid, once it has left the group.

    Returns:
        The script's source, ready for ``python -c``.
    """
    return (
        "import os, time; "
        "os.setsid(); "
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


def script_exiting_on_second_request(pid_file: Path, asked_file: Path) -> str:
    """Build a script that creates ``asked_file`` on the first SIGTERM and exits on the second.

    Args:
        pid_file: Where the script writes its own pid, once its handler is in place.
        asked_file: The file the first SIGTERM creates.

    Returns:
        The script's source, ready for ``python -c``.
    """
    return (
        "import os, signal, sys, time\n"
        "asked = []\n"
        "def on_request(_signal_number, _frame):\n"
        "    if asked:\n"
        "        sys.exit(0)\n"
        "    asked.append(True)\n"
        f"    open({str(asked_file)!r}, 'w').close()\n"
        "signal.signal(signal.SIGTERM, on_request)\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()) + '\\n')\n"
        "time.sleep(60)\n"
    )


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
        pytest.param(Teardown(_GRANDCHILD_TIMEOUT_MS, leave_to_timeout), id="timeout"),
        pytest.param(Teardown(None, set_abort), id="abort"),
    ],
)
async def test_exec_when_torn_down_with_a_descendant_holding_the_pipes_does_close_them(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    stray_process_ids: list[int],
    make_opts: Callable[..., ExecOptions],
    teardown: Teardown,
) -> None:
    # The escaped descendant survives the group kill with stdout and stderr
    # still open, so the readers reach EOF only if the teardown drops the pipes.
    escaped_pid_file = tmp_path / "escaped.pid"
    escaping = shlex.quote(script_escaping_the_group(escaped_pid_file))
    abort = asyncio.Event()
    task = asyncio.create_task(
        run_exec(
            f"{shlex.quote(sys.executable)} -c {escaping} & sleep 30",
            make_opts(timeout_ms=teardown.timeout_ms, abort=abort),
        )
    )
    proc = await wait_for_spawned(spawned_processes)
    stray_process_ids.append(await wait_for_pid_file(escaped_pid_file))

    teardown.trigger(task, proc, abort)
    await asyncio.wait_for(task, 5)

    assert proc.stdout is not None
    assert proc.stderr is not None
    assert proc.stdout.at_eof()
    assert proc.stderr.at_eof()


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
async def test_exec_when_torn_down_after_shell_exits_does_kill_descendants_leaving_the_reap_to_asyncio(
    tmp_path: Path,
    reap_at_group_signals: Callable[[asyncio.subprocess.Process], list[ReapSnapshot]],
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    *,
    caplog: pytest.LogCaptureFixture,
    teardown: Teardown,
) -> None:
    # The background sleep holds stderr open, so exec is still running when the
    # shell exits and the teardown lands on a shell nobody has reaped yet.
    abort = asyncio.Event()
    task = asyncio.create_task(
        run_exec(
            "sleep 30 >/dev/null & echo $! > grandchild.pid; exit 0",
            make_opts(timeout_ms=teardown.timeout_ms, abort=abort),
        ),
    )
    proc = await wait_for_spawned(spawned_processes)
    snapshots = reap_at_group_signals(proc)
    grandchild = await wait_for_pid_file(tmp_path / "grandchild.pid")
    await wait_for_shell_exit(proc)

    teardown.trigger(task, proc, abort)
    await settle(task)
    await asyncio.wait_for(proc.wait(), 5)

    assert snapshots[:1] == [(True, True)]
    await wait_until_dead(grandchild, timeout_s=3.0)
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
    ],
)
async def test_exec_when_torn_down_after_shell_exits_does_not_warn_about_killpg(
    reap_at_group_signals: Callable[[asyncio.subprocess.Process], list[ReapSnapshot]],
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
    snapshots = reap_at_group_signals(proc)
    await wait_for_shell_exit(proc)

    teardown.trigger(task, proc, abort)
    await settle(task)
    await asyncio.wait_for(proc.wait(), 5)

    assert snapshots[:1] == [(True, True)]
    assert killpg_warnings(recwarn) == []


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
    reap_after_teardown = (held_reaper.is_holding(proc.pid), proc.returncode)
    # Let the held reap finish inside the test, while its loop is still open.
    held_reaper.release.set()
    await asyncio.wait_for(proc.wait(), 5)

    assert reap_after_teardown == (True, None)
    assert done == {task}
    assert task.cancelled()
    # The shell not yet reaped is the group's only member, a zombie with nothing
    # left to stop, so the group refusing the kill is not worth a warning.
    assert killpg_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# a descendant left behind once the shell has exited
# ---------------------------------------------------------------------------


def shell_leaving_a_descendant(pid_file: Path, *, detached: bool = False) -> str:
    """Build a shell command that backgrounds a ``sleep``, writes its pid to ``pid_file``, and exits.

    Args:
        pid_file: Where the ``sleep`` writes its pid.
        detached: Send all three of the ``sleep``'s streams to ``/dev/null``;
            otherwise it holds the shell's stderr open.

    Returns:
        The shell command line.
    """
    redirects = "</dev/null >/dev/null 2>&1" if detached else ">/dev/null"
    return f"sleep 30 {redirects} & echo $! > {shlex.quote(str(pid_file))}; exit 0"


def argv_leaving_a_descendant(pid_file: Path, *, detached: bool = False) -> tuple[str, ...]:
    """Build an argv whose child starts a sleeper, writes its pid to ``pid_file``, and exits.

    Args:
        pid_file: Where the child writes the sleeper's pid.
        detached: Send all three of the sleeper's streams to ``DEVNULL``;
            otherwise it holds the child's stderr open.

    Returns:
        The argv to run.
    """
    streams = (
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
        if detached
        else "stdout=subprocess.DEVNULL"
    )
    script = (
        "import subprocess\n"
        f"p = subprocess.Popen({list(SLEEPER_ARGV)!r}, {streams})\n"
        f"open({str(pid_file)!r}, 'w').write(str(p.pid) + '\\n')\n"
    )
    return (sys.executable, "-c", script)


@pytest.mark.parametrize(
    ("run", "leaving_a_descendant"),
    [
        pytest.param(run_shell, shell_leaving_a_descendant, id="exec"),
        pytest.param(run_argv, argv_leaving_a_descendant, id="exec_argv"),
    ],
)
async def test_exec_when_cancelled_after_shell_is_reaped_does_kill_the_descendant_holding_its_pipes(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    stray_process_ids: list[int],
    make_opts: Callable[..., ExecOptions],
    *,
    run: ExecRun,
    leaving_a_descendant: Callable[[Path], str | tuple[str, ...]],
) -> None:
    # The descendant holds stderr open, so the run is still reading when the
    # shell exits and asyncio's own child watcher collects its exit status.
    pid_file = tmp_path / "descendant.pid"
    task = asyncio.create_task(run(leaving_a_descendant(pid_file), make_opts()))
    proc = await wait_for_spawned(spawned_processes)
    descendant = await wait_for_pid_file(pid_file)
    stray_process_ids.append(descendant)
    await poll_until(
        lambda: proc.returncode is not None,
        3.0,
        lambda: TimeoutError(f"shell {proc.pid} was never reaped"),
    )

    await cancel_and_settle(task)

    # Blocking, so the loop runs nothing more: the kill must already be sent.
    wait_until_dead_blocking(descendant, timeout_s=3.0)


@pytest.mark.parametrize(
    ("run", "detaching_a_descendant"),
    [
        pytest.param(
            run_shell, functools.partial(shell_leaving_a_descendant, detached=True), id="exec"
        ),
        pytest.param(
            run_argv, functools.partial(argv_leaving_a_descendant, detached=True), id="exec_argv"
        ),
    ],
)
async def test_exec_when_completed_with_a_detached_descendant_does_not_signal_its_group(
    tmp_path: Path,
    stray_process_ids: list[int],
    make_opts: Callable[..., ExecOptions],
    record_killpg: Callable[[], list[int]],
    *,
    run: ExecRun,
    detaching_a_descendant: Callable[[Path], str | tuple[str, ...]],
) -> None:
    # The descendant holds none of the run's pipes, so the run completes on its
    # own once the shell exits, with the descendant still in the shell's group.
    pid_file = tmp_path / "descendant.pid"
    signalled = record_killpg()

    await run(detaching_a_descendant(pid_file), make_opts())
    stray_process_ids.append(await wait_for_pid_file(pid_file))

    assert signalled == []


# ---------------------------------------------------------------------------
# the graceful request that precedes the kill
# ---------------------------------------------------------------------------


async def test_exec_argv_when_aborted_child_writes_stderr_on_request_does_settle_failed_with_it(
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
    assert result == expected_result("", "BYE\n", 1)


async def test_exec_argv_when_graceful_signal_refused_does_kill_group_without_warning(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    # macOS answers EPERM for a group that is on its way out, which cannot be
    # told apart from a genuine refusal until the leader is reaped; the stop
    # request earns the same deferral the kill already gets.
    refusal = refuse_killpg(monkeypatch, lambda signal_number: signal_number == signal.SIGTERM)
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
    assert signal.SIGTERM in refusal.signals
    assert result == expected_result("", "", 1)
    assert killpg_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# cancellation keeps the pid registered until its group is killed
# ---------------------------------------------------------------------------


# Long enough that only the live-group sweep, never the run's own teardown, can
# end a child within the test.
_LONG_GRACE_S = 30.0


async def test_kill_live_process_groups_when_cancelled_exec_argv_run_is_mid_grace_does_kill_its_child(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A pid dropped from the registry before its group is killed is a pid the
    # interpreter-exit sweep can no longer reach. The child marks the first stop
    # request, which is the cancelled run's teardown starting its grace, and
    # exits on the second, which only the sweep sends.
    monkeypatch.setattr(exec_mod, "TERMINATE_GRACE_S", _LONG_GRACE_S)
    pid_file = tmp_path / "child.pid"
    asked_file = tmp_path / "asked.marker"
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, "-c", script_exiting_on_second_request(pid_file, asked_file)],
            make_opts(),
        ),
    )
    proc = await wait_for_spawned(spawned_processes, spawner="exec_argv")
    await wait_for_pid_file(pid_file)
    task.cancel()
    await wait_for_file(asked_file)

    exec_mod.kill_live_process_groups()

    await wait_until_dead(proc.pid, timeout_s=3.0)
    await settle(task)
    assert task.cancelled()


# ---------------------------------------------------------------------------
# cancellation with only a zombie left in the group
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
