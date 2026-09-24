"""Tree-teardown hardening: a bench in its own session dies with the run that started it.

Every test drives the production chain out of process. A tool child runs
``run_with_signal_abort`` and starts its bench through ``exec_argv``, exactly as
a gymrat run does, so the bench lands in its own POSIX session — or its own
Windows job — instead of in the tool child's process group. Tearing the outer
run down has to reach it anyway, whether the teardown comes from an abort, a
timeout, a cancellation, or a signal to the supervisor. The supervisor's stdio
session child is contained the same way, so its descendants die with the session.

Liveness is read from heartbeat files rather than process IDs: ``os.kill(pid, 0)``
terminates the target on Windows, so a pid probe cannot be shared across
platforms. A process that has stopped rewriting its heartbeat file has stopped
running.

The module carries no platform skip. The few sub-cases that can only hold on
one platform are gated one by one.
"""

import asyncio
import contextlib
import ctypes
import dataclasses
import importlib.util
import os
import signal
import subprocess
import sys
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from gymrat import exec as gymrat_exec
from gymrat import process_group
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError, exec_argv
from gymrat.supervisor import create_stdio_driver
from gymrat.supervisor.driver import SessionOutcome
from tests._process_helpers import (
    SLEEPER_ARGV,
    capture_spawns,
    refuse_resume,
    wait_for_pid_file_blocking,
    wait_until_dead_blocking,
)
from tests.supervisor._fixtures import collecting_observer, make_prompt

# How often a heartbeat script rewrites its file.
_BEAT_PERIOD_S = 0.05

# How long a frozen heartbeat is watched before it counts as stopped: many
# beats wide, so a process that is merely slow still moves within it.
_STOPPED_WINDOW_S = 0.75

# Upper bound for the first heartbeat of a tree that has to start two
# interpreters and import gymrat on the way.
_BEAT_TIMEOUT_S = 20.0

# Timeout given to the outer run of the cross-platform teardown test. It fires
# well after the bench is beating, so the timeout case always lands mid-bench.
_OUTER_TIMEOUT_MS = 5000

# Upper bound for a torn-down outer run to settle, including its grace.
_SETTLE_TIMEOUT_S = 15.0

# A child that acts on the graceful request settles at its own exit speed; a
# fixed sleep for the grace would blow through this.
_CHILD_EXIT_SETTLE_S = 0.5

# A child that ignores the graceful request is killed once the grace elapses.
# The grace is at most a second, and killing plus reaping a sleeping child is
# milliseconds, so a longer grace cannot fit inside this bound.
_GRACE_BOUND_S = 1.5

# Cancellation additionally waits for the reap, bounded at two seconds in exec.
_CANCEL_BOUND_S = 2.5

_SUPERVISOR_EXIT_TIMEOUT_S = 30.0

# How long the late-containment test holds the job assignment back. A venv
# launcher starts its real interpreter within tens of milliseconds, so a full
# second guarantees the interpreter exists before the launcher joins the job.
_LATE_ATTACH_DELAY_S = 1.0

# The job tests fake ``sys.platform``, so a check against the host this suite is
# really running on has to read the platform recorded before any faking.
_HOST_IS_WINDOWS = sys.platform == "win32"

# Stand-in win32 handles and pid for the job tests, which never touch a real
# process: distinct values so a close can be told apart from a job close.
_JOB_HANDLE = 777
_PROCESS_HANDLE = 4242
_CHILD_PID = 4321

# Grace given to a faked job that never empties, kept far below the real one so
# the bound is reached in milliseconds.
_FAKE_JOB_GRACE_S = 0.05

# Hard cap on how often a faked job may be polled for its live process count.
# The wait sleeps between polls, so any grace admits a poll count of roughly
# grace / poll period — a handful under ``_FAKE_JOB_GRACE_S``, and fewer still
# on a slow machine. Only a wait that stopped honouring its deadline reaches
# this, and it turns that runaway loop into a failure instead of a hung suite.
_FAKE_JOB_QUERY_CAP = 200

# ``ctypes.WinError`` is bound only on Windows, but the production refusal paths
# format it into their warning, so the faked platform hands them a stand-in.
_FAKE_WIN_ERROR = "the host refused the call"

# Win32 ABI values, spelled out here rather than read back from the module under
# test: a build that set the wrong ones has to fail this, not agree with itself.
# ``JobObjectExtendedLimitInformation`` and ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``.
_WIN32_EXTENDED_LIMIT_INFORMATION = 9
_WIN32_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

# ``CREATE_SUSPENDED``: the child exists but runs no instruction until resumed.
_WIN32_CREATE_SUSPENDED = 0x4

# NTSTATUS codes the faked ``NtResumeProcess`` answers with: ``STATUS_SUCCESS``
# and the ``STATUS_ACCESS_DENIED`` a locked-down host replies with.
_NT_STATUS_SUCCESS = 0
_NT_STATUS_ACCESS_DENIED = 0xC0000022

_BEAT = '''"""Rewrite the file named in argv[1] with this pid and a rising counter."""

import os
import sys
import time
from pathlib import Path

PERIOD_S = 0.05


def beat_forever(path: str) -> None:
    """Rewrite ``path`` with ``"<pid> <counter>"`` until this process is killed."""
    target = Path(path)
    pid = os.getpid()
    count = 0
    while True:
        count += 1
        target.write_text(f"{pid} {count}", encoding="utf-8")
        time.sleep(PERIOD_S)


if __name__ == "__main__":
    beat_forever(sys.argv[1])
'''

_BENCH = '''"""Spawn a heartbeat grandchild, then heartbeat forever.

argv: ``<own-beat-file> <grandchild-beat-file>``
"""

import subprocess
import sys
from pathlib import Path

from beat import beat_forever

if __name__ == "__main__":
    here = Path(__file__).parent
    subprocess.Popen([sys.executable, str(here / "beat.py"), sys.argv[2]])
    beat_forever(sys.argv[1])
'''

_TOOL = '''"""Record this pid in ``argv[1]``, then run ``argv[2:]`` as a bench.

``run_with_signal_abort`` wires the termination cleanup and hands an abort
event to the body, and the bench itself is started with ``exec_argv`` — the
same two steps a gymrat run takes around every sample.
"""

import asyncio
import os
import sys
from pathlib import Path

from gymrat.cli.shared import run_with_signal_abort
from gymrat.exec import ExecOptions, exec_argv


async def main() -> None:
    """Record this pid, then start the bench named in ``argv[2:]`` and wait for it."""
    Path(sys.argv[1]).write_text(f"{os.getpid()}\\n", encoding="utf-8")
    argv = sys.argv[2:]
    cwd = str(Path(__file__).parent)

    async def execute(abort: asyncio.Event) -> object:
        return await exec_argv(argv, ExecOptions(cwd=cwd, abort=abort))

    await run_with_signal_abort(execute)


if __name__ == "__main__":
    asyncio.run(main())
'''

_IGNORE = '''"""Ignore the graceful termination signal, then heartbeat forever."""

import signal
import sys

from beat import beat_forever

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    beat_forever(sys.argv[1])
'''

_ORPHAN = '''"""Spawn a heartbeat grandchild, then exit at once, orphaning it."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).parent
    subprocess.Popen([sys.executable, str(here / "beat.py"), sys.argv[1]])
'''

_SCRIPTS = {"beat": _BEAT, "bench": _BENCH, "tool": _TOOL, "ignore": _IGNORE, "orphan": _ORPHAN}

type ExecTask = asyncio.Task[ExecResult | ExecTimeoutError]


def read_beat(path: Path) -> str | None:
    """Read a heartbeat file, or ``None`` while it is absent or unreadable."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        # Windows refuses the read while the writer holds the file; that only
        # happens while the writer is alive, which reads as "still beating".
        return None


def beat_pid(path: Path) -> int | None:
    """The pid recorded in a heartbeat file, or ``None`` when none is readable yet."""
    raw = read_beat(path)
    first = raw.split()[0] if raw else ""
    return int(first) if first.isdigit() else None


def wait_for_beat(path: Path, timeout_s: float = _BEAT_TIMEOUT_S) -> str:
    """Poll ``path`` until a heartbeat lands there, then return it."""
    deadline = time.monotonic() + timeout_s
    while True:
        beat = read_beat(path)
        if beat:
            return beat
        if time.monotonic() > deadline:
            message = f"no heartbeat appeared at {path} within {timeout_s}s"
            raise AssertionError(message)
        time.sleep(_BEAT_PERIOD_S)


def still_beating(path: Path) -> bool:
    """Whether something is writing ``path`` right now, sampled across two beat periods.

    The pid in a heartbeat file outlives the process that wrote it, and Windows
    hands a freed pid to the next process within milliseconds, so signalling a
    pid read from a stale file can take down an unrelated process. A file that
    is still advancing belongs to a live writer, which makes its pid current.
    """
    first = read_beat(path)
    time.sleep(_BEAT_PERIOD_S * 2)
    return read_beat(path) != first


def heartbeat_stopped(path: Path) -> bool:
    """Whether nothing is writing ``path`` any more: two samples a window apart match."""
    first = read_beat(path)
    time.sleep(_STOPPED_WINDOW_S)
    return read_beat(path) == first


def tool_argv(scripts: dict[str, Path], pid_file: Path, *bench_argv: str) -> list[str]:
    """The argv of a tool child that records its pid in ``pid_file``, then runs ``bench_argv``."""
    return [sys.executable, str(scripts["tool"]), str(pid_file), *bench_argv]


def bench_argv(scripts: dict[str, Path], bench_beat: Path, grandchild_beat: Path) -> list[str]:
    """The argv of a bench that heartbeats and spawns a heartbeat grandchild."""
    return [sys.executable, str(scripts["bench"]), str(bench_beat), str(grandchild_beat)]


def bench_beat_paths(tmp_path: Path, reap_beats: list[Path]) -> tuple[Path, Path]:
    """Bench and grandchild heartbeat paths under ``tmp_path``, tracked for reap-on-teardown."""
    bench_beat = tmp_path / "bench.beat"
    grandchild_beat = tmp_path / "grandchild.beat"
    reap_beats.extend([bench_beat, grandchild_beat])
    return bench_beat, grandchild_beat


def run_signalled_supervisor(
    tmp_path: Path,
    scripts: dict[str, Path],
    bench_beat: Path,
    grandchild_beat: Path,
    stop: Callable[[subprocess.Popen[str]], None],
) -> int:
    """Run a tool-in-tool supervisor tree until the bench beats, then apply ``stop`` to it.

    Args:
        tmp_path: Directory the supervisor runs in and the pid files land in.
        scripts: The helper scripts, by name.
        bench_beat: Heartbeat file the bench writes.
        grandchild_beat: Heartbeat file the bench's own child writes.
        stop: What ends the supervisor once the bench is beating.

    Returns:
        The pid of the intermediate tool child that started the bench.
    """
    inner_pid_file = tmp_path / "inner_tool.pid"
    proc = subprocess.Popen(  # noqa: S603 -- argv is a fixed list, not shell-injected
        tool_argv(
            scripts,
            tmp_path / "outer_tool.pid",
            *tool_argv(
                scripts,
                inner_pid_file,
                *bench_argv(scripts, bench_beat, grandchild_beat),
            ),
        ),
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_beat(bench_beat)
        tool_pid = wait_for_pid_file_blocking(inner_pid_file, timeout_s=_BEAT_TIMEOUT_S)

        stop(proc)

        proc.communicate(timeout=_SUPERVISOR_EXIT_TIMEOUT_S)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    return tool_pid


def leave_to_timeout(task: ExecTask, abort: asyncio.Event) -> None:
    """Leave the run alone: its own timeout tears it down."""


def set_abort(task: ExecTask, abort: asyncio.Event) -> None:
    """Tear the run down through its abort event."""
    abort.set()


def cancel_task(task: ExecTask, abort: asyncio.Event) -> None:
    """Tear the run down by cancelling the task awaiting it."""
    task.cancel()


@dataclasses.dataclass(frozen=True, slots=True)
class Teardown:
    """How a test ends the outer run: an optional timeout plus an action on the running task."""

    timeout_ms: int | None
    trigger: Callable[[ExecTask, asyncio.Event], None]


@pytest.fixture
def scripts(tmp_path: Path) -> dict[str, Path]:
    """Write the helper scripts into the test's ``tmp_path`` and return their paths."""
    written: dict[str, Path] = {}
    for name, source in _SCRIPTS.items():
        path = tmp_path / f"{name}.py"
        path.write_text(source, encoding="utf-8")
        written[name] = path
    return written


@pytest.fixture
def reap_beats() -> Iterator[list[Path]]:
    """Track heartbeat files and hard-kill whatever is still writing one on teardown."""
    beats: list[Path] = []
    try:
        yield beats
    finally:
        for path in beats:
            if not still_beating(path):
                continue
            pid = beat_pid(path)
            if pid is None:
                continue
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL)


# ---------------------------------------------------------------------------
# tearing the outer run down reaches the bench in its own session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "teardown",
    [
        pytest.param(Teardown(None, set_abort), id="abort"),
        pytest.param(Teardown(_OUTER_TIMEOUT_MS, leave_to_timeout), id="timeout"),
        pytest.param(Teardown(None, cancel_task), id="cancelled"),
    ],
)
async def test_exec_argv_when_outer_run_torn_down_does_leave_no_bench_alive(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
    teardown: Teardown,
) -> None:
    bench_beat, grandchild_beat = bench_beat_paths(tmp_path, reap_beats)
    abort = asyncio.Event()
    task = asyncio.create_task(
        exec_argv(
            tool_argv(
                scripts,
                tmp_path / "tool.pid",
                *bench_argv(scripts, bench_beat, grandchild_beat),
            ),
            ExecOptions(cwd=str(tmp_path), timeout_ms=teardown.timeout_ms, abort=abort),
        ),
    )
    await asyncio.to_thread(wait_for_beat, bench_beat)
    await asyncio.to_thread(wait_for_beat, grandchild_beat)

    teardown.trigger(task, abort)

    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, _SETTLE_TIMEOUT_S)
    assert await asyncio.to_thread(heartbeat_stopped, bench_beat), "the bench outlived the run"
    assert await asyncio.to_thread(heartbeat_stopped, grandchild_beat), (
        "the bench's grandchild outlived the run"
    )


async def test_exec_argv_when_child_exits_on_graceful_request_does_settle_at_child_speed(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
) -> None:
    beat = tmp_path / "child.beat"
    reap_beats.append(beat)
    abort = asyncio.Event()
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, str(scripts["beat"]), str(beat)],
            ExecOptions(cwd=str(tmp_path), abort=abort),
        ),
    )
    await asyncio.to_thread(wait_for_beat, beat)
    loop = asyncio.get_running_loop()
    started = loop.time()

    abort.set()

    await asyncio.wait_for(task, _SETTLE_TIMEOUT_S)
    assert loop.time() - started < _CHILD_EXIT_SETTLE_S


# ---------------------------------------------------------------------------
# a child that ignores the graceful request is killed once the grace elapses
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="win32 has no ignorable termination signal")
async def test_exec_argv_when_aborted_and_child_ignores_graceful_signal_does_kill_after_grace(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
) -> None:
    beat = tmp_path / "child.beat"
    reap_beats.append(beat)
    abort = asyncio.Event()
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, str(scripts["ignore"]), str(beat)],
            ExecOptions(cwd=str(tmp_path), abort=abort),
        ),
    )
    await asyncio.to_thread(wait_for_beat, beat)
    loop = asyncio.get_running_loop()
    started = loop.time()

    abort.set()

    result = await asyncio.wait_for(task, _SETTLE_TIMEOUT_S)
    assert result == ExecResult(stdout="", stderr="", exit_code=1, stdout_bytes=0, stderr_bytes=0)
    assert loop.time() - started < _GRACE_BOUND_S


@pytest.mark.skipif(sys.platform == "win32", reason="win32 has no ignorable termination signal")
async def test_exec_argv_when_timed_out_and_child_ignores_graceful_signal_does_kill_after_grace(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
) -> None:
    beat = tmp_path / "child.beat"
    reap_beats.append(beat)
    timeout_ms = 1000
    loop = asyncio.get_running_loop()
    started = loop.time()

    result = await asyncio.wait_for(
        exec_argv(
            [sys.executable, str(scripts["ignore"]), str(beat)],
            ExecOptions(cwd=str(tmp_path), timeout_ms=timeout_ms),
        ),
        _SETTLE_TIMEOUT_S,
    )

    assert isinstance(result, ExecTimeoutError)
    assert result.timeout_ms == timeout_ms
    assert loop.time() - started < timeout_ms / 1000 + _GRACE_BOUND_S


@pytest.mark.skipif(sys.platform == "win32", reason="win32 has no ignorable termination signal")
async def test_exec_argv_when_cancelled_and_child_ignores_graceful_signal_does_settle_in_reap_bound(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
) -> None:
    beat = tmp_path / "child.beat"
    reap_beats.append(beat)
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, str(scripts["ignore"]), str(beat)],
            ExecOptions(cwd=str(tmp_path)),
        ),
    )
    await asyncio.to_thread(wait_for_beat, beat)
    loop = asyncio.get_running_loop()
    started = loop.time()

    task.cancel()

    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, _SETTLE_TIMEOUT_S)
    assert loop.time() - started < _CANCEL_BOUND_S
    assert await asyncio.to_thread(heartbeat_stopped, beat), "the child outlived its cancellation"


# ---------------------------------------------------------------------------
# a signalled — or hard-killed — supervisor takes the whole tree with it
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal delivery")
def test_supervisor_when_signalled_mid_bench_does_leave_no_bench_alive(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
) -> None:
    bench_beat, grandchild_beat = bench_beat_paths(tmp_path, reap_beats)

    tool_pid = run_signalled_supervisor(
        tmp_path, scripts, bench_beat, grandchild_beat, lambda p: p.send_signal(signal.SIGTERM)
    )

    wait_until_dead_blocking(tool_pid, timeout_s=_SETTLE_TIMEOUT_S)
    assert heartbeat_stopped(bench_beat), "the bench outlived the signalled supervisor"


@pytest.mark.skipif(sys.platform != "win32", reason="kill-on-close is a Job Object guarantee")
def test_supervisor_when_killed_without_cleanup_does_leave_no_bench_alive(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
) -> None:
    bench_beat, grandchild_beat = bench_beat_paths(tmp_path, reap_beats)

    run_signalled_supervisor(tmp_path, scripts, bench_beat, grandchild_beat, lambda p: p.kill())

    assert heartbeat_stopped(bench_beat), "the bench outlived the hard-killed supervisor"


# ---------------------------------------------------------------------------
# win32 Job Objects: orphaned descendants and the taskkill fallback
# ---------------------------------------------------------------------------


def delay_attach(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delay ``attach_process_group`` so descendants can start before containment lands."""
    attach = gymrat_exec.attach_process_group

    def attach_late(pid: int) -> None:
        time.sleep(_LATE_ATTACH_DELAY_S)
        attach(pid)

    monkeypatch.setattr(gymrat_exec, "attach_process_group", attach_late)


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are win32-only")
async def test_exec_argv_when_child_already_exited_does_kill_orphaned_grandchild(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
) -> None:
    beat = tmp_path / "grandchild.beat"
    reap_beats.append(beat)
    abort = asyncio.Event()
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, str(scripts["orphan"]), str(beat)],
            ExecOptions(cwd=str(tmp_path), abort=abort),
        ),
    )
    await asyncio.to_thread(wait_for_beat, beat)

    abort.set()

    await asyncio.wait_for(task, _SETTLE_TIMEOUT_S)
    assert await asyncio.to_thread(heartbeat_stopped, beat), (
        "a grandchild whose parent had already exited outlived the run"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are win32-only")
async def test_exec_argv_when_containment_lands_late_does_kill_descendants_started_before_it(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bench_beat, grandchild_beat = bench_beat_paths(tmp_path, reap_beats)
    delay_attach(monkeypatch)
    abort = asyncio.Event()
    task = asyncio.create_task(
        exec_argv(
            bench_argv(scripts, bench_beat, grandchild_beat),
            ExecOptions(cwd=str(tmp_path), abort=abort),
        ),
    )
    await asyncio.to_thread(wait_for_beat, bench_beat)
    await asyncio.to_thread(wait_for_beat, grandchild_beat)

    abort.set()

    await asyncio.wait_for(task, _SETTLE_TIMEOUT_S)
    assert await asyncio.to_thread(heartbeat_stopped, grandchild_beat), (
        "a grandchild started before containment landed outlived the run"
    )
    assert await asyncio.to_thread(heartbeat_stopped, bench_beat), (
        "a bench started before containment landed outlived the run"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are win32-only")
@pytest.mark.parametrize(
    "tree",
    [
        pytest.param(("bench", "bench.beat", "grandchild.beat"), id="live-descendants"),
        pytest.param(("orphan", "grandchild.beat"), id="descendant-whose-parent-exited"),
    ],
)
async def test_stdio_driver_when_containment_lands_late_does_leave_no_descendant_alive(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    tree: tuple[str, ...],
) -> None:
    script, *beat_names = tree
    beats = [tmp_path / name for name in beat_names]
    reap_beats.extend(beats)
    delay_attach(monkeypatch)
    abort = asyncio.Event()
    session = create_stdio_driver([sys.executable, str(scripts[script]), *map(str, beats)]).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer, abort
    )
    await asyncio.gather(*(asyncio.to_thread(wait_for_beat, beat) for beat in beats))

    abort.set()

    await asyncio.wait_for(session.outcome, _SETTLE_TIMEOUT_S)
    survivors = [b.name for b in beats if not await asyncio.to_thread(heartbeat_stopped, b)]
    assert survivors == [], "a descendant of the session child outlived the session"


async def test_stdio_driver_when_child_cannot_be_resumed_does_settle_error_with_child_torn_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = capture_spawns(monkeypatch, "create_subprocess_exec")
    # On win32 the child really is left suspended; elsewhere it runs, but only teardown ends it.
    monkeypatch.setattr(gymrat_exec, "resume_process_group", refuse_resume)
    session = create_stdio_driver(list(SLEEPER_ARGV)).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )

    outcome = await asyncio.wait_for(session.outcome, _SETTLE_TIMEOUT_S)

    assert (outcome.reason, outcome.cost_usd) == ("error", 0.0)
    assert "could not be resumed" in (outcome.message or "")
    assert spawned[0].returncode is not None, "the child that could not resume was never killed"


# ---------------------------------------------------------------------------
# win32 Job Objects: teardown observed from a POSIX run
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class FakeJobs:
    """Answers the Job Object calls of one child and records what was asked of it.

    ``active_counts`` is handed out one per accounting query, so a test can walk
    a job down to empty; once it runs out, every further query reports
    ``final_active``.
    """

    active_counts: list[int] = dataclasses.field(default_factory=list)
    final_active: int = 0
    creation_granted: bool = True
    """Whether ``CreateJobObjectW`` hands out a job, as a locked-down host would not."""

    assignment_granted: bool = True
    """Whether ``AssignProcessToJobObject`` accepts the child, as a locked-down host would not."""

    resume_granted: bool = True
    """Whether ``NtResumeProcess`` accepts the handle, as a locked-down host would not."""

    queried: list[int] = dataclasses.field(default_factory=list)
    closed: list[int] = dataclasses.field(default_factory=list)
    terminated: list[int] = dataclasses.field(default_factory=list)
    limited: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    """Info class and ``BasicLimitInformation.LimitFlags`` of each limits call."""

    opened: dict[int, int] = dataclasses.field(default_factory=dict)
    """Pid behind each process handle handed out by ``OpenProcess``."""

    assigned: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    """Job handle and pid of each successful assignment."""

    resumed: list[int] = dataclasses.field(default_factory=list)
    """Pid behind the handle of each successful resume."""


def fake_kernel32(jobs: FakeJobs) -> types.SimpleNamespace:
    """A ``kernel32`` stub that grants every job call and answers queries from ``jobs``."""

    def query_information_job_object(
        job: object,
        info_class: object,
        info: Any,
        *_args: object,
    ) -> int:
        active = jobs.active_counts.pop(0) if jobs.active_counts else jobs.final_active
        jobs.queried.append(active)
        assert len(jobs.queried) <= _FAKE_JOB_QUERY_CAP, (
            "the wait polled the job past every plausible grace instead of giving up"
        )
        # The production call passes ``ctypes.byref(struct)``; the struct it
        # reads the count back out of is what ``_obj`` reaches.
        info._obj.ActiveProcesses = active
        return 1

    def close_handle(handle: int) -> int:
        jobs.closed.append(handle)
        return 1

    def terminate_job_object(job: int, *_args: object) -> int:
        jobs.terminated.append(job)
        return 1

    def create_job_object(*_args: object) -> int:
        # A NULL handle is how ``CreateJobObjectW`` reports a refusal.
        return _JOB_HANDLE if jobs.creation_granted else 0

    def open_process(_access: int, _inherit: bool, pid: int) -> int:
        jobs.opened[_PROCESS_HANDLE] = pid
        return _PROCESS_HANDLE

    def set_information_job_object(
        _job: int,
        info_class: int,
        info: Any,
        *_args: object,
    ) -> int:
        # ``ctypes.byref(struct)`` again: ``_obj`` is the struct the production
        # code filled in before handing it over.
        jobs.limited.append((info_class, info._obj.BasicLimitInformation.LimitFlags))
        return 1

    def assign_process_to_job_object(job: int, process: int) -> int:
        if not jobs.assignment_granted:
            return 0
        jobs.assigned.append((job, jobs.opened[process]))
        return 1

    return types.SimpleNamespace(
        CreateJobObjectW=create_job_object,
        SetInformationJobObject=set_information_job_object,
        OpenProcess=open_process,
        AssignProcessToJobObject=assign_process_to_job_object,
        CloseHandle=close_handle,
        TerminateJobObject=terminate_job_object,
        QueryInformationJobObject=query_information_job_object,
    )


def fake_ntdll(jobs: FakeJobs) -> types.SimpleNamespace:
    """An ``ntdll`` stub that resumes the process behind a handle ``jobs`` handed out."""

    def resume_process(process: int) -> int:
        if not jobs.resume_granted:
            return _NT_STATUS_ACCESS_DENIED
        jobs.resumed.append(jobs.opened[process])
        return _NT_STATUS_SUCCESS

    return types.SimpleNamespace(NtResumeProcess=resume_process)


def win32_process_group(monkeypatch: pytest.MonkeyPatch, jobs: FakeJobs) -> types.ModuleType:
    """Load a private copy of ``gymrat.process_group`` with its win32 job path bound.

    The job functions exist only under ``sys.platform == "win32"``, so reaching
    them from a POSIX run means executing the module source again with the
    platform faked and ``kernel32`` stubbed. The copy is the test's own; the
    imported module keeps its POSIX bindings.
    """
    windll = types.SimpleNamespace(kernel32=fake_kernel32(jobs), ntdll=fake_ntdll(jobs))
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda: OSError(_FAKE_WIN_ERROR), raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    spec = importlib.util.spec_from_file_location("process_group_win32", process_group.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_subprocess_runs(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub ``subprocess.run`` to record every call's argv instead of running it."""
    argv_calls: list[list[str]] = []

    def record_run(args: list[str], *_args: object, **_kwargs: object) -> object:
        argv_calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", record_run)
    return argv_calls


def test_attach_process_group_when_child_gets_a_job_does_limit_it_to_kill_on_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs()
    module = win32_process_group(monkeypatch, jobs)

    module.attach_process_group(_CHILD_PID)

    assert len(jobs.limited) == 1, "the child's job was created without a limits call"
    info_class, limit_flags = jobs.limited[0]
    assert info_class == _WIN32_EXTENDED_LIMIT_INFORMATION
    assert limit_flags & _WIN32_LIMIT_KILL_ON_JOB_CLOSE, (
        "the job does not kill its members when its last handle closes"
    )


def test_terminate_process_group_when_job_assignment_refused_does_warn_and_fall_back_to_taskkill(
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    jobs = FakeJobs(assignment_granted=False)
    module = win32_process_group(monkeypatch, jobs)
    argv_calls = record_subprocess_runs(monkeypatch)
    module.attach_process_group(_CHILD_PID)

    module.terminate_process_group(_CHILD_PID)

    refusals = [w for w in recwarn if "job assignment refused" in str(w.message)]
    assert len(refusals) == 1, "a refused assignment has to warn exactly once"
    assert refusals[0].category is RuntimeWarning
    assert argv_calls == [["taskkill", "/F", "/T", "/PID", str(_CHILD_PID)]], (
        "a child that never reached a job was not torn down through taskkill"
    )
    assert jobs.closed == [_PROCESS_HANDLE, _JOB_HANDLE], "the refused job handle was leaked"
    assert _CHILD_PID not in module._job_handles


async def test_exec_argv_when_run_settles_on_win32_does_job_the_child_then_close_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs()
    group = win32_process_group(monkeypatch, jobs)
    module = win32_exec(monkeypatch, group)

    result = await module.exec_argv(
        [sys.executable, "-c", "import os; print(os.getpid(), os.getppid())"],
        module.ExecOptions(cwd=str(tmp_path)),
    )

    assert isinstance(result, module.ExecResult)
    assert result.exit_code == 0
    reporting_pid, parent_pid = (int(field) for field in result.stdout.split())
    # A Windows virtual environment reaches the interpreter through a launcher
    # that runs it as a child of its own, so there the pid exec spawned is the
    # reporting process's parent — the job holds both either way. On POSIX the
    # parent is this test process, which is never the one jobbed.
    spawned_tree = {reporting_pid, parent_pid} if _HOST_IS_WINDOWS else {reporting_pid}
    assert [job for job, _ in jobs.assigned] == [_JOB_HANDLE], (
        "the spawned child never reached a job"
    )
    jobbed_pid = jobs.assigned[0][1]
    assert jobbed_pid in spawned_tree, "a process outside the spawned tree was put in the job"
    assert jobs.resumed == [jobbed_pid], "the contained child was left suspended"
    assert jobs.closed == [_PROCESS_HANDLE, _PROCESS_HANDLE, _JOB_HANDLE], (
        "the settled run left the child's job open, so a descendant survives it"
    )
    assert jobbed_pid not in group._job_handles


def test_kill_process_group_when_host_has_no_sigkill_does_terminate_the_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs()
    monkeypatch.delattr(signal, "SIGKILL", raising=False)
    module = win32_process_group(monkeypatch, jobs)
    module.attach_process_group(_CHILD_PID)

    module.kill_process_group(_CHILD_PID)

    assert jobs.terminated == [_JOB_HANDLE], (
        "the kill never reached the job: a host without SIGKILL cannot be asked for one"
    )


def test_terminate_process_group_when_job_still_emptying_does_wait_for_its_last_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs(active_counts=[2, 1, 0])
    module = win32_process_group(monkeypatch, jobs)
    module.attach_process_group(_CHILD_PID)

    module.terminate_process_group(_CHILD_PID)

    assert jobs.terminated == [_JOB_HANDLE]
    assert jobs.queried == [2, 1, 0], "the terminate returned before the job reported itself empty"


def test_terminate_process_group_when_job_never_empties_does_give_up_at_the_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs(final_active=1)
    module = win32_process_group(monkeypatch, jobs)
    monkeypatch.setattr(module, "TERMINATE_GRACE_S", _FAKE_JOB_GRACE_S)
    module.attach_process_group(_CHILD_PID)
    started = time.monotonic()

    module.terminate_process_group(_CHILD_PID)

    elapsed = time.monotonic() - started
    assert len(jobs.queried) > 1, "the wait gave up without ever polling the job again"
    assert elapsed < _GRACE_BOUND_S, "a job that never empties held the teardown open"


def test_release_process_group_when_run_settles_does_close_the_job_and_forget_the_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs()
    module = win32_process_group(monkeypatch, jobs)
    module.attach_process_group(_CHILD_PID)

    module.release_process_group(_CHILD_PID)

    assert jobs.closed == [_PROCESS_HANDLE, _JOB_HANDLE], (
        "the settle path left the job handle open, so a descendant survives it"
    )
    assert _CHILD_PID not in module._job_handles


def test_release_process_group_when_job_still_emptying_does_drain_it_before_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs(active_counts=[2, 1, 0])
    module = win32_process_group(monkeypatch, jobs)
    module.attach_process_group(_CHILD_PID)

    module.release_process_group(_CHILD_PID)

    assert jobs.terminated == [_JOB_HANDLE], (
        "the release left the job's members to die on their own after the handle closed"
    )
    assert jobs.queried == [2, 1, 0], (
        "the release closed the job handle before its members had reported themselves gone"
    )
    assert jobs.closed == [_PROCESS_HANDLE, _JOB_HANDLE], (
        "the drained job handle was leaked instead of closed"
    )
    assert _CHILD_PID not in module._job_handles


# ---------------------------------------------------------------------------
# win32 Job Objects: the child is contained before it runs
# ---------------------------------------------------------------------------


def bind_group_seams(
    monkeypatch: pytest.MonkeyPatch,
    module: types.ModuleType,
    group: types.ModuleType,
) -> None:
    """Rebind ``module``'s process-group seams to the faked-win32 ``group`` copy."""
    for seam in ("attach_process_group", "release_process_group", "resume_process_group"):
        monkeypatch.setattr(module, seam, getattr(group, seam))


def win32_exec(monkeypatch: pytest.MonkeyPatch, group: types.ModuleType) -> types.ModuleType:
    """Load a private copy of ``gymrat.exec`` with its win32 creation flags bound.

    The flags a child is created with are decided when the module is imported,
    so a POSIX run reaches the win32 ones only by executing the module source
    again with the platform faked — which ``win32_process_group`` has already
    done for its caller. The copy's process-group seams are rebound to
    ``group``, the faked-win32 copy of the job code.

    The copy asks for its children suspended, and the faked win32 layer resumes
    nothing, so a real child created that way would stay suspended for good on
    a Windows host and be refused outright on a POSIX one. The spawn the copy
    reaches therefore drops the creation flags, and every child it starts runs
    from the beginning.

    Args:
        monkeypatch: Used to rebind the copy's process-group seams and the
            spawn calls it reaches.
        group: The faked-win32 ``process_group`` copy the seams are bound to.

    Returns:
        The private ``gymrat.exec`` copy.
    """
    spec = importlib.util.spec_from_file_location("exec_win32", gymrat_exec.__file__)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bind_group_seams(monkeypatch, module, group)
    spawn_children_running(monkeypatch)
    return module


def spawn_children_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the creation flags from every asyncio spawn, so no child starts suspended."""
    spawn_exec = asyncio.create_subprocess_exec
    spawn_shell = asyncio.create_subprocess_shell

    async def exec_running(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
        kwargs.pop("creationflags", None)
        return await spawn_exec(*args, **kwargs)

    async def shell_running(command: str, **kwargs: Any) -> asyncio.subprocess.Process:
        kwargs.pop("creationflags", None)
        return await spawn_shell(command, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", exec_running)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", shell_running)


def trace_containment(
    monkeypatch: pytest.MonkeyPatch,
    module: types.ModuleType,
) -> list[tuple[str, int]]:
    """Record every containment step ``module`` takes around a spawn, in order.

    The recorded number is the child's creation flags for the spawn itself and
    the pid for the seams that follow it, so a test reads both the order of the
    steps and what each one asked for.

    Args:
        monkeypatch: Used to wrap the spawn call and the copy's seams.
        module: The ``gymrat.exec`` copy whose spawn is traced.

    Returns:
        The list the steps are appended to, in the order they happen.
    """
    steps: list[tuple[str, int]] = []
    spawn = asyncio.create_subprocess_exec
    attach = module.attach_process_group
    resume = module.resume_process_group

    async def record_spawn(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
        steps.append(("spawn", kwargs.get("creationflags", 0)))
        return await spawn(*args, **kwargs)

    def record_attach(pid: int) -> None:
        steps.append(("attach", pid))
        attach(pid)

    def record_resume(pid: int) -> bool:
        steps.append(("resume", pid))
        return resume(pid)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record_spawn)
    monkeypatch.setattr(module, "attach_process_group", record_attach)
    monkeypatch.setattr(module, "resume_process_group", record_resume)
    return steps


def test_resume_process_group_when_child_is_suspended_does_resume_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs()
    module = win32_process_group(monkeypatch, jobs)

    resumed = module.resume_process_group(_CHILD_PID)

    assert resumed is True
    assert jobs.resumed == [_CHILD_PID], "the suspended child was never resumed"
    assert jobs.closed == [_PROCESS_HANDLE], "the resumed child's handle was leaked"


def test_resume_process_group_when_host_refuses_resume_does_warn_and_report_the_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = FakeJobs(resume_granted=False)
    module = win32_process_group(monkeypatch, jobs)

    with pytest.warns(RuntimeWarning, match="could not resume"):
        resumed = module.resume_process_group(_CHILD_PID)

    assert resumed is False
    assert jobs.closed == [_PROCESS_HANDLE], "the refused child's handle was leaked"


@pytest.mark.parametrize(
    "assignment_granted",
    [pytest.param(True, id="job-granted"), pytest.param(False, id="job-refused")],
)
async def test_exec_argv_when_spawning_on_win32_does_suspend_the_child_until_it_is_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
    assignment_granted: bool,
) -> None:
    jobs = FakeJobs(assignment_granted=assignment_granted)
    group = win32_process_group(monkeypatch, jobs)
    module = win32_exec(monkeypatch, group)
    steps = trace_containment(monkeypatch, module)

    result = await module.exec_argv(
        [sys.executable, "-c", "pass"],
        module.ExecOptions(cwd=str(tmp_path)),
    )

    assert result.exit_code == 0, "a contained child never ran to completion"
    assert [name for name, _ in steps] == ["spawn", "attach", "resume"], (
        "the child was left to run before it had joined its job"
    )
    creation_flags = steps[0][1]
    assert creation_flags & _WIN32_CREATE_SUSPENDED, (
        "the child was created running, so it acts before containment lands"
    )
    assert jobs.resumed == [steps[2][1]], "the contained child was left suspended"


# ---------------------------------------------------------------------------
# win32 Job Objects: the stdio session child is contained like an exec child
# ---------------------------------------------------------------------------

# A session child that answers the start command with a completed outcome.
_COMPLETES = """input(); print('{"type":"outcome","reason":"completed","cost_usd":0.25}')"""


def win32_stdio(
    monkeypatch: pytest.MonkeyPatch,
    jobs: FakeJobs,
) -> tuple[list[asyncio.subprocess.Process], list[list[str]]]:
    """Bind the stdio session's containment to a faked-win32 job layer answering from ``jobs``.

    The spawn, resume, and release seams of ``gymrat.exec`` and the session's own
    group kill all reach the faked ``process_group`` copy, while the child itself
    is a real process started running. ``subprocess.run`` — the ``taskkill``
    fallback — is recorded instead of run.

    Args:
        monkeypatch: Used to rebind the seams, the spawn, and ``subprocess.run``.
        jobs: The faked job layer's answers and records.

    Returns:
        Every child the session spawned, and the argv of every ``subprocess.run`` call.
    """
    group = win32_process_group(monkeypatch, jobs)
    bind_group_seams(monkeypatch, gymrat_exec, group)
    monkeypatch.setattr("gymrat.supervisor.stdio.kill_process_group", group.kill_process_group)
    argv_calls = record_subprocess_runs(monkeypatch)
    spawn_children_running(monkeypatch)
    return capture_spawns(monkeypatch, "create_subprocess_exec"), argv_calls


@pytest.mark.parametrize(
    ("refusal", "warning"),
    [
        pytest.param(
            lambda: FakeJobs(creation_granted=False),
            "job creation refused",
            id="job-creation-refused",
        ),
        pytest.param(
            lambda: FakeJobs(assignment_granted=False),
            "job assignment refused",
            id="job-assignment-refused",
        ),
    ],
)
async def test_stdio_driver_when_host_refuses_the_job_does_fall_back_to_taskkill_with_one_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
    refusal: Callable[[], FakeJobs],
    warning: str,
) -> None:
    spawned, argv_calls = win32_stdio(monkeypatch, refusal())
    session = create_stdio_driver([sys.executable, "-c", _COMPLETES]).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )

    outcome = await asyncio.wait_for(session.outcome, _SETTLE_TIMEOUT_S)

    assert outcome == SessionOutcome(reason="completed", cost_usd=0.25)
    runtime_warnings = [str(w.message) for w in recwarn if w.category is RuntimeWarning]
    assert len(runtime_warnings) == 1, "a refused job warns once"
    assert warning in runtime_warnings[0], "the one warning is not the job refusal"
    assert ["taskkill", "/F", "/T", "/PID", str(spawned[0].pid)] in argv_calls, "no taskkill ran"
