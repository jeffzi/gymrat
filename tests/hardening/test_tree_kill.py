"""Tree-teardown hardening: a bench in its own session dies with the run that started it.

The heartbeat tests drive the production chain out of process. A tool child
runs ``run_with_signal_abort`` and starts its bench through ``exec_argv``,
exactly as a gymrat run does, so the bench lands in its own POSIX session — or
its own Windows job — instead of in the tool child's process group. Tearing the
outer run down has to reach it anyway, whether the teardown comes from an
abort, a timeout, a cancellation, or a signal to the supervisor.

The win32 Job Object tests at the end run in process instead: they drive a
private copy of ``gymrat.exec`` whose job layer is faked, so the order in which
a child is contained, and the taskkill fallback when the host refuses a job, are
pinned from any host.

Liveness is read from heartbeat files rather than process IDs: ``os.kill(pid, 0)``
terminates the target on Windows, so a pid probe cannot be shared across
platforms. A process that has stopped rewriting its heartbeat file has stopped
running.

The module carries no platform skip. The few sub-cases that can only hold on
one platform are gated one by one.
"""

import asyncio
import contextlib
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
from gymrat.exec import (
    ExecOptions,
    SpawnError,
    exec_argv,
    release_contained,
    spawn_contained,
)
from tests._exec_fixtures import Teardown, cancel_task, leave_to_timeout, set_abort, settle
from tests._process_helpers import (
    JOB_HANDLE,
    PROCESS_HANDLE,
    SLEEPER_ARGV,
    FakeJobs,
    capture_spawns,
    reaped,
    record_subprocess_runs,
    refuse_resume,
    wait_for_pid_file_blocking,
    wait_until_dead_blocking,
    win32_process_group,
)

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

# How long the late-containment rows hold the job assignment back. A venv
# launcher starts its real interpreter within tens of milliseconds, so a full
# second guarantees the interpreter exists before the launcher joins the job.
_LATE_ATTACH_DELAY_S = 1.0

# The job tests fake ``sys.platform``, so a check against the host this suite is
# really running on has to read the platform recorded before any faking.
_HOST_IS_WINDOWS = sys.platform == "win32"

# ``CREATE_SUSPENDED``, spelled out here rather than read back from the module
# under test: the child exists but runs no instruction until resumed.
_WIN32_CREATE_SUSPENDED = 0x4

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

from gymrat.cli.run_setup import run_with_signal_abort
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

    Args:
        path: The heartbeat file to sample.

    Returns:
        True when the file changed between the two samples.
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
    """Run a tool-in-tool supervisor tree until the bench and its child beat, then apply ``stop``.

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
    argv = tool_argv(
        scripts,
        tmp_path / "outer_tool.pid",
        *tool_argv(scripts, inner_pid_file, *bench_argv(scripts, bench_beat, grandchild_beat)),
    )
    with reaped(
        subprocess.Popen(  # noqa: S603 -- argv is a fixed list, not shell-injected
            argv,
            cwd=str(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    ) as proc:
        wait_for_beat(bench_beat)
        wait_for_beat(grandchild_beat)
        tool_pid = wait_for_pid_file_blocking(inner_pid_file, timeout_s=_BEAT_TIMEOUT_S)

        stop(proc)

        proc.communicate(timeout=_SUPERVISOR_EXIT_TIMEOUT_S)
    return tool_pid


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

    teardown.trigger(task, None, abort)

    await settle(task, _SETTLE_TIMEOUT_S)
    assert await asyncio.to_thread(heartbeat_stopped, bench_beat), "the bench outlived the run"
    assert await asyncio.to_thread(heartbeat_stopped, grandchild_beat), (
        "the bench's grandchild outlived the run"
    )


#: The timeout an ignoring child runs under, short enough to keep the timeout row quick.
_IGNORING_CHILD_TIMEOUT_MS = 1000

_POSIX_ONLY_IGNORE = pytest.mark.skipif(
    sys.platform == "win32", reason="win32 has no ignorable termination signal"
)


@pytest.mark.parametrize(
    ("script", "teardown", "bound_s", "outcome"),
    [
        pytest.param(
            "beat",
            Teardown(None, set_abort),
            _CHILD_EXIT_SETTLE_S,
            "ExecResult",
            id="graceful-child-settles-at-child-speed",
        ),
        pytest.param(
            "ignore",
            Teardown(None, set_abort),
            _GRACE_BOUND_S,
            "ExecResult",
            id="ignoring-child-aborted-is-killed-after-grace",
            marks=_POSIX_ONLY_IGNORE,
        ),
        pytest.param(
            "ignore",
            Teardown(_IGNORING_CHILD_TIMEOUT_MS, leave_to_timeout),
            _IGNORING_CHILD_TIMEOUT_MS / 1000 + _GRACE_BOUND_S,
            "ExecTimeoutError",
            id="ignoring-child-timed-out-is-killed-after-grace",
            marks=_POSIX_ONLY_IGNORE,
        ),
        pytest.param(
            "ignore",
            Teardown(None, cancel_task),
            _CANCEL_BOUND_S,
            "CancelledError",
            id="ignoring-child-cancelled-settles-in-reap-bound",
            marks=_POSIX_ONLY_IGNORE,
        ),
    ],
)
async def test_exec_argv_when_torn_down_does_settle_within_the_bound_with_the_child_dead(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
    *,
    script: str,
    teardown: Teardown,
    bound_s: float,
    outcome: str,
) -> None:
    beat = tmp_path / "child.beat"
    reap_beats.append(beat)
    abort = asyncio.Event()
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, str(scripts[script]), str(beat)],
            ExecOptions(cwd=str(tmp_path), abort=abort, timeout_ms=teardown.timeout_ms),
        ),
    )
    await asyncio.to_thread(wait_for_beat, beat)
    loop = asyncio.get_running_loop()
    started = loop.time()

    teardown.trigger(task, None, abort)

    try:
        settled: object = await asyncio.wait_for(task, _SETTLE_TIMEOUT_S)
    except asyncio.CancelledError as cancelled:
        settled = cancelled
    assert loop.time() - started < bound_s
    assert type(settled).__name__ == outcome
    assert await asyncio.to_thread(heartbeat_stopped, beat), "the child outlived its teardown"


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
    assert heartbeat_stopped(grandchild_beat), (
        "the bench's grandchild outlived the signalled supervisor"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="kill-on-close is a Job Object guarantee")
def test_supervisor_when_killed_without_cleanup_does_leave_no_bench_alive(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
) -> None:
    bench_beat, grandchild_beat = bench_beat_paths(tmp_path, reap_beats)

    run_signalled_supervisor(tmp_path, scripts, bench_beat, grandchild_beat, lambda p: p.kill())

    assert heartbeat_stopped(bench_beat), "the bench outlived the hard-killed supervisor"
    assert heartbeat_stopped(grandchild_beat), (
        "the bench's grandchild outlived the hard-killed supervisor"
    )


# ---------------------------------------------------------------------------
# win32 Job Objects: orphaned descendants, late containment, and a refused resume
# ---------------------------------------------------------------------------


def delay_attach(monkeypatch: pytest.MonkeyPatch, delay_s: float) -> None:
    """Delay ``attach_process_group`` so descendants can start before containment lands.

    Args:
        monkeypatch: Used to wrap ``gymrat.exec``'s attach seam.
        delay_s: Seconds each attach is held back; ``0`` lets it land at once.
    """
    attach = gymrat_exec.attach_process_group

    def attach_late(pid: int) -> None:
        time.sleep(delay_s)
        attach(pid)

    monkeypatch.setattr(gymrat_exec, "attach_process_group", attach_late)


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are win32-only")
@pytest.mark.parametrize(
    ("tree", "attach_delay_s"),
    [
        pytest.param(("orphan", "grandchild.beat"), 0, id="descendant-whose-parent-exited"),
        pytest.param(
            ("bench", "bench.beat", "grandchild.beat"),
            _LATE_ATTACH_DELAY_S,
            id="live-descendants-contained-late",
        ),
        pytest.param(
            ("orphan", "grandchild.beat"),
            _LATE_ATTACH_DELAY_S,
            id="descendant-whose-parent-exited-contained-late",
        ),
    ],
)
async def test_exec_argv_when_child_contained_does_leave_no_descendant_alive(
    tmp_path: Path,
    scripts: dict[str, Path],
    reap_beats: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    *,
    tree: tuple[str, ...],
    attach_delay_s: float,
) -> None:
    script, *beat_names = tree
    beats = [tmp_path / name for name in beat_names]
    reap_beats.extend(beats)
    delay_attach(monkeypatch, attach_delay_s)
    abort = asyncio.Event()
    task = asyncio.create_task(
        exec_argv(
            [sys.executable, str(scripts[script]), *map(str, beats)],
            ExecOptions(cwd=str(tmp_path), abort=abort),
        ),
    )
    await asyncio.gather(*(asyncio.to_thread(wait_for_beat, beat) for beat in beats))

    abort.set()

    await asyncio.wait_for(task, _SETTLE_TIMEOUT_S)
    survivors = [b.name for b in beats if not await asyncio.to_thread(heartbeat_stopped, b)]
    assert survivors == [], "a descendant of the contained child outlived the run"


# The POSIX side of a refused resume is pinned in tests/exec/test_spawn_contained.py.
@pytest.mark.skipif(
    sys.platform != "win32",
    reason="a refused resume leaves a real child suspended only under Job Objects",
)
async def test_spawn_contained_when_child_cannot_be_resumed_does_raise_with_child_torn_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = capture_spawns(monkeypatch, "create_subprocess_exec")
    # The child really is left suspended; only teardown ends it.
    monkeypatch.setattr(gymrat_exec, "resume_process_group", refuse_resume)

    with pytest.raises(SpawnError, match="could not be resumed"):
        await asyncio.wait_for(
            spawn_contained(asyncio.create_subprocess_exec, *SLEEPER_ARGV, cwd=str(tmp_path)),
            _SETTLE_TIMEOUT_S,
        )

    assert spawned[0].returncode is not None, "the child that could not resume was never killed"


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

    The flags a child is created with are decided by the platform at each spawn,
    and the process-group seams by the platform at import, so a POSIX run
    reaches the win32 ones only by executing the module source again with the
    platform faked — which ``win32_process_group`` has already done for its
    caller, and which stays faked for the rest of the test. The copy's process-group seams are rebound to
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


@dataclasses.dataclass(slots=True)
class ContainmentTrace:
    """What a traced ``gymrat.exec`` copy did around each spawn."""

    steps: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    """Every containment step in order: the spawn with the child's creation
    flags, then each seam that follows it with the pid it was handed."""

    session_requests: list[set[str]] = dataclasses.field(default_factory=list)
    """The POSIX session arguments each spawn asked for."""


def trace_containment(
    monkeypatch: pytest.MonkeyPatch, module: types.ModuleType
) -> ContainmentTrace:
    """Record every containment step ``module`` takes around a spawn, in order.

    Args:
        monkeypatch: Used to wrap the spawn call and the copy's seams.
        module: The ``gymrat.exec`` copy whose spawn is traced.

    Returns:
        The trace the steps and spawn arguments are recorded into as they happen.
    """
    trace = ContainmentTrace()
    spawn = asyncio.create_subprocess_exec
    attach = module.attach_process_group
    resume = module.resume_process_group

    async def record_spawn(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
        trace.steps.append(("spawn", kwargs.get("creationflags", 0)))
        trace.session_requests.append(kwargs.keys() & {"start_new_session", "preexec_fn"})
        return await spawn(*args, **kwargs)

    def record_attach(pid: int) -> None:
        trace.steps.append(("attach", pid))
        attach(pid)

    def record_resume(pid: int) -> bool:
        trace.steps.append(("resume", pid))
        return resume(pid)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record_spawn)
    monkeypatch.setattr(module, "attach_process_group", record_attach)
    monkeypatch.setattr(module, "resume_process_group", record_resume)
    return trace


@pytest.mark.parametrize(
    ("assignment_granted", "warning", "assigned_jobs", "closed"),
    [
        pytest.param(
            True,
            None,
            [JOB_HANDLE],
            [PROCESS_HANDLE, PROCESS_HANDLE, JOB_HANDLE],
            id="job-granted-held-until-the-run-settles",
        ),
        pytest.param(
            False,
            "job assignment refused",
            [],
            [PROCESS_HANDLE, JOB_HANDLE, PROCESS_HANDLE],
            id="job-refused-released-at-once",
        ),
    ],
)
async def test_exec_argv_when_run_on_win32_does_contain_the_suspended_child_until_it_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    assignment_granted: bool,
    warning: str | None,
    assigned_jobs: list[int],
    closed: list[int],
) -> None:
    jobs = FakeJobs(assignment_granted=assignment_granted)
    group = win32_process_group(monkeypatch, jobs)
    module = win32_exec(monkeypatch, group)
    trace = trace_containment(monkeypatch, module)
    expect_warning = (
        pytest.warns(RuntimeWarning, match=warning)
        if warning is not None
        else contextlib.nullcontext()
    )

    with expect_warning:
        result = await module.exec_argv(
            [sys.executable, "-c", "import os; print(os.getpid(), os.getppid())"],
            module.ExecOptions(cwd=str(tmp_path)),
        )

    assert result.exit_code == 0, "a contained child never ran to completion"
    assert [name for name, _ in trace.steps] == ["spawn", "attach", "resume"], (
        "the child was left to run before it had joined its job"
    )
    creation_flags = trace.steps[0][1]
    assert creation_flags & _WIN32_CREATE_SUSPENDED, (
        "the child was created running, so it acts before containment lands"
    )
    assert trace.session_requests == [set()], "a win32 spawn asked for a POSIX session"
    child_pid = trace.steps[2][1]
    reporting_pid, parent_pid = (int(field) for field in result.stdout.split())
    # A Windows virtual environment reaches the interpreter through a launcher
    # that runs it as a child of its own, so there the pid exec spawned is the
    # reporting process's parent — the job holds both either way. On POSIX the
    # parent is this test process, which is never the one contained.
    spawned_tree = {reporting_pid, parent_pid} if _HOST_IS_WINDOWS else {reporting_pid}
    assert child_pid in spawned_tree, "a process outside the spawned tree was contained"
    assert jobs.assigned == [(job, child_pid) for job in assigned_jobs]
    assert jobs.resumed == [child_pid], "the contained child was left suspended"
    assert jobs.closed == closed, "the settled run left a handle open"


# ---------------------------------------------------------------------------
# win32 Job Objects: a contained child the host refuses a job falls back to taskkill
# ---------------------------------------------------------------------------


async def test_kill_process_group_when_job_creation_was_refused_does_fall_back_to_taskkill_with_one_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    # The child is real and started running; its job layer is the faked win32
    # one, and the taskkill it falls back to is recorded instead of run. A
    # refused assignment takes the same fallback through the stop entry points.
    group = win32_process_group(monkeypatch, FakeJobs(creation_granted=False))
    bind_group_seams(monkeypatch, gymrat_exec, group)
    argv_calls = record_subprocess_runs(monkeypatch)
    spawn_children_running(monkeypatch)
    child = await spawn_contained(
        asyncio.create_subprocess_exec, sys.executable, "-c", "pass", cwd=str(tmp_path)
    )
    await asyncio.wait_for(child.wait(), _SETTLE_TIMEOUT_S)

    try:
        group.kill_process_group(child.pid)
    finally:
        release_contained(child.pid)

    runtime_warnings = [str(w.message) for w in recwarn if w.category is RuntimeWarning]
    assert len(runtime_warnings) == 1, "a refused job warns once"
    assert "job creation refused" in runtime_warnings[0], "the one warning is not the job refusal"
    assert ["taskkill", "/F", "/T", "/PID", str(child.pid)] in argv_calls, "no taskkill ran"
