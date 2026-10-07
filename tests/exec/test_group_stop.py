"""Behavioral tests for stopping ``exec``'s process groups on a termination signal.

A second signal kills every live group at once, after a shortened grace that a
nested gymrat run spends finishing its own teardown; stopping a group waits for
every running member, not just its leader; a nested gymrat run gets to finish
its own bench sweep before the run above it kills it.
"""

import asyncio
import shlex
import signal
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from gymrat import exec as exec_mod
from gymrat.exec import ExecOptions
from gymrat.exec import exec as run_exec
from gymrat.signals import install_termination_cleanup
from tests._exec_fixtures import (
    isolate_live_groups as _isolate_live_groups,  # noqa: F401 -- registers the autouse fixture
)
from tests._process_helpers import wait_for_pid_file, wait_until_dead

# exec drives POSIX process groups (killpg) and sh-only shell syntax; neither
# works under cmd.exe, so the whole module is POSIX-only.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only shell and process groups"
)


# ---------------------------------------------------------------------------
# a second termination signal kills every live group at once
# ---------------------------------------------------------------------------

_SHELL_PID_FILE = "shell.pid"

# Ignores SIGTERM along with its foreground child, so only a SIGKILL ends the
# group. It holds back its pid until exec feeds its stdin: exec does that only
# once the spawn has returned, so a signal raised after the pid appears is never
# deferred by a spawn still in progress.
_STARTED_TERM_IGNORING_COMMAND = f"trap '' TERM; read -r _; echo $$ > {_SHELL_PID_FILE}; sleep 30"


@pytest.fixture(autouse=True)
def _top_level_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as a top-level gymrat, whose escalation grace the cleanup times are sized against."""
    monkeypatch.setattr(exec_mod, "_NESTING_DEPTH", 0)


# How long the cleaner takes to clean up once asked to stop: well past the few
# milliseconds a kill lands in, well inside the one-second grace a stop allows.
_CLEANUP_S = 0.5

# The same for a stop escalated by a second signal: well inside the escalation
# grace those tests run with.
_ESCALATED_CLEANUP_S = 0.1

# The escalation grace the escalated-cleanup tests run with. The real 0.2 s at
# the top nesting level leaves the cleaner only 0.1 s to spare, which a loaded
# CI runner's scheduling delays can use up; the halving tests below pin the
# real grace instead.
_ROOMY_ESCALATION_GRACE_S = 1.0


@pytest.fixture
def roomy_escalation_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give a second signal's kill sweep a grace with room for a loaded runner's delays."""
    monkeypatch.setattr(exec_mod, "_ESCALATION_GRACE_S", _ROOMY_ESCALATION_GRACE_S)


# A timeout that fires once the cleaner is up and waiting to be stopped.
_TIMEOUT_AFTER_START_MS = 1500

_CLEANER_PID_FILE = "cleaner.pid"

_CLEANED_MARKER = "cleaned.marker"

# A stand-in for a nested gymrat run: on SIGTERM it spends ``argv[1]`` seconds
# cleaning up, writes ``argv[2]``, then exits. It writes its pid to ``argv[3]``
# once its handler is in place.
_CLEANER_SCRIPT = """
import os, signal, sys, time

def clean_up(_signal_number, _frame):
    time.sleep(float(sys.argv[1]))
    with open(sys.argv[2], "w") as marker:
        marker.write("cleaned\\n")
    os._exit(0)

signal.signal(signal.SIGTERM, clean_up)
with open(sys.argv[3], "w") as pid_file:
    pid_file.write(str(os.getpid()) + "\\n")
time.sleep(30)
"""


def slow_cleanup_command(
    *, nested: bool, cleanup_s: float = _CLEANUP_S, release_stdio: bool = False
) -> str:
    """Return a shell command running the cleaner, as the group's leader or nested under the shell.

    Nested, the cleaner runs in the foreground of a shell that dies on the
    first SIGTERM while the cleaner is still cleaning up; the trailing ``true``
    keeps the shell from exec-ing into it.

    Args:
        nested: Run the cleaner under the shell instead of as the group's leader.
        cleanup_s: Seconds the cleaner spends cleaning up once asked to stop.
        release_stdio: Point the cleaner's stdio at ``/dev/null``. exec's pipes
            then close with the shell, so waiting on the shell alone no longer
            waits for the cleaner as a side effect.

    Returns:
        The command, for ``exec`` to run under ``sh -c``.
    """
    cleaner = shlex.join([
        sys.executable,
        "-c",
        _CLEANER_SCRIPT,
        str(cleanup_s),
        _CLEANED_MARKER,
        _CLEANER_PID_FILE,
    ])
    if release_stdio:
        cleaner = f"{cleaner} </dev/null >/dev/null 2>&1"
    return f"{cleaner}; true" if nested else f"exec {cleaner}"


def escalate_termination(raise_signal: Callable[[int], int]) -> None:
    """Deliver a second termination signal before the live-group kill sweep has run."""

    def interrupt() -> None:
        raise_signal(signal.SIGTERM)

    install_termination_cleanup(interrupt)
    install_termination_cleanup(exec_mod.kill_live_process_groups)
    raise_signal(signal.SIGINT)


async def test_exec_when_second_signal_arrives_before_kill_sweep_runs_does_kill_live_group(
    tmp_path: Path,
    spawned_processes: list[asyncio.subprocess.Process],
    make_opts: Callable[..., ExecOptions],
    raise_signal: Callable[[int], int],
) -> None:
    task = asyncio.create_task(run_exec(_STARTED_TERM_IGNORING_COMMAND, make_opts(stdin="go\n")))
    shell = await wait_for_pid_file(tmp_path / _SHELL_PID_FILE)

    def interrupt() -> None:
        raise_signal(signal.SIGTERM)

    install_termination_cleanup(interrupt)
    install_termination_cleanup(exec_mod.kill_live_process_groups)

    code = raise_signal(signal.SIGINT)

    await wait_until_dead(shell, timeout_s=3.0)
    await task
    assert code == 128 + signal.SIGINT


@pytest.mark.usefixtures("roomy_escalation_grace")
async def test_exec_when_second_signal_arrives_before_kill_sweep_runs_does_let_leader_clean_up_first(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    raise_signal: Callable[[int], int],
) -> None:
    command = slow_cleanup_command(nested=False, cleanup_s=_ESCALATED_CLEANUP_S)
    task = asyncio.create_task(run_exec(command, make_opts()))
    await wait_for_pid_file(tmp_path / _CLEANER_PID_FILE)
    escalate_termination(raise_signal)

    await task

    assert (tmp_path / _CLEANED_MARKER).exists(), (
        "the group was killed before its leader cleaned up"
    )


@pytest.mark.usefixtures("roomy_escalation_grace")
async def test_exec_when_second_signal_arrives_before_kill_sweep_runs_does_let_nested_child_clean_up(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    raise_signal: Callable[[int], int],
) -> None:
    command = slow_cleanup_command(nested=True, cleanup_s=_ESCALATED_CLEANUP_S)
    task = asyncio.create_task(run_exec(command, make_opts()))
    await wait_for_pid_file(tmp_path / _CLEANER_PID_FILE)
    escalate_termination(raise_signal)

    await task

    assert (tmp_path / _CLEANED_MARKER).exists(), "the group was killed once its leader died"


# ---------------------------------------------------------------------------
# a nested run's stop and escalation graces halve per nesting level
# ---------------------------------------------------------------------------


@pytest.fixture
def group_wait_graces(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the grace each signal-path group wait is given, still waiting it out for real."""
    graces: list[float] = []
    real_wait = exec_mod.wait_for_process_group_exit

    def record(leaders: Iterable[int], timeout_s: float) -> None:
        graces.append(timeout_s)
        real_wait(leaders, timeout_s)

    monkeypatch.setattr(exec_mod, "wait_for_process_group_exit", record)
    return graces


@pytest.mark.parametrize(
    ("nesting_depth", "expected_grace_s"),
    [
        pytest.param(0, 0.2, id="top-level"),
        pytest.param(1, 0.1, id="nested-once"),
        pytest.param(2, 0.05, id="nested-twice"),
        pytest.param(3, 0.025, id="nested-three-times"),
    ],
)
async def test_exec_when_second_signal_arrives_in_nested_run_does_halve_grace_per_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    raise_signal: Callable[[int], int],
    group_wait_graces: list[float],
    *,
    nesting_depth: int,
    expected_grace_s: float,
) -> None:
    monkeypatch.setattr(exec_mod, "_NESTING_DEPTH", nesting_depth)
    task = asyncio.create_task(run_exec(_STARTED_TERM_IGNORING_COMMAND, make_opts(stdin="go\n")))
    await wait_for_pid_file(tmp_path / _SHELL_PID_FILE)

    def interrupt() -> None:
        raise_signal(signal.SIGTERM)

    install_termination_cleanup(interrupt)

    raise_signal(signal.SIGINT)

    await task
    assert group_wait_graces == [pytest.approx(expected_grace_s)]


@pytest.mark.parametrize(
    ("nesting_depth", "expected_grace_s"),
    [
        pytest.param(0, 1.0, id="top-level"),
        pytest.param(1, 0.5, id="nested-once"),
        pytest.param(2, 0.25, id="nested-twice"),
        pytest.param(3, 0.125, id="nested-three-times"),
    ],
)
async def test_kill_live_process_groups_when_run_is_nested_does_halve_grace_per_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    group_wait_graces: list[float],
    *,
    nesting_depth: int,
    expected_grace_s: float,
) -> None:
    monkeypatch.setattr(exec_mod, "_NESTING_DEPTH", nesting_depth)
    task = asyncio.create_task(run_exec(_STARTED_TERM_IGNORING_COMMAND, make_opts(stdin="go\n")))
    await wait_for_pid_file(tmp_path / _SHELL_PID_FILE)

    exec_mod.kill_live_process_groups()

    await task
    assert group_wait_graces == [pytest.approx(expected_grace_s)]


@pytest.fixture
def abort_wait_graces(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the grace each aborted run's stop is given, still waiting it out for real."""
    graces: list[float] = []
    real_wait = exec_mod._wait_for_exit

    async def record(proc: asyncio.subprocess.Process, grace_s: float) -> bool:
        graces.append(grace_s)
        return await real_wait(proc, grace_s)

    monkeypatch.setattr(exec_mod, "_wait_for_exit", record)
    return graces


@pytest.mark.parametrize(
    ("nesting_depth", "expected_grace_s"),
    [
        pytest.param(0, 1.0, id="top-level"),
        pytest.param(1, 0.5, id="nested-once"),
        pytest.param(2, 0.25, id="nested-twice"),
        pytest.param(3, 0.125, id="nested-three-times"),
    ],
)
async def test_exec_when_aborted_in_nested_run_does_halve_grace_per_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    abort_wait_graces: list[float],
    *,
    nesting_depth: int,
    expected_grace_s: float,
) -> None:
    monkeypatch.setattr(exec_mod, "_NESTING_DEPTH", nesting_depth)
    abort = asyncio.Event()
    options = make_opts(stdin="go\n", abort=abort)
    task = asyncio.create_task(run_exec(_STARTED_TERM_IGNORING_COMMAND, options))
    await wait_for_pid_file(tmp_path / _SHELL_PID_FILE)

    abort.set()

    await task
    assert abort_wait_graces == [pytest.approx(expected_grace_s)]


# ---------------------------------------------------------------------------
# stopping a group waits for every member, not just the leader
# ---------------------------------------------------------------------------


async def test_kill_live_process_groups_when_leader_dies_before_nested_child_does_let_child_clean_up(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    task = asyncio.create_task(run_exec(slow_cleanup_command(nested=True), make_opts()))
    await wait_for_pid_file(tmp_path / _CLEANER_PID_FILE)

    exec_mod.kill_live_process_groups()

    await task
    assert (tmp_path / _CLEANED_MARKER).exists(), "the group was killed once its leader died"


async def test_exec_when_aborted_and_leader_dies_before_nested_child_does_let_child_clean_up(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    abort = asyncio.Event()
    command = slow_cleanup_command(nested=True, release_stdio=True)
    task = asyncio.create_task(run_exec(command, make_opts(abort=abort)))
    await wait_for_pid_file(tmp_path / _CLEANER_PID_FILE)

    abort.set()

    await task
    assert (tmp_path / _CLEANED_MARKER).exists(), "the group was killed once its leader died"


async def test_exec_when_timed_out_and_leader_dies_before_nested_child_does_let_child_clean_up(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
) -> None:
    command = slow_cleanup_command(nested=True, release_stdio=True)

    await run_exec(command, make_opts(timeout_ms=_TIMEOUT_AFTER_START_MS))

    assert (tmp_path / _CLEANED_MARKER).exists(), "the group was killed once its leader died"


# ---------------------------------------------------------------------------
# a nested gymrat run finishes its own bench sweep before it is killed
# ---------------------------------------------------------------------------

# A real nested gymrat run: it installs the termination cleanup, then runs the
# bench named in ``argv[1]`` through ``exec``, which puts the bench in a session
# of its own. Killing this run's group therefore never reaches the bench: only
# the run's own sweep does.
_NESTED_RUN_SCRIPT = """
import asyncio, os, sys

from gymrat.exec import ExecOptions, exec, kill_live_process_groups
from gymrat.signals import install_termination_cleanup

install_termination_cleanup(kill_live_process_groups)
asyncio.run(exec(sys.argv[1], ExecOptions(cwd=os.getcwd(), stdin="go\\n")))
"""

# Upper bound for the nested run to start an interpreter, import gymrat and get
# its bench going.
_NESTED_START_TIMEOUT_S = 20.0

# Upper bound for a stopped outer run to settle, including its grace.
_NESTED_SETTLE_TIMEOUT_S = 10.0

# How long a bench killed by the nested run's sweep may take to show as dead
# once the outer run has settled: a kill lands in milliseconds, while a bench
# the sweep never reached sleeps on for half a minute.
_KILLED_BENCH_SETTLE_S = 0.5


def nested_run_command() -> str:
    """Return a shell command running a nested gymrat whose bench ignores SIGTERM.

    Returns:
        The command, for ``exec`` to run under ``sh -c`` with the nested run as
        the group's leader.
    """
    nested_run = shlex.join([
        sys.executable,
        "-c",
        _NESTED_RUN_SCRIPT,
        _STARTED_TERM_IGNORING_COMMAND,
    ])
    return f"exec {nested_run}"


async def test_kill_live_process_groups_when_nested_run_sweeps_term_ignoring_bench_does_leave_bench_dead(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    reap_groups: list[int],
) -> None:
    task = asyncio.create_task(run_exec(nested_run_command(), make_opts()))
    bench = await wait_for_pid_file(tmp_path / _SHELL_PID_FILE, _NESTED_START_TIMEOUT_S)
    reap_groups.append(bench)

    exec_mod.kill_live_process_groups()

    await asyncio.wait_for(task, _NESTED_SETTLE_TIMEOUT_S)
    await wait_until_dead(bench, timeout_s=_KILLED_BENCH_SETTLE_S)


async def test_exec_when_aborted_and_nested_run_sweeps_term_ignoring_bench_does_leave_bench_dead(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    reap_groups: list[int],
) -> None:
    abort = asyncio.Event()
    task = asyncio.create_task(run_exec(nested_run_command(), make_opts(abort=abort)))
    bench = await wait_for_pid_file(tmp_path / _SHELL_PID_FILE, _NESTED_START_TIMEOUT_S)
    reap_groups.append(bench)

    abort.set()

    await asyncio.wait_for(task, _NESTED_SETTLE_TIMEOUT_S)
    await wait_until_dead(bench, timeout_s=_KILLED_BENCH_SETTLE_S)
