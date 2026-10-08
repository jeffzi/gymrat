"""Behavioral tests for stopping ``exec``'s process groups on a termination signal.

A second signal kills every live group at once, after a shortened grace that a
nested gymrat run spends finishing its own teardown; stopping a group waits for
every running member, not just its leader; a nested gymrat run gets to finish
its own bench sweep before the run above it kills it.
"""

import asyncio
import dataclasses
import shlex
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat import exec as exec_mod
from gymrat import process_group
from gymrat.exec import ExecOptions
from gymrat.exec import exec as run_exec
from gymrat.signals import install_termination_cleanup
from tests._exec_fixtures import ExecTask, Teardown, leave_to_timeout, set_abort
from tests._process_helpers import wait_for_pid_file, wait_until_dead

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
    """Deliver a second termination signal before the live-group kill sweep has run.

    Args:
        raise_signal: Runs the installed handler for a signal and reports the exit code.
    """

    def interrupt() -> None:
        raise_signal(signal.SIGTERM)

    install_termination_cleanup(interrupt)
    install_termination_cleanup(exec_mod.kill_live_process_groups)
    raise_signal(signal.SIGINT)


@pytest.mark.usefixtures("roomy_escalation_grace")
@pytest.mark.parametrize("nested", [False, True], ids=["leader", "nested-child"])
async def test_exec_when_second_signal_arrives_before_kill_sweep_runs_does_let_the_cleaner_finish(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    raise_signal: Callable[[int], int],
    *,
    nested: bool,
) -> None:
    command = slow_cleanup_command(nested=nested, cleanup_s=_ESCALATED_CLEANUP_S)
    task = asyncio.create_task(run_exec(command, make_opts()))
    await wait_for_pid_file(tmp_path / _CLEANER_PID_FILE)

    escalate_termination(raise_signal)

    await task
    assert (tmp_path / _CLEANED_MARKER).exists(), "the group was killed before the cleaner finished"


# ---------------------------------------------------------------------------
# a nested run's stop and escalation graces halve per nesting level
# ---------------------------------------------------------------------------


# How far past its grace a group wait may run on the fake clock: one poll of the
# group, which the wait sleeps between checks.
_GROUP_POLL_S = 0.01


@dataclasses.dataclass
class FakeClock:
    """Stand-in for the ``time`` module the process-group waits read, advancing only when slept."""

    now: float = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def group_wait_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Run the signal path's group wait on a fake clock, so a grace is waited out without sleeping."""
    clock = FakeClock()
    monkeypatch.setattr(process_group, "time", clock)
    return clock


@pytest.mark.parametrize(
    ("nesting_depth", "expected_grace_s"),
    [
        pytest.param(0, 0.2, id="top-level"),
        pytest.param(1, 0.1, id="nested-once"),
        pytest.param(2, 0.05, id="nested-twice"),
        pytest.param(3, 0.025, id="nested-three-times"),
    ],
)
@pytest.mark.usefixtures("spawned_processes")
async def test_exec_when_second_signal_arrives_in_nested_run_does_kill_live_group_after_halved_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    raise_signal: Callable[[int], int],
    group_wait_clock: FakeClock,
    *,
    nesting_depth: int,
    expected_grace_s: float,
) -> None:
    monkeypatch.setattr(exec_mod, "_NESTING_DEPTH", nesting_depth)
    task = asyncio.create_task(run_exec(_STARTED_TERM_IGNORING_COMMAND, make_opts(stdin="go\n")))
    shell = await wait_for_pid_file(tmp_path / _SHELL_PID_FILE)

    escalate_termination(raise_signal)
    await task

    await wait_until_dead(shell, timeout_s=3.0)
    assert group_wait_clock.now == pytest.approx(expected_grace_s, abs=_GROUP_POLL_S)


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
    group_wait_clock: FakeClock,
    *,
    nesting_depth: int,
    expected_grace_s: float,
) -> None:
    monkeypatch.setattr(exec_mod, "_NESTING_DEPTH", nesting_depth)
    task = asyncio.create_task(run_exec(_STARTED_TERM_IGNORING_COMMAND, make_opts(stdin="go\n")))
    await wait_for_pid_file(tmp_path / _SHELL_PID_FILE)

    exec_mod.kill_live_process_groups()

    await task
    assert group_wait_clock.now == pytest.approx(expected_grace_s, abs=_GROUP_POLL_S)


# asyncio may fire a timer up to a clock tick early, so a wait can end a hair
# before its grace by the test's own clock.
_TIMER_SLACK_S = 0.01

# The stop grace the abort-timing test runs with. At the real 1 s, the deepest
# level's 0.25 s grace leaves the kill, pipe close and reap a quarter second
# before the bound, which a loaded CI runner's scheduling delays can use up.
_ROOMY_TERMINATE_GRACE_S = 2.0


@pytest.fixture
def roomy_terminate_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give a stopped run's grace room for a loaded runner's delays at every nesting level."""
    monkeypatch.setattr(exec_mod, "TERMINATE_GRACE_S", _ROOMY_TERMINATE_GRACE_S)


@pytest.mark.parametrize(
    ("nesting_depth", "expected_grace_s"),
    [
        pytest.param(0, 2.0, id="top-level"),
        pytest.param(1, 1.0, id="nested-once"),
        pytest.param(2, 0.5, id="nested-twice"),
    ],
)
@pytest.mark.usefixtures("roomy_terminate_grace")
async def test_exec_when_aborted_in_nested_run_does_halve_grace_per_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    *,
    nesting_depth: int,
    expected_grace_s: float,
) -> None:
    monkeypatch.setattr(exec_mod, "_NESTING_DEPTH", nesting_depth)
    abort = asyncio.Event()
    options = make_opts(stdin="go\n", abort=abort)
    task = asyncio.create_task(run_exec(_STARTED_TERM_IGNORING_COMMAND, options))
    await wait_for_pid_file(tmp_path / _SHELL_PID_FILE)
    started = time.monotonic()

    abort.set()
    await task
    elapsed = time.monotonic() - started

    # The group ignores the polite request, so it stands for the whole grace;
    # a grace twice as long, the level above's, would outlast the bound.
    assert expected_grace_s - _TIMER_SLACK_S <= elapsed < 2 * expected_grace_s


# ---------------------------------------------------------------------------
# stopping a group waits for every member, not just the leader
# ---------------------------------------------------------------------------


def _sweep_live_groups(
    task: ExecTask, proc: asyncio.subprocess.Process | None, abort: asyncio.Event
) -> None:
    """Stop the run through the live-group kill sweep a termination signal runs."""
    exec_mod.kill_live_process_groups()


@pytest.mark.parametrize(
    "teardown",
    [
        pytest.param(Teardown(None, _sweep_live_groups), id="kill-sweep"),
        pytest.param(Teardown(None, set_abort), id="abort"),
        pytest.param(Teardown(_TIMEOUT_AFTER_START_MS, leave_to_timeout), id="timeout"),
    ],
)
async def test_exec_when_leader_dies_before_nested_child_does_let_child_clean_up(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    teardown: Teardown,
) -> None:
    abort = asyncio.Event()
    command = slow_cleanup_command(nested=True, release_stdio=True)
    options = make_opts(abort=abort, timeout_ms=teardown.timeout_ms)
    task = asyncio.create_task(run_exec(command, options))
    await wait_for_pid_file(tmp_path / _CLEANER_PID_FILE)

    teardown.trigger(task, None, abort)

    await task
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


@pytest.mark.parametrize(
    "teardown",
    [
        pytest.param(Teardown(None, _sweep_live_groups), id="kill-sweep"),
        pytest.param(Teardown(None, set_abort), id="abort"),
    ],
)
async def test_exec_when_nested_run_sweeps_term_ignoring_bench_does_leave_bench_dead(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    reap_groups: list[int],
    teardown: Teardown,
) -> None:
    abort = asyncio.Event()
    task = asyncio.create_task(run_exec(nested_run_command(), make_opts(abort=abort)))
    bench = await wait_for_pid_file(tmp_path / _SHELL_PID_FILE, _NESTED_START_TIMEOUT_S)
    reap_groups.append(bench)

    teardown.trigger(task, None, abort)

    await asyncio.wait_for(task, _NESTED_SETTLE_TIMEOUT_S)
    await wait_until_dead(bench, timeout_s=_KILLED_BENCH_SETTLE_S)
