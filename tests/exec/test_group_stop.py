"""Behavioral tests for stopping ``exec``'s process groups on a termination signal.

A second signal kills every live group at once, after a shortened grace that a
nested gymrat run spends finishing its own teardown; stopping a group waits for
every running member, not just its leader.
"""

import asyncio
import shlex
import signal
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat import exec as exec_mod
from gymrat import process_group
from gymrat.exec import ExecOptions
from gymrat.exec import exec as run_exec
from gymrat.signals import install_termination_cleanup
from tests._clock import WaitClock, install_wait_clock, within_hang_guard
from tests._exec_fixtures import wait_for_spawned
from tests._process_helpers import wait_for_pid_file, wait_until_dead

# ---------------------------------------------------------------------------
# a second termination signal kills every live group at once
# ---------------------------------------------------------------------------

_SHELL_PID_FILE = "shell.pid"

# Ignores SIGTERM, so only a SIGKILL ends the group. It holds back its pid until
# exec feeds its stdin: exec does that only once the spawn has returned, so a
# signal raised after the pid appears is never deferred by a spawn still in
# progress.
#
# The shell replaces itself with its sleep instead of forking it, so the group
# is one process once the pid appears and the stop it receives acts on exactly
# the process whose pid the test read.
_STARTED_TERM_IGNORING_COMMAND = (
    f"trap '' TERM; read -r _; echo $$ > {_SHELL_PID_FILE}; exec sleep 30"
)


@pytest.fixture(autouse=True)
def _top_level_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as a top-level gymrat, whose escalation grace the cleanup times are sized against."""
    monkeypatch.setattr(exec_mod, "NESTING_DEPTH", 0)


# How long the cleaner takes to clean up once asked to stop: well past the few
# milliseconds a kill lands in, well inside the grace the stop tests run with.
_CLEANUP_S = 0.5

# The same for a stop escalated by a second signal: well inside the escalation
# grace those tests run with.
_ESCALATED_CLEANUP_S = 0.1

# The escalation grace the escalated-cleanup tests run with. The real 0.2 s at
# the top nesting level leaves the cleaner only 0.1 s to spare, which a loaded
# CI runner's scheduling delays can use up.
_ROOMY_ESCALATION_GRACE_S = 1.0


@pytest.fixture
def roomy_escalation_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give a second signal's kill sweep a grace with room for a loaded runner's delays."""
    monkeypatch.setattr(exec_mod, "ESCALATION_GRACE_S", _ROOMY_ESCALATION_GRACE_S)


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
async def test_exec_when_second_signal_arrives_before_kill_sweep_runs_does_let_the_cleaner_finish(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    raise_signal: Callable[[int], int],
) -> None:
    command = slow_cleanup_command(nested=False, cleanup_s=_ESCALATED_CLEANUP_S)
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

# Real seconds a stop on the fake clock may take: a few seconds above the
# spawn and kill it does on real time, so a sweep that waited its grace on
# real time instead fails rather than passing slowly.
_HANG_GUARD_S = 5.0


# The unscaled grace the halving tests patch in, so the expected waits follow
# from it rather than from a copy of the real constant.
_BASE_GRACE_S = 0.8


@pytest.fixture
def group_wait_clock(monkeypatch: pytest.MonkeyPatch) -> WaitClock:
    """Run every process-group grace wait on a fake clock, so a grace is waited out without sleeping."""
    # The kill's settle loop polls the same clock, and a real SIGKILL lands in
    # real time while a fake sleep returns at once, so the loop would run the
    # fake clock past the grace these tests measure.
    monkeypatch.setattr(process_group, "KILL_SETTLE_S", 0.0)
    return install_wait_clock(monkeypatch, process_group)


# Two levels down: one level is the only depth where halving once per level
# and halving once in all agree, so it cannot tell them apart.
_NESTING_DEPTH = 2
_NESTED_GRACE_S = _BASE_GRACE_S / 4


@pytest.mark.usefixtures("spawned_processes")
async def test_exec_when_second_signal_arrives_in_nested_run_does_kill_live_group_after_grace_halved_per_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    raise_signal: Callable[[int], int],
    group_wait_clock: WaitClock,
) -> None:
    monkeypatch.setattr(exec_mod, "ESCALATION_GRACE_S", _BASE_GRACE_S)
    monkeypatch.setattr(exec_mod, "NESTING_DEPTH", _NESTING_DEPTH)
    task = asyncio.create_task(run_exec(_STARTED_TERM_IGNORING_COMMAND, make_opts(stdin="go\n")))
    shell = await wait_for_pid_file(tmp_path / _SHELL_PID_FILE)

    with within_hang_guard(_HANG_GUARD_S):
        escalate_termination(raise_signal)
        await task

    await wait_until_dead(shell, timeout_s=3.0)
    assert group_wait_clock.now == pytest.approx(_NESTED_GRACE_S, abs=_GROUP_POLL_S)


async def test_kill_live_process_groups_when_run_is_nested_does_halve_grace_per_level(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    group_wait_clock: WaitClock,
) -> None:
    monkeypatch.setattr(exec_mod, "TERMINATE_GRACE_S", _BASE_GRACE_S)
    monkeypatch.setattr(exec_mod, "NESTING_DEPTH", _NESTING_DEPTH)
    task = asyncio.create_task(run_exec(_STARTED_TERM_IGNORING_COMMAND, make_opts(stdin="go\n")))
    await wait_for_pid_file(tmp_path / _SHELL_PID_FILE)

    with within_hang_guard(_HANG_GUARD_S):
        exec_mod.kill_live_process_groups()

    await task
    assert group_wait_clock.now == pytest.approx(_NESTED_GRACE_S, abs=_GROUP_POLL_S)


# The stop grace the stop-timing tests run with. The real 1 s leaves the kill,
# pipe close and reap, or a cleaner's half-second cleanup, only half a second
# to spare, which a loaded CI runner's scheduling delays can use up.
_ROOMY_TERMINATE_GRACE_S = 2.0


@pytest.fixture
def roomy_terminate_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give a stopped run's grace room for a loaded runner's delays."""
    monkeypatch.setattr(exec_mod, "TERMINATE_GRACE_S", _ROOMY_TERMINATE_GRACE_S)


# The abort-timing test's unscaled grace and nesting depth: two levels scale the
# grace to a quarter, which a wait one level short or unscaled would overshoot.
_ABORT_TERMINATE_GRACE_S = 1.0
_ABORT_NESTING_DEPTH = 2
_ABORT_NESTED_GRACE_S = 0.25


@pytest.mark.usefixtures("spawned_processes")
async def test_exec_when_aborted_in_nested_run_does_wait_the_nested_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_opts: Callable[..., ExecOptions],
    group_wait_clock: WaitClock,
) -> None:
    monkeypatch.setattr(exec_mod, "TERMINATE_GRACE_S", _ABORT_TERMINATE_GRACE_S)
    monkeypatch.setattr(exec_mod, "NESTING_DEPTH", _ABORT_NESTING_DEPTH)
    abort = asyncio.Event()
    options = make_opts(stdin="go\n", abort=abort)
    task = asyncio.create_task(run_exec(_STARTED_TERM_IGNORING_COMMAND, options))
    await wait_for_pid_file(tmp_path / _SHELL_PID_FILE)

    with within_hang_guard(_HANG_GUARD_S):
        abort.set()
        await task

    assert group_wait_clock.now == pytest.approx(_ABORT_NESTED_GRACE_S, abs=_GROUP_POLL_S)


# ---------------------------------------------------------------------------
# stopping a group waits for every member, not just the leader
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("roomy_terminate_grace")
async def test_exec_when_leader_dies_before_nested_child_does_let_child_clean_up(
    tmp_path: Path,
    make_opts: Callable[..., ExecOptions],
    spawned_processes: list[asyncio.subprocess.Process],
) -> None:
    abort = asyncio.Event()
    command = slow_cleanup_command(nested=True, release_stdio=True)
    task = asyncio.create_task(run_exec(command, make_opts(abort=abort)))
    await wait_for_spawned(spawned_processes)
    await wait_for_pid_file(tmp_path / _CLEANER_PID_FILE)

    abort.set()

    await task
    assert (tmp_path / _CLEANED_MARKER).exists(), "the group was killed once its leader died"
