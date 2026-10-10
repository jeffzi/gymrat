"""End-to-end signal-cleanup hardening tests over real subprocesses.

Each test runs the CLI out of process against a throwaway repo and a real shell
bench, then delivers a real signal while the bench is mid-run. Together they pin
the guarantees that must outlive an interrupted run:

- no bench process — nor a grandchild it spawned — survives the CLI,
- a lock stranded by a hard kill never wedges the next run,
- the terminal status line is cleared on a TTY and left untouched off one,
- every worktree is swept, even when a second signal lands during cleanup,
- a signal that lands during the normal sweep never starts a second sweep,
- and still reports the worktree that sweep had failed to remove.

The whole module is POSIX-only: it relies on real signals, ``sh`` bench scripts,
and process-group tree-kill. Every test that spawns a real tree registers its
group with the ``reap_groups`` fixture so no orphan bench survives a failed
assertion.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests._cli import no_color_env as _env
from tests._cli import run_cli
from tests._git import (
    BENCH_ONCE_FLAGS,
    EMIT_ONE_BENCH,
    list_worktree_dirs,
    register_absent_worktree,
    wait_for_worktrees,
)
from tests._git import write_committed_bench as _write_committed_bench
from tests._process_helpers import poll_until_blocking
from tests._process_helpers import read_pid_file as _read_pid_file
from tests._process_helpers import (
    wait_for_pid_file_blocking as _wait_for_pid_file_blocking,
)
from tests._process_helpers import (
    wait_until_dead_blocking as _wait_until_dead_blocking,
)
from tests._rich import cursor_hidden, screen_lines
from tests.cli._signalled_cli import (
    SETTLE_TIMEOUT_S,
    pid_recording_script,
    spawned_gymrat,
    stop_by_signal,
)
from tests.hardening._pty import pty_capture

if TYPE_CHECKING:
    import subprocess
    from collections.abc import Callable, Generator


# Fixed pty dimensions for the pyte screen replay. The slave pty is sized to
# these values via TIOCSWINSZ so Rich in the child renders at a known geometry.
_PTY_WIDTH = 80
_PTY_HEIGHT = 24

# A bench that records its own process-group leader pid and a background
# grandchild pid, emits one metric, then blocks forever on the grandchild. The
# sample never completes on its own, so the run is always mid-bench when signalled.
_TRACKED_BENCH = pid_recording_script(
    "bench.pid", "sleep 120 &\necho $! > grandchild.pid\necho 'METRIC x=1'\nwait\n"
)


#: The CLI invocation that runs the tracked bench once, in place.
_MEASURE_ONCE = ["measure", *BENCH_ONCE_FLAGS]


@contextlib.contextmanager
def _cli_mid_bench(
    repo: str, reap_groups: list[int], **popen_kwargs: Any
) -> Generator[tuple[subprocess.Popen[Any], int]]:
    """Run ``measure`` over the tracked bench and yield once the bench is running.

    The bench's group is registered for reaping, and the CLI is killed and reaped
    on the way out if the test left it running.

    Args:
        repo: The scratch repository the bench is committed into.
        reap_groups: The fixture list the bench's process group is added to.
        **popen_kwargs: Overrides for the default piped, text-mode ``Popen``.

    Yields:
        The CLI process and the bench's process-group leader pid.
    """
    _write_committed_bench(repo, _TRACKED_BENCH)
    with spawned_gymrat(_MEASURE_ONCE, repo, **popen_kwargs) as proc:
        bench_pid = _wait_for_pid_file_blocking(
            Path(repo) / "bench.pid", timeout_s=SETTLE_TIMEOUT_S
        )
        reap_groups.append(bench_pid)
        yield proc, bench_pid


# ---------------------------------------------------------------------------
# bench process tree does not outlive the CLI
# ---------------------------------------------------------------------------


def test_measure_when_signalled_mid_bench_does_kill_the_bench_tree(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
):
    repo = create_scratch_repo()

    with _cli_mid_bench(repo, reap_groups) as (proc, bench_pid):
        grandchild = _wait_for_pid_file_blocking(
            Path(repo) / "grandchild.pid", timeout_s=SETTLE_TIMEOUT_S
        )

        stop_by_signal(proc, signal.SIGTERM)

    assert proc.returncode == 128 + signal.SIGTERM
    _wait_until_dead_blocking(grandchild, timeout_s=SETTLE_TIMEOUT_S)
    # Polled rather than checked once: a SIGKILLed leader stays visible to
    # ``os.kill(pid, 0)`` as a zombie until its parent reaps it.
    _wait_until_dead_blocking(bench_pid, timeout_s=SETTLE_TIMEOUT_S)


# ---------------------------------------------------------------------------
# a stranded lock never wedges the next run
# ---------------------------------------------------------------------------


def test_measure_when_prior_run_hard_killed_does_take_over_stale_lock_on_rerun(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
):
    repo = create_scratch_repo()

    with _cli_mid_bench(repo, reap_groups) as (first, bench_pid):
        first.kill()  # SIGKILL runs no cleanup, so the lock is left behind
        first.communicate(timeout=30)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(bench_pid), signal.SIGKILL)
    _wait_until_dead_blocking(bench_pid, timeout_s=SETTLE_TIMEOUT_S)

    (Path(repo) / "bench.sh").write_text(EMIT_ONE_BENCH, encoding="utf-8")

    rerun = run_cli(
        ["measure", "--bench", "sh bench.sh", "--samples", "2", "--format", "json"],
        repo,
        check=False,
        timeout=60,
    )

    assert rerun.returncode == 0, rerun.stderr
    assert list(json.loads(rerun.stdout)["metrics"]) == ["x"]


# ---------------------------------------------------------------------------
# the status line is cleared on a TTY and untouched off one
# ---------------------------------------------------------------------------


def _wait_for_drawn(chunks: list[bytes], marker: bytes) -> None:
    """Block until the pty output gathered so far contains ``marker``.

    Args:
        chunks: The pty output a reader thread is appending to.
        marker: The bytes that show the awaited draw has landed.

    Raises:
        AssertionError: ``marker`` has not drawn within ``SETTLE_TIMEOUT_S``.
    """
    poll_until_blocking(
        lambda: marker in b"".join(chunks),
        SETTLE_TIMEOUT_S,
        lambda: AssertionError(f"{marker!r} never drew within {SETTLE_TIMEOUT_S:g} s: {chunks!r}"),
    )


def test_measure_when_signalled_off_a_tty_does_not_clear_a_line(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
):
    repo = create_scratch_repo()

    with _cli_mid_bench(repo, reap_groups) as (proc, _bench_pid):
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=30)

    assert proc.returncode == 130
    # Off a TTY the status line is never mounted, so no control sequence of any
    # kind (an erase, a cursor move, a cursor show) reaches either stream.
    assert "\x1b[" not in stdout + stderr


def test_measure_when_signalled_on_a_tty_does_clear_the_status_line(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
):
    repo = create_scratch_repo()

    # A known window size, so Rich renders at _PTY_WIDTH x _PTY_HEIGHT instead
    # of falling back on its defaults from a 0x0 pty.
    with (
        pty_capture(size=(_PTY_HEIGHT, _PTY_WIDTH)) as terminal,
        _cli_mid_bench(
            repo,
            reap_groups,
            stdin=terminal.slave,
            stdout=terminal.slave,
            stderr=terminal.slave,
            text=False,
            start_new_session=True,
            close_fds=True,
        ) as (proc, _bench_pid),
    ):
        _wait_for_drawn(terminal.chunks, b"sampling")

        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
    output = terminal.output

    assert proc.returncode == 130
    # The pty stream replayed at the pty's own geometry: the status line drew
    # (the wait above), so a cleared screen with the cursor shown is the erase.
    assert (
        screen_lines(output, width=_PTY_WIDTH, height=_PTY_HEIGHT),
        cursor_hidden(output, width=_PTY_WIDTH, height=_PTY_HEIGHT),
    ) == ([], False), output


# ---------------------------------------------------------------------------
# every worktree is swept on a signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("signal_number", [signal.SIGINT, signal.SIGHUP], ids=["SIGINT", "SIGHUP"])
def test_compare_when_signalled_with_many_worktrees_does_sweep_all_of_them(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
    signal_number: signal.Signals,
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _TRACKED_BENCH, branches=("candidate-one", "candidate-two"))

    argv = ["compare", "main", "candidate-one", "candidate-two", *BENCH_ONCE_FLAGS]

    with spawned_gymrat(argv, repo) as proc:
        for wt in wait_for_worktrees(repo, 2):
            pid = _read_pid_file(Path(wt) / "bench.pid")
            if pid is not None:
                reap_groups.append(pid)

        stop_by_signal(proc, signal_number)

    assert proc.returncode == 128 + signal_number
    assert list_worktree_dirs(repo, include_main=False) == []


# ---------------------------------------------------------------------------
# a signal during the normal sweep never starts a second one
# ---------------------------------------------------------------------------

# A stand-in ``git`` that appends each call's arguments to a log, runs the real
# git, and sends its parent one SIGTERM as the first ``worktree remove`` ends.
# The signal leaves before the stand-in exits, so it always lands while the CLI
# is still inside that git call: no sleep or poll decides the timing. The
# ``mkdir`` succeeds once, which keeps a later removal from signalling again.
_SIGNALLING_GIT = """#!/bin/sh
printf '%s\\n' "$*" >> "{log}"
"{git}" "$@"
status=$?
if printf '%s' " $* " | grep -q " worktree remove "; then
    mkdir "{sent}" 2>/dev/null && kill -TERM "$PPID"
fi
exit $status
"""


# The same stand-in with one more turn: the first ``worktree remove`` is refused
# without reaching the real git, so its worktree stays on disk, and the signal
# leaves as the second one ends. Each ``mkdir`` succeeds once, which is what
# makes "first" and "second" hold whatever else git is asked.
_REFUSING_THEN_SIGNALLING_GIT = """#!/bin/sh
printf '%s\\n' "$*" >> "{log}"
if printf '%s' " $* " | grep -q " worktree remove "; then
    if mkdir "{refused}" 2>/dev/null; then
        echo "fatal: the stand-in refuses this removal" >&2
        exit 1
    fi
    "{git}" "$@"
    status=$?
    mkdir "{sent}" 2>/dev/null && kill -TERM "$PPID"
    exit $status
fi
exec "{git}" "$@"
"""


def _env_with_path(directory: Path) -> dict[str, str]:
    """The color-free test environment with ``directory`` first on ``PATH``.

    Args:
        directory: The directory holding a stand-in ``git``.

    Returns:
        The environment a CLI run under the stand-in is started with.
    """
    env = _env()
    env["PATH"] = f"{directory}{os.pathsep}{env['PATH']}"
    return env


def _install_signalling_git(directory: Path, script: str = _SIGNALLING_GIT) -> Path:
    """Write a stand-in ``git`` into a directory.

    Args:
        directory: Where the stand-in is written; put it first on ``PATH``.
        script: The stand-in's text, with ``log``, ``git``, ``sent``,
            ``refused``, ``held``, ``entered`` and ``release`` placeholders.

    Returns:
        The log the stand-in appends to, one git call per line.
    """
    real_git = shutil.which("git")
    assert real_git is not None
    log = directory / "git-calls.log"
    stand_in = directory / "git"
    stand_in.write_text(
        script.format(
            log=log,
            git=real_git,
            sent=directory / "signal-sent",
            refused=directory / "removal-refused",
            held=directory / "removal-held",
            entered=directory / "removal-entered",
            release=directory / "removal-released",
        ),
        encoding="utf-8",
    )
    stand_in.chmod(0o755)
    return log


def test_compare_when_signalled_during_the_normal_sweep_does_remove_each_worktree_once_without_pruning(
    create_scratch_repo: Callable[[], str],
    tmp_path: Path,
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, EMIT_ONE_BENCH, branches=("candidate",))
    absent = register_absent_worktree(repo)
    git_log = _install_signalling_git(tmp_path)
    env = _env_with_path(tmp_path)

    # ``timeout=60`` is the hang guard: ``run`` kills and reaps the CLI on expiry, so a
    # switch to ``Popen`` needs its own kill-and-reap.
    proc = run_cli(
        ["compare", "main", "candidate", *BENCH_ONCE_FLAGS],
        repo,
        check=False,
        timeout=60,
        env=env,
    )

    calls = git_log.read_text(encoding="utf-8").splitlines()
    added = [call for call in calls if " worktree add " in f" {call} "]
    removed = [call for call in calls if " worktree remove " in f" {call} "]
    assert proc.returncode == 128 + signal.SIGTERM, proc.stderr
    assert len(set(removed)) == len(removed) == len(added)
    assert list_worktree_dirs(repo, include_main=False) == [absent]


def test_compare_when_signalled_after_the_normal_sweep_left_a_worktree_does_name_it_without_pruning(
    create_scratch_repo: Callable[[], str],
    tmp_path: Path,
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, EMIT_ONE_BENCH, branches=("candidate-one", "candidate-two"))
    absent = register_absent_worktree(repo)
    git_log = _install_signalling_git(tmp_path, _REFUSING_THEN_SIGNALLING_GIT)
    env = _env_with_path(tmp_path)

    # ``timeout=60`` is the hang guard: ``run`` kills and reaps the CLI on expiry, so a
    # switch to ``Popen`` needs its own kill-and-reap.
    proc = run_cli(
        ["compare", "main", "candidate-one", "candidate-two", *BENCH_ONCE_FLAGS],
        repo,
        check=False,
        timeout=60,
        env=env,
    )

    calls = git_log.read_text(encoding="utf-8").splitlines()
    removed = [call for call in calls if " worktree remove " in f" {call} "]
    assert proc.returncode == 128 + signal.SIGTERM, proc.stderr
    assert "cleanup did not finish:" in proc.stderr
    assert removed[0].split()[-1] in proc.stderr
    assert absent in list_worktree_dirs(repo, include_main=False)


# ---------------------------------------------------------------------------
# a second signal during the cleanup sweep exits without sweeping again
# ---------------------------------------------------------------------------

# A stand-in ``git`` whose first ``worktree remove`` records its pid and then
# holds until the test drops the release file. gymrat masks termination signals
# for the length of each git call, so a second signal sent while the removal is
# held stays pending until it returns; the test thereby knows the second signal
# lands mid-sweep, after one removal and before the next. The hold is bounded so
# an orphaned stand-in never outlives a failed test by long.
_HOLDING_GIT = """#!/bin/sh
printf '%s\\n' "$*" >> "{log}"
if printf '%s' " $* " | grep -q " worktree remove "; then
    if mkdir "{held}" 2>/dev/null; then
        echo $$ > "{entered}"
        tries=0
        while [ ! -e "{release}" ] && [ "$tries" -lt 600 ]; do
            sleep 0.05
            tries=$((tries + 1))
        done
    fi
fi
exec "{git}" "$@"
"""


def test_compare_when_signalled_again_during_the_cleanup_sweep_does_stop_after_the_removal_in_flight(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
    tmp_path: Path,
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _TRACKED_BENCH, branches=("candidate-one", "candidate-two"))
    absent = register_absent_worktree(repo)
    git_log = _install_signalling_git(tmp_path, _HOLDING_GIT)
    release = tmp_path / "removal-released"
    argv = ["compare", "main", "candidate-one", "candidate-two", *BENCH_ONCE_FLAGS]

    try:
        with spawned_gymrat(argv, repo, env=_env_with_path(tmp_path)) as proc:
            # Three: the absent user worktree plus two the run added.
            for wt in wait_for_worktrees(repo, 3):
                pid = _read_pid_file(Path(wt) / "bench.pid")
                if pid is not None:
                    reap_groups.append(pid)
            proc.send_signal(signal.SIGINT)
            _wait_for_pid_file_blocking(tmp_path / "removal-entered", timeout_s=SETTLE_TIMEOUT_S)

            proc.send_signal(signal.SIGINT)
            release.touch()
            proc.communicate(timeout=30)
    finally:
        release.touch()

    calls = git_log.read_text(encoding="utf-8").splitlines()
    assert proc.returncode == 130
    assert len([call for call in calls if " worktree remove " in f" {call} "]) == 1
    assert absent in list_worktree_dirs(repo, include_main=False)
