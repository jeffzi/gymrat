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
import os
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

if sys.platform != "win32":
    import fcntl
    import pty
    import termios

from tests._cli import ENTRY as _ENTRY
from tests._cli import no_color_env as _env
from tests._git import EMIT_ONE_BENCH
from tests._git import run_git as _git
from tests._git import write_committed_bench as _write_committed_bench
from tests._process_helpers import read_pid_file as _read_pid_file
from tests._process_helpers import (
    wait_for_pid_file_blocking as _wait_for_pid_file_blocking,
)
from tests._process_helpers import (
    wait_until_dead_blocking as _wait_until_dead_blocking,
)
from tests._rich import screen_lines
from tests.conftest import list_worktree_dirs, register_absent_worktree, wait_for_worktrees
from tests.hardening._bench_helpers import drain as _drain

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")

# The ANSI sequence that makes the cursor visible again; the live status line
# hides it while it draws.
_SHOW_CURSOR = "\x1b[?25h"

# Fixed pty dimensions for the pyte screen replay. The slave pty is sized to
# these values via TIOCSWINSZ so Rich in the child renders at a known geometry.
_PTY_WIDTH = 80
_PTY_HEIGHT = 24

# Budget for every pid-file and death-wait poll below.
_SETTLE_TIMEOUT_S = 30.0

# A bench that records its own process-group leader pid and a background
# grandchild pid, emits one metric, then blocks forever on the grandchild. The
# sample never completes on its own, so the run is always mid-bench when signalled.
_TRACKED_BENCH = """#!/bin/sh
echo $$ > bench.pid
sleep 120 &
echo $! > grandchild.pid
echo 'METRIC x=1'
wait
"""


# ---------------------------------------------------------------------------
# bench process tree does not outlive the CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("signal_number", "expected_code"),
    [
        pytest.param(signal.SIGINT, 130, id="sigint"),
        pytest.param(signal.SIGTERM, 143, id="sigterm"),
    ],
)
def test_measure_when_signalled_mid_bench_does_kill_bench_grandchild_before_exit(
    signal_number: int,
    expected_code: int,
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _TRACKED_BENCH)

    proc = subprocess.Popen(  # noqa: S603
        [*_ENTRY, "measure", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        bench_pid = _wait_for_pid_file_blocking(
            Path(repo) / "bench.pid", timeout_s=_SETTLE_TIMEOUT_S
        )
        reap_groups.append(bench_pid)
        grandchild = _wait_for_pid_file_blocking(
            Path(repo) / "grandchild.pid", timeout_s=_SETTLE_TIMEOUT_S
        )
        proc.send_signal(signal_number)
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == expected_code
    _wait_until_dead_blocking(grandchild, timeout_s=_SETTLE_TIMEOUT_S)
    # Polled rather than checked once: a SIGKILLed leader stays visible to
    # ``os.kill(pid, 0)`` as a zombie until its parent reaps it.
    _wait_until_dead_blocking(bench_pid, timeout_s=_SETTLE_TIMEOUT_S)


# ---------------------------------------------------------------------------
# a stranded lock never wedges the next run
# ---------------------------------------------------------------------------


def test_measure_when_prior_run_hard_killed_does_take_over_stale_lock_on_rerun(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _TRACKED_BENCH)

    first = subprocess.Popen(  # noqa: S603
        [*_ENTRY, "measure", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        bench_pid = _wait_for_pid_file_blocking(
            Path(repo) / "bench.pid", timeout_s=_SETTLE_TIMEOUT_S
        )
        reap_groups.append(bench_pid)
        first.kill()  # SIGKILL runs no cleanup, so the lock is left behind
        first.communicate(timeout=30)
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(bench_pid), signal.SIGKILL)
    _wait_until_dead_blocking(bench_pid, timeout_s=_SETTLE_TIMEOUT_S)
    # The lock left behind above is now stale; the rerun below must take it over.

    (Path(repo) / "bench.sh").write_text(EMIT_ONE_BENCH, encoding="utf-8")
    rerun = subprocess.run(  # noqa: S603
        [*_ENTRY, "measure", "--bench", "sh bench.sh", "--samples", "2"],
        cwd=repo,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert rerun.returncode == 0, rerun.stderr
    assert "x" in rerun.stdout


# ---------------------------------------------------------------------------
# the status line is cleared on a TTY and untouched off one
# ---------------------------------------------------------------------------


def test_measure_when_signalled_on_a_tty_does_clear_the_status_line(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _TRACKED_BENCH)

    master, slave = pty.openpty()

    # Set a known window size on the slave so Rich renders at _PTY_WIDTH x
    # _PTY_HEIGHT instead of falling back on its defaults from a 0x0 pty.
    ws = struct.pack("HHHH", _PTY_HEIGHT, _PTY_WIDTH, 0, 0)  # cspell:disable-line
    fcntl.ioctl(slave, termios.TIOCSWINSZ, ws)

    proc = subprocess.Popen(  # noqa: S603
        [*_ENTRY, "measure", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=_env(),
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave)
    chunks: list[bytes] = []
    reader = threading.Thread(target=_drain, args=(master, chunks))
    reader.start()
    try:
        bench_pid = _wait_for_pid_file_blocking(
            Path(repo) / "bench.pid", timeout_s=_SETTLE_TIMEOUT_S
        )
        reap_groups.append(bench_pid)
        time.sleep(0.75)  # let the status line draw progress before the signal
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        reader.join(timeout=10)
        os.close(master)
    output = b"".join(chunks).decode("utf-8", "replace")

    assert proc.returncode == 130
    assert "sampling" in output, f"status line never drew progress: {output!r}"
    assert _SHOW_CURSOR in output, f"signal left the cursor hidden: {output!r}"

    # Replay the pty stream through a pyte emulated screen at the same
    # dimensions. screen_lines strips trailing blank rows, so a properly
    # cleared status area at the bottom of the screen simply disappears from
    # the result. If progress content survived the signal, it would remain as
    # the last visible row — the "━" bar character is unambiguous.
    visible = screen_lines(output, width=_PTY_WIDTH, height=_PTY_HEIGHT)
    last = visible[-1] if visible else ""
    assert "━" not in last, f"progress bar survived signal cleanup: {last!r}"


def test_measure_when_signalled_off_a_tty_does_not_emit_terminal_clear_codes(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _TRACKED_BENCH)

    proc = subprocess.Popen(  # noqa: S603
        [*_ENTRY, "measure", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        bench_pid = _wait_for_pid_file_blocking(
            Path(repo) / "bench.pid", timeout_s=_SETTLE_TIMEOUT_S
        )
        reap_groups.append(bench_pid)
        time.sleep(0.75)
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 130
    assert "\x1b[K" not in stderr
    assert "\x1b[K" not in stdout


# ---------------------------------------------------------------------------
# every worktree is swept, even under a second signal during cleanup
# ---------------------------------------------------------------------------


def test_compare_when_signalled_with_many_worktrees_does_sweep_all_of_them(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _TRACKED_BENCH)
    _git(["switch", "-c", "candidate-one"], repo)
    _git(["switch", "-c", "candidate-two"], repo)
    _git(["switch", "main"], repo)

    proc = subprocess.Popen(  # noqa: S603
        [
            *_ENTRY,
            "compare",
            "main",
            "candidate-one",
            "candidate-two",
            "--bench",
            "sh bench.sh",
            "--samples",
            "1",
        ],
        cwd=repo,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for wt in wait_for_worktrees(repo, 2):
            pid = _read_pid_file(Path(wt) / "bench.pid")
            if pid is not None:
                reap_groups.append(pid)
        proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 130
    assert list_worktree_dirs(repo, include_main=False) == []


def test_compare_when_signalled_twice_during_cleanup_does_exit_promptly(
    create_scratch_repo: Callable[[], str],
    reap_groups: list[int],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _TRACKED_BENCH)
    _git(["switch", "-c", "candidate"], repo)
    _git(["switch", "main"], repo)

    proc = subprocess.Popen(  # noqa: S603
        [*_ENTRY, "compare", "main", "candidate", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_for_worktrees(repo, 1)
        proc.send_signal(signal.SIGINT)
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(signal.SIGINT)
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    # The second signal's contract is a prompt exit that does not re-run
    # cleanups: it can take the re-entry guard's fast path and exit before the
    # first signal's worktree sweep finishes, so the surviving-worktree state is
    # timing-dependent here by design. The bounded communicate above already
    # pins "promptly"; the deterministic single-signal test above owns the
    # all-worktrees-swept guarantee.
    assert proc.returncode == 130


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


def _install_signalling_git(directory: Path, script: str = _SIGNALLING_GIT) -> Path:
    """Write the signalling stand-in ``git`` into a directory.

    Args:
        directory: Where the stand-in is written; put it first on ``PATH``.
        script: The stand-in's text, with ``log``, ``git``, ``sent`` and
            ``refused`` placeholders.

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
    _write_committed_bench(repo, EMIT_ONE_BENCH)
    _git(["switch", "-c", "candidate"], repo)
    _git(["switch", "main"], repo)
    absent = register_absent_worktree(repo)
    git_log = _install_signalling_git(tmp_path)
    env = _env()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"

    # ``timeout=60`` is the hang guard: ``run`` kills and reaps the CLI on expiry, so a
    # switch to ``Popen`` needs its own kill-and-reap.
    proc = subprocess.run(  # noqa: S603
        [*_ENTRY, "compare", "main", "candidate", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    calls = git_log.read_text(encoding="utf-8").splitlines()
    added = [call for call in calls if " worktree add " in f" {call} "]
    removed = [call for call in calls if " worktree remove " in f" {call} "]
    assert proc.returncode == 128 + signal.SIGTERM, proc.stderr
    assert len(set(removed)) == len(removed) == len(added)
    assert [call for call in calls if " worktree prune" in f" {call}"] == []
    assert list_worktree_dirs(repo, include_main=False) == [absent]


def test_compare_when_signalled_after_the_normal_sweep_left_a_worktree_does_name_it_without_pruning(
    create_scratch_repo: Callable[[], str],
    tmp_path: Path,
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, EMIT_ONE_BENCH)
    _git(["switch", "-c", "candidate-one"], repo)
    _git(["switch", "-c", "candidate-two"], repo)
    _git(["switch", "main"], repo)
    absent = register_absent_worktree(repo)
    git_log = _install_signalling_git(tmp_path, _REFUSING_THEN_SIGNALLING_GIT)
    env = _env()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"

    # ``timeout=60`` is the hang guard: ``run`` kills and reaps the CLI on expiry, so a
    # switch to ``Popen`` needs its own kill-and-reap.
    proc = subprocess.run(  # noqa: S603
        [
            *_ENTRY,
            "compare",
            "main",
            "candidate-one",
            "candidate-two",
            "--bench",
            "sh bench.sh",
            "--samples",
            "1",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    calls = git_log.read_text(encoding="utf-8").splitlines()
    removed = [call for call in calls if " worktree remove " in f" {call} "]
    assert proc.returncode == 128 + signal.SIGTERM, proc.stderr
    assert "cleanup did not finish:" in proc.stderr
    assert removed[0].split()[-1] in proc.stderr
    assert [call for call in calls if " worktree prune" in f" {call}"] == []
    assert absent in list_worktree_dirs(repo, include_main=False)
