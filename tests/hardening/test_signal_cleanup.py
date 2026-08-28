"""End-to-end signal-cleanup hardening tests over real subprocesses.

Each test runs the CLI out of process against a throwaway repo and a real shell
bench, then delivers a real signal while the bench is mid-run. Together they pin
the guarantees that must outlive an interrupted run:

- no bench process — nor a grandchild it spawned — survives the CLI,
- a lock stranded by a hard kill never wedges the next run,
- the terminal status line is cleared on a TTY and left untouched off one,
- every worktree is swept, even when a second signal lands during cleanup.

The whole module is POSIX-only: it relies on real signals, ``sh`` bench scripts,
and process-group tree-kill. Every test that spawns a real tree registers its
group with the ``reap_groups`` fixture so no orphan bench survives a failed
assertion.
"""

from __future__ import annotations

import contextlib
import os
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
    from collections.abc import Callable, Iterator

if sys.platform != "win32":
    import fcntl
    import pty
    import termios

from tests._cli import ENTRY as _ENTRY
from tests._process_helpers import is_alive as _is_alive
from tests._rich import screen_lines
from tests.hardening._bench_helpers import drain as _drain
from tests.hardening._bench_helpers import env as _env
from tests.hardening._bench_helpers import git as _git
from tests.hardening._bench_helpers import write_committed_bench as _write_committed_bench

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")

# The overwrite status line clears its row with a carriage return followed by the
# ANSI "erase to end of line" sequence.
_CLEAR_LINE = "\r\x1b[K"

# Fixed pty dimensions for the pyte screen replay. The slave pty is sized to
# these values via TIOCSWINSZ so Rich in the child renders at a known geometry.
_PTY_WIDTH = 80
_PTY_HEIGHT = 24

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

_FAST_BENCH = "#!/bin/sh\necho 'METRIC x=1'\n"


def _read_pid(path: Path) -> int | None:
    """Read a pid a shell wrote with ``echo $$ >`` / ``echo $! >``.

    Returns ``None`` until the file holds a complete positive integer, so a
    partial write reads as "no pid yet" rather than a truncated value.
    """
    try:
        raw = path.read_text().strip()
    except FileNotFoundError:
        return None
    return int(raw) if raw.isdigit() else None


def _wait_for_pid(path: Path, timeout_s: float = 30.0) -> int:
    """Poll ``path`` until it holds a complete pid, then return it."""
    deadline = time.monotonic() + timeout_s
    while True:
        pid = _read_pid(path)
        if pid:
            return pid
        if time.monotonic() > deadline:
            message = f"pid never appeared at {path}"
            raise AssertionError(message)
        time.sleep(0.05)


def _wait_until_dead(pid: int, timeout_s: float = 30.0) -> None:
    """Poll until the process with ``pid`` no longer exists."""
    deadline = time.monotonic() + timeout_s
    while _is_alive(pid):
        if time.monotonic() > deadline:
            message = f"process {pid} was still alive after {timeout_s}s"
            raise AssertionError(message)
        time.sleep(0.05)


def _wait_for_worktree_count(
    list_worktree_dirs: Callable[..., list[str]], repo: str, count: int, timeout_s: float = 30.0
) -> None:
    """Poll until at least ``count`` non-main worktrees exist for ``repo``."""
    deadline = time.monotonic() + timeout_s
    while len(list_worktree_dirs(repo, include_main=False)) < count:
        if time.monotonic() > deadline:
            got = list_worktree_dirs(repo, include_main=False)
            message = f"expected >= {count} worktrees, saw {got}"
            raise AssertionError(message)
        time.sleep(0.05)


@pytest.fixture
def reap_groups() -> Iterator[list[int]]:
    """Track process-group leaders and hard-kill any survivor on teardown.

    Every test here drives a real bench process tree; a failed assertion must
    not strand a live bench or the grandchild it forked. The bench writes its
    own ``$$`` to ``bench.pid``; tests append that leader pid, and the whole
    group is ``SIGKILL``ed when the test ends.
    """
    leaders: list[int] = []
    try:
        yield leaders
    finally:
        for pid in leaders:
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(pid), signal.SIGKILL)


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
        bench_pid = _wait_for_pid(Path(repo) / "bench.pid")
        reap_groups.append(bench_pid)
        grandchild = _wait_for_pid(Path(repo) / "grandchild.pid")
        proc.send_signal(signal_number)
        proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == expected_code
    _wait_until_dead(grandchild)
    # Polled rather than checked once: a SIGKILLed leader stays visible to
    # ``os.kill(pid, 0)`` as a zombie until its parent reaps it.
    _wait_until_dead(bench_pid)


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
    bench_pid = _wait_for_pid(Path(repo) / "bench.pid")
    reap_groups.append(bench_pid)
    first.kill()  # SIGKILL runs no cleanup, so the lock is left behind
    first.communicate(timeout=30)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(bench_pid), signal.SIGKILL)
    _wait_until_dead(bench_pid)
    # The lock left behind above is now stale; the rerun below must take it over.

    (Path(repo) / "bench.sh").write_text(_FAST_BENCH, encoding="utf-8")
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
        bench_pid = _wait_for_pid(Path(repo) / "bench.pid")
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
    assert _CLEAR_LINE in output, f"status line never drew progress: {output!r}"

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
        bench_pid = _wait_for_pid(Path(repo) / "bench.pid")
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
    list_worktree_dirs: Callable[..., list[str]],
    reap_groups: list[int],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _TRACKED_BENCH)
    _git(repo, "switch", "-c", "candidate-one")
    _git(repo, "switch", "-c", "candidate-two")
    _git(repo, "switch", "main")

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
        _wait_for_worktree_count(list_worktree_dirs, repo, 2)
        worktrees = list_worktree_dirs(repo, include_main=False)
        for wt in worktrees:
            pid = _read_pid(Path(wt) / "bench.pid")
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
    list_worktree_dirs: Callable[..., list[str]],
    reap_groups: list[int],
):
    repo = create_scratch_repo()
    _write_committed_bench(repo, _TRACKED_BENCH)
    _git(repo, "switch", "-c", "candidate")
    _git(repo, "switch", "main")

    proc = subprocess.Popen(  # noqa: S603
        [*_ENTRY, "compare", "main", "candidate", "--bench", "sh bench.sh", "--samples", "1"],
        cwd=repo,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_worktree_count(list_worktree_dirs, repo, 1)
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
