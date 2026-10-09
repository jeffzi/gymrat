"""Contention-hardening tests for the single-flight lock and its command wiring.

Where :mod:`tests.session.test_lock` drives the lock through patched system
calls in one process, these tests pin the guarantees that only surface under
genuine pressure:

- a burst of real processes racing for one lockfile grants exactly one holder
  and hands every loser the contention error, with the lockfile never torn,
- the repository lock and the supervise lock are independent, so holding one
  never blocks the command guarded by the other.

The multi-process tests are POSIX-only: they rendezvous children on a named
pipe.
"""

import contextlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from gymrat.session.lock import acquire_lock, is_held
from gymrat.session.paths import lockfile_path, supervise_lockfile_path
from tests.hardening._barrier import CHILD_BARRIER, racing_children

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only named pipes and hard links"
)

# Ceiling for a race to resolve and for children to exit once released; generous
# enough to absorb CI scheduling jitter without masking a real deadlock.
RACE_TIMEOUT_SECONDS = 30

# A child that races for the lock, records its verdict under ``results``, and —
# when it wins — holds the lock until the parent drops the release flag, so
# every rival races against a genuinely live holder.
_RACE_CHILD = (
    CHILD_BARRIER
    + """\
import sys
import time

from gymrat.errors import GymratError
from gymrat.session.lock import acquire_lock


def main() -> int:
    lock_path, barrier_path, results_dir, release_flag, command = sys.argv[1:6]
    pid = os.getpid()

    wait_at_barrier(barrier_path)

    try:
        release = acquire_lock(lock_path, command)
    except GymratError as error:
        payload = f"{error}\\n{error.hint or ''}"
        Path(results_dir, f"lost.{pid}").write_text(payload, encoding="utf-8")
        return 3

    Path(results_dir, f"won.{pid}").write_text(str(pid), encoding="utf-8")
    while not Path(release_flag).exists():
        time.sleep(0.02)
    release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
)


# A holder that takes the lock and dies without releasing it. ``os._exit`` skips
# every cleanup path, so the kernel drops the OS lock while its file, the
# holder record and the publish lock file stay on disk, as after a real crash.
_CRASHED_HOLDER_CHILD = """\
import os
import sys

from gymrat.session.lock import acquire_lock

acquire_lock(sys.argv[1], "measure")
os._exit(0)
"""


def _write_stale_lock(lock_path: str) -> None:
    """Leave behind the lock files of a holder that crashed while holding the lock."""
    subprocess.run(  # noqa: S603 -- argv is a fixed list, not shell-injected
        [sys.executable, "-c", _CRASHED_HOLDER_CHILD, lock_path],
        check=True,
        timeout=RACE_TIMEOUT_SECONDS,
    )


@dataclass(frozen=True, slots=True)
class _RaceOutcome:
    """What a whole race left behind, once every child has exited."""

    won_pid_values: list[str]
    lost_payloads: list[str]
    holder_snapshot: str | None
    exit_codes: list[int]
    child_errors: list[str]


def _surface_child_crashes(children: list[subprocess.Popen[str]]) -> None:
    """Fail loudly if a child died for any reason other than win (0) or loss (3)."""
    for child in children:
        if child.poll() is not None and child.returncode not in (0, 3):
            stderr = child.stderr.read() if child.stderr else ""
            message = f"race child crashed (exit {child.returncode}): {stderr}"
            raise AssertionError(message)


def _run_race(tmp_path: Path, lock_path: str, count: int, command: str = "measure") -> _RaceOutcome:
    """Race ``count`` child processes for ``lock_path`` behind a shared barrier.

    All children open a named pipe, leave a ``ready.<pid>`` marker beside it and
    block on it; the parent writes the go bytes only once every marker exists,
    so every child calls ``acquire_lock`` within the same scheduling window. The
    winner holds the lock until the parent drops the release flag, guaranteeing
    every loser contends a live holder. The lockfile is snapshotted while the
    winner still holds it, so a torn or vanished lock is caught.

    Args:
        tmp_path: Where the barrier, the result markers and the child script live.
        lock_path: The lockfile every child races for.
        count: How many children race.
        command: The command name each child acquires the lock for.

    Returns:
        What the race left behind once every child exited.
    """
    results = tmp_path / "results"
    results.mkdir()
    release_flag = tmp_path / "release.flag"

    with racing_children(
        tmp_path,
        _RACE_CHILD,
        lambda barrier, _index: (lock_path, str(barrier), str(results), str(release_flag), command),
        count,
        on_poll=_surface_child_crashes,
        timeout_s=RACE_TIMEOUT_SECONDS,
    ) as children:
        deadline = time.monotonic() + RACE_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            won = sorted(results.glob("won.*"))
            lost = sorted(results.glob("lost.*"))
            if len(won) == 1 and len(lost) == count - 1:
                break
            _surface_child_crashes(children)
            time.sleep(0.02)

        won = sorted(results.glob("won.*"))
        lost = sorted(results.glob("lost.*"))
        try:
            snapshot = Path(lock_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            snapshot = None

        release_flag.write_text("go", encoding="utf-8")
        exit_codes = [child.wait(timeout=RACE_TIMEOUT_SECONDS) for child in children]
        child_errors = [child.stderr.read() if child.stderr else "" for child in children]

    return _RaceOutcome(
        won_pid_values=[path.read_text(encoding="utf-8").strip() for path in won],
        lost_payloads=[path.read_text(encoding="utf-8") for path in lost],
        holder_snapshot=snapshot,
        exit_codes=exit_codes,
        child_errors=child_errors,
    )


# ---------------------------------------------------------------------------
# real multi-process contention over one lockfile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("initial_state", "count"),
    [
        pytest.param("fresh", 5, id="fresh-burst"),
        pytest.param("stale", 2, id="stale-steal"),
    ],
)
def test_acquire_lock_when_processes_race_does_admit_exactly_one_holder(
    initial_state: str, count: int, tmp_path: Path
):
    lock_path = str(tmp_path / "gymrat.lock.json")
    if initial_state == "stale":
        _write_stale_lock(lock_path)

    outcome = _run_race(tmp_path, lock_path, count)

    assert len(outcome.won_pid_values) == 1, outcome.child_errors
    assert len(outcome.lost_payloads) == count - 1
    assert sorted(outcome.exit_codes) == sorted([0, *([3] * (count - 1))])

    winner_pid = outcome.won_pid_values[0]
    assert outcome.holder_snapshot is not None, "lockfile vanished while a holder was active"
    assert json.loads(outcome.holder_snapshot)["pid"] == int(winner_pid)

    for payload in outcome.lost_payloads:
        assert f"PID {winner_pid}" in payload


# ---------------------------------------------------------------------------
# the repository lock and the supervise lock are independent
# ---------------------------------------------------------------------------


def test_acquire_lock_when_repository_lock_is_held_does_not_block_the_supervise_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    root = str(tmp_path / "checkout")
    repository_lock = lockfile_path(root)
    supervise_lock = supervise_lockfile_path(root)

    with contextlib.ExitStack() as releases:
        releases.callback(acquire_lock(repository_lock, "measure"))

        releases.callback(acquire_lock(supervise_lock, "supervise"))

        assert (is_held(repository_lock), is_held(supervise_lock)) == (True, True)
