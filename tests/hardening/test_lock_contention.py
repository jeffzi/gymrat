"""Contention-hardening tests for the single-flight lock and its command wiring.

Where :mod:`tests.session.test_lock` drives the lock through patched system
calls in one process, these tests pin the guarantees that only surface under
genuine pressure:

- a burst of real processes racing for one lockfile grants exactly one holder
  and hands every loser the contention error, with the lockfile never torn,
- two processes racing to steal the same stale lock never both end up holding
  it, and never delete a lock a live winner is using,
- a leftover file another user owns at any of the lock's auxiliary scratch
  paths (the scratch record, the claim link, the stale-aside) yields the
  "belongs to another user" remedy rather than a raw permission traceback,
- the repository lock and the supervise lock are independent, so holding one
  never blocks the command guarded by the other,
- ``supervise`` run outside a git repository exits cleanly, naming the
  requirement, rather than crashing with an unhandled error.

The multi-process tests are POSIX-only: they rendezvous children on a named
pipe and rely on hard links for the claim protocol.
"""

import errno
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.errors import GymratError
from gymrat.session.lock import acquire_lock
from gymrat.session.paths import lockfile_path, supervise_lockfile_path
from tests._process_helpers import dead_pid

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only named pipes and hard links"
)

# The ISO-8601 shape a freshly published holder record stamps into ``at``.
AT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

# The fixed timestamp planted into a stale lockfile fixture; its exact value is
# immaterial, only that a holder record is well-formed.
WRITTEN_LOCK_AT = "2026-01-01T00:00:00.000Z"

# A child that races for the lock, records its verdict under ``results``, and —
# when it wins — holds the lock until the parent drops the release flag, so
# every rival races against a genuinely live holder.
_RACE_CHILD = """\
import os
import sys
import time
from pathlib import Path

from gymrat.errors import GymratError
from gymrat.session.lock import acquire_lock


def main() -> int:
    lock_path, barrier_path, results_dir, release_flag, command = sys.argv[1:6]
    pid = os.getpid()

    # Unbuffered read so each child consumes exactly one go byte; a buffered
    # reader would slurp the whole pipe and starve its siblings.
    barrier_fd = os.open(barrier_path, os.O_RDONLY)
    os.read(barrier_fd, 1)
    os.close(barrier_fd)

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


def _write_stale_lock(lock_path: str) -> None:
    """Plant a holder record whose owning process has already exited."""
    Path(lock_path).write_text(
        json.dumps({"pid": dead_pid(), "command": "measure", "at": WRITTEN_LOCK_AT}),
        encoding="utf-8",
    )


@dataclass
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

    All children block on a named pipe until the parent writes the go bytes, so
    the race starts together. The winner holds the lock until the parent drops
    the release flag, guaranteeing every loser contends a live holder. The
    lockfile is snapshotted while the winner still holds it, so a torn or
    vanished lock is caught.
    """
    barrier_path = tmp_path / "barrier.pipe"
    os.mkfifo(barrier_path)
    results = tmp_path / "results"
    results.mkdir()
    release_flag = tmp_path / "release.flag"
    script = tmp_path / "race_child.py"
    script.write_text(_RACE_CHILD, encoding="utf-8")

    children = [
        subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                str(script),
                lock_path,
                str(barrier_path),
                str(results),
                str(release_flag),
                command,
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(count)
    ]

    go_fd = os.open(str(barrier_path), os.O_RDWR)
    try:
        os.write(go_fd, b"\x00" * count)

        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            won = sorted(results.glob("won.*"))
            lost = sorted(results.glob("lost.*"))
            if len(won) == 1 and len(lost) == count - 1:
                break
            _surface_child_crashes(children)
            time.sleep(0.02)

        won = sorted(results.glob("won.*"))
        lost = sorted(results.glob("lost.*"))
        snapshot = Path(lock_path).read_text(encoding="utf-8") if Path(lock_path).exists() else None

        release_flag.write_text("go", encoding="utf-8")
        exit_codes = [child.wait(timeout=30) for child in children]
        child_errors = [child.stderr.read() if child.stderr else "" for child in children]
    finally:
        os.close(go_fd)
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
            if child.stderr is not None:
                child.stderr.close()

    return _RaceOutcome(
        won_pid_values=[path.read_text(encoding="utf-8").strip() for path in won],
        lost_payloads=[path.read_text(encoding="utf-8") for path in lost],
        holder_snapshot=snapshot,
        exit_codes=exit_codes,
        child_errors=child_errors,
    )


def _assert_belongs_to_other_user(error: GymratError, lock_path: str) -> None:
    """Assert ``error`` is the manual-remedy reframing for a lock another user owns."""
    assert lock_path in str(error)
    hint = error.hint or ""
    assert re.search("remove", hint, re.IGNORECASE)
    assert lock_path in hint


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
def test_acquire_lock_when_processes_race_does_grant_one_holder_and_contend_the_rest(
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
    record = json.loads(outcome.holder_snapshot)
    assert set(record) == {"pid", "command", "at"}
    assert record["pid"] == int(winner_pid)
    assert record["command"] == "measure"
    assert AT_PATTERN.match(record["at"])

    for payload in outcome.lost_payloads:
        assert f"PID {winner_pid}" in payload


# ---------------------------------------------------------------------------
# a leftover another user owns at an auxiliary scratch path
# ---------------------------------------------------------------------------


def test_acquire_lock_when_scratch_record_owned_by_other_user_does_reframe_with_manual_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A sticky /tmp left a read-only ``<lock>.<pid>.record`` behind, so opening
    # the scratch file the holder record is staged through is refused.
    lock_path = str(tmp_path / "gymrat.lock.json")
    real_open = os.open

    def refuse_scratch_record(path: str, *args: int) -> int:
        if path.startswith(lock_path) and path.endswith(".record"):
            raise OSError(errno.EPERM, "operation not permitted")
        return real_open(path, *args)

    monkeypatch.setattr(os, "open", refuse_scratch_record)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    _assert_belongs_to_other_user(caught.value, lock_path)


def test_acquire_lock_when_claim_link_owned_by_other_user_does_reframe_with_manual_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The stale lock is another user's, so hard-linking it to the claim path is
    # refused mid-takeover.
    lock_path = str(tmp_path / "gymrat.lock.json")
    _write_stale_lock(lock_path)
    real_link = os.link

    def refuse_claim_link(src: str, dst: str) -> None:
        if dst.endswith(".claim"):
            raise OSError(errno.EPERM, "operation not permitted")
        real_link(src, dst)

    monkeypatch.setattr(os, "link", refuse_claim_link)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    _assert_belongs_to_other_user(caught.value, lock_path)


def test_acquire_lock_when_stale_aside_owned_by_other_user_does_reframe_with_manual_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The stale lock is another user's, so renaming it aside to take it over is
    # refused.
    lock_path = str(tmp_path / "gymrat.lock.json")
    _write_stale_lock(lock_path)

    def refuse_rename(src: str, dst: str) -> None:
        raise OSError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(os, "rename", refuse_rename)

    with pytest.raises(GymratError) as caught:
        acquire_lock(lock_path, "compare")

    _assert_belongs_to_other_user(caught.value, lock_path)


# ---------------------------------------------------------------------------
# the repository lock and the supervise lock are independent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("held_role", "acquired_role"),
    [
        pytest.param("repo", "supervise", id="repo-held-then-supervise"),
        pytest.param("supervise", "repo", id="supervise-held-then-repo"),
    ],
)
def test_acquire_lock_when_one_command_lock_is_held_does_not_block_the_other(
    held_role: str, acquired_role: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    root = str(tmp_path / "checkout")
    paths = {"repo": lockfile_path(root), "supervise": supervise_lockfile_path(root)}

    held = acquire_lock(paths[held_role], held_role)
    other = acquire_lock(paths[acquired_role], acquired_role)

    assert Path(paths[held_role]).exists()
    assert Path(paths[acquired_role]).exists()
    held()
    other()


# ---------------------------------------------------------------------------
# supervise run outside a git repository
# ---------------------------------------------------------------------------


def test_supervise_when_run_outside_a_git_repository_does_exit_two_naming_the_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["supervise", "optimize it", "--max-minutes", "10"])

    assert result.exit_code == 2
    from tests._ansi import strip_ansi

    combined = strip_ansi((result.stdout or "") + (result.stderr or ""))
    assert re.search("git repository", combined, re.IGNORECASE)
    assert "Traceback" not in combined
