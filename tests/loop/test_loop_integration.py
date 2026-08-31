"""End-to-end integration tests for the whole gymrat optimization loop.

These drive the six loop commands against real scratch repositories, real
worktrees, and real bench subprocesses — nothing about the measurement stack is
faked. Three flows are covered:

- A whole session driven command by command, each step a fresh cold-start
  process, so every command has to rebuild the session from the log on disk.
- Lock contention: a gated bench holds the first ``iterate`` open while a second
  one collides with the repository lock and is refused.
- A restart after a session finalized without its worktree on disk, which must
  open a fresh session rather than resume the closed one.

POSIX-only: the flows lean on real subprocesses, worktrees, and file gating.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.loop.finalize import finalize_session
from gymrat.loop.start import start_session
from gymrat.session.paths import (
    archived_session_path,
    baseline_worktree_dir,
    experiment_worktree_dir,
    lockfile_path,
    session_jsonl_path,
)
from gymrat.session.records import (
    BaselineRecord,
    DiscardRecord,
    IterationRecord,
    KeepRecord,
    SessionLogRecord,
    SessionRecord,
)
from gymrat.session.store import append_record, read_records
from tests._git import run_git as _run_git
from tests.loop._bench import BASELINE_LATENCY, TUNING_FILE, commit_project
from tests.loop.iterate._fixtures import resolved_config
from tests.session.records._fixtures import committed_keep, iteration_record

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only worktrees and gating")

from tests._ansi import strip_sgr as _strip_ansi
from tests._cli import ENTRY as _ENTRY
from tests._cli import no_color_env as _env

#: Generous budget: every command creates real worktrees and spawns real benches.
LONG_RUN_TIMEOUT = 180

#: Paired samples per iteration — a real measurement, but few enough to stay quick.
SAMPLES = 5

#: The latency the first edit tunes to, and the one the keep commits.
KEPT_LATENCY = 90

#: The latency the second edit tunes to, and the one the discard throws away.
DISCARDED_LATENCY = 80

#: The throwaway file the discard must erase, distinctive enough to grep history for.
DISCARD_MARKER = "discarded-edit-marker"

DISCARDED_FILE = "discarded-note.txt"


def _run_cli(repo: str, *argv: str) -> subprocess.CompletedProcess[str]:
    """Run one loop command from a cold-start process rooted in ``repo``.

    A fresh process per call is what makes the sequence a restart test: nothing a
    command computed survives into the next one, so every command has to rebuild
    the session from the log on disk.
    """
    return subprocess.run(  # noqa: S603
        [*_ENTRY, *argv],
        cwd=repo,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=LONG_RUN_TIMEOUT,
        check=False,
    )


def _git(repo: str, *args: str) -> str:
    """Run git in ``repo`` for test setup and inspection, returning trimmed stdout."""
    return _run_git(list(args), repo).strip()


def _tune_experiment(repo: str, latency: int) -> None:
    """Tune the experiment worktree to ``latency``, the edit an agent would make."""
    (Path(experiment_worktree_dir(repo)) / TUNING_FILE).write_text(f"{latency}\n", encoding="utf-8")


def _pick[R: SessionLogRecord](records: list[SessionLogRecord], record_type: type[R]) -> list[R]:
    """The records of one class, in file order, narrowed to that record's shape."""
    return [record for record in records if isinstance(record, record_type)]


def _latency_samples(latency: int, count: int = SAMPLES) -> tuple[dict[str, float], ...]:
    """``count`` sample rounds, each reporting ``latency``."""
    return tuple({"latency": float(latency)} for _ in range(count))


# ---------------------------------------------------------------------------
# a whole session, driven command by command
# ---------------------------------------------------------------------------


def test_loop_when_driven_command_by_command_does_run_the_whole_session(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    commit_project(repo, samples=SAMPLES)

    # Each command runs from a cold start, in the order an agent would drive
    # them: open the session, pin the baseline, edit, measure, keep, edit,
    # measure, throw away.
    exit_codes: list[int] = [
        _run_cli(repo, "start", "main").returncode,
        _run_cli(repo, "measure", "main", "--record").returncode,
    ]
    _tune_experiment(repo, KEPT_LATENCY)
    exit_codes.append(_run_cli(repo, "iterate").returncode)
    exit_codes.append(_run_cli(repo, "keep", "-m", "tune latency to 90").returncode)

    status_report = _strip_ansi(_run_cli(repo, "status", "--no-color").stdout)

    _tune_experiment(repo, DISCARDED_LATENCY)
    (Path(experiment_worktree_dir(repo)) / DISCARDED_FILE).write_text(
        f"{DISCARD_MARKER}\n", encoding="utf-8"
    )
    exit_codes.append(_run_cli(repo, "iterate").returncode)
    exit_codes.append(_run_cli(repo, "discard").returncode)

    records = read_records(session_jsonl_path(repo))
    session = _pick(records, SessionRecord)[0]
    keep = _pick(records, KeepRecord)[0]
    branch = session.branch
    kept_commit = keep.commit
    assert kept_commit is not None

    # Every command in the sequence succeeded.
    assert exit_codes == [0, 0, 0, 0, 0, 0]

    # The log holds the session, the baseline, both iterations, the keep, and
    # the discard in exact order.
    assert [record.type for record in records] == [
        "session",
        "baseline",
        "iteration",
        "keep",
        "iteration",
        "discard",
    ]

    # The second iteration numbers from the log left by the first — seq 2 can
    # only come from reading the log, nothing carried over in memory.
    iterations = _pick(records, IterationRecord)
    assert [record.seq for record in iterations] == [1, 2]
    assert keep.seq == 1
    assert keep.status == "committed"
    assert _pick(records, DiscardRecord)[0].seq == 2

    # The records hold what the real bench printed in each worktree it ran in.
    baseline = _pick(records, BaselineRecord)[0]
    assert baseline.label == "main"
    assert baseline.samples == _latency_samples(BASELINE_LATENCY)
    first, second = iterations
    assert first.samples.experiment == _latency_samples(KEPT_LATENCY)
    assert first.samples.baseline == _latency_samples(BASELINE_LATENCY)
    assert second.samples.experiment == _latency_samples(DISCARDED_LATENCY)
    assert second.samples.baseline == _latency_samples(KEPT_LATENCY)

    # status rebuilds the whole session out of the log alone.
    lines = status_report.split("\n")
    assert f"session {session.session_id}" in lines[0]
    assert f"baseline main · latency {BASELINE_LATENCY}" in lines
    assert re.search(rf"^iteration 1 · .* · kept {kept_commit[:7]}$", status_report, re.MULTILINE)

    # The main working tree ends clean.
    assert _git(repo, "status", "--porcelain") == ""

    # The kept edit is the only commit on the experiment branch, and the branch
    # carries the tuned latency.
    assert _git(repo, "log", "--format=%H", f"main..{branch}").split("\n") == [kept_commit]
    assert _git(repo, "show", f"{branch}:{TUNING_FILE}") == str(KEPT_LATENCY)

    # The discarded edit is nowhere on disk or in history, and the worktree is
    # back to the kept latency.
    worktree = Path(experiment_worktree_dir(repo))
    assert DISCARD_MARKER not in _git(repo, "log", "--all", "-p")
    assert not (worktree / DISCARDED_FILE).exists()
    assert (worktree / TUNING_FILE).read_text(encoding="utf-8").strip() == str(KEPT_LATENCY)


# ---------------------------------------------------------------------------
# lock contention between two iterate runs
# ---------------------------------------------------------------------------


def test_loop_when_second_iterate_collides_with_the_lock_does_refuse_it(
    create_scratch_repo: Callable[[], str],
    tmp_path: Path,
):
    gate_file = str(tmp_path / "release")
    repo = create_scratch_repo()
    commit_project(repo, samples=SAMPLES, gate_file=gate_file)

    assert _run_cli(repo, "start", "main").returncode == 0
    _tune_experiment(repo, KEPT_LATENCY)

    lock_path = lockfile_path(repo)
    first = subprocess.Popen(  # noqa: S603
        [*_ENTRY, "iterate"],
        cwd=repo,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # The gated bench holds the first run open; wait for it to grab the lock.
        deadline = time.monotonic() + 30
        while not os.path.exists(lock_path):  # noqa: PTH110
            if first.poll() is not None:
                pytest.fail(f"first iterate exited early: {first.communicate()}")
            if time.monotonic() > deadline:
                first.kill()
                pytest.fail("first iterate never grabbed the lock")
            time.sleep(0.025)

        second = _run_cli(repo, "iterate")
        Path(gate_file).write_text("", encoding="utf-8")
        first_stdout, first_stderr = first.communicate(timeout=LONG_RUN_TIMEOUT)
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate()

    assert second.returncode == 2, second.stderr
    assert first.returncode == 0, first_stderr or first_stdout
    assert not os.path.exists(lock_path)  # noqa: PTH110
    assert len(_pick(read_records(session_jsonl_path(repo)), IterationRecord)) == 1


# ---------------------------------------------------------------------------
# restart after a finalize whose worktree was deleted first
# ---------------------------------------------------------------------------


def test_loop_when_restarted_after_a_finalize_without_worktree_does_open_fresh(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()

    # Drive a whole session by hand: open it, commit the edit a keep would
    # commit, log the iteration and the keep behind it, then close it. The bench
    # never runs — the iteration record stands in for what iterate measured.
    first = start_session(repo, "main", resolved_config())
    closed_session_id = first.session.session_id

    worktree = experiment_worktree_dir(repo)
    (Path(worktree) / TUNING_FILE).write_text(f"{KEPT_LATENCY}\n", encoding="utf-8")
    _git(worktree, "add", "-A")
    _git(worktree, "commit", "-m", "tune latency to 90")
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    append_record(
        session_jsonl_path(repo),
        committed_keep(1, commit=_git(worktree, "rev-parse", "HEAD")),
    )

    # The directory goes before finalize does, so ``git worktree remove`` finds
    # nothing to take and git keeps its entry for the path.
    shutil.rmtree(worktree)
    finalize_session(repo)
    closed_log = read_records(session_jsonl_path(repo))

    restarted = start_session(repo, "main", resolved_config())

    # A fresh session opens rather than resuming the closed one.
    assert restarted.resumed is False
    assert restarted.session.session_id != closed_session_id

    # Both worktrees of the fresh session check out.
    assert Path(experiment_worktree_dir(repo)).exists()
    assert Path(baseline_worktree_dir(repo)).exists()

    # The closed session's log is archived under the id it belonged to.
    assert read_records(archived_session_path(repo, closed_session_id)) == closed_log
