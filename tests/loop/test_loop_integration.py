"""End-to-end integration tests for the whole gymrat optimization loop.

These drive the six loop commands against real scratch repositories, real
worktrees, and real bench subprocesses — nothing about the measurement stack is
faked. Two flows are covered:

- A whole session driven command by command, each step a fresh cold-start
  process, so every command has to rebuild the session from the log on disk.
- Lock contention: a gated bench holds the first ``iterate`` open while a second
  one collides with the repository lock and is refused.

POSIX-only: the flows lean on real subprocesses, worktrees, and file gating.
"""

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.report.loop import (
    SettleKept,
    StatusIteration,
    format_status_baseline,
    format_status_iteration,
)
from gymrat.session.lock import read_holder
from gymrat.session.paths import (
    experiment_worktree_dir,
    lockfile_path,
)
from gymrat.session.records import (
    BaselineRecord,
    DiscardRecord,
    IterationRecord,
    KeepRecord,
    SessionRecord,
)
from tests._ansi import (
    strip_sgr,
)
from tests._cli import ENTRY as _ENTRY
from tests._cli import no_color_env as _env
from tests._cli import run_cli
from tests._git import run_git as _git
from tests._git import status_of
from tests._platform import needs_posix_worktrees
from tests._process_helpers import reaped
from tests.loop._bench import (
    BASELINE_LATENCY,
    LONG_RUN_TIMEOUT,
    TUNING_FILE,
    commit_project,
    tune_experiment,
)
from tests.report._assertions import render_plain
from tests.session.records._fixtures import (
    log_records,
    records_of_type,
)

pytestmark = needs_posix_worktrees

#: Paired samples per iteration — a real measurement, but few enough to stay quick.
SAMPLES = 4

#: The latency the first edit tunes to, and the one the keep commits.
KEPT_LATENCY = 90

#: The latency the second edit tunes to, and the one the discard throws away.
DISCARDED_LATENCY = 80

#: The throwaway file the discard must erase, distinctive enough to grep history for.
DISCARD_MARKER = "discarded-edit-marker"

DISCARDED_FILE = "discarded-note.txt"


def _latency_samples(latency: int, count: int = SAMPLES) -> tuple[dict[str, float], ...]:
    """``count`` sample rounds, each reporting ``latency``."""
    return tuple({"latency": float(latency)} for _ in range(count))


# ---------------------------------------------------------------------------
# a whole session, driven command by command
# ---------------------------------------------------------------------------


def _run_command(repo: str, argv: list[str]) -> subprocess.CompletedProcess[str]:
    return run_cli(argv, repo, check=False, timeout=LONG_RUN_TIMEOUT)


def _drive_session(repo: str) -> tuple[list[int], str]:
    """Start, measure, iterate and keep one edit, then iterate and discard a second.

    Each command runs as a fresh process, so every step has to rebuild the session
    from the log on disk.

    Args:
        repo: The scratch repository holding the committed bench project.

    Returns:
        The exit code of every loop command in run order, and the colorless
        ``status`` report taken between the keep and the second edit.
    """
    exit_codes = [
        _run_command(repo, ["start", "--baseline", "main"]).returncode,
        _run_command(repo, ["measure", "main", "--record"]).returncode,
    ]
    tune_experiment(repo, KEPT_LATENCY)
    exit_codes.append(_run_command(repo, ["iterate"]).returncode)
    exit_codes.append(_run_command(repo, ["keep", "-m", "tune latency to 90"]).returncode)
    status_report = strip_sgr(_run_command(repo, ["status", "--no-color"]).stdout)
    tune_experiment(repo, DISCARDED_LATENCY)
    (Path(experiment_worktree_dir(repo)) / DISCARDED_FILE).write_text(
        f"{DISCARD_MARKER}\n", encoding="utf-8"
    )
    exit_codes.append(_run_command(repo, ["iterate"]).returncode)
    exit_codes.append(_run_command(repo, ["discard"]).returncode)
    return exit_codes, status_report


def test_loop_when_driven_command_by_command_does_run_the_whole_session(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    commit_project(repo, samples=SAMPLES)

    exit_codes, status_report = _drive_session(repo)

    assert exit_codes == [0, 0, 0, 0, 0, 0]
    records = log_records(repo)
    session = records_of_type(repo, SessionRecord)[0]
    keep = records_of_type(repo, KeepRecord)[0]
    branch = session.branch
    kept_commit = keep.commit
    assert kept_commit is not None

    # The log holds the session, the starting baseline, both iterations, the keep
    # with the baseline it appends, and the discard in exact order (command
    # records interleaved by the seam are filtered out — they are verified by
    # their own tests).
    domain_types = [record.type for record in records if record.type != "command"]
    assert domain_types == [
        "session",
        "baseline",
        "iteration",
        "keep",
        "baseline",
        "iteration",
        "discard",
    ]

    # The second iteration numbers from the log left by the first — seq 2 can
    # only come from reading the log, nothing carried over in memory.
    iterations = records_of_type(repo, IterationRecord)
    assert [record.seq for record in iterations] == [1, 2]
    assert keep.seq == 1
    assert keep.status == "committed"
    assert records_of_type(repo, DiscardRecord)[0].seq == 2

    baseline = records_of_type(repo, BaselineRecord)[0]
    assert baseline.label == "main"
    assert baseline.samples == _latency_samples(BASELINE_LATENCY)
    first, second = iterations
    assert first.samples.experiment == _latency_samples(KEPT_LATENCY)
    assert first.samples.baseline == _latency_samples(BASELINE_LATENCY)
    assert second.samples.experiment == _latency_samples(DISCARDED_LATENCY)
    assert second.samples.baseline == _latency_samples(KEPT_LATENCY)

    # status rebuilds the whole session out of the log alone.
    lines = status_report.split("\n")
    assert session.session_id in lines[0]
    assert render_plain(format_status_baseline(baseline)) in lines
    kept_line = format_status_iteration(
        StatusIteration(
            seq=1,
            delta_pct=first.primary.delta_pct,
            outcome=first.outcome,
            settle=SettleKept(commit=kept_commit, outcome=first.outcome),
        )
    )
    assert render_plain(kept_line) in lines

    assert status_of(repo) == ""

    assert _git(["log", "--format=%H", f"main..{branch}"], repo).split("\n") == [kept_commit]
    assert _git(["show", f"{branch}:{TUNING_FILE}"], repo) == str(KEPT_LATENCY)

    worktree = Path(experiment_worktree_dir(repo))
    assert DISCARD_MARKER not in _git(["log", "--all", "-p"], repo)
    assert not (worktree / DISCARDED_FILE).exists()
    assert (worktree / TUNING_FILE).read_text(encoding="utf-8").strip() == str(KEPT_LATENCY)


# ---------------------------------------------------------------------------
# lock contention between two iterate runs
# ---------------------------------------------------------------------------


def _wait_until_lock_held(first: subprocess.Popen[str], lock_path: str) -> None:
    """Block until ``first`` holds the repository lock, failing the test if it never does."""
    deadline = time.monotonic() + 30
    while True:
        if first.poll() is not None:
            pytest.fail(f"first iterate exited early: {first.communicate()}")
        if time.monotonic() > deadline:
            first.kill()
            pytest.fail("first iterate never grabbed the lock")
        # A pure read on purpose: probing with flock would itself take the
        # lock whenever it is still free, stealing it from the first run.
        holder = read_holder(lock_path)
        if holder is not None and holder.pid == first.pid:
            return
        time.sleep(0.025)


def test_loop_when_second_iterate_collides_with_the_lock_does_refuse_it(
    create_scratch_repo: Callable[[], str],
    tmp_path: Path,
):
    gate_file = str(tmp_path / "release")
    repo = create_scratch_repo()
    commit_project(repo, samples=SAMPLES, gate_file=gate_file)

    run_cli(["start", "--baseline", "main"], repo, timeout=LONG_RUN_TIMEOUT)
    tune_experiment(repo, KEPT_LATENCY)

    lock_path = lockfile_path(repo)
    with reaped(
        subprocess.Popen(  # noqa: S603 -- fixed argv, interpreter is sys.executable
            [*_ENTRY, "iterate"],
            cwd=repo,
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    ) as first:
        _wait_until_lock_held(first, lock_path)

        second = run_cli(["iterate"], repo, check=False, timeout=LONG_RUN_TIMEOUT)
        Path(gate_file).write_text("", encoding="utf-8")
        first_stdout, first_stderr = first.communicate(timeout=LONG_RUN_TIMEOUT)

    assert second.returncode == 2, second.stderr
    assert str(first.pid) in second.stderr
    assert "iterate" in second.stderr
    assert first.returncode == 0, first_stderr or first_stdout
    assert len(records_of_type(repo, IterationRecord)) == 1
