"""Behavioral tests for closing a session (``finalize_session``).

Every test drives the real ``finalize_session`` against a throwaway repository
from the shared ``create_scratch_repo`` factory, so the suite is order-independent
and safe under ``pytest-xdist`` / ``pytest-randomly``. Every git operation runs
against real worktrees, and the assertions read commit SHAs straight out of the
repository git laid down.
"""

import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.config import StopConfig
from gymrat.loop.finalize import (
    FinalizeOptions,
    finalize_session,
)
from gymrat.loop.start import start_session
from gymrat.session.paths import baseline_worktree_dir, experiment_worktree_dir, session_jsonl_path
from gymrat.session.records import FinalizeRecord, SessionLogRecord
from tests._config import resolved_config
from tests._git import commit_all, head_of, list_worktree_dirs
from tests._git import run_git as _git
from tests.loop._settle import (
    capture_error,
    keep_iteration,
)
from tests.session.records._fixtures import (
    append_records,
    committed_keep,
    iteration_record,
    log_records,
    session_header_of,
    stop_record,
)

# A settled run config carrying the keys the session header snapshots; it drives
# ``start_session`` without ever being benched against.
CONFIG = resolved_config(
    prepare="npm run build",
    filter="npm run bench -- {names}",
    stop=StopConfig(max_iterations=20),
)


def _last_record(root: str) -> SessionLogRecord:
    """The record ``root``'s log ends on, failing when the log is empty."""
    records = log_records(root)
    assert records, f"expected a record in {session_jsonl_path(root)}"
    return records[-1]


def _commit_iteration(root: str, seq: int, message: str) -> str:
    """Commit one edit in the experiment worktree and log the iteration behind it.

    The experiment worktree is checked out on the session branch, so each call
    moves that branch forward exactly as a real ``gymrat keep`` would. The keep
    record is left to the caller.
    """
    commit = commit_all(experiment_worktree_dir(root), message, file=f"step-{seq}.txt")
    append_records(root, iteration_record(seq=seq))
    return commit


def _keep_iteration(root: str, seq: int, message: str) -> str:
    """Commit one edit and log the iteration and the committed keep that settled it."""
    commit = commit_all(experiment_worktree_dir(root), message, file=f"step-{seq}.txt")
    keep_iteration(root, seq, commit=commit, message=message)
    return commit


@pytest.fixture
def session_repo(repo: str) -> str:
    """The scratch repository with an open session on ``main``."""
    start_session(repo, "main", CONFIG)
    return repo


# ---------------------------------------------------------------------------
# when nothing has been kept
# ---------------------------------------------------------------------------


def test_finalize_session_when_nothing_kept_does_refuse_creating_no_branch_and_no_record(
    session_repo: str,
):
    before = len(log_records(session_repo))

    error = capture_error(lambda: finalize_session(session_repo))

    assert error.reason == "nothing-kept"
    assert error.hint is not None
    assert re.search(r"keep", error.hint, re.IGNORECASE)
    assert _git(["branch", "--list", "*-final"], session_repo) == ""
    assert len(log_records(session_repo)) == before


# ---------------------------------------------------------------------------
# when the last iteration is neither kept nor discarded
# ---------------------------------------------------------------------------


def test_finalize_session_when_last_iteration_unsettled_does_refuse_writing_no_record(
    session_repo: str,
):
    _keep_iteration(session_repo, 1, "cache the regex")
    append_records(session_repo, iteration_record(seq=2))
    before = len(log_records(session_repo))

    error = capture_error(lambda: finalize_session(session_repo))

    assert error.reason == "unsettled"
    assert error.hint is not None
    assert re.search(r"keep", error.hint, re.IGNORECASE)
    assert re.search(r"discard", error.hint, re.IGNORECASE)
    assert len(log_records(session_repo)) == before


# ---------------------------------------------------------------------------
# when the experiment worktree carries uncommitted work
# ---------------------------------------------------------------------------


def test_finalize_session_when_experiment_worktree_dirty_does_refuse_writing_no_record(
    session_repo: str,
):
    _keep_iteration(session_repo, 1, "cache the regex")
    (Path(experiment_worktree_dir(session_repo)) / "scratch.txt").write_text(
        "notes\n", encoding="utf-8"
    )
    before = len(log_records(session_repo))

    error = capture_error(lambda: finalize_session(session_repo))

    assert error.reason == "dirty-worktree"
    assert error.hint is not None
    assert re.search(r"keep", error.hint, re.IGNORECASE)
    assert re.search(r"discard", error.hint, re.IGNORECASE)
    assert len(log_records(session_repo)) == before


# ---------------------------------------------------------------------------
# when the experiment worktree HEAD is ahead of the last kept commit
# ---------------------------------------------------------------------------


def test_finalize_session_when_experiment_head_ahead_of_last_keep_does_refuse_hinting_keep_or_discard(
    session_repo: str,
):
    _keep_iteration(session_repo, 1, "cache the regex")
    commit_all(
        experiment_worktree_dir(session_repo), "extra commit", file="extra.txt", content="extra\n"
    )
    before = len(log_records(session_repo))

    error = capture_error(lambda: finalize_session(session_repo))

    assert error.reason == "unkept-commits"
    assert error.hint is not None
    assert re.search(r"keep", error.hint, re.IGNORECASE)
    assert re.search(r"discard", error.hint, re.IGNORECASE)
    assert len(log_records(session_repo)) == before


# ---------------------------------------------------------------------------
# when the experiment worktree is already gone from disk
# ---------------------------------------------------------------------------


def test_finalize_session_when_worktree_gone_and_unkept_commits_exist_does_finalize_squashing_last_kept_tree(
    session_repo: str,
):
    last_kept_commit = _keep_iteration(session_repo, 1, "cache the regex")
    last_kept_tree = _git(["rev-parse", f"{last_kept_commit}^{{tree}}"], session_repo)

    worktree = experiment_worktree_dir(session_repo)
    commit_all(worktree, "unkept commit", file="unkept.txt", content="unkept work\n")
    shutil.rmtree(worktree)

    result = finalize_session(session_repo)

    squash_tree = _git(["rev-parse", f"{result.record.branch}^{{tree}}"], session_repo)
    assert squash_tree == last_kept_tree
    assert _last_record(session_repo) == result.record


# ---------------------------------------------------------------------------
# when a committed keep carries no message
# ---------------------------------------------------------------------------


def _keep_without_message(root: str) -> list[str]:
    """Keep one described edit and one bare commit; return the body finalize should write."""
    _keep_iteration(root, 1, "cache the regex")
    commit = _commit_iteration(root, 2, "hoist the loop")
    append_records(root, committed_keep(2, commit=commit, message=None))
    return ["cache the regex", commit[:7]]


def _keep_without_message_or_commit(root: str) -> list[str]:
    """Keep one iteration with neither message nor commit; return the expected body."""
    _commit_iteration(root, 1, "cache the regex")
    append_records(root, committed_keep(1, commit=None, message=None))
    return ["(no message)"]


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_keep_without_message, id="short-commit-stands-in"),
        pytest.param(_keep_without_message_or_commit, id="placeholder-stands-in"),
    ],
)
def test_finalize_session_when_keep_has_no_message_does_stand_a_fallback_line_in(
    session_repo: str, arrange: Callable[[str], list[str]]
):
    expected_body = arrange(session_repo)

    result = finalize_session(session_repo)

    body = _git(["log", "-1", "--format=%b", result.record.branch], session_repo)
    assert body.split("\n") == expected_body


# ---------------------------------------------------------------------------
# when the session has committed keeps
# ---------------------------------------------------------------------------

MESSAGES = ["cache the regex", "hoist the loop"]


@pytest.fixture
def kept_repo(session_repo: str) -> str:
    """A repository whose open session has two committed keeps ready to squash."""
    for index, message in enumerate(MESSAGES):
        _keep_iteration(session_repo, index + 1, message)
    return session_repo


@pytest.fixture
def final_branch(kept_repo: str) -> str:
    """The branch finalize names when the caller does not."""
    return f"{session_header_of(kept_repo).branch}-final"


def test_finalize_session_when_committed_keeps_exist_does_close_the_session_on_one_squash_commit(
    kept_repo: str, repo_head: str, final_branch: str
):
    session_branch = session_header_of(kept_repo).branch
    session_tree = _git(["rev-parse", f"{session_branch}^{{tree}}"], kept_repo)
    session_head = _git(["rev-parse", session_branch], kept_repo)

    result = finalize_session(kept_repo)

    record = result.record
    subject = _git(["log", "-1", "--format=%s", final_branch], kept_repo)
    body = _git(["log", "-1", "--format=%b", final_branch], kept_repo)
    assert _git(["rev-parse", f"{final_branch}^{{tree}}"], kept_repo) == session_tree
    assert _git(["rev-parse", f"{final_branch}^"], kept_repo) == repo_head
    assert _git(["rev-parse", final_branch], kept_repo) == record.commit
    assert (record.type, record.branch) == ("finalize", final_branch)
    assert record.at > 0
    assert _last_record(kept_repo) == record
    assert "2 kept iterations" in subject
    assert body.split("\n") == MESSAGES
    assert record.message == f"{subject}\n\n{body}"
    assert head_of(kept_repo) == repo_head
    assert _git(["rev-parse", "--abbrev-ref", "HEAD"], kept_repo) == "main"
    assert _git(["rev-parse", session_branch], kept_repo) == session_head
    assert not Path(experiment_worktree_dir(kept_repo)).exists()
    assert not Path(baseline_worktree_dir(kept_repo)).exists()
    assert list_worktree_dirs(kept_repo, include_main=False) == []
    assert final_branch in result.report
    assert record.commit[:7] in result.report
    assert "2 kept" in result.report
    assert re.search(r"closed", result.report, re.IGNORECASE)


def test_finalize_session_when_baseline_commit_unknown_does_raise_naming_the_check_command(
    kept_repo: str,
):
    unknown = "0" * 40
    pinned = session_header_of(kept_repo).baseline.sha
    log = Path(session_jsonl_path(kept_repo))
    header, rest = log.read_text(encoding="utf-8").split("\n", 1)
    log.write_text(f"{header.replace(pinned, unknown)}\n{rest}", encoding="utf-8")

    error = capture_error(lambda: finalize_session(kept_repo))

    assert error.hint == (
        f"Check that {unknown} is a commit this repository has: git cat-file -t {unknown}"
    )


def test_finalize_session_when_callers_message_given_does_commit_it_verbatim(
    kept_repo: str, final_branch: str
):
    result = finalize_session(kept_repo, FinalizeOptions(message="squash the tuning session"))

    assert (
        _git(["log", "-1", "--format=%B", final_branch], kept_repo) == "squash the tuning session"
    )
    assert result.record.message == "squash the tuning session"


def test_finalize_session_when_callers_branch_name_given_does_point_it_at_the_squash_commit(
    kept_repo: str,
):
    result = finalize_session(kept_repo, FinalizeOptions(branch="perf/regex-cache"))

    assert result.record.branch == "perf/regex-cache"
    assert _git(["rev-parse", "perf/regex-cache"], kept_repo) == result.record.commit


def test_finalize_session_when_branch_name_looks_like_flag_does_refuse_creating_nothing(
    kept_repo: str,
):
    branches_before = _git(["branch", "--format=%(refname:short)"], kept_repo)
    before = len(log_records(kept_repo))

    error = capture_error(lambda: finalize_session(kept_repo, FinalizeOptions(branch="-m")))

    assert "-m" in str(error)
    assert re.search(r"flag", str(error), re.IGNORECASE)
    assert error.reason == "bad-branch"
    assert error.hint is not None
    assert _git(["branch", "--format=%(refname:short)"], kept_repo) == branches_before
    assert len(log_records(kept_repo)) == before


def test_finalize_session_when_target_branch_exists_does_refuse_creating_nothing(
    kept_repo: str, repo_head: str, final_branch: str
):
    _git(["branch", final_branch, repo_head], kept_repo)
    before = len(log_records(kept_repo))

    error = capture_error(lambda: finalize_session(kept_repo))

    assert final_branch in str(error)
    assert error.reason == "branch-exists"
    assert _git(["rev-parse", final_branch], kept_repo) == repo_head
    assert len(log_records(kept_repo)) == before


def test_finalize_session_when_worktree_removal_refused_does_close_the_session(
    kept_repo: str,
):
    # A locked worktree is the one git declines to take with a single --force,
    # standing in for any removal the filesystem blocks.
    experiment = experiment_worktree_dir(kept_repo)
    _git(["worktree", "lock", experiment], kept_repo)

    result = finalize_session(kept_repo)

    assert experiment in result.report
    assert re.search(r"git worktree remove", result.report, re.IGNORECASE)
    assert _last_record(kept_repo) == result.record


# ---------------------------------------------------------------------------
# when the session has a stop record followed by kept work
# ---------------------------------------------------------------------------


def test_finalize_session_when_stopped_and_has_kept_work_does_close_the_session(kept_repo: str):
    append_records(kept_repo, stop_record())

    result = finalize_session(kept_repo)

    record = _last_record(kept_repo)
    assert isinstance(record, FinalizeRecord)
    assert record == result.record
