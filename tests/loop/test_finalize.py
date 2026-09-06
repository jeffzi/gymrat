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

from gymrat.config import ResolvedConfig, StopConfig
from gymrat.errors import GymratError
from gymrat.loop.finalize import (
    FinalizeOptions,
    finalize_session,
)
from gymrat.loop.start import start_session
from gymrat.session import (
    FinalizeRecord,
    SessionLogRecord,
    SessionRecord,
    append_record,
    baseline_worktree_dir,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests._git import run_git
from tests.session.records._fixtures import committed_keep, iteration_record, stop_record

ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

# A settled run config carrying the keys the session header snapshots; it drives
# ``start_session`` without ever being benched against.
CONFIG = ResolvedConfig(
    bench="npm run bench",
    prepare="npm run build",
    adapter="metric-lines",
    samples=10,
    timeout_seconds=1800,
    unstable_noise_pct=200.0,
    primary="geomean",
    filter="npm run bench -- {names}",
    stop=StopConfig(max_iterations=20),
)


def _git(args: list[str], cwd: str) -> str:
    """Run git in ``cwd`` for test setup and assertions, returning trimmed stdout."""
    return run_git(args, cwd).strip()


def _records(root: str) -> list[SessionLogRecord]:
    """The full session log at ``root``."""
    return read_records(session_jsonl_path(root))


def _session_header(root: str) -> SessionRecord:
    """The session header ``root``'s log opens with, failing when there is none."""
    records = _records(root)
    assert records, f"expected a session header in {session_jsonl_path(root)}"
    first = records[0]
    assert isinstance(first, SessionRecord)
    return first


def _last_record(root: str) -> SessionLogRecord:
    """The record ``root``'s log ends on, failing when the log is empty."""
    records = _records(root)
    assert records, f"expected a record in {session_jsonl_path(root)}"
    return records[-1]


def _commit_iteration(root: str, seq: int, message: str) -> str:
    """Commit one edit in the experiment worktree and log the iteration behind it.

    The experiment worktree is checked out on the session branch, so each call
    moves that branch forward exactly as a real ``gymrat keep`` would. The keep
    record is left to the caller.
    """
    worktree = experiment_worktree_dir(root)
    (Path(worktree) / f"step-{seq}.txt").write_text(f"{message}\n", encoding="utf-8")
    _git(["add", "-A"], worktree)
    _git(["commit", "-m", message], worktree)
    commit = _git(["rev-parse", "HEAD"], worktree)
    append_record(session_jsonl_path(root), iteration_record(seq=seq))
    return commit


def _keep_iteration(root: str, seq: int, message: str) -> str:
    """Commit one edit and log the iteration and the committed keep that settled it."""
    commit = _commit_iteration(root, seq, message)
    append_record(session_jsonl_path(root), committed_keep(seq, commit=commit, message=message))
    return commit


def _capture_error(action: Callable[[], object]) -> GymratError:
    """Run ``action`` expecting a :class:`GymratError`, returning the raised error."""
    with pytest.raises(GymratError) as excinfo:
        action()
    return excinfo.value


def _mentions_keep_and_discard(hint: str) -> bool:
    """Whether ``hint`` names both settling commands."""
    return bool(
        re.search(r"keep", hint, re.IGNORECASE) and re.search(r"discard", hint, re.IGNORECASE)
    )


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str]) -> str:
    """A scratch repository with an open session on ``main``."""
    root = create_scratch_repo()
    start_session(root, "main", CONFIG)
    return root


@pytest.fixture
def baseline_sha(repo: str) -> str:
    """The commit ``main`` sits on — the baseline every squash hangs from.

    ``start_session`` never moves ``main``, so reading it after the session opens
    yields the same commit the baseline is pinned to.
    """
    return _git(["rev-parse", "HEAD"], repo)


# ---------------------------------------------------------------------------
# when the repository holds no session
# ---------------------------------------------------------------------------


def test_finalize_when_no_session_does_refuse_pointing_at_the_command_that_opens_one(
    create_scratch_repo: Callable[[], str],
):
    empty = create_scratch_repo()

    error = _capture_error(lambda: finalize_session(empty))

    assert error.hint is not None
    assert "gymrat start" in error.hint


# ---------------------------------------------------------------------------
# when the session was already finalized
# ---------------------------------------------------------------------------


def test_finalize_when_already_finalized_does_refuse_naming_closed_session_and_fresh_start(
    repo: str,
):
    _keep_iteration(repo, 1, "cache the regex")
    finalize_session(repo)

    error = _capture_error(lambda: finalize_session(repo))

    assert _session_header(repo).session_id in str(error)
    assert error.hint is not None
    assert "gymrat start" in error.hint


# ---------------------------------------------------------------------------
# when nothing has been kept
# ---------------------------------------------------------------------------


def test_finalize_when_nothing_kept_does_refuse_creating_no_branch_and_no_record(repo: str):
    before = len(_records(repo))

    error = _capture_error(lambda: finalize_session(repo))

    assert error.hint is not None
    assert re.search(r"keep", error.hint, re.IGNORECASE)
    assert _git(["branch", "--list", "*-final"], repo) == ""
    assert len(_records(repo)) == before


# ---------------------------------------------------------------------------
# when the last iteration is neither kept nor discarded
# ---------------------------------------------------------------------------


def test_finalize_when_last_iteration_unsettled_does_refuse_writing_no_record(repo: str):
    _keep_iteration(repo, 1, "cache the regex")
    append_record(session_jsonl_path(repo), iteration_record(seq=2))
    before = len(_records(repo))

    error = _capture_error(lambda: finalize_session(repo))

    assert error.hint is not None
    assert _mentions_keep_and_discard(error.hint)
    assert len(_records(repo)) == before


# ---------------------------------------------------------------------------
# when the experiment worktree carries uncommitted work
# ---------------------------------------------------------------------------


def test_finalize_when_experiment_worktree_dirty_does_refuse_writing_no_record(repo: str):
    _keep_iteration(repo, 1, "cache the regex")
    (Path(experiment_worktree_dir(repo)) / "scratch.txt").write_text("notes\n", encoding="utf-8")
    before = len(_records(repo))

    error = _capture_error(lambda: finalize_session(repo))

    assert error.hint is not None
    assert _mentions_keep_and_discard(error.hint)
    assert len(_records(repo)) == before


# ---------------------------------------------------------------------------
# when the experiment worktree HEAD is ahead of the last kept commit
# ---------------------------------------------------------------------------


def test_finalize_when_experiment_head_ahead_of_last_keep_does_refuse_hinting_keep_or_discard(
    repo: str,
):
    _keep_iteration(repo, 1, "cache the regex")
    worktree = experiment_worktree_dir(repo)
    (Path(worktree) / "extra.txt").write_text("extra\n", encoding="utf-8")
    _git(["add", "-A"], worktree)
    _git(["commit", "-m", "extra commit"], worktree)
    before = len(_records(repo))

    error = _capture_error(lambda: finalize_session(repo))

    assert error.hint is not None
    assert _mentions_keep_and_discard(error.hint)
    assert len(_records(repo)) == before


# ---------------------------------------------------------------------------
# when the experiment worktree is already gone from disk
# ---------------------------------------------------------------------------


def test_finalize_when_experiment_worktree_gone_does_finalize_anyway(repo: str):
    _keep_iteration(repo, 1, "cache the regex")
    shutil.rmtree(experiment_worktree_dir(repo))

    result = finalize_session(repo)

    assert _last_record(repo) == result.record


def test_finalize_when_worktree_gone_and_unkept_commits_exist_does_squash_last_kept_tree(
    repo: str,
):
    last_kept_commit = _keep_iteration(repo, 1, "cache the regex")
    last_kept_tree = _git(["rev-parse", f"{last_kept_commit}^{{tree}}"], repo)

    worktree = experiment_worktree_dir(repo)
    (Path(worktree) / "unkept.txt").write_text("unkept work\n", encoding="utf-8")
    _git(["add", "-A"], worktree)
    _git(["commit", "-m", "unkept commit"], worktree)
    session_branch = _session_header(repo).branch
    branch_tip_tree = _git(["rev-parse", f"{session_branch}^{{tree}}"], repo)
    assert last_kept_tree != branch_tip_tree, "precondition: unkept commit changed the tree"

    shutil.rmtree(worktree)

    result = finalize_session(repo)

    squash_tree = _git(["rev-parse", f"{result.record.branch}^{{tree}}"], repo)
    assert squash_tree == last_kept_tree


# ---------------------------------------------------------------------------
# when a committed keep carries no message
# ---------------------------------------------------------------------------


def test_finalize_when_keep_has_no_message_does_stand_short_commit_in(repo: str):
    _keep_iteration(repo, 1, "cache the regex")
    commit = _commit_iteration(repo, 2, "hoist the loop")
    append_record(session_jsonl_path(repo), committed_keep(2, commit=commit, message=None))

    result = finalize_session(repo)

    subject = _git(["log", "-1", "--format=%s", result.record.branch], repo)
    body = _git(["log", "-1", "--format=%b", result.record.branch], repo)
    assert "2 kept iterations" in subject
    assert body.split("\n") == ["cache the regex", commit[:7]]


def test_finalize_when_keep_has_no_message_and_no_commit_does_stand_placeholder_in(repo: str):
    _commit_iteration(repo, 1, "cache the regex")
    append_record(session_jsonl_path(repo), committed_keep(1, commit=None, message=None))

    result = finalize_session(repo)

    body = _git(["log", "-1", "--format=%b", result.record.branch], repo)
    assert body.split("\n") == ["(no message)"]


# ---------------------------------------------------------------------------
# when the session has committed keeps
# ---------------------------------------------------------------------------

MESSAGES = ["cache the regex", "hoist the loop"]


@pytest.fixture
def kept_repo(repo: str) -> str:
    """A repository whose open session has two committed keeps ready to squash."""
    for index, message in enumerate(MESSAGES):
        _keep_iteration(repo, index + 1, message)
    return repo


@pytest.fixture
def final_branch(kept_repo: str) -> str:
    """The branch finalize names when the caller does not."""
    return f"{_session_header(kept_repo).branch}-final"


def test_finalize_when_committed_keeps_exist_does_build_one_commit_carrying_session_tree_on_pinned_baseline(
    kept_repo: str, baseline_sha: str, final_branch: str
):
    session_branch = _session_header(kept_repo).branch
    session_tree = _git(["rev-parse", f"{session_branch}^{{tree}}"], kept_repo)

    result = finalize_session(kept_repo)

    assert _git(["rev-parse", f"{final_branch}^{{tree}}"], kept_repo) == session_tree
    assert _git(["rev-parse", f"{final_branch}^"], kept_repo) == baseline_sha
    assert _git(["rev-parse", final_branch], kept_repo) == result.record.commit


def test_finalize_when_committed_keeps_exist_does_move_neither_checkout_nor_session_branch(
    kept_repo: str, baseline_sha: str
):
    session_branch = _session_header(kept_repo).branch
    session_head = _git(["rev-parse", session_branch], kept_repo)

    finalize_session(kept_repo)

    assert _git(["rev-parse", "HEAD"], kept_repo) == baseline_sha
    assert _git(["rev-parse", "--abbrev-ref", "HEAD"], kept_repo) == "main"
    assert _git(["rev-parse", session_branch], kept_repo) == session_head


def test_finalize_when_committed_keeps_exist_does_append_a_finalize_record_naming_branch_and_squash_commit(
    kept_repo: str, final_branch: str
):
    result = finalize_session(kept_repo)

    record = result.record
    assert record.type == "finalize"
    assert ISO_PATTERN.match(record.at)
    assert record.branch == final_branch
    assert record.commit == _git(["rev-parse", final_branch], kept_repo)
    assert isinstance(record.message, str)
    assert _last_record(kept_repo) == record


def test_finalize_when_committed_keeps_exist_does_take_both_worktrees_off_disk_and_out_of_git(
    kept_repo: str, list_worktree_dirs: Callable[..., list[str]]
):
    finalize_session(kept_repo)

    assert not Path(experiment_worktree_dir(kept_repo)).exists()
    assert not Path(baseline_worktree_dir(kept_repo)).exists()
    assert list_worktree_dirs(kept_repo, include_main=False) == []


def test_finalize_when_committed_keeps_exist_does_report_branch_short_commit_kept_count_and_closed_session(
    kept_repo: str, final_branch: str
):
    result = finalize_session(kept_repo)

    assert final_branch in result.report
    assert result.record.commit[:7] in result.report
    assert "2 kept" in result.report
    assert re.search(r"closed", result.report, re.IGNORECASE)


def test_finalize_when_committed_keeps_exist_does_generate_a_message_naming_kept_count_over_kept_messages(
    kept_repo: str, final_branch: str
):
    result = finalize_session(kept_repo)

    subject = _git(["log", "-1", "--format=%s", final_branch], kept_repo)
    body = _git(["log", "-1", "--format=%b", final_branch], kept_repo)
    assert "2 kept iterations" in subject
    assert body.split("\n") == MESSAGES
    assert result.record.message == f"{subject}\n\n{body}"


def test_finalize_when_callers_message_given_does_commit_it_verbatim(
    kept_repo: str, final_branch: str
):
    result = finalize_session(kept_repo, FinalizeOptions(message="squash the tuning session"))

    assert (
        _git(["log", "-1", "--format=%B", final_branch], kept_repo) == "squash the tuning session"
    )
    assert result.record.message == "squash the tuning session"


def test_finalize_when_callers_branch_name_given_does_point_it_at_the_squash_commit(kept_repo: str):
    result = finalize_session(kept_repo, FinalizeOptions(branch="perf/regex-cache"))

    assert result.record.branch == "perf/regex-cache"
    assert _git(["rev-parse", "perf/regex-cache"], kept_repo) == result.record.commit


def test_finalize_when_branch_name_looks_like_flag_does_refuse_creating_nothing(kept_repo: str):
    branches_before = _git(["branch", "--format=%(refname:short)"], kept_repo)
    before = len(_records(kept_repo))

    error = _capture_error(lambda: finalize_session(kept_repo, FinalizeOptions(branch="-m")))

    assert "-m" in str(error)
    assert re.search(r"flag", str(error), re.IGNORECASE)
    assert error.hint is not None
    assert _git(["branch", "--format=%(refname:short)"], kept_repo) == branches_before
    assert len(_records(kept_repo)) == before


def test_finalize_does_refuse_when_the_target_branch_already_exists_creating_nothing(
    kept_repo: str, baseline_sha: str, final_branch: str
):
    _git(["branch", final_branch, baseline_sha], kept_repo)
    before = len(_records(kept_repo))

    error = _capture_error(lambda: finalize_session(kept_repo))

    assert final_branch in str(error)
    assert _git(["rev-parse", final_branch], kept_repo) == baseline_sha
    assert len(_records(kept_repo)) == before


def test_finalize_does_close_the_session_even_when_git_refuses_to_remove_a_worktree(
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


def test_finalize_when_stopped_and_has_kept_work_does_close_the_session(kept_repo: str):
    append_record(session_jsonl_path(kept_repo), stop_record())

    result = finalize_session(kept_repo)

    record = _last_record(kept_repo)
    assert isinstance(record, FinalizeRecord)
    assert record == result.record
