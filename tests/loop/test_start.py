"""Behavioral tests for starting or resuming a session (``start_session``).

Every test drives the real ``start_session`` against a throwaway repository from
the shared ``create_scratch_repo`` factory, so the suite is order-independent and
safe under ``pytest-xdist`` / ``pytest-randomly``. No git call is mocked: the
worktree checkouts, resume recreation, and finalize archive-and-recreate only
reveal their behavior against real worktrees, and the assertions read commit SHAs
straight out of the worktrees git laid down.
"""

import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.config import HooksConfig, ResolvedConfig, StopConfig
from gymrat.errors import GymratError
from gymrat.loop.start import StartResult, start_session
from gymrat.session import (
    BaselineRef,
    SessionConfig,
    SessionHooks,
    SessionRecord,
    Worktrees,
    append_record,
    archived_session_path,
    baseline_worktree_dir,
    experiment_worktree_dir,
    fold_session,
    read_records,
    remove_worktrees,
    session_jsonl_path,
)
from tests.session._records import committed_keep, finalize_record, iteration_record

SESSION_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
BRANCH_PATTERN = re.compile(r"^gymrat/\d{8}-\d{6}-[0-9a-f]{4}$")
ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

HOOKS = HooksConfig(before="npm run warm-cache", after="npm run cool-down")

# A settled run config carrying both the keys the session header snapshots and
# keys it must leave out (``unstable_noise_pct``, ``stop``).
CONFIG_WITHOUT_HOOKS = ResolvedConfig(
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
    hooks=HOOKS,
)

# The subset of ``CONFIG_WITHOUT_HOOKS`` the header keeps as provenance.
CONFIG_SNAPSHOT_WITHOUT_HOOKS = SessionConfig(
    bench="npm run bench",
    adapter="metric-lines",
    samples=10,
    timeout_seconds=1800,
    primary="geomean",
    prepare="npm run build",
    filter="npm run bench -- {names}",
)

# The provenance the header keeps once hooks are configured.
CONFIG_SNAPSHOT = SessionConfig(
    bench="npm run bench",
    adapter="metric-lines",
    samples=10,
    timeout_seconds=1800,
    primary="geomean",
    prepare="npm run build",
    filter="npm run bench -- {names}",
    hooks=SessionHooks(before="npm run warm-cache", after="npm run cool-down"),
)


def _git(args: list[str], cwd: str) -> str:
    """Run git in ``cwd`` for test setup and assertions, returning trimmed stdout."""
    from tests._git import run_git

    return run_git(args, cwd).strip()


def _session_header_of(root: str) -> SessionRecord:
    """The session header ``root``'s log opens with, failing when there is none."""
    records = read_records(session_jsonl_path(root))
    assert records, f"expected a session header in {session_jsonl_path(root)}"
    first = records[0]
    assert isinstance(first, SessionRecord)
    return first


def _commit_in_experiment(root: str, message: str) -> str:
    """Commit an edit on the session branch from the experiment worktree, returning its SHA.

    Stands in for the commit a keep makes, so the SHA is a real commit a worktree
    can later be checked out at.
    """
    worktree = experiment_worktree_dir(root)
    (Path(worktree) / "README.md").write_text(f"# {message}\n", encoding="utf-8")
    _git(["add", "README.md"], worktree)
    _git(["commit", "-m", message], worktree)
    return _git(["rev-parse", "HEAD"], worktree)


def _close_session_with_one_keep(root: str) -> str:
    """Keep one commit on the open session and close it, returning the id it closed on.

    Mirrors what a real finalize leaves behind — worktrees off disk and a finalize
    record ending the log — by committing a keep, taking the worktrees down, and
    appending a finalize record, so the next ``start_session`` meets a settled,
    closed session.
    """
    header = _session_header_of(root)
    commit = _commit_in_experiment(root, "cache the regex")
    jsonl = session_jsonl_path(root)
    append_record(jsonl, iteration_record(seq=1))
    append_record(jsonl, committed_keep(1, commit=commit))
    remove_worktrees(root, header.worktrees)
    append_record(jsonl, finalize_record())
    return header.session_id


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str]) -> str:
    return create_scratch_repo()


@pytest.fixture
def head_sha(repo: str) -> str:
    return _git(["rev-parse", "HEAD"], repo)


# ---------------------------------------------------------------------------
# when the repository holds no session yet
# ---------------------------------------------------------------------------


def test_start_session_when_no_session_yet_does_write_header_naming_baseline_branch_worktrees_and_config(
    repo: str, head_sha: str
):
    start_session(repo, "main", CONFIG)

    header = _session_header_of(repo)
    assert read_records(session_jsonl_path(repo)) == [header]
    assert header.type == "session"
    assert header.schema_version == 1
    assert SESSION_ID_PATTERN.match(header.session_id)
    assert ISO_PATTERN.match(header.created_at)
    assert header.baseline == BaselineRef(ref="main", sha=head_sha)
    assert BRANCH_PATTERN.match(header.branch)
    assert header.worktrees == Worktrees(
        experiment=experiment_worktree_dir(repo),
        baseline=baseline_worktree_dir(repo),
    )
    assert header.config == CONFIG_SNAPSHOT


def test_start_session_when_no_hooks_configured_does_leave_hooks_out_of_the_config_snapshot(
    repo: str,
):
    start_session(repo, "main", CONFIG_WITHOUT_HOOKS)

    header = _session_header_of(repo)
    assert header.config.hooks is None
    assert header.config == CONFIG_SNAPSHOT_WITHOUT_HOOKS


def test_start_session_when_new_does_name_the_branch_after_the_session_id(repo: str):
    result = start_session(repo, "main", CONFIG)

    assert result.session.branch == f"gymrat/{result.session.session_id}"


def test_start_session_when_new_does_check_out_the_experiment_and_baseline_worktrees(repo: str):
    start_session(repo, "main", CONFIG)

    assert Path(experiment_worktree_dir(repo)).exists()
    assert Path(baseline_worktree_dir(repo)).exists()


def test_start_session_when_new_does_return_the_recorded_session_with_no_history(repo: str):
    result = start_session(repo, "main", CONFIG)

    header = _session_header_of(repo)
    assert result == StartResult(
        session=header,
        state=fold_session([header]),
        resumed=False,
    )


def test_start_session_when_no_ref_given_does_pin_the_baseline_at_head(repo: str, head_sha: str):
    result = start_session(repo, None, CONFIG)

    assert result.session.baseline == BaselineRef(ref="HEAD", sha=head_sha)


# ---------------------------------------------------------------------------
# when a session is already on disk
# ---------------------------------------------------------------------------


def test_start_session_when_session_on_disk_does_resume_returning_counts_without_appending(
    repo: str,
):
    created = start_session(repo, "main", CONFIG).session
    jsonl = session_jsonl_path(repo)
    append_record(jsonl, iteration_record(seq=1))
    append_record(jsonl, committed_keep(1))

    result = start_session(repo, "main", CONFIG)

    assert result.session == created
    assert result.resumed is True
    assert result.state.iteration_count == 1
    assert result.state.keep_count == 1
    assert len(read_records(jsonl)) == 3


def test_start_session_when_experiment_worktree_missing_does_put_it_back(repo: str):
    start_session(repo, "main", CONFIG)
    shutil.rmtree(experiment_worktree_dir(repo))

    start_session(repo, "main", CONFIG)

    assert Path(experiment_worktree_dir(repo)).exists()


# ---------------------------------------------------------------------------
# when the session on disk was finalized
# ---------------------------------------------------------------------------


def test_start_session_when_finalized_does_move_the_closed_log_aside_under_its_session_id(
    repo: str,
):
    start_session(repo, "main", CONFIG)
    closed = _close_session_with_one_keep(repo)
    closed_log = read_records(session_jsonl_path(repo))

    start_session(repo, "main", CONFIG)

    assert read_records(archived_session_path(repo, closed)) == closed_log


def test_start_session_when_finalized_does_open_a_fresh_session_in_the_vacated_log(repo: str):
    start_session(repo, "main", CONFIG)
    closed = _close_session_with_one_keep(repo)

    result = start_session(repo, "main", CONFIG)

    assert result.archived == closed
    assert result.archived_path == archived_session_path(repo, closed)
    assert result.resumed is False
    assert result.state.finalized is None
    assert result.session.session_id != closed
    assert read_records(session_jsonl_path(repo)) == [result.session]


def test_start_session_when_finalized_does_check_out_both_worktrees_at_the_pinned_baseline(
    repo: str, head_sha: str
):
    start_session(repo, "main", CONFIG)
    _close_session_with_one_keep(repo)

    start_session(repo, "main", CONFIG)

    assert _git(["rev-parse", "HEAD"], experiment_worktree_dir(repo)) == head_sha
    assert _git(["rev-parse", "HEAD"], baseline_worktree_dir(repo)) == head_sha


@pytest.mark.skipif(sys.platform == "win32", reason="post-checkout SIGKILL is POSIX-only")
def test_start_session_when_fresh_workspace_after_finalize_dies_does_put_the_closed_log_back(
    repo: str, kill_git_during_worktree_add: Callable[[str], None]
):
    start_session(repo, "main", CONFIG)
    closed = _close_session_with_one_keep(repo)
    closed_log = read_records(session_jsonl_path(repo))
    kill_git_during_worktree_add(repo)

    with pytest.raises(GymratError) as excinfo:
        start_session(repo, "main", CONFIG)

    # The rollback's own failure never speaks for the start's.
    assert re.search(r"cannot create the experiment worktree", str(excinfo.value), re.IGNORECASE)
    assert read_records(session_jsonl_path(repo)) == closed_log
    assert not Path(archived_session_path(repo, closed)).exists()


# ---------------------------------------------------------------------------
# when the baseline worktree went missing
# ---------------------------------------------------------------------------


def test_start_session_when_baseline_worktree_missing_does_put_it_back_at_the_last_kept_commit(
    repo: str,
):
    start_session(repo, "main", CONFIG)
    kept = _commit_in_experiment(repo, "cache the regex")
    jsonl = session_jsonl_path(repo)
    append_record(jsonl, iteration_record(seq=1))
    append_record(jsonl, committed_keep(1, commit=kept))
    shutil.rmtree(baseline_worktree_dir(repo))

    start_session(repo, "main", CONFIG)

    assert _git(["rev-parse", "HEAD"], baseline_worktree_dir(repo)) == kept


def test_start_session_when_baseline_worktree_missing_and_nothing_kept_does_put_it_back_at_pinned_sha(
    repo: str, head_sha: str
):
    start_session(repo, "main", CONFIG)
    _commit_in_experiment(repo, "work the agent has not kept")
    shutil.rmtree(baseline_worktree_dir(repo))

    start_session(repo, "main", CONFIG)

    assert _git(["rev-parse", "HEAD"], baseline_worktree_dir(repo)) == head_sha


# ---------------------------------------------------------------------------
# when the baseline ref cannot be used
# ---------------------------------------------------------------------------


def test_start_session_when_baseline_ref_does_not_resolve_does_raise_and_leave_no_session(
    repo: str,
):
    with pytest.raises(GymratError) as excinfo:
        start_session(repo, "no-such-ref", CONFIG)

    assert "no-such-ref" in str(excinfo.value)
    assert not Path(session_jsonl_path(repo)).exists()


def test_start_session_when_baseline_ref_is_a_directory_does_raise_naming_the_ref(
    repo: str, create_in_place_target_dir: Callable[[str, str, str], str]
):
    target_dir = create_in_place_target_dir(repo, "bench-dir", "echo hi\n")

    with pytest.raises(GymratError) as excinfo:
        start_session(repo, target_dir, CONFIG)

    assert target_dir in str(excinfo.value)
    assert not Path(session_jsonl_path(repo)).exists()
