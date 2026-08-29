"""Behavioral tests for syncing uncommitted changes to the experiment worktree.

Every test drives the real ``sync_to_experiment`` against a throwaway repository
from the shared ``create_scratch_repo`` factory, so the suite is order-independent
and safe under ``pytest-xdist`` / ``pytest-randomly``. No git call is mocked: the
sync only reveals its behavior against real worktrees and real dirty files.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.config import ResolvedConfig
from gymrat.errors import GymratError, hint_of
from gymrat.loop.start import start_session
from gymrat.loop.sync import SyncResult, sync_to_experiment
from gymrat.session import experiment_worktree_dir

CONFIG = ResolvedConfig(
    bench="echo ok",
    adapter="metric-lines",
    samples=1,
    timeout_seconds=60,
    unstable_noise_pct=200.0,
    primary="geomean",
)


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str]) -> str:
    return create_scratch_repo()


@pytest.fixture
def session(repo: str) -> str:
    """A scratch repo with an open session."""
    start_session(repo, None, CONFIG)
    return repo


# ---------------------------------------------------------------------------
# sync with changes
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_tracked_file_modified_does_apply_change_to_experiment(
    session: str,
):
    (Path(session) / "README.md").write_text("# Modified\n", encoding="utf-8")

    result = sync_to_experiment(session)

    experiment = experiment_worktree_dir(session)
    assert (Path(experiment) / "README.md").read_text(encoding="utf-8") == "# Modified\n"
    assert isinstance(result, SyncResult)
    assert "README.md" in result.files


def test_sync_to_experiment_when_untracked_file_added_does_copy_it_to_experiment(
    session: str,
):
    (Path(session) / "new_file.txt").write_text("hello\n", encoding="utf-8")

    result = sync_to_experiment(session)

    experiment = experiment_worktree_dir(session)
    assert (Path(experiment) / "new_file.txt").read_text(encoding="utf-8") == "hello\n"
    assert "new_file.txt" in result.files


def test_sync_to_experiment_when_changes_present_does_return_all_synced_paths(
    session: str,
):
    (Path(session) / "README.md").write_text("# Changed\n", encoding="utf-8")
    (Path(session) / "extra.py").write_text("x = 1\n", encoding="utf-8")

    result = sync_to_experiment(session)

    assert set(result.files) == {"README.md", "extra.py"}


def test_sync_to_experiment_when_changes_present_does_not_sync_gymrat_dir(
    session: str,
):
    gymrat_dir = Path(session) / ".gymrat"
    (gymrat_dir / "should-not-sync.txt").write_text("nope\n", encoding="utf-8")
    (Path(session) / "real.txt").write_text("yes\n", encoding="utf-8")

    result = sync_to_experiment(session)

    experiment = experiment_worktree_dir(session)
    assert not (Path(experiment) / ".gymrat" / "should-not-sync.txt").exists()
    assert "real.txt" in result.files
    assert all(".gymrat" not in f for f in result.files)


# ---------------------------------------------------------------------------
# nothing to sync
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_working_tree_clean_does_return_empty_file_list(
    session: str,
):
    result = sync_to_experiment(session)

    assert isinstance(result, SyncResult)
    assert result.files == ()


# ---------------------------------------------------------------------------
# conflict refusal
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_experiment_has_conflicting_changes_does_raise_naming_files(
    session: str,
):
    (Path(session) / "README.md").write_text("# Main change\n", encoding="utf-8")
    experiment = experiment_worktree_dir(session)
    (Path(experiment) / "README.md").write_text("# Experiment change\n", encoding="utf-8")

    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(session)

    assert "README.md" in str(excinfo.value)


def test_sync_to_experiment_when_experiment_has_conflicting_changes_does_leave_worktree_unchanged(
    session: str,
):
    experiment = experiment_worktree_dir(session)
    (Path(session) / "README.md").write_text("# Main change\n", encoding="utf-8")
    (Path(experiment) / "README.md").write_text("# Experiment change\n", encoding="utf-8")

    with pytest.raises(GymratError):
        sync_to_experiment(session)

    assert (Path(experiment) / "README.md").read_text(encoding="utf-8") == "# Experiment change\n"


def test_sync_to_experiment_when_experiment_has_conflicting_changes_does_hint_to_settle_or_revert(
    session: str,
):
    (Path(session) / "README.md").write_text("# Main change\n", encoding="utf-8")
    experiment = experiment_worktree_dir(session)
    (Path(experiment) / "README.md").write_text("# Experiment change\n", encoding="utf-8")

    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(session)

    hint = hint_of(excinfo.value) or ""
    assert "settle" in hint.lower() or "revert" in hint.lower()


# ---------------------------------------------------------------------------
# no open session
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_no_session_does_raise_pointing_at_start(
    repo: str,
):
    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(repo)

    assert "gymrat start" in (hint_of(excinfo.value) or "")
