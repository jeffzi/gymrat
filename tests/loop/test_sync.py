"""Behavioral tests for syncing uncommitted changes to the experiment worktree.

Every test drives the real ``sync_to_experiment`` against a throwaway repository
from the shared ``create_scratch_repo`` factory, so the suite is order-independent
and safe under ``pytest-xdist`` / ``pytest-randomly``. No git call is mocked: the
sync only reveals its behavior against real worktrees and real dirty files.
"""

import shutil
import stat
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.config import ResolvedConfig
from gymrat.errors import GymratError, hint_of
from gymrat.loop.start import start_session
from gymrat.loop.sync import SyncResult, sync_to_experiment
from gymrat.session import experiment_worktree_dir
from tests._git import git as run_git

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


def test_sync_to_experiment_when_experiment_has_conflicting_changes_does_refuse_leaving_worktree_intact(
    session: str,
):
    experiment = experiment_worktree_dir(session)
    (Path(session) / "README.md").write_text("# Main change\n", encoding="utf-8")
    (Path(experiment) / "README.md").write_text("# Experiment change\n", encoding="utf-8")

    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(session)

    assert "README.md" in str(excinfo.value)
    hint = hint_of(excinfo.value) or ""
    assert "settle" in hint.lower() or "revert" in hint.lower()
    assert (Path(experiment) / "README.md").read_text(encoding="utf-8") == "# Experiment change\n"


# ---------------------------------------------------------------------------
# no open session
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_no_session_does_raise_pointing_at_start(
    repo: str,
):
    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(repo)

    assert "gymrat start" in (hint_of(excinfo.value) or "")


# ---------------------------------------------------------------------------
# non-ASCII / quoted-by-git filenames
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_filename_contains_non_ascii_does_sync_real_path(
    session: str,
):
    """Sync uses the real filesystem path, not the C-quoted string.

    Git with core.quotePath=true C-quotes non-ASCII names.
    """
    run_git(session, "config", "core.quotePath", "true")
    non_ascii_name = "été.txt"  # ete with accents
    (Path(session) / non_ascii_name).write_text("summer\n", encoding="utf-8")

    result = sync_to_experiment(session)

    experiment = experiment_worktree_dir(session)
    assert (Path(experiment) / non_ascii_name).read_text(encoding="utf-8") == "summer\n"
    assert non_ascii_name in result.files


# ---------------------------------------------------------------------------
# renames
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_file_renamed_does_remove_old_path_from_experiment(
    session: str,
):
    """A staged rename removes the old path and syncs the new one."""
    run_git(session, "mv", "README.md", "GUIDE.md")

    result = sync_to_experiment(session)

    experiment = experiment_worktree_dir(session)
    assert not (Path(experiment) / "README.md").exists()
    assert (Path(experiment) / "GUIDE.md").read_text(encoding="utf-8") == "# Test Repo\n"
    assert "GUIDE.md" in result.files


@pytest.mark.skipif(sys.platform == "win32", reason="'>' is illegal in Windows filenames")
def test_sync_to_experiment_when_path_contains_arrow_literal_does_not_misparse_as_rename(  # cspell:disable-line
    session: str,
):
    arrow_name = "a -> b.txt"
    (Path(session) / arrow_name).write_text("literal arrow\n", encoding="utf-8")

    result = sync_to_experiment(session)

    experiment = experiment_worktree_dir(session)
    assert (Path(experiment) / arrow_name).read_text(encoding="utf-8") == "literal arrow\n"
    assert arrow_name in result.files


# ---------------------------------------------------------------------------
# file metadata preservation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_sync_to_experiment_when_file_is_executable_does_preserve_exec_bit(
    session: str,
):
    script = Path(session) / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    script.chmod(0o755)

    sync_to_experiment(session)

    experiment = experiment_worktree_dir(session)
    synced = Path(experiment) / "run.sh"
    assert synced.exists()
    assert synced.stat().st_mode & stat.S_IXUSR


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_sync_to_experiment_when_file_is_symlink_does_sync_as_symlink(
    session: str,
):
    target = Path(session) / "target.txt"
    target.write_text("real content\n", encoding="utf-8")
    link = Path(session) / "link.txt"
    link.symlink_to("target.txt")

    sync_to_experiment(session)

    experiment = experiment_worktree_dir(session)
    synced_link = Path(experiment) / "link.txt"
    assert synced_link.is_symlink()
    assert synced_link.readlink() == Path("target.txt")


# ---------------------------------------------------------------------------
# error wrapping
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_experiment_worktree_missing_does_raise_gymrat_error_with_hint(
    session: str,
):
    experiment = experiment_worktree_dir(session)
    shutil.rmtree(experiment)
    (Path(session) / "change.txt").write_text("trigger\n", encoding="utf-8")

    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(session)

    assert hint_of(excinfo.value) is not None


def test_sync_to_experiment_when_git_status_fails_does_raise_gymrat_error(
    session: str,
):
    """A git failure surfaces as GymratError, not CalledProcessError."""
    index = Path(session) / ".git" / "index"
    index.write_bytes(b"corrupt")
    (Path(session) / "change.txt").write_text("trigger\n", encoding="utf-8")

    with pytest.raises(GymratError):
        sync_to_experiment(session)


# ---------------------------------------------------------------------------
# file-vs-directory error
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_source_is_directory_does_report_expected_file_and_submodule_hint(
    session: str,
):
    readme = Path(session) / "README.md"
    readme.unlink()
    readme.mkdir()
    (readme / "nested.txt").write_text("inside\n", encoding="utf-8")

    with pytest.raises(GymratError, match="expected a file but found a directory") as excinfo:
        sync_to_experiment(session)

    assert "README.md" in str(excinfo.value)
    hint = hint_of(excinfo.value) or ""
    assert "submodule" in hint


# ---------------------------------------------------------------------------
# git status -z parsing: various output shapes
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_file_deleted_does_remove_from_experiment(
    session: str,
):
    run_git(session, "rm", "README.md")

    sync_to_experiment(session)

    experiment = experiment_worktree_dir(session)
    assert not (Path(experiment) / "README.md").exists()


def test_sync_to_experiment_when_mixed_status_types_does_sync_all(
    session: str,
):
    (Path(session) / "README.md").write_text("# Changed\n", encoding="utf-8")
    (Path(session) / "added.py").write_text("x = 1\n", encoding="utf-8")
    run_git(session, "add", ".")
    run_git(session, "mv", "README.md", "GUIDE.md")

    result = sync_to_experiment(session)

    experiment = experiment_worktree_dir(session)
    assert (Path(experiment) / "GUIDE.md").read_text(encoding="utf-8") == "# Changed\n"
    assert (Path(experiment) / "added.py").read_text(encoding="utf-8") == "x = 1\n"
    assert not (Path(experiment) / "README.md").exists()
    assert "GUIDE.md" in result.files
    assert "added.py" in result.files
