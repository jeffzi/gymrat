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

from gymrat.errors import GymratError
from gymrat.loop.sync import sync_to_experiment
from gymrat.session.paths import experiment_worktree_dir
from tests._git import run_git


def _tree_snapshot(root: str) -> dict[str, bytes | str | None]:
    # Maps every path under root to its bytes (file), link target (symlink) or
    # None (directory), so two snapshots compare equal only for identical trees.
    # The worktree's own `.git` pointer file is not part of the synced content.
    snapshot: dict[str, bytes | str | None] = {}
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == ".git":
            continue
        if path.is_symlink():
            snapshot[relative] = str(path.readlink())
        elif path.is_dir():
            snapshot[relative] = None
        else:
            snapshot[relative] = path.read_bytes()
    return snapshot


# ---------------------------------------------------------------------------
# sync with changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "files",
    [
        pytest.param({"README.md": "# Changed\n", "extra.py": "x = 1\n"}, id="several-files"),
        pytest.param({"été.txt": "summer\n"}, id="non-ascii-name"),
        pytest.param(
            {"a -> b.txt": "literal arrow\n"},
            id="arrow-literal-in-name",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="'>' is illegal in Windows filenames"
            ),
        ),
    ],
)
def test_sync_to_experiment_when_changes_present_does_copy_each_changed_file(
    session_repo: str, files: dict[str, str]
):
    # core.quotePath=true C-quotes non-ASCII names; pinned so a developer's global
    # config cannot turn it off, since sync must still copy the real path.
    run_git(["config", "core.quotePath", "true"], session_repo)
    for name, content in files.items():
        (Path(session_repo) / name).write_text(content, encoding="utf-8")

    result = sync_to_experiment(session_repo)

    experiment = experiment_worktree_dir(session_repo)
    synced = {name: (Path(experiment) / name).read_text(encoding="utf-8") for name in files}
    assert synced == files
    assert result.files == tuple(sorted(files))


def _show_session_dir_to_git(root: str) -> None:
    """Drop the session directory's line from the git exclude file the start wrote."""
    exclude = Path(root) / ".git" / "info" / "exclude"
    kept = [line for line in exclude.read_text(encoding="utf-8").splitlines() if line != ".gymrat/"]
    exclude.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")


def test_sync_to_experiment_when_git_reports_session_dir_does_not_sync_it(
    session_repo: str,
):
    # The start hides `.gymrat/` from git; un-hiding it makes git report the
    # session files, so only the sync's own filter keeps them out.
    _show_session_dir_to_git(session_repo)
    gymrat_dir = Path(session_repo) / ".gymrat"
    (gymrat_dir / "should-not-sync.txt").write_text("nope\n", encoding="utf-8")
    (Path(session_repo) / "real.txt").write_text("yes\n", encoding="utf-8")

    result = sync_to_experiment(session_repo)

    experiment = experiment_worktree_dir(session_repo)
    assert not (Path(experiment) / ".gymrat" / "should-not-sync.txt").exists()
    assert "real.txt" in result.files
    assert all(".gymrat" not in f for f in result.files)


# ---------------------------------------------------------------------------
# nothing to sync
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_working_tree_clean_does_return_empty_file_list(
    session_repo: str,
):
    result = sync_to_experiment(session_repo)

    assert result.files == ()


# ---------------------------------------------------------------------------
# conflict refusal
# ---------------------------------------------------------------------------


def _both_modify_readme(session_repo: str) -> str:
    """Edit README.md in the main tree and, differently, in the experiment."""
    (Path(session_repo) / "README.md").write_text("# Main change\n", encoding="utf-8")
    (Path(experiment_worktree_dir(session_repo)) / "README.md").write_text(
        "# Experiment change\n", encoding="utf-8"
    )
    return "README.md"


def _rename_with_dirty_source(session_repo: str) -> str:
    """Rename README.md in the main tree while the experiment edits the source."""
    run_git(["mv", "README.md", "GUIDE.md"], session_repo)
    (Path(experiment_worktree_dir(session_repo)) / "README.md").write_text(
        "# Experiment change\n", encoding="utf-8"
    )
    return "README.md"


def _rename_with_dirty_destination(session_repo: str) -> str:
    """Rename README.md in the main tree while the experiment writes the destination."""
    run_git(["mv", "README.md", "GUIDE.md"], session_repo)
    (Path(experiment_worktree_dir(session_repo)) / "GUIDE.md").write_text(
        "# Experiment change\n", encoding="utf-8"
    )
    return "GUIDE.md"


def _rename_then_delete_with_dirty_source(session_repo: str) -> str:
    """Rename then delete README.md in the main tree while the experiment edits the source."""
    run_git(["mv", "README.md", "GUIDE.md"], session_repo)
    (Path(session_repo) / "GUIDE.md").unlink()
    (Path(experiment_worktree_dir(session_repo)) / "README.md").write_text(
        "# Experiment change\n", encoding="utf-8"
    )
    return "README.md"


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_both_modify_readme, id="modified-on-both-sides"),
        pytest.param(_rename_with_dirty_source, id="rename-source"),
        pytest.param(_rename_with_dirty_destination, id="rename-destination"),
        pytest.param(_rename_then_delete_with_dirty_source, id="rename-then-delete-source"),
    ],
)
def test_sync_to_experiment_when_experiment_has_conflicting_changes_does_refuse_leaving_worktree_intact(
    session_repo: str,
    arrange: Callable[[str], str],
):
    conflicting_path = arrange(session_repo)
    experiment = experiment_worktree_dir(session_repo)
    before = _tree_snapshot(experiment)

    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(session_repo)

    assert str(excinfo.value) == (
        f"Cannot sync — the experiment worktree has uncommitted changes in: {conflicting_path}"
    )
    assert excinfo.value.reason == "dirty-worktree"
    assert excinfo.value.hint == "Settle or revert the experiment worktree first."
    assert _tree_snapshot(experiment) == before


# ---------------------------------------------------------------------------
# renames
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_renamed_file_then_deleted_does_remove_old_path_from_experiment(
    session_repo: str,
):
    run_git(["mv", "README.md", "GUIDE.md"], session_repo)
    (Path(session_repo) / "GUIDE.md").unlink()

    sync_to_experiment(session_repo)

    experiment = experiment_worktree_dir(session_repo)
    assert not (Path(experiment) / "README.md").exists()
    assert not (Path(experiment) / "GUIDE.md").exists()


# ---------------------------------------------------------------------------
# file metadata preservation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_sync_to_experiment_when_file_is_executable_does_preserve_exec_bit(
    session_repo: str,
):
    script = Path(session_repo) / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    script.chmod(0o755)

    sync_to_experiment(session_repo)

    experiment = experiment_worktree_dir(session_repo)
    synced = Path(experiment) / "run.sh"
    assert synced.exists()
    assert synced.stat().st_mode & stat.S_IXUSR


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
@pytest.mark.parametrize(
    "create_target",
    [
        pytest.param(Path.touch, id="file-target"),
        pytest.param(Path.mkdir, id="directory-target"),
    ],
)
def test_sync_to_experiment_when_entry_is_symlink_does_sync_as_symlink(
    session_repo: str,
    create_target: Callable[[Path], None],
):
    create_target(Path(session_repo) / "target")
    (Path(session_repo) / "link").symlink_to("target")

    sync_to_experiment(session_repo)

    experiment = experiment_worktree_dir(session_repo)
    synced_link = Path(experiment) / "link"
    assert synced_link.is_symlink()
    assert synced_link.readlink() == Path("target")


# ---------------------------------------------------------------------------
# error wrapping
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_experiment_worktree_missing_does_raise_gymrat_error_with_hint(
    session_repo: str,
):
    experiment = experiment_worktree_dir(session_repo)
    shutil.rmtree(experiment)
    (Path(session_repo) / "change.txt").write_text("trigger\n", encoding="utf-8")

    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(session_repo)

    assert str(excinfo.value).startswith("Cannot read experiment worktree: ")
    assert excinfo.value.hint == (
        "The experiment worktree may have been deleted. Run 'gymrat start' to begin a new session."
    )


def test_sync_to_experiment_when_git_status_fails_does_raise_gymrat_error(
    session_repo: str,
):
    index = Path(session_repo) / ".git" / "index"
    index.write_bytes(b"corrupt")
    (Path(session_repo) / "change.txt").write_text("trigger\n", encoding="utf-8")

    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(session_repo)

    assert str(excinfo.value).startswith("Cannot read dirty files: ")
    assert excinfo.value.hint == "Check that the repository is not corrupt."


# ---------------------------------------------------------------------------
# file-vs-directory error
# ---------------------------------------------------------------------------


SUBMODULE_HINT = "If this is a submodule, commit or remove it before syncing."


def _readme_replaced_by_directory(session_repo: str) -> str:
    """Replace the tracked README.md in the main tree with a directory."""
    readme = Path(session_repo) / "README.md"
    readme.unlink()
    readme.mkdir()
    (readme / "nested.txt").write_text("inside\n", encoding="utf-8")
    return "README.md"


def _nested_repository(session_repo: str) -> str:
    """Modify README.md and add an untracked nested repository beside it."""
    (Path(session_repo) / "README.md").write_text("# Modified\n", encoding="utf-8")
    nested = Path(session_repo) / "vendor"
    nested.mkdir()
    run_git(["init"], str(nested))
    (nested / "lib.py").write_text("x = 1\n", encoding="utf-8")
    return "vendor/"


def _destination_is_directory(session_repo: str) -> str:
    """Add a main-tree file whose path is a directory in the experiment."""
    (Path(session_repo) / "AAA.txt").write_text("lands first\n", encoding="utf-8")
    (Path(session_repo) / "notes").write_text("a file\n", encoding="utf-8")
    (Path(experiment_worktree_dir(session_repo)) / "notes").mkdir()
    return "notes"


def _rename_source_is_directory(session_repo: str) -> str:
    """Rename README.md in the main tree while the experiment commits it as a directory.

    The experiment commits the swap so its worktree is clean: the refusal comes
    from the directory itself, not from a dirty path.

    Args:
        session_repo: The repository whose session the rename is staged in.

    Returns:
        The path the refusal names.
    """
    experiment = experiment_worktree_dir(session_repo)
    run_git(["mv", "README.md", "GUIDE.md"], session_repo)
    run_git(["rm", "README.md"], experiment)
    (Path(experiment) / "README.md").mkdir()
    (Path(experiment) / "README.md" / "inner.txt").write_text("inside\n", encoding="utf-8")
    run_git(["add", "README.md"], experiment)
    run_git(["commit", "-m", "replace file with directory"], experiment)
    return "README.md"


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_readme_replaced_by_directory, id="tracked-file-became-directory"),
        pytest.param(_nested_repository, id="nested-repository"),
        pytest.param(_destination_is_directory, id="experiment-destination"),
        pytest.param(_rename_source_is_directory, id="experiment-rename-source"),
    ],
)
def test_sync_to_experiment_when_a_path_is_a_directory_on_either_side_does_refuse_leaving_experiment_untouched(
    session_repo: str,
    arrange: Callable[[str], str],
):
    offending = arrange(session_repo)
    experiment = experiment_worktree_dir(session_repo)
    before = _tree_snapshot(experiment)

    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(session_repo)

    assert str(excinfo.value) == f"Cannot sync '{offending}': expected a file but found a directory"
    assert excinfo.value.hint == SUBMODULE_HINT
    assert _tree_snapshot(experiment) == before


def test_sync_to_experiment_when_destination_ancestor_is_file_does_leave_experiment_untouched(
    session_repo: str,
):
    experiment = experiment_worktree_dir(session_repo)
    (Path(session_repo) / "AAA.txt").write_text("lands first\n", encoding="utf-8")
    (Path(session_repo) / "notes").mkdir()
    (Path(session_repo) / "notes" / "x.txt").write_text("nested\n", encoding="utf-8")
    (Path(experiment) / "notes").write_text("a file\n", encoding="utf-8")
    before = _tree_snapshot(experiment)

    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(session_repo)

    assert str(excinfo.value) == (
        "Cannot sync 'notes/x.txt': 'notes' is not a directory in the experiment worktree"
    )
    assert _tree_snapshot(experiment) == before


# ---------------------------------------------------------------------------
# git status -z parsing: various output shapes
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_file_deleted_does_remove_from_experiment(
    session_repo: str,
):
    run_git(["rm", "README.md"], session_repo)

    sync_to_experiment(session_repo)

    experiment = experiment_worktree_dir(session_repo)
    assert not (Path(experiment) / "README.md").exists()


def test_sync_to_experiment_when_mixed_status_types_does_sync_all(
    session_repo: str,
):
    (Path(session_repo) / "README.md").write_text("# Changed\n", encoding="utf-8")
    (Path(session_repo) / "added.py").write_text("x = 1\n", encoding="utf-8")
    run_git(["add", "."], session_repo)
    run_git(["mv", "README.md", "GUIDE.md"], session_repo)

    result = sync_to_experiment(session_repo)

    experiment = experiment_worktree_dir(session_repo)
    assert (Path(experiment) / "GUIDE.md").read_text(encoding="utf-8") == "# Changed\n"
    assert (Path(experiment) / "added.py").read_text(encoding="utf-8") == "x = 1\n"
    assert not (Path(experiment) / "README.md").exists()
    assert "GUIDE.md" in result.files
    assert "added.py" in result.files
