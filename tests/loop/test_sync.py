"""Behavioral tests for syncing uncommitted changes to the experiment worktree.

Every test drives the real ``sync_to_experiment`` against a throwaway repository
from the shared ``create_scratch_repo`` factory, so the suite is order-independent
and safe under ``pytest-xdist`` / ``pytest-randomly``. No git call is mocked: the
sync only reveals its behavior against real worktrees and real dirty files.
The one exception is a copy whose source the main tree leaves unchanged: git
never reports that shape, so that test stubs the main tree's status output.
"""

import shutil
import stat
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.git import run_git_step
from gymrat.loop.sync import sync_to_experiment
from gymrat.session.paths import experiment_worktree_dir
from tests._git import commit_all, git_exclude_path, run_git
from tests.loop._settle import start_with


def _tree_snapshot(root: str) -> dict[str, bytes | str | None]:
    """Snapshot the tree under ``root``, so two snapshots compare equal only for identical trees.

    Args:
        root: The directory walked. The worktree's own `.git` pointer file is
            skipped: it is not part of the synced content.

    Returns:
        Every path under ``root`` mapped to its bytes (file), link target
        (symlink) or ``None`` (directory).
    """
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
    exclude = git_exclude_path(root)
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


def _rename_readme(repo: str) -> None:
    """Rename ``README.md`` to ``GUIDE.md`` in the main tree, keeping its content."""
    run_git(["mv", "README.md", "GUIDE.md"], repo)


def _rename_readme_then_delete(repo: str) -> None:
    """Rename ``README.md`` to ``GUIDE.md``, then delete ``GUIDE.md`` from the working tree."""
    _rename_readme(repo)
    (Path(repo) / "GUIDE.md").unlink()


def _text_or_none(path: Path) -> str | None:
    """The text at ``path``, or ``None`` when nothing is there."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


@pytest.mark.parametrize(
    ("arrange", "expected_guide"),
    [
        pytest.param(_rename_readme, "# Test Repo\n", id="renamed"),
        pytest.param(_rename_readme_then_delete, None, id="renamed-then-deleted"),
    ],
)
def test_sync_to_experiment_when_file_renamed_does_remove_old_path_and_sync_new_one(
    session_repo: str, arrange: Callable[[str], None], expected_guide: str | None
):
    arrange(session_repo)

    sync_to_experiment(session_repo)

    experiment = Path(experiment_worktree_dir(session_repo))
    assert _text_or_none(experiment / "README.md") is None
    assert _text_or_none(experiment / "GUIDE.md") == expected_guide


# ---------------------------------------------------------------------------
# copies
# ---------------------------------------------------------------------------


def test_sync_to_experiment_when_file_copied_does_sync_copy_and_keep_source(
    session_repo: str,
):
    # Git reports a copy only with copy detection on and only from a source
    # changed in the same changeset, so the source is edited and staged too.
    run_git(["config", "status.renames", "copies"], session_repo)
    readme = Path(session_repo) / "README.md"
    shutil.copy(readme, Path(session_repo) / "COPY.md")
    readme.write_text("# Test Repo\nedited\n", encoding="utf-8")
    run_git(["add", "."], session_repo)

    sync_to_experiment(session_repo)

    experiment = experiment_worktree_dir(session_repo)
    assert (Path(experiment) / "COPY.md").read_text(encoding="utf-8") == "# Test Repo\n"
    assert (Path(experiment) / "README.md").read_text(encoding="utf-8") == "# Test Repo\nedited\n"


@pytest.mark.parametrize(
    "experiment_readme",
    [
        pytest.param("# Test Repo\n", id="source-clean-in-experiment"),
        pytest.param("# Experiment change\n", id="source-edited-in-experiment"),
    ],
)
def test_sync_to_experiment_when_copy_reported_does_sync_copy_leaving_source_untouched(
    session_repo: str, monkeypatch: pytest.MonkeyPatch, experiment_readme: str
):
    experiment = experiment_worktree_dir(session_repo)
    shutil.copy(Path(session_repo) / "README.md", Path(session_repo) / "COPY.md")
    (Path(experiment) / "README.md").write_text(experiment_readme, encoding="utf-8")
    main_tree = Path(session_repo).resolve()

    def status_reports_copy(
        args: list[str], cwd: str, message: str, hint: str | None = None
    ) -> str:
        if "status" in args and Path(cwd).resolve() == main_tree:
            return "C  COPY.md\0README.md\0"
        return run_git_step(args, cwd, message, hint)

    monkeypatch.setattr("gymrat.loop.sync.run_git_step", status_reports_copy)

    sync_to_experiment(session_repo)

    assert (Path(experiment) / "COPY.md").read_text(encoding="utf-8") == "# Test Repo\n"
    assert (Path(experiment) / "README.md").read_text(encoding="utf-8") == experiment_readme


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


_NEEDS_POSIX_SYMLINKS = pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")


@_NEEDS_POSIX_SYMLINKS
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


def _outside_file(outside: Path) -> Path:
    """Create a file outside the experiment for a symlink to point at."""
    target = outside / "target.txt"
    target.write_text("outside\n", encoding="utf-8")
    return target


def _outside_directory(outside: Path) -> Path:
    """Create a directory outside the experiment for a symlink to point at."""
    (outside / "x.txt").write_text("outside\n", encoding="utf-8")
    return outside


@_NEEDS_POSIX_SYMLINKS
@pytest.mark.parametrize(
    "create_outside_target",
    [
        pytest.param(_outside_file, id="link-to-outside-file"),
        pytest.param(_outside_directory, id="link-to-outside-directory"),
    ],
)
def test_sync_to_experiment_when_experiment_path_is_symlink_does_replace_link_leaving_target_untouched(
    session_repo: str,
    tmp_path: Path,
    create_outside_target: Callable[[Path], Path],
):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = create_outside_target(outside)
    experiment = Path(experiment_worktree_dir(session_repo))
    (experiment / "notes").symlink_to(target, target_is_directory=target.is_dir())
    commit_all(str(experiment), "add notes as a symlink")
    (Path(session_repo) / "notes").write_text("main\n", encoding="utf-8")
    outside_before = _tree_snapshot(str(outside))

    sync_to_experiment(session_repo)

    synced = experiment / "notes"
    assert not synced.is_symlink()
    assert synced.read_text(encoding="utf-8") == "main\n"
    assert _tree_snapshot(str(outside)) == outside_before


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
        "The experiment worktree may have been deleted. Run gymrat start to begin a new session."
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
    commit_all(experiment, "replace file with directory")
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


def _write_below_notes(repo: str) -> str:
    """Open a session, then add ``notes/x.txt`` in the main tree."""
    start_with(repo)
    (Path(repo) / "AAA.txt").write_text("lands first\n", encoding="utf-8")
    (Path(repo) / "notes").mkdir()
    (Path(repo) / "notes" / "x.txt").write_text("nested\n", encoding="utf-8")
    return "notes/x.txt"


def _remove_below_notes(repo: str) -> str:
    """Commit ``notes/x.txt``, open a session, then delete it in the main tree."""
    (Path(repo) / "notes").mkdir()
    (Path(repo) / "notes" / "x.txt").write_text("nested\n", encoding="utf-8")
    commit_all(repo, "add notes")
    start_with(repo)
    run_git(["rm", "-q", "notes/x.txt"], repo)
    return "notes/x.txt"


def _rename_from_below_notes(repo: str) -> str:
    """Commit ``notes/x.txt``, open a session, then rename it out of ``notes`` in the main tree."""
    (Path(repo) / "notes").mkdir()
    (Path(repo) / "notes" / "x.txt").write_text("nested\n", encoding="utf-8")
    commit_all(repo, "add notes")
    start_with(repo)
    run_git(["mv", "notes/x.txt", "moved.txt"], repo)
    return "notes/x.txt"


def _notes_as_file(experiment: Path, outside: Path) -> None:
    """Make ``notes`` a plain file in the experiment."""
    shutil.rmtree(experiment / "notes", ignore_errors=True)
    (experiment / "notes").write_text("a file\n", encoding="utf-8")


def _notes_as_symlink_to_outside_directory(experiment: Path, outside: Path) -> None:
    """Make ``notes`` in the experiment a committed symlink to a directory outside it."""
    (outside / "x.txt").write_text("outside\n", encoding="utf-8")
    shutil.rmtree(experiment / "notes", ignore_errors=True)
    (experiment / "notes").symlink_to(outside, target_is_directory=True)
    commit_all(str(experiment), "replace notes with a symlink")


@pytest.mark.parametrize(
    ("arrange_main", "arrange_experiment"),
    [
        pytest.param(_write_below_notes, _notes_as_file, id="write-below-file"),
        pytest.param(
            _write_below_notes,
            _notes_as_symlink_to_outside_directory,
            id="write-below-directory-symlink",
            marks=_NEEDS_POSIX_SYMLINKS,
        ),
        pytest.param(
            _remove_below_notes,
            _notes_as_symlink_to_outside_directory,
            id="remove-below-directory-symlink",
            marks=_NEEDS_POSIX_SYMLINKS,
        ),
        pytest.param(
            _rename_from_below_notes,
            _notes_as_symlink_to_outside_directory,
            id="rename-from-below-directory-symlink",
            marks=_NEEDS_POSIX_SYMLINKS,
        ),
    ],
)
def test_sync_to_experiment_when_ancestor_is_not_a_real_directory_does_refuse_leaving_both_trees_untouched(
    repo: str,
    tmp_path: Path,
    arrange_main: Callable[[str], str],
    arrange_experiment: Callable[[Path, Path], None],
):
    offending = arrange_main(repo)
    experiment = experiment_worktree_dir(repo)
    outside = tmp_path / "outside"
    outside.mkdir()
    arrange_experiment(Path(experiment), outside)
    before = (_tree_snapshot(experiment), _tree_snapshot(str(outside)))

    with pytest.raises(GymratError) as excinfo:
        sync_to_experiment(repo)

    assert str(excinfo.value) == (
        f"Cannot sync '{offending}': 'notes' is not a directory in the experiment worktree"
    )
    assert (_tree_snapshot(experiment), _tree_snapshot(str(outside))) == before


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
