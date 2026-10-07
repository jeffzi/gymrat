"""Behavioral tests for the session git-workspace orchestration.

Every test runs real git against a throwaway repository from the shared
``create_scratch_repo`` factory, so the suite is order-independent and safe
under ``pytest-xdist`` / ``pytest-randomly``. No git call is mocked: the module
under test is pure git orchestration, and only real worktrees reveal the
pruning, unwind, and detachment behavior these tests pin.
"""

import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.session.paths import baseline_worktree_dir, experiment_worktree_dir
from gymrat.session.workspace import (
    BaselineRef,
    WorkspaceResult,
    advance_baseline,
    commit_workspace,
    create_workspace,
    dirty_file_count,
    ensure_git_exclude,
    recreate_workspace,
    remove_worktrees,
    revert_workspace,
    worktree_fingerprint,
    worktree_head,
)
from tests._git import (
    commit_all,
    head_of,
    install_git_hook,
    kill_git_during_worktree_add,
    list_worktree_dirs,
    register_absent_worktree,
    session_branches,
    status_of,
)
from tests._git import run_git as _git
from tests.session.records._fixtures import (
    SESSION_ID,
    worktrees_at,
)

BRANCH = f"gymrat/{SESSION_ID}"
BASELINE_REF = "main"

# The id the session after SESSION_ID opens on, for a workspace built over an
# earlier one's leftovers.
NEXT_SESSION_ID = "20260808-152045-b7c1"
NEXT_BRANCH = f"gymrat/{NEXT_SESSION_ID}"


def _checked_out_ref(worktree: str) -> str:
    """The ref a worktree has checked out: a branch name, or ``HEAD`` when detached."""
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], worktree)


def _exclude_path(root: str) -> Path:
    return Path(root) / ".git" / "info" / "exclude"


def _both_worktrees_exist(root: str) -> bool:
    return (
        Path(experiment_worktree_dir(root)).exists() and Path(baseline_worktree_dir(root)).exists()
    )


def _edit_worktree(worktree: str, edit: str | None) -> None:
    """Write agent-edit content to ``edit`` in ``worktree``, or leave it clean when ``None``."""
    if edit is not None:
        (Path(worktree) / edit).write_text("# edited by the agent\n", encoding="utf-8")


@pytest.fixture
def baseline(repo_head: str) -> BaselineRef:
    """A ``BaselineRef`` pointing at the scratch repo's initial commit on ``main``."""
    return BaselineRef(ref=BASELINE_REF, sha=repo_head)


# ---------------------------------------------------------------------------
# create_workspace
# ---------------------------------------------------------------------------


def test_create_workspace_when_no_session_workspace_does_build_branch_worktrees_and_descriptor(
    repo: str, repo_head: str, baseline: BaselineRef
):
    result = create_workspace(repo, SESSION_ID, baseline)

    exp = experiment_worktree_dir(repo)
    bl = baseline_worktree_dir(repo)
    assert _git(["rev-parse", BRANCH], repo) == repo_head
    assert Path(exp).exists()
    assert _checked_out_ref(exp) == BRANCH
    assert head_of(bl) == repo_head
    assert _checked_out_ref(bl) == "HEAD"
    assert ".gymrat/" in _exclude_path(repo).read_text(encoding="utf-8").split("\n")
    assert result == WorkspaceResult(branch=BRANCH, worktrees=worktrees_at(repo))


def test_create_workspace_when_branch_already_exists_does_raise_naming_branch_and_hint(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)

    with pytest.raises(GymratError) as excinfo:
        create_workspace(repo, SESSION_ID, baseline)

    assert BRANCH in str(excinfo.value)
    assert re.search(r"git branch -D", excinfo.value.hint or "", re.IGNORECASE)


@pytest.mark.skipif(sys.platform == "win32", reason="post-checkout SIGKILL is POSIX-only")
def test_create_workspace_when_worktree_add_dies_does_unwind_and_fail_on_that_step(
    repo: str,
    baseline: BaselineRef,
):
    # Installed after the scratch repo's own commit so only the worktree
    # checkouts under test die.
    kill_git_during_worktree_add(repo)

    with pytest.raises(GymratError) as excinfo:
        create_workspace(repo, SESSION_ID, baseline)

    # The unwind's own git steps never speak for it.
    assert re.search(r"cannot create the experiment worktree", str(excinfo.value), re.IGNORECASE)
    assert session_branches(repo) == []
    assert list_worktree_dirs(repo, include_main=False) == []
    assert not Path(experiment_worktree_dir(repo)).exists()


def test_create_workspace_when_registry_entries_are_stale_does_check_out_over_them(
    repo: str, repo_head: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    shutil.rmtree(experiment_worktree_dir(repo))
    shutil.rmtree(baseline_worktree_dir(repo))

    result = create_workspace(repo, NEXT_SESSION_ID, baseline)

    assert _checked_out_ref(result.worktrees.experiment) == NEXT_BRANCH
    assert head_of(result.worktrees.baseline) == repo_head


def test_create_workspace_when_registry_stale_does_leave_the_users_own_worktrees_registered(
    repo: str,
    repo_head: str,
    baseline: BaselineRef,
):
    create_workspace(repo, SESSION_ID, baseline)
    shutil.rmtree(experiment_worktree_dir(repo))
    shutil.rmtree(baseline_worktree_dir(repo))
    live = str(Path(repo) / "live-worktree")
    _git(["worktree", "add", "--detach", live, repo_head], repo)
    absent = register_absent_worktree(repo)

    create_workspace(repo, NEXT_SESSION_ID, baseline)

    registered = list_worktree_dirs(repo, include_main=False)
    assert Path(live).exists()
    assert (live in registered, absent in registered) == (True, True)


def test_create_workspace_when_earlier_worktree_still_on_disk_does_leave_its_work_and_name_the_path(
    repo: str, baseline: BaselineRef
):
    # The earlier session's log is gone, so nothing told this run the workspace
    # was already there; its worktree still holds uncommitted work.
    create_workspace(repo, SESSION_ID, baseline)
    stranded = Path(experiment_worktree_dir(repo)) / "README.md"
    stranded.write_text("# work from the earlier session\n", encoding="utf-8")

    with pytest.raises(GymratError) as excinfo:
        create_workspace(repo, NEXT_SESSION_ID, baseline)

    # Only this attempt's own branch is unwound.
    assert stranded.read_text(encoding="utf-8") == "# work from the earlier session\n"
    assert session_branches(repo) == [BRANCH]
    assert experiment_worktree_dir(repo) in str(excinfo.value)


def test_create_workspace_when_directory_is_not_a_git_repository_does_raise(
    tmp_path: Path, repo_head: str
):
    outside = str(tmp_path)

    with pytest.raises(GymratError) as excinfo:
        create_workspace(outside, SESSION_ID, BaselineRef(ref=BASELINE_REF, sha=repo_head))

    assert re.search(r"not a git repository", str(excinfo.value), re.IGNORECASE)
    assert re.search(r"git repository", excinfo.value.hint or "", re.IGNORECASE)


# ---------------------------------------------------------------------------
# ensure_git_exclude
# ---------------------------------------------------------------------------


def _seed_exclude(repo: str, content: bytes | None) -> Path:
    """Leave the repo's exclude file holding *content*, or absent when it is ``None``."""
    path = _exclude_path(repo)
    path.unlink(missing_ok=True)
    if content is not None:
        path.write_bytes(content)
    return path


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            b"node_modules/\n.gymrat/\n", b"node_modules/\n.gymrat/\n", id="already-listed"
        ),
        pytest.param(
            b"node_modules/\nbuild/\n", b"node_modules/\nbuild/\n.gymrat/\n", id="entry-missing"
        ),
        pytest.param(None, b".gymrat/\n", id="file-missing"),
        pytest.param(b"build/\n# \xe9\n", b"build/\n# \xe9\n.gymrat/\n", id="non-utf8-bytes"),
        pytest.param(b"build/", b"build/\n.gymrat/\n", id="last-line-unterminated"),
        pytest.param(
            b"# \xe9\n.gymrat/\n", b"# \xe9\n.gymrat/\n", id="listed-beside-non-utf8-bytes"
        ),
    ],
)
def test_ensure_git_exclude_when_called_does_list_the_session_dir_once_keeping_other_bytes(
    repo: str, before: bytes | None, after: bytes
):
    path = _seed_exclude(repo, before)

    ensure_git_exclude(repo)

    assert path.read_bytes() == after


def test_ensure_git_exclude_when_file_cannot_be_read_does_raise_naming_the_file(repo: str):
    path = _exclude_path(repo)
    path.unlink(missing_ok=True)
    path.mkdir()

    with pytest.raises(GymratError) as excinfo:
        ensure_git_exclude(repo)

    assert str(path) in str(excinfo.value)


# ---------------------------------------------------------------------------
# remove_worktrees
# ---------------------------------------------------------------------------


def _nothing_gone(_repo: str) -> None:
    """Leave both worktrees on disk."""


def _experiment_gone(repo: str) -> None:
    """Delete the experiment worktree's directory, leaving its registration."""
    shutil.rmtree(experiment_worktree_dir(repo))


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_nothing_gone, id="both-on-disk"),
        pytest.param(_experiment_gone, id="one-directory-already-gone"),
    ],
)
def test_remove_worktrees_when_called_does_remove_both_without_warning(
    repo: str, baseline: BaselineRef, arrange: Callable[[str], None]
):
    create_workspace(repo, SESSION_ID, baseline)
    arrange(repo)

    warnings = remove_worktrees(repo, worktrees_at(repo))

    assert warnings == []
    assert not Path(experiment_worktree_dir(repo)).exists()
    assert not Path(baseline_worktree_dir(repo)).exists()
    assert list_worktree_dirs(repo, include_main=False) == []


def test_remove_worktrees_when_one_gone_does_deregister_by_name_only(
    repo: str,
    baseline: BaselineRef,
):
    create_workspace(repo, SESSION_ID, baseline)
    # The user's own worktree, absent only for the moment.
    absent = register_absent_worktree(repo)
    shutil.rmtree(experiment_worktree_dir(repo))

    remove_worktrees(repo, worktrees_at(repo))

    listed = list_worktree_dirs(repo)
    assert experiment_worktree_dir(repo) not in listed
    assert absent in listed


def test_remove_worktrees_when_git_refuses_does_warn_naming_it_and_remove_the_other(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    # git declines a locked worktree unless --force is passed twice.
    _git(["worktree", "lock", experiment_worktree_dir(repo)], repo)

    warnings = remove_worktrees(repo, worktrees_at(repo))

    assert len(warnings) == 1
    assert experiment_worktree_dir(repo) in warnings[0]
    assert not Path(baseline_worktree_dir(repo)).exists()


# ---------------------------------------------------------------------------
# dirty_file_count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        pytest.param(None, 0, id="nothing-touched"),
        pytest.param("README.md", 1, id="tracked-file-edited"),
        pytest.param("scratch.txt", 1, id="untracked-file-added"),
    ],
)
def test_dirty_file_count_when_worktree_edited_does_return_entry_count(
    repo: str, baseline: BaselineRef, edit: str | None, expected: int
):
    create_workspace(repo, SESSION_ID, baseline)
    worktree = experiment_worktree_dir(repo)
    _edit_worktree(worktree, edit)

    count = dirty_file_count(worktree)

    assert count == expected


def test_dirty_file_count_when_untracked_directory_does_expand_to_individual_files(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    worktree = experiment_worktree_dir(repo)
    subdir = Path(worktree) / "extras"
    subdir.mkdir()
    (subdir / "alpha.py").write_text("# alpha\n", encoding="utf-8")
    (subdir / "beta.py").write_text("# beta\n", encoding="utf-8")
    (subdir / "gamma.py").write_text("# gamma\n", encoding="utf-8")

    count = dirty_file_count(worktree)

    assert count == 3


def test_dirty_file_count_when_directory_missing_does_return_zero(repo: str):
    count = dirty_file_count(experiment_worktree_dir(repo))

    assert count == 0


# ---------------------------------------------------------------------------
# recreate_workspace
# ---------------------------------------------------------------------------


def test_recreate_workspace_when_experiment_gone_does_put_it_back_on_the_branch(
    repo: str, repo_head: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    shutil.rmtree(experiment_worktree_dir(repo))

    recreate_workspace(repo, BRANCH, repo_head)

    assert _checked_out_ref(experiment_worktree_dir(repo)) == BRANCH
    assert _both_worktrees_exist(repo)


def test_recreate_workspace_when_experiment_gone_does_leave_absent_user_worktree_registered(
    repo: str,
    repo_head: str,
    baseline: BaselineRef,
):
    create_workspace(repo, SESSION_ID, baseline)
    user_worktree = str(Path(repo) / "user-worktree")
    _git(["worktree", "add", "--detach", user_worktree, repo_head], repo)
    shutil.rmtree(user_worktree)
    shutil.rmtree(experiment_worktree_dir(repo))

    recreate_workspace(repo, BRANCH, repo_head)

    assert user_worktree in list_worktree_dirs(repo, include_main=False)


def test_recreate_workspace_when_baseline_gone_does_put_it_back_detached_at_sha(
    repo: str, repo_head: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    shutil.rmtree(baseline_worktree_dir(repo))

    recreate_workspace(repo, BRANCH, repo_head)

    worktree = baseline_worktree_dir(repo)
    assert head_of(worktree) == repo_head
    assert _checked_out_ref(worktree) == "HEAD"


def test_recreate_workspace_when_both_on_disk_does_leave_experiment_work_untouched(
    repo: str, repo_head: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    edited = Path(experiment_worktree_dir(repo)) / "README.md"
    edited.write_text("# edited by the agent\n", encoding="utf-8")

    recreate_workspace(repo, BRANCH, repo_head)

    assert edited.read_text(encoding="utf-8") == "# edited by the agent\n"
    assert _both_worktrees_exist(repo)


# ---------------------------------------------------------------------------
# commit_workspace / revert_workspace / worktree_head / advance_baseline
# ---------------------------------------------------------------------------


def test_commit_workspace_when_changes_staged_and_untracked_does_commit_and_return_new_head(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)
    (Path(experiment) / "README.md").write_text("# edited by the agent\n", encoding="utf-8")
    (Path(experiment) / "new-file.txt").write_text("brand new\n", encoding="utf-8")
    before = head_of(experiment)

    sha = commit_workspace(experiment, "agent change")

    assert sha != before
    assert sha == head_of(experiment)
    committed = _git(["show", "--name-only", "--format=", "HEAD"], experiment).split("\n")
    assert "README.md" in committed
    assert "new-file.txt" in committed


def test_commit_workspace_when_nothing_to_commit_does_raise_gymrat_error_leaving_head(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)
    before = head_of(experiment)

    with pytest.raises(GymratError) as excinfo:
        commit_workspace(experiment, "no-op")

    # The wrapper surfaces git's failure as "<step message>: <diagnostic>".
    assert ": " in str(excinfo.value)
    assert head_of(experiment) == before


@pytest.mark.skipif(sys.platform == "win32", reason="the post-commit hook is a POSIX shell script")
def test_commit_workspace_when_head_unreadable_after_commit_does_raise_the_worktree_head_error(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)
    (Path(experiment) / "new-file.txt").write_text("brand new\n", encoding="utf-8")
    # The commit lands, then the hook leaves HEAD on a branch that does not exist.
    install_git_hook(repo, "post-commit", "git symbolic-ref HEAD refs/heads/missing-branch\n")

    with pytest.raises(GymratError) as excinfo:
        commit_workspace(experiment, "agent change")

    assert str(excinfo.value).startswith(f"Cannot read the HEAD of the worktree at {experiment}: ")
    assert excinfo.value.hint == "Inspect what is standing there with: git log -1"


def test_revert_workspace_when_worktree_dirty_does_restore_head_and_drop_untracked_files(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)
    (Path(experiment) / "README.md").write_text("# dirtied\n", encoding="utf-8")
    (Path(experiment) / "untracked.txt").write_text("junk\n", encoding="utf-8")

    revert_workspace(experiment, target="HEAD")

    assert (Path(experiment) / "README.md").read_text(encoding="utf-8") == "# Test Repo\n"
    assert not (Path(experiment) / "untracked.txt").exists()


def test_revert_workspace_when_target_sha_given_does_reset_head_and_tree_to_that_commit(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)
    original = head_of(experiment)
    commit_all(experiment, "agent step", file="step.txt", content="agent change\n")

    revert_workspace(experiment, target=original)

    assert head_of(experiment) == original
    assert not (Path(experiment) / "step.txt").exists()
    assert status_of(experiment) == ""


def test_worktree_head_when_worktree_on_branch_does_return_the_checked_out_sha(
    repo: str, repo_head: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)

    head = worktree_head(experiment_worktree_dir(repo))

    assert head == repo_head


def test_worktree_head_when_directory_is_not_a_repository_does_raise_naming_the_directory(
    tmp_path: Path,
):
    with pytest.raises(GymratError) as excinfo:
        worktree_head(str(tmp_path))

    assert str(excinfo.value).startswith(f"Cannot read the HEAD of the worktree at {tmp_path}: ")
    assert excinfo.value.hint == "Inspect what is standing there with: git log -1"


def test_advance_baseline_when_target_sha_given_does_land_the_baseline_detached_at_it(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)
    (Path(experiment) / "new-file.txt").write_text("advance\n", encoding="utf-8")
    target = commit_workspace(experiment, "advance the branch")
    baseline_dir = baseline_worktree_dir(repo)

    advance_baseline(baseline_dir, target)

    assert head_of(baseline_dir) == target
    assert _checked_out_ref(baseline_dir) == "HEAD"
    assert _checked_out_ref(experiment) == BRANCH


_UNKNOWN_COMMIT = "0" * 40


def _nothing_to_arrange(_repo: str) -> None:
    pass


def _remove_baseline_worktree(repo: str) -> None:
    _git(["worktree", "remove", "--force", baseline_worktree_dir(repo)], repo)


def _recreate_on_unknown_commit(repo: str) -> None:
    recreate_workspace(repo, BRANCH, _UNKNOWN_COMMIT)


def _advance_to_unknown_commit(repo: str) -> None:
    advance_baseline(baseline_worktree_dir(repo), _UNKNOWN_COMMIT)


@pytest.mark.parametrize(
    ("arrange", "act"),
    [
        pytest.param(
            _remove_baseline_worktree, _recreate_on_unknown_commit, id="recreate-workspace"
        ),
        pytest.param(_nothing_to_arrange, _advance_to_unknown_commit, id="advance-baseline"),
    ],
)
def test_baseline_move_when_commit_unknown_does_raise_naming_the_check_command(
    repo: str,
    baseline: BaselineRef,
    arrange: Callable[[str], None],
    act: Callable[[str], None],
):
    create_workspace(repo, SESSION_ID, baseline)
    arrange(repo)

    with pytest.raises(GymratError) as excinfo:
        act(repo)

    assert excinfo.value.hint == (
        f"Check that {_UNKNOWN_COMMIT} is a commit this repository has: "
        f"git cat-file -t {_UNKNOWN_COMMIT}"
    )


# ---------------------------------------------------------------------------
# worktree_fingerprint
# ---------------------------------------------------------------------------


def test_worktree_fingerprint_when_content_unchanged_does_hash_alike(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = Path(experiment_worktree_dir(repo))
    first = worktree_fingerprint(experiment)

    second = worktree_fingerprint(experiment)

    assert first is not None
    assert second == first


def test_worktree_fingerprint_when_content_edited_does_hash_apart(repo: str, baseline: BaselineRef):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = Path(experiment_worktree_dir(repo))
    before = worktree_fingerprint(experiment)
    (experiment / "README.md").write_text("# changed\n", encoding="utf-8")

    after = worktree_fingerprint(experiment)

    assert after is not None
    assert after != before


def test_worktree_fingerprint_when_called_does_leave_index_untouched(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)

    (Path(experiment) / "staged.txt").write_text("staged content\n", encoding="utf-8")
    _git(["add", "staged.txt"], experiment)
    (Path(experiment) / "unstaged.txt").write_text("unstaged content\n", encoding="utf-8")

    index_path = Path(_git(["rev-parse", "--git-path", "index"], experiment))
    index_before = index_path.read_bytes()

    worktree_fingerprint(Path(experiment))

    index_after = index_path.read_bytes()
    assert index_before == index_after
    staged = _git(["diff", "--cached", "--name-only"], experiment)
    assert staged.strip() == "staged.txt"


def test_worktree_fingerprint_when_git_fails_does_return_none(tmp_path: Path):
    result = worktree_fingerprint(tmp_path)

    assert result is None


def test_worktree_fingerprint_when_scratch_directory_cannot_be_created_does_return_none(
    repo: str, baseline: BaselineRef, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    create_workspace(repo, SESSION_ID, baseline)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "missing"))

    result = worktree_fingerprint(Path(experiment_worktree_dir(repo)))

    assert result is None


def test_worktree_fingerprint_when_scratch_cleanup_fails_does_still_return_the_hash(
    repo: str, baseline: BaselineRef, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = Path(experiment_worktree_dir(repo))
    # The worktree is a fresh, clean checkout of the baseline commit, so its
    # tree hash already equals what ``worktree_fingerprint`` will compute.
    expected = _git(["rev-parse", "HEAD^{tree}"], str(experiment))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def _refuse_rmdir(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        msg = f"directory is busy: {path}"
        raise OSError(msg)

    monkeypatch.setattr(os, "rmdir", _refuse_rmdir)

    result = worktree_fingerprint(experiment)

    assert result == expected
