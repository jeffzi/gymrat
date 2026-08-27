"""Behavioral tests for the session git-workspace orchestration.

Every test runs real git against a throwaway repository from the shared
``create_scratch_repo`` factory, so the suite is order-independent and safe
under ``pytest-xdist`` / ``pytest-randomly``. No git call is mocked: the module
under test is pure git orchestration, and only real worktrees reveal the
pruning, unwind, and detachment behavior these tests pin.
"""

import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat_py.errors import GymratError, hint_of
from gymrat_py.session.paths import baseline_worktree_dir, experiment_worktree_dir
from gymrat_py.session.workspace import (
    BaselineRef,
    WorkspaceResult,
    Worktrees,
    advance_baseline,
    commit_workspace,
    create_workspace,
    ensure_git_exclude,
    is_worktree_dirty,
    recreate_workspace,
    remove_worktrees,
    revert_workspace,
    worktree_head,
)

SESSION_ID = "20260808-141530-a3f2"
BRANCH = f"gymrat/{SESSION_ID}"
BASELINE_REF = "main"

# The id the session after SESSION_ID opens on, for a workspace built over an
# earlier one's leftovers.
NEXT_SESSION_ID = "20260808-152045-b7c1"
NEXT_BRANCH = f"gymrat/{NEXT_SESSION_ID}"


def _git(args: list[str], cwd: str) -> str:
    """Run git in ``cwd`` for test setup and assertions, returning trimmed stdout."""
    from tests._git import run_git

    return run_git(args, cwd).strip()


def _checked_out_ref(worktree: str) -> str:
    """The ref a worktree has checked out: a branch name, or ``HEAD`` when detached."""
    return _git(["rev-parse", "--abbrev-ref", "HEAD"], worktree)


def _exclude_path(root: str) -> Path:
    return Path(root) / ".git" / "info" / "exclude"


def _session_branches(root: str) -> list[str]:
    """The session branches ``root`` still holds, one per ``gymrat/…`` ref."""
    output = _git(["for-each-ref", "--format=%(refname:short)", "refs/heads/gymrat"], root)
    return [line for line in output.split("\n") if line != ""]


def _both_worktrees_exist(root: str) -> bool:
    return (
        Path(experiment_worktree_dir(root)).exists() and Path(baseline_worktree_dir(root)).exists()
    )


def _worktrees(root: str) -> Worktrees:
    """The two worktree paths ``create_workspace`` laid down in ``root``."""
    return Worktrees(
        experiment=experiment_worktree_dir(root),
        baseline=baseline_worktree_dir(root),
    )


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str]) -> str:
    return create_scratch_repo()


@pytest.fixture
def baseline_sha(repo: str) -> str:
    return _git(["rev-parse", "HEAD"], repo)


@pytest.fixture
def baseline(baseline_sha: str) -> BaselineRef:
    return BaselineRef(ref=BASELINE_REF, sha=baseline_sha)


# ---------------------------------------------------------------------------
# create_workspace
# ---------------------------------------------------------------------------


def test_create_workspace_when_no_session_workspace_does_build_branch_worktrees_and_descriptor(
    repo: str, baseline_sha: str, baseline: BaselineRef
):
    result = create_workspace(repo, SESSION_ID, baseline)

    exp = experiment_worktree_dir(repo)
    bl = baseline_worktree_dir(repo)
    assert _git(["rev-parse", BRANCH], repo) == baseline_sha
    assert Path(exp).exists()
    assert _checked_out_ref(exp) == BRANCH
    assert _git(["rev-parse", "HEAD"], bl) == baseline_sha
    assert _checked_out_ref(bl) == "HEAD"
    assert ".gymrat/" in _exclude_path(repo).read_text(encoding="utf-8").split("\n")
    assert result == WorkspaceResult(
        branch=BRANCH,
        worktrees=Worktrees(experiment=exp, baseline=bl),
        baseline=BaselineRef(ref=BASELINE_REF, sha=baseline_sha),
    )


def test_create_workspace_when_branch_already_exists_does_raise_naming_branch_and_hint(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)

    with pytest.raises(GymratError) as excinfo:
        create_workspace(repo, SESSION_ID, baseline)

    assert BRANCH in str(excinfo.value)
    assert re.search(r"git branch -D", hint_of(excinfo.value) or "", re.IGNORECASE)


@pytest.mark.skipif(sys.platform == "win32", reason="post-checkout SIGKILL is POSIX-only")
def test_create_workspace_when_worktree_add_dies_does_unwind_and_fail_on_that_step(
    repo: str,
    baseline: BaselineRef,
    kill_git_during_worktree_add: Callable[[str], None],
    list_worktree_dirs: Callable[..., list[str]],
):
    # Installed after the scratch repo's own commit so only the worktree
    # checkouts under test die.
    kill_git_during_worktree_add(repo)

    with pytest.raises(GymratError) as excinfo:
        create_workspace(repo, SESSION_ID, baseline)

    # The unwind's own git steps never speak for it.
    assert re.search(r"cannot create the experiment worktree", str(excinfo.value), re.IGNORECASE)
    assert _session_branches(repo) == []
    assert list_worktree_dirs(repo, include_main=False) == []
    assert not Path(experiment_worktree_dir(repo)).exists()


def test_create_workspace_when_registry_entries_are_stale_does_check_out_over_them(
    repo: str, baseline_sha: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    shutil.rmtree(experiment_worktree_dir(repo))
    shutil.rmtree(baseline_worktree_dir(repo))

    result = create_workspace(repo, NEXT_SESSION_ID, baseline)

    assert _checked_out_ref(result.worktrees.experiment) == NEXT_BRANCH
    assert _git(["rev-parse", "HEAD"], result.worktrees.baseline) == baseline_sha


def test_create_workspace_when_registry_stale_does_leave_a_live_worktree_registered(
    repo: str,
    baseline_sha: str,
    baseline: BaselineRef,
    list_worktree_dirs: Callable[..., list[str]],
):
    create_workspace(repo, SESSION_ID, baseline)
    shutil.rmtree(experiment_worktree_dir(repo))
    shutil.rmtree(baseline_worktree_dir(repo))
    live = str(Path(repo) / "live-worktree")
    _git(["worktree", "add", "--detach", live, baseline_sha], repo)

    create_workspace(repo, NEXT_SESSION_ID, baseline)

    assert Path(live).exists()
    assert live in list_worktree_dirs(repo, include_main=False)


def test_create_workspace_when_registry_stale_does_leave_a_temporarily_absent_user_worktree_registered(
    repo: str,
    baseline_sha: str,
    baseline: BaselineRef,
    list_worktree_dirs: Callable[..., list[str]],
):
    create_workspace(repo, SESSION_ID, baseline)
    shutil.rmtree(experiment_worktree_dir(repo))
    shutil.rmtree(baseline_worktree_dir(repo))
    user_worktree = str(Path(repo) / "user-worktree")
    _git(["worktree", "add", "--detach", user_worktree, baseline_sha], repo)
    shutil.rmtree(user_worktree)

    create_workspace(repo, NEXT_SESSION_ID, baseline)

    assert user_worktree in list_worktree_dirs(repo, include_main=False)


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
    assert _session_branches(repo) == [BRANCH]
    assert experiment_worktree_dir(repo) in str(excinfo.value)


def test_create_workspace_when_directory_is_not_a_git_repository_does_raise(
    tmp_path: Path, baseline_sha: str
):
    outside = str(tmp_path)

    with pytest.raises(GymratError) as excinfo:
        create_workspace(outside, SESSION_ID, BaselineRef(ref=BASELINE_REF, sha=baseline_sha))

    assert re.search(r"not a git repository", str(excinfo.value), re.IGNORECASE)
    assert re.search(r"git repository", hint_of(excinfo.value) or "", re.IGNORECASE)


# ---------------------------------------------------------------------------
# ensure_git_exclude
# ---------------------------------------------------------------------------


def test_ensure_git_exclude_when_already_listed_does_leave_file_byte_for_byte_unchanged(repo: str):
    before = "node_modules/\n.gymrat/\n"
    path = _exclude_path(repo)
    path.write_text(before, encoding="utf-8")

    ensure_git_exclude(repo)

    assert path.read_text(encoding="utf-8") == before


def test_ensure_git_exclude_when_entry_missing_does_append_it_after_existing_content(repo: str):
    existing = "node_modules/\nbuild/\n"
    path = _exclude_path(repo)
    path.write_text(existing, encoding="utf-8")

    ensure_git_exclude(repo)

    assert path.read_text(encoding="utf-8") == f"{existing}.gymrat/\n"


def test_ensure_git_exclude_when_file_missing_does_create_it_holding_the_line(repo: str):
    path = _exclude_path(repo)
    path.unlink(missing_ok=True)

    ensure_git_exclude(repo)

    assert ".gymrat/" in path.read_text(encoding="utf-8").split("\n")


# ---------------------------------------------------------------------------
# remove_worktrees
# ---------------------------------------------------------------------------


def test_remove_worktrees_when_both_on_disk_does_remove_them_without_warning(
    repo: str, baseline: BaselineRef, list_worktree_dirs: Callable[..., list[str]]
):
    create_workspace(repo, SESSION_ID, baseline)

    warnings = remove_worktrees(repo, _worktrees(repo))

    assert warnings == []
    assert not Path(experiment_worktree_dir(repo)).exists()
    assert not Path(baseline_worktree_dir(repo)).exists()
    assert list_worktree_dirs(repo, include_main=False) == []


def test_remove_worktrees_when_one_directory_already_gone_does_remove_the_other_without_warning(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    shutil.rmtree(experiment_worktree_dir(repo))

    warnings = remove_worktrees(repo, _worktrees(repo))

    assert warnings == []
    assert not Path(baseline_worktree_dir(repo)).exists()


def test_remove_worktrees_when_one_gone_does_deregister_by_name_only(
    repo: str,
    baseline: BaselineRef,
    register_absent_worktree: Callable[[str], str],
    list_worktree_dirs: Callable[..., list[str]],
):
    create_workspace(repo, SESSION_ID, baseline)
    # The user's own worktree, absent only for the moment.
    absent = register_absent_worktree(repo)
    shutil.rmtree(experiment_worktree_dir(repo))

    remove_worktrees(repo, _worktrees(repo))

    listed = list_worktree_dirs(repo)
    assert experiment_worktree_dir(repo) not in listed
    assert absent in listed


def test_remove_worktrees_when_git_refuses_does_warn_naming_it_and_remove_the_other(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    # git declines a locked worktree unless --force is passed twice.
    _git(["worktree", "lock", experiment_worktree_dir(repo)], repo)

    warnings = remove_worktrees(repo, _worktrees(repo))

    assert len(warnings) == 1
    assert experiment_worktree_dir(repo) in warnings[0]
    assert not Path(baseline_worktree_dir(repo)).exists()


# ---------------------------------------------------------------------------
# is_worktree_dirty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        pytest.param(None, False, id="nothing-touched"),
        pytest.param("README.md", True, id="tracked-file-edited"),
        pytest.param("scratch.txt", True, id="untracked-file-added"),
    ],
)
def test_is_worktree_dirty_when_worktree_edited_does_report_dirty(
    repo: str, baseline: BaselineRef, edit: str | None, expected: bool
):
    create_workspace(repo, SESSION_ID, baseline)
    worktree = experiment_worktree_dir(repo)
    if edit is not None:
        (Path(worktree) / edit).write_text("# edited by the agent\n", encoding="utf-8")

    assert is_worktree_dirty(worktree) is expected


def test_is_worktree_dirty_when_directory_missing_does_report_clean(repo: str):
    assert is_worktree_dirty(experiment_worktree_dir(repo)) is False


# ---------------------------------------------------------------------------
# recreate_workspace
# ---------------------------------------------------------------------------


def test_recreate_workspace_when_experiment_gone_does_put_it_back_on_the_branch(
    repo: str, baseline_sha: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    shutil.rmtree(experiment_worktree_dir(repo))

    recreate_workspace(repo, BRANCH, baseline_sha)

    assert _checked_out_ref(experiment_worktree_dir(repo)) == BRANCH
    assert _both_worktrees_exist(repo)


def test_recreate_workspace_when_experiment_gone_does_leave_absent_user_worktree_registered(
    repo: str,
    baseline_sha: str,
    baseline: BaselineRef,
    list_worktree_dirs: Callable[..., list[str]],
):
    create_workspace(repo, SESSION_ID, baseline)
    user_worktree = str(Path(repo) / "user-worktree")
    _git(["worktree", "add", "--detach", user_worktree, baseline_sha], repo)
    shutil.rmtree(user_worktree)
    shutil.rmtree(experiment_worktree_dir(repo))

    recreate_workspace(repo, BRANCH, baseline_sha)

    assert user_worktree in list_worktree_dirs(repo, include_main=False)


def test_recreate_workspace_when_baseline_gone_does_put_it_back_detached_at_sha(
    repo: str, baseline_sha: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    shutil.rmtree(baseline_worktree_dir(repo))

    recreate_workspace(repo, BRANCH, baseline_sha)

    worktree = baseline_worktree_dir(repo)
    assert _git(["rev-parse", "HEAD"], worktree) == baseline_sha
    assert _checked_out_ref(worktree) == "HEAD"


def test_recreate_workspace_when_both_on_disk_does_leave_experiment_work_untouched(
    repo: str, baseline_sha: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    edited = Path(experiment_worktree_dir(repo)) / "README.md"
    edited.write_text("# edited by the agent\n", encoding="utf-8")

    recreate_workspace(repo, BRANCH, baseline_sha)

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
    before = _git(["rev-parse", "HEAD"], experiment)

    sha = commit_workspace(experiment, "agent change")

    assert sha != before
    assert sha == _git(["rev-parse", "HEAD"], experiment)
    committed = _git(["show", "--name-only", "--format=", "HEAD"], experiment).split("\n")
    assert "README.md" in committed
    assert "new-file.txt" in committed


def test_commit_workspace_when_nothing_to_commit_does_raise_gymrat_error_leaving_head(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)
    before = _git(["rev-parse", "HEAD"], experiment)

    with pytest.raises(GymratError) as excinfo:
        commit_workspace(experiment, "no-op")

    # The wrapper surfaces git's failure as "<step message>: <diagnostic>".
    assert ": " in str(excinfo.value)
    assert _git(["rev-parse", "HEAD"], experiment) == before


def test_revert_workspace_when_worktree_dirty_does_restore_head_and_drop_untracked_files(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)
    (Path(experiment) / "README.md").write_text("# dirtied\n", encoding="utf-8")
    (Path(experiment) / "untracked.txt").write_text("junk\n", encoding="utf-8")

    revert_workspace(experiment)

    assert (Path(experiment) / "README.md").read_text(encoding="utf-8") == "# Test Repo\n"
    assert not (Path(experiment) / "untracked.txt").exists()


def test_revert_workspace_when_target_sha_given_does_reset_head_and_tree_to_that_commit(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)
    original = _git(["rev-parse", "HEAD"], experiment)
    (Path(experiment) / "step.txt").write_text("agent change\n", encoding="utf-8")
    _git(["add", "-A"], experiment)
    _git(["commit", "-m", "agent step"], experiment)
    assert _git(["rev-parse", "HEAD"], experiment) != original

    revert_workspace(experiment, target=original)

    assert _git(["rev-parse", "HEAD"], experiment) == original
    assert not (Path(experiment) / "step.txt").exists()
    assert _git(["status", "--porcelain"], experiment) == ""


def test_worktree_head_when_worktree_on_branch_does_return_the_checked_out_sha(
    repo: str, baseline_sha: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)

    assert worktree_head(experiment_worktree_dir(repo)) == baseline_sha


def test_advance_baseline_when_target_sha_given_does_land_the_baseline_detached_at_it(
    repo: str, baseline: BaselineRef
):
    create_workspace(repo, SESSION_ID, baseline)
    experiment = experiment_worktree_dir(repo)
    (Path(experiment) / "new-file.txt").write_text("advance\n", encoding="utf-8")
    target = commit_workspace(experiment, "advance the branch")
    baseline_dir = baseline_worktree_dir(repo)

    advance_baseline(baseline_dir, target)

    assert _git(["rev-parse", "HEAD"], baseline_dir) == target
    assert _checked_out_ref(baseline_dir) == "HEAD"
    assert _checked_out_ref(experiment) == BRANCH
