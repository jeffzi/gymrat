"""Behavioral tests for target resolution and the worktree lifecycle.

Real-subprocess tests are parallel-safe: the ``create_scratch_repo`` factory
(see ``conftest.py``) gives every test its own temp git repository.
"""

import contextlib
import dataclasses
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.targets import (
    InPlaceTarget,
    RefTarget,
    WorktreeInfo,
    cleanup_worktrees,
    materialize_worktree,
    plan_worktree,
    resolve_target,
)

# A sha no repository holds, so ``git worktree add`` rejects it outright.
UNKNOWN_SHA = "0" * 40

# Hint gymrat attaches to every unresolvable target, duplicated here so the test
# asserts against the same string production emits.
RESOLVE_TARGET_HINT = "Pass an existing directory, or a git ref that resolves to a commit."

_IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0

skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX signal delivery to git is required"
)
skip_on_windows_or_root = pytest.mark.skipif(
    sys.platform == "win32" or _IS_ROOT,
    reason="Windows lacks EACCES from chmod and root bypasses the mode bits",
)


from tests._git import run_git as _run_git


def _get_head_sha(repo_dir: str) -> str:
    return _run_git(["rev-parse", "HEAD"], repo_dir).strip()


def _plan_and_attempt_materialize(target: RefTarget, repo_dir: str) -> tuple[WorktreeInfo, bool]:
    """Plan a worktree and materialize it, reporting failure instead of raising."""
    worktree = plan_worktree(target)
    try:
        materialize_worktree(worktree, repo_dir)
    except GymratError:
        return worktree, True
    return worktree, False


def _create_head_worktree(repo_dir: str) -> WorktreeInfo:
    sha = _get_head_sha(repo_dir)
    worktree = plan_worktree(RefTarget(ref=sha, resolved_sha=sha))
    materialize_worktree(worktree, repo_dir)
    return worktree


def _leave_interrupted_worktree(repo_dir: str) -> WorktreeInfo:
    """Leave a worktree a killed ``git worktree add`` never returned success for.

    Raises rather than returning a half-arranged fixture, so a git version that
    cleaned up despite the kill fails the test that asked for this state.
    """
    sha = _get_head_sha(repo_dir)
    worktree, failed = _plan_and_attempt_materialize(RefTarget(ref=sha, resolved_sha=sha), repo_dir)
    if failed and Path(worktree.dir).exists():
        return worktree
    shutil.rmtree(worktree.dir, ignore_errors=True)
    message = f"expected an interrupted worktree left at {worktree.dir}"
    raise AssertionError(message)


def _plan_rejected_worktree(repo_dir: str) -> WorktreeInfo:
    """Plan a worktree whose ``git worktree add`` fails before creating anything."""
    worktree, failed = _plan_and_attempt_materialize(
        RefTarget(ref="missing", resolved_sha=UNKNOWN_SHA), repo_dir
    )
    if failed and not Path(worktree.dir).exists():
        return worktree
    message = f"expected 'git worktree add' to create nothing at {worktree.dir}"
    raise AssertionError(message)


def _create_stray_worktree() -> WorktreeInfo:
    """A real directory that is not a git worktree, so removal fails."""
    stray_dir = tempfile.mkdtemp(prefix="gymrat-stray-")
    return WorktreeInfo(dir=stray_dir, sha=UNKNOWN_SHA, created=True)


# ---------------------------------------------------------------------------
# Target dataclasses
# ---------------------------------------------------------------------------


def test_ref_target_when_constructed_does_carry_ref_and_resolved_sha():
    target = RefTarget(ref="feature", resolved_sha="deadbeef")

    assert target.ref == "feature"
    assert target.resolved_sha == "deadbeef"


def test_in_place_target_when_constructed_does_carry_dir():
    target = InPlaceTarget(dir="/work")

    assert target.dir == "/work"


def test_ref_target_when_field_assigned_does_raise_frozen():
    target = RefTarget(ref="feature", resolved_sha="deadbeef")

    with pytest.raises(dataclasses.FrozenInstanceError):
        target.ref = "other"  # type: ignore[misc]


def test_in_place_target_when_field_assigned_does_raise_frozen():
    target = InPlaceTarget(dir="/work")

    with pytest.raises(dataclasses.FrozenInstanceError):
        target.dir = "/elsewhere"  # type: ignore[misc]


def test_targets_when_discriminated_by_type_does_distinguish_variants():
    variants = [InPlaceTarget(dir="/work"), RefTarget(ref="main", resolved_sha="abc123")]

    kinds = [isinstance(target, RefTarget) for target in variants]

    assert kinds == [False, True]


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


def test_resolve_target_when_input_is_existing_directory_does_return_in_place_target():
    temp_dir = tempfile.mkdtemp(prefix="gymrat-test-")

    try:
        result = resolve_target(temp_dir, tempfile.gettempdir())

        assert result == InPlaceTarget(dir=os.path.realpath(temp_dir))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_resolve_target_when_input_is_relative_directory_does_resolve_to_absolute():
    temp_dir = tempfile.mkdtemp(prefix="gymrat-test-")
    original_cwd = Path.cwd()

    try:
        os.chdir(Path(temp_dir).parent)
        relative_path = Path(temp_dir).name

        result = resolve_target(relative_path, tempfile.gettempdir())

        assert result == InPlaceTarget(dir=os.path.realpath(temp_dir))
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.parametrize("ref_kind", ["commit-sha", "head", "tag"])
def test_resolve_target_when_input_is_valid_git_ref_does_return_ref_target(
    create_scratch_repo: Callable[[], str], ref_kind: str
):
    repo = create_scratch_repo()
    sha = _get_head_sha(repo)
    if ref_kind == "commit-sha":
        ref = sha
    elif ref_kind == "head":
        ref = "HEAD"
    else:
        _run_git(["tag", "v1.0.0"], repo)
        ref = "v1.0.0"

    result = resolve_target(ref, repo)

    assert result == RefTarget(ref=ref, resolved_sha=sha)


def test_resolve_target_when_input_is_existing_dir_matching_a_ref_does_prefer_directory(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    _run_git(["branch", "shared-name"], repo)
    shared_dir = Path(repo) / "shared-name"
    shared_dir.mkdir()

    result = resolve_target(str(shared_dir), repo)

    assert result == InPlaceTarget(dir=os.path.realpath(shared_dir))


def test_resolve_target_when_input_is_existing_file_does_fall_through_to_ref(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    sha = _get_head_sha(repo)
    _run_git(["branch", "myfile"], repo)
    (Path(repo) / "myfile").write_text("not a directory\n")
    original_cwd = Path.cwd()

    try:
        # The input names an existing regular file relative to the process cwd;
        # a non-directory must fall through to ref resolution, not resolve
        # in place and not raise.
        os.chdir(repo)

        result = resolve_target("myfile", repo)

        assert result == RefTarget(ref="myfile", resolved_sha=sha)
    finally:
        os.chdir(original_cwd)


def test_resolve_target_when_input_neither_dir_nor_ref_does_raise_naming_input(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()

    with pytest.raises(GymratError, match=r"Cannot resolve target 'nonexistent-ref-xyz'"):
        resolve_target("nonexistent-ref-xyz", repo)


def test_resolve_target_when_input_not_a_ref_does_carry_git_stderr(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()

    with pytest.raises(GymratError) as exc_info:
        resolve_target("definitely-not-a-ref", repo)

    message = str(exc_info.value)
    assert "fatal:" in message
    assert exc_info.value.hint == RESOLVE_TARGET_HINT


@pytest.mark.parametrize(
    "rev",
    [pytest.param("HEAD^{tree}", id="tree"), pytest.param("HEAD:README.md", id="blob")],
)
def test_resolve_target_when_input_is_non_commit_object_sha_does_reject(
    create_scratch_repo: Callable[[], str], rev: str
):
    repo = create_scratch_repo()
    sha = _run_git(["rev-parse", rev], repo).strip()

    with pytest.raises(GymratError, match=r"Cannot resolve target"):
        resolve_target(sha, repo)


@skip_on_windows_or_root
def test_resolve_target_when_probe_hits_symlink_loop_does_raise_resolve_error(
    create_scratch_repo: Callable[[], str], tmp_path: Path
):
    repo = create_scratch_repo()
    loop = tmp_path / "loop"
    loop.symlink_to(loop)

    with pytest.raises(GymratError) as exc_info:
        resolve_target(str(loop), repo)

    message = str(exc_info.value)
    assert f"Cannot resolve target '{loop}'" in message
    assert "fatal:" not in message
    assert exc_info.value.hint == RESOLVE_TARGET_HINT


@skip_on_windows_or_root
def test_resolve_target_when_probe_hits_unsearchable_parent_does_raise_resolve_error(
    create_scratch_repo: Callable[[], str], tmp_path: Path
):
    repo = create_scratch_repo()
    parent = tmp_path / "parent"
    target = parent / "target"
    target.mkdir(parents=True)
    parent.chmod(0o000)

    try:
        with pytest.raises(GymratError) as exc_info:
            resolve_target(str(target), repo)

        message = str(exc_info.value)
        assert f"Cannot resolve target '{target}'" in message
        assert "fatal:" not in message
        assert exc_info.value.hint == RESOLVE_TARGET_HINT
    finally:
        parent.chmod(0o700)


# ---------------------------------------------------------------------------
# plan_worktree
# ---------------------------------------------------------------------------


def test_plan_worktree_when_given_ref_target_does_name_absolute_uncreated_tmpdir_path():
    ref_target = RefTarget(ref="my-tag", resolved_sha=UNKNOWN_SHA)

    worktree = plan_worktree(ref_target)

    assert worktree.sha == UNKNOWN_SHA
    assert worktree.created is False
    assert str(Path(worktree.dir).parent) == os.path.realpath(tempfile.gettempdir())
    assert not Path(worktree.dir).exists()


def test_plan_worktree_when_tmpdir_nonexistent_does_raise_naming_temp_dir(
    monkeypatch: pytest.MonkeyPatch,
):
    bogus_dir = str(Path(tempfile.gettempdir()) / f"gymrat-no-such-tmpdir-{os.getpid()}")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: bogus_dir)

    with pytest.raises(GymratError) as exc_info:
        plan_worktree(RefTarget(ref="v1", resolved_sha=UNKNOWN_SHA))

    assert bogus_dir in str(exc_info.value)


# ---------------------------------------------------------------------------
# materialize_worktree
# ---------------------------------------------------------------------------


def test_materialize_worktree_when_given_planned_worktree_does_check_out_ref_files(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    sha = _get_head_sha(repo)
    worktree = plan_worktree(RefTarget(ref=sha, resolved_sha=sha))

    materialize_worktree(worktree, repo)

    assert (Path(worktree.dir) / "README.md").read_text(encoding="utf-8") == "# Test Repo\n"


def test_materialize_worktree_when_git_rejects_sha_does_raise_gymrat_error_with_stderr(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    worktree = plan_worktree(RefTarget(ref="bad-sha", resolved_sha=UNKNOWN_SHA))

    with pytest.raises(GymratError) as exc_info:
        materialize_worktree(worktree, repo)

    message = str(exc_info.value)
    assert UNKNOWN_SHA in message
    assert re.search(r"not a valid object name|invalid reference", message)
    assert "returned non-zero exit status" not in message


@skip_on_windows
def test_materialize_worktree_when_add_interrupted_does_set_created_from_disk_state(
    create_scratch_repo: Callable[[], str],
    kill_git_during_worktree_add: Callable[[str], None],
):
    repo = create_scratch_repo()
    kill_git_during_worktree_add(repo)
    sha = _get_head_sha(repo)
    worktree = plan_worktree(RefTarget(ref=sha, resolved_sha=sha))

    with contextlib.suppress(GymratError):
        materialize_worktree(worktree, repo)

    assert worktree.created is True
    assert Path(worktree.dir).exists()


# ---------------------------------------------------------------------------
# cleanup_worktrees
# ---------------------------------------------------------------------------


def test_cleanup_worktrees_when_given_non_empty_list_does_remove_and_report(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    worktree = _create_head_worktree(repo)

    result = cleanup_worktrees([worktree], repo)

    assert result.removed == 1
    assert not result.failures
    assert result.prune_error is None
    assert not Path(worktree.dir).exists()


def test_cleanup_worktrees_when_list_empty_does_leave_registry_untouched(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
):
    repo = create_scratch_repo()

    cleanup_worktrees([], repo)

    assert len(list_worktree_dirs(repo)) == 1


def test_cleanup_worktrees_when_list_empty_outside_repo_does_skip_prune_and_report_no_error():
    non_repo_dir = tempfile.mkdtemp(prefix="gymrat-not-a-repo-")

    try:
        result = cleanup_worktrees([], non_repo_dir)

        assert result.removed == 0
        assert not result.failures
        assert result.prune_error is None
    finally:
        shutil.rmtree(non_repo_dir, ignore_errors=True)


@skip_on_windows
def test_cleanup_worktrees_when_add_was_killed_does_remove_like_a_normal_worktree(
    create_scratch_repo: Callable[[], str],
    kill_git_during_worktree_add: Callable[[str], None],
):
    repo = create_scratch_repo()
    kill_git_during_worktree_add(repo)
    worktree = _leave_interrupted_worktree(repo)

    try:
        result = cleanup_worktrees([worktree], repo)

        assert result.removed == 1
        assert not Path(worktree.dir).exists()
    finally:
        shutil.rmtree(worktree.dir, ignore_errors=True)


def test_cleanup_worktrees_when_add_left_nothing_counts_as_neither_removed_nor_failure(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    worktree = _plan_rejected_worktree(repo)

    result = cleanup_worktrees([worktree], repo)

    assert result.removed == 0
    assert not result.failures
    assert result.prune_error is None


def test_cleanup_worktrees_when_never_created_worktree_does_not_touch_unrelated_absent(
    create_scratch_repo: Callable[[], str],
    register_absent_worktree: Callable[[str], str],
    list_worktree_dirs: Callable[..., list[str]],
):
    repo = create_scratch_repo()
    absent = register_absent_worktree(repo)
    worktree = _plan_rejected_worktree(repo)

    result = cleanup_worktrees([worktree], repo)

    assert result.removed == 0
    assert absent in list_worktree_dirs(repo)


def test_cleanup_worktrees_when_all_removals_succeed_leaves_unasked_registry_entries(
    create_scratch_repo: Callable[[], str],
    register_absent_worktree: Callable[[str], str],
    list_worktree_dirs: Callable[..., list[str]],
):
    repo = create_scratch_repo()
    absent = register_absent_worktree(repo)
    worktree = _create_head_worktree(repo)

    result = cleanup_worktrees([worktree], repo)

    assert result.removed == 1
    assert absent in list_worktree_dirs(repo)


def test_cleanup_worktrees_when_dir_gone_deregisters_only_that_worktree(
    create_scratch_repo: Callable[[], str],
    register_absent_worktree: Callable[[str], str],
    list_worktree_dirs: Callable[..., list[str]],
):
    repo = create_scratch_repo()
    absent = register_absent_worktree(repo)
    worktree = _create_head_worktree(repo)
    shutil.rmtree(worktree.dir, ignore_errors=True)

    cleanup_worktrees([worktree], repo)

    listed = list_worktree_dirs(repo)
    assert worktree.dir not in listed
    assert absent in listed


def test_cleanup_worktrees_when_removal_fails_reports_dir_with_git_error_text(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    stray = _create_stray_worktree()

    try:
        result = cleanup_worktrees([stray], repo)

        assert [failure.dir for failure in result.failures] == [stray.dir]
        assert "is not a working tree" in result.failures[0].error
        assert "returned non-zero exit status" not in result.failures[0].error
    finally:
        shutil.rmtree(stray.dir, ignore_errors=True)


def test_cleanup_worktrees_when_removal_fails_does_not_prune_unrelated_entries(
    create_scratch_repo: Callable[[], str],
    register_absent_worktree: Callable[[str], str],
    list_worktree_dirs: Callable[..., list[str]],
):
    repo = create_scratch_repo()
    absent = register_absent_worktree(repo)
    stray = _create_stray_worktree()

    try:
        result = cleanup_worktrees([stray], repo)

        assert len(result.failures) == 1
        assert absent in list_worktree_dirs(repo)
    finally:
        shutil.rmtree(stray.dir, ignore_errors=True)


def test_cleanup_worktrees_when_removal_fails_still_removes_later_worktrees(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    stray = _create_stray_worktree()
    worktree = _create_head_worktree(repo)

    try:
        result = cleanup_worktrees([stray, worktree], repo)

        assert result.removed == 1
        assert not Path(worktree.dir).exists()
    finally:
        shutil.rmtree(stray.dir, ignore_errors=True)


def test_cleanup_worktrees_when_prune_sweep_fails_reports_prune_error_not_raises():
    non_repo_dir = tempfile.mkdtemp(prefix="gymrat-not-a-repo-")

    try:
        vanished = WorktreeInfo(dir=str(Path(non_repo_dir) / "gone"), sha=UNKNOWN_SHA, created=True)

        result = cleanup_worktrees([vanished], non_repo_dir)

        assert result.removed == 0
        assert not result.failures
        assert result.prune_error is not None
        assert "not a git repository" in result.prune_error
    finally:
        shutil.rmtree(non_repo_dir, ignore_errors=True)


def test_cleanup_worktrees_when_swept_twice_reports_removed_zero_and_no_failures(
    create_scratch_repo: Callable[[], str],
    list_worktree_dirs: Callable[..., list[str]],
):
    repo = create_scratch_repo()
    worktree = _create_head_worktree(repo)

    cleanup_worktrees([worktree], repo)
    second = cleanup_worktrees([worktree], repo)

    assert not Path(worktree.dir).exists()
    assert len(list_worktree_dirs(repo)) == 1
    assert second.removed == 0
    assert not second.failures


def test_cleanup_worktrees_when_swept_twice_does_not_prune_unrelated_absent(
    create_scratch_repo: Callable[[], str],
    register_absent_worktree: Callable[[str], str],
    list_worktree_dirs: Callable[..., list[str]],
):
    repo = create_scratch_repo()
    worktree = _create_head_worktree(repo)
    absent = register_absent_worktree(repo)

    cleanup_worktrees([worktree], repo)
    second = cleanup_worktrees([worktree], repo)

    assert second.removed == 0
    assert not second.failures
    assert second.prune_error is None
    assert absent in list_worktree_dirs(repo)
