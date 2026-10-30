"""Behavioral tests for ``resolve_target``: a directory runs in place, a ref resolves to a commit.

Real-subprocess tests are parallel-safe: the ``repo`` fixture gives every test
its own temp git repository. The worktree lifecycle a ref target goes through is
tested beside its source in ``tests/sampling/test_worktree_run.py``.
"""

import os
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.targets import InPlaceTarget, RefTarget, resolve_target
from tests._git import (
    head_of,
)
from tests._git import run_git as _run_git
from tests._platform import needs_symlinks

# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


def test_resolve_target_when_input_is_existing_directory_does_return_absolute_in_place_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "bench").mkdir()
    # A repo directory with no ``bench`` inside: a relative spelling resolves
    # against the cwd, never against the repository.
    elsewhere = tmp_path / "repo"
    elsewhere.mkdir()
    monkeypatch.chdir(tmp_path)

    result = resolve_target("bench", str(elsewhere))

    assert result == InPlaceTarget(dir=os.path.realpath(tmp_path / "bench"))


def test_resolve_target_when_input_is_valid_git_ref_does_return_ref_target(repo: str):
    sha = head_of(repo)

    result = resolve_target(sha, repo)

    assert result == RefTarget(ref=sha, resolved_sha=sha)


def test_resolve_target_when_input_is_existing_dir_matching_a_ref_does_prefer_directory(
    repo: str,
):
    _run_git(["branch", "shared-name"], repo)
    shared_dir = Path(repo) / "shared-name"
    shared_dir.mkdir()

    result = resolve_target("shared-name", repo)

    assert result == InPlaceTarget(dir=os.path.realpath(shared_dir))


@pytest.mark.parametrize(
    "ref",
    [
        pytest.param("myfile", id="file-itself"),
        pytest.param("myfile/typo", id="path-under-file"),
    ],
)
def test_resolve_target_when_input_is_existing_file_does_fall_through_to_ref(repo: str, ref: str):
    # The input names an existing regular file relative to the process cwd (the
    # repo), or a path underneath one; neither is a directory, so both must fall
    # through to ref resolution, not resolve in place and not raise.
    sha = head_of(repo)
    _run_git(["branch", ref], repo)
    (Path(repo) / "myfile").write_text("not a directory\n")

    result = resolve_target(ref, repo)

    assert result == RefTarget(ref=ref, resolved_sha=sha)


def test_resolve_target_when_input_is_not_a_commit_does_raise_naming_it(repo: str):
    tree = _run_git(["rev-parse", "HEAD^{tree}"], repo)

    with pytest.raises(GymratError) as exc_info:
        resolve_target(tree, repo)

    assert f"'{tree}'" in str(exc_info.value)


@pytest.fixture
def stat_failing_target(tmp_path: Path) -> Path:
    """A path whose stat fails with an errno other than ENOENT or ENOTDIR: a symlink loop."""
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    return loop


@needs_symlinks
def test_resolve_target_when_path_stat_fails_does_raise_resolve_error(
    repo: str, stat_failing_target: Path
):
    with pytest.raises(GymratError) as exc_info:
        resolve_target(str(stat_failing_target), repo)

    message = str(exc_info.value)
    assert f"'{stat_failing_target}'" in message
    assert isinstance(exc_info.value.__cause__, OSError)
