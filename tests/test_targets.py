"""Behavioral tests for ``resolve_target``: a directory runs in place, a ref resolves to a commit.

Real-subprocess tests are parallel-safe: the ``repo`` fixture gives every test
its own temp git repository. The worktree lifecycle a ref target goes through is
tested beside its source in ``tests/sampling/test_worktree_run.py``.
"""

import os
import sys
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.targets import InPlaceTarget, RefTarget, resolve_target
from tests._git import (
    head_of,
)
from tests._git import run_git as _run_git

# Hint gymrat attaches to every unresolvable target, duplicated here so the test
# asserts against the same string production emits.
RESOLVE_TARGET_HINT = "Pass an existing directory, or a git ref that resolves to a commit."

_IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0

skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows ignores the execute bit"
)
skip_on_windows_or_root = pytest.mark.skipif(
    sys.platform == "win32" or _IS_ROOT,
    reason="Windows lacks EACCES from chmod and root bypasses the mode bits",
)


@pytest.fixture(
    params=[
        pytest.param(False, id="git-missing"),
        pytest.param(True, id="git-not-executable", marks=skip_on_windows),
    ]
)
def unusable_git(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave ``PATH`` holding one directory where git is absent or lacks the execute bit."""
    if request.param:
        blocked = tmp_path / "git"
        blocked.write_text("#!/bin/sh\n", encoding="utf-8")
        blocked.chmod(0o644)
    monkeypatch.setenv("PATH", str(tmp_path))


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


def test_resolve_target_when_input_is_existing_directory_does_return_in_place_target(
    tmp_path: Path,
):
    result = resolve_target(str(tmp_path), str(tmp_path))

    assert result == InPlaceTarget(dir=os.path.realpath(tmp_path))


def test_resolve_target_when_input_is_relative_directory_does_resolve_to_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "bench").mkdir()
    monkeypatch.chdir(tmp_path)

    result = resolve_target("bench", str(tmp_path))

    assert result == InPlaceTarget(dir=os.path.realpath(tmp_path / "bench"))


@pytest.mark.parametrize("ref_kind", ["commit-sha", "head", "tag"])
def test_resolve_target_when_input_is_valid_git_ref_does_return_ref_target(
    repo: str, ref_kind: str
):
    sha = head_of(repo)
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
    repo: str,
):
    _run_git(["branch", "shared-name"], repo)
    shared_dir = Path(repo) / "shared-name"
    shared_dir.mkdir()

    result = resolve_target(str(shared_dir), repo)

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


def test_resolve_target_when_input_neither_dir_nor_ref_does_raise_with_git_stderr(
    repo: str,
):
    with pytest.raises(GymratError) as exc_info:
        resolve_target("nonexistent-ref-xyz", repo)

    message = str(exc_info.value)
    assert "Cannot resolve target 'nonexistent-ref-xyz'" in message
    assert "fatal:" in message
    assert exc_info.value.hint == RESOLVE_TARGET_HINT


@pytest.mark.usefixtures("unusable_git")
def test_resolve_target_when_git_cannot_be_started_does_raise_gymrat_error(tmp_path: Path):
    with pytest.raises(GymratError, match=r"^Cannot resolve target 'main': .+") as exc_info:
        resolve_target("main", str(tmp_path))

    assert exc_info.value.hint == RESOLVE_TARGET_HINT


@pytest.mark.parametrize(
    "rev",
    [pytest.param("HEAD^{tree}", id="tree"), pytest.param("HEAD:README.md", id="blob")],
)
def test_resolve_target_when_input_is_non_commit_object_sha_does_reject(repo: str, rev: str):
    sha = _run_git(["rev-parse", rev], repo)

    with pytest.raises(GymratError, match=r"Cannot resolve target"):
        resolve_target(sha, repo)


@skip_on_windows_or_root
def test_resolve_target_when_probe_hits_symlink_loop_does_raise_resolve_error(
    repo: str, tmp_path: Path
):
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
    repo: str, tmp_path: Path
):
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
