"""Behavioral tests for ``resolve_target``: a directory runs in place, a ref resolves to a commit.

Real-subprocess tests are parallel-safe: the ``repo`` fixture gives every test
its own temp git repository. The worktree lifecycle a ref target goes through is
tested beside its source in ``tests/sampling/test_worktree_run.py``.
"""

import contextlib
import os
from collections.abc import Callable, Generator, Iterator
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.targets import InPlaceTarget, RefTarget, resolve_target
from tests._git import (
    head_of,
)
from tests._git import run_git as _run_git
from tests._mode_bits import needs_mode_bits
from tests._platform import needs_symlinks

# Hint gymrat attaches to every unresolvable target, duplicated here so the test
# asserts against the same string production emits.
RESOLVE_TARGET_HINT = "Pass an existing directory, or a git ref that resolves to a commit."


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "absolute", [pytest.param(False, id="relative"), pytest.param(True, id="absolute")]
)
def test_resolve_target_when_input_is_existing_directory_does_return_absolute_in_place_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, absolute: bool
):
    (tmp_path / "bench").mkdir()
    monkeypatch.chdir(tmp_path)
    spelling = str(tmp_path / "bench") if absolute else "bench"

    result = resolve_target(spelling, str(tmp_path))

    assert result == InPlaceTarget(dir=os.path.realpath(tmp_path / "bench"))


def _commit_sha(repo: str) -> str:
    return head_of(repo)


def _head(_repo: str) -> str:
    return "HEAD"


def _tag(repo: str) -> str:
    _run_git(["tag", "v1.0.0"], repo)
    return "v1.0.0"


@pytest.mark.parametrize(
    "make_ref",
    [
        pytest.param(_commit_sha, id="commit-sha"),
        pytest.param(_head, id="head"),
        pytest.param(_tag, id="tag"),
    ],
)
def test_resolve_target_when_input_is_valid_git_ref_does_return_ref_target(
    repo: str, make_ref: Callable[[str], str]
):
    sha = head_of(repo)
    ref = make_ref(repo)

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


def _unknown_ref(_repo: str) -> str:
    return "banana"


def _tree_sha(repo: str) -> str:
    return _run_git(["rev-parse", "HEAD^{tree}"], repo)


def _blob_sha(repo: str) -> str:
    return _run_git(["rev-parse", "HEAD:README.md"], repo)


@pytest.mark.parametrize(
    "make_ref",
    [
        pytest.param(_unknown_ref, id="unknown-ref"),
        pytest.param(_tree_sha, id="tree"),
        pytest.param(_blob_sha, id="blob"),
    ],
)
def test_resolve_target_when_input_is_not_a_commit_does_raise_with_git_stderr(
    repo: str, make_ref: Callable[[str], str]
):
    ref = make_ref(repo)

    with pytest.raises(GymratError) as exc_info:
        resolve_target(ref, repo)

    message = str(exc_info.value)
    assert f"Cannot resolve target '{ref}'" in message
    assert "fatal:" in message
    assert exc_info.value.hint == RESOLVE_TARGET_HINT


@pytest.mark.usefixtures("unusable_git")
def test_resolve_target_when_git_cannot_be_started_does_raise_gymrat_error(tmp_path: Path):
    with pytest.raises(GymratError, match=r"^Cannot resolve target 'main': .+") as exc_info:
        resolve_target("main", str(tmp_path))

    assert exc_info.value.hint == RESOLVE_TARGET_HINT


@contextlib.contextmanager
def _symlink_loop(tmp_path: Path) -> Generator[Path]:
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    yield loop


@contextlib.contextmanager
def _unsearchable_parent(tmp_path: Path) -> Generator[Path]:
    parent = tmp_path / "parent"
    target = parent / "target"
    target.mkdir(parents=True)
    parent.chmod(0o000)
    try:
        yield target
    finally:
        parent.chmod(0o700)


@pytest.fixture(
    params=[
        pytest.param(_symlink_loop, id="symlink-loop", marks=needs_symlinks),
        pytest.param(_unsearchable_parent, id="unsearchable-parent", marks=needs_mode_bits),
    ]
)
def stat_failing_target(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Path]:
    """A path whose stat fails with an errno other than ENOENT or ENOTDIR."""
    with request.param(tmp_path) as target:
        yield target


def test_resolve_target_when_path_stat_fails_does_raise_resolve_error(
    repo: str, stat_failing_target: Path
):
    with pytest.raises(GymratError) as exc_info:
        resolve_target(str(stat_failing_target), repo)

    message = str(exc_info.value)
    assert f"Cannot resolve target '{stat_failing_target}'" in message
    assert "fatal:" not in message
    assert exc_info.value.hint == RESOLVE_TARGET_HINT
