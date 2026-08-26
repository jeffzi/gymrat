"""Behavioral tests for session repository paths and the derived layout.

``repo_root`` runs real git against throwaway repositories from the shared
``create_scratch_repo`` factory, so the tests are parallel-safe under
``pytest-xdist``. The derivation helpers never touch the filesystem, so they are
exercised against an arbitrary absolute root.
"""

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat_py.errors import GymratError
from gymrat_py.session.paths import (
    archived_session_path,
    baseline_worktree_dir,
    experiment_worktree_dir,
    lockfile_path,
    repo_root,
    session_dir,
    session_jsonl_path,
    supervise_lockfile_path,
)

SESSION_ID = "20260808-141530-a3f2"

# An arbitrary absolute root: the derivation helpers never touch the filesystem.
ROOT = str(Path(tempfile.gettempdir()) / "repo-root")

# Repo roots paired with the lockfile name gymrat has always given them. The
# names are golden values, a cross-implementation contract rather than a
# recomputed detail: two runs over the same checkout must land on the same lock.
LOCKFILE_NAMES = [
    ("/srv/projects/demo", "gymrat-lock-9fe2fb7fa4f9.json"),
    ("/srv/projects/other", "gymrat-lock-4ff7d20c47bc.json"),
]


# ---------------------------------------------------------------------------
# repo_root
# ---------------------------------------------------------------------------


def test_repo_root_when_probed_from_nested_subdir_does_return_top_level(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    nested = Path(repo) / "packages" / "core"
    nested.mkdir(parents=True)

    root = repo_root(str(nested))

    assert os.path.normpath(root) == os.path.normpath(repo)


def test_repo_root_when_no_directory_given_does_use_cwd():
    expected = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    root = repo_root()

    assert os.path.normpath(root) == os.path.normpath(expected)


def test_repo_root_when_directory_not_in_repo_does_raise_gymrat_error():
    outside = tempfile.mkdtemp(prefix="not-a-repo-")
    try:
        with pytest.raises(GymratError, match=r"(?i)git repository"):
            repo_root(outside)
    finally:
        shutil.rmtree(outside, ignore_errors=True)


# ---------------------------------------------------------------------------
# session layout
# ---------------------------------------------------------------------------


def _derive_archived(root: str) -> str:
    return archived_session_path(root, SESSION_ID)


@pytest.mark.parametrize(
    ("derive", "relative"),
    [
        (session_dir, (".gymrat",)),
        (session_jsonl_path, (".gymrat", "session.jsonl")),
        (experiment_worktree_dir, (".gymrat", "worktrees", "experiment")),
        (baseline_worktree_dir, (".gymrat", "worktrees", "baseline")),
        (_derive_archived, (".gymrat", f"session-{SESSION_ID}.jsonl")),
    ],
)
def test_session_layout_when_deriving_path_does_place_under_root(
    derive: Callable[[str], str], relative: tuple[str, ...]
):
    result = derive(ROOT)

    assert result == str(Path(ROOT, *relative))


# ---------------------------------------------------------------------------
# lockfile_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("root", "name"), LOCKFILE_NAMES)
def test_lockfile_path_when_given_root_does_map_to_golden_name(root: str, name: str):
    assert lockfile_path(root) == str(Path(tempfile.gettempdir()) / name)


# The supervise lock shares the repo digest (it is keyed on the root, not the
# prefix), so the golden names are the lockfile names with the supervise prefix.
SUPERVISE_LOCKFILE_NAMES = [
    ("/srv/projects/demo", "gymrat-supervise-lock-9fe2fb7fa4f9.json"),
    ("/srv/projects/other", "gymrat-supervise-lock-4ff7d20c47bc.json"),
]


@pytest.mark.parametrize(("root", "name"), SUPERVISE_LOCKFILE_NAMES)
def test_supervise_lockfile_path_when_given_root_does_map_to_golden_name(root: str, name: str):
    assert supervise_lockfile_path(root) == str(Path(tempfile.gettempdir()) / name)
