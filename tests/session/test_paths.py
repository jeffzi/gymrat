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

from gymrat.errors import GymratError
from gymrat.session.paths import (
    SESSION_LOG_NAME,
    archived_session_path,
    baseline_worktree_dir,
    budget_path,
    experiment_worktree_dir,
    lockfile_path,
    repo_root,
    session_dir,
    session_jsonl_path,
    supervise_lockfile_path,
    supervisor_log_name,
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
        (budget_path, (".gymrat", "budget.json")),
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
    (root, name.replace("gymrat-lock-", "gymrat-supervise-lock-")) for root, name in LOCKFILE_NAMES
]


@pytest.mark.parametrize(("root", "name"), SUPERVISE_LOCKFILE_NAMES)
def test_supervise_lockfile_path_when_given_root_does_map_to_golden_name(root: str, name: str):
    assert supervise_lockfile_path(root) == str(Path(tempfile.gettempdir()) / name)


# ---------------------------------------------------------------------------
# log file names — single-source naming for session and supervisor logs
# ---------------------------------------------------------------------------


def test_session_log_name_when_accessed_does_return_bare_filename():
    assert SESSION_LOG_NAME == "session.jsonl"


def test_supervisor_log_name_when_given_timestamp_does_return_filename_with_ms():
    name = supervisor_log_name(1723123456789)

    assert name == "supervisor-1723123456789.jsonl"


def test_session_jsonl_path_when_derived_does_end_with_session_log_name():
    path = session_jsonl_path(ROOT)

    assert path.endswith(SESSION_LOG_NAME)
