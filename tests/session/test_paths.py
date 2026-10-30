"""Behavioral tests for session repository paths and the derived layout.

``repo_root`` runs real git against throwaway repositories from the shared
``create_scratch_repo`` factory, so the tests are parallel-safe under
``pytest-xdist``. The derivation helpers never touch the filesystem, so they are
exercised against an arbitrary absolute root.
"""

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.errors import GymratError
from gymrat.session.paths import (
    NotAGitRepositoryError,
    archived_session_path,
    git_common_dir,
    lockfile_path,
    repo_root,
    repository_lookup_error,
    session_dir,
    supervisor_log_name,
)
from tests._git import add_worktree, run_git
from tests.session.records._fixtures import (
    SESSION_ID,
)

# An arbitrary absolute root: the derivation helpers never touch the filesystem.
ROOT = str(Path(tempfile.gettempdir()) / "repo-root")

# A repo root paired with the lockfile name gymrat has always given it. The
# name is a golden value, a cross-implementation contract rather than a
# recomputed detail: two runs over the same checkout must land on the same lock.
LOCKFILE_ROOT = "/srv/projects/demo"
LOCKFILE_NAME = "gymrat-lock-9fe2fb7fa4f9.json"


# ---------------------------------------------------------------------------
# repository_lookup_error
# ---------------------------------------------------------------------------


def test_repository_lookup_error_when_phrase_only_inside_path_does_not_classify_as_missing():
    cause = subprocess.CalledProcessError(
        128,
        ["git"],
        stderr="error: cannot open /tmp/not a git repository/config: No such file\n",
    )

    error = repository_lookup_error("/some/dir", cause)

    assert not isinstance(error, NotAGitRepositoryError)
    assert "/some/dir" in str(error)
    assert "cannot open /tmp/not a git repository/config" in str(error)


# ---------------------------------------------------------------------------
# repo_root
# ---------------------------------------------------------------------------


def test_repo_root_when_no_directory_given_does_use_cwd(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    repo = create_scratch_repo()
    nested = Path(repo) / "packages" / "core"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    root = repo_root()

    assert Path(root) == Path(repo)


def test_repo_root_when_directory_not_in_repo_does_raise_not_a_git_repository(tmp_path: Path):
    with pytest.raises(NotAGitRepositoryError) as excinfo:
        repo_root(str(tmp_path))

    assert str(tmp_path) in str(excinfo.value)


# ---------------------------------------------------------------------------
# git_common_dir
# ---------------------------------------------------------------------------


def _main_checkout(repo: str) -> str:
    return repo


def _linked_worktree(repo: str) -> str:
    return add_worktree(repo, "linked")


@pytest.mark.parametrize(
    "checkout",
    [
        pytest.param(_main_checkout, id="main-checkout"),
        pytest.param(_linked_worktree, id="linked-worktree"),
    ],
)
def test_git_common_dir_when_called_does_return_the_git_directory_of_the_owning_repository(
    create_scratch_repo: Callable[[], str], checkout: Callable[[str], str]
):
    repo = create_scratch_repo()
    directory = checkout(repo)

    common = git_common_dir(directory)

    assert Path(common).resolve() == Path(repo, ".git").resolve()


# ---------------------------------------------------------------------------
# repo_root — gymrat worktree ownership
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "checkout",
    [
        pytest.param(_main_checkout, id="main-checkout"),
        pytest.param(_linked_worktree, id="linked-worktree"),
    ],
)
def test_repo_root_when_probed_inside_a_gymrat_worktree_does_return_the_owning_checkout(
    create_scratch_repo: Callable[[], str],
    checkout: Callable[[str], str],
):
    owner = checkout(create_scratch_repo())
    probe = add_worktree(owner, ".gymrat/worktrees/experiment")

    root = repo_root(probe)

    assert Path(root) == Path(owner)


def test_repo_root_when_directory_above_gymrat_dir_is_not_a_repository_does_return_the_toplevel(
    tmp_path: Path,
):
    standalone = tmp_path / "plain" / ".gymrat" / "worktrees" / "standalone"
    standalone.mkdir(parents=True)
    run_git(["init"], str(standalone))

    root = repo_root(str(standalone))

    assert Path(root) == Path(standalone)


def test_repo_root_when_git_declines_for_directory_above_gymrat_dir_does_raise_naming_the_directory_as_a_declined_lookup(
    tmp_path: Path,
):
    above = tmp_path / "plain"
    standalone = above / ".gymrat" / "worktrees" / "standalone"
    standalone.mkdir(parents=True)
    run_git(["init"], str(standalone))
    malformed = above / ".git"
    malformed.write_text("banana\n", encoding="utf-8")

    with pytest.raises(GymratError) as exc:
        repo_root(str(standalone))

    assert not isinstance(exc.value, NotAGitRepositoryError)
    assert str(above) in str(exc.value)


def test_repo_root_when_directory_above_gymrat_dir_is_below_a_checkout_top_does_return_the_toplevel(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    worktree = add_worktree(repo, "packages/.gymrat/worktrees/experiment")

    root = repo_root(worktree)

    assert Path(root) == Path(worktree)


def test_repo_root_when_gymrat_worktree_reached_through_a_symlink_does_return_the_owning_repository(
    create_scratch_repo: Callable[[], str], tmp_path: Path
):
    repo = create_scratch_repo()
    add_worktree(repo, ".gymrat/worktrees/experiment")
    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo)

    root = repo_root(str(alias / ".gymrat" / "worktrees" / "experiment"))

    assert Path(root) == Path(repo)


def test_repo_root_when_foreign_worktree_sits_in_gymrat_dir_does_return_that_worktree(
    create_scratch_repo: Callable[[], str],
):
    host = create_scratch_repo()
    foreign = create_scratch_repo()
    target = str(Path(host) / ".gymrat" / "worktrees" / "impostor")
    add_worktree(foreign, target)

    root = repo_root(target)

    assert Path(root) == Path(target)


# ---------------------------------------------------------------------------
# session layout
# ---------------------------------------------------------------------------


def test_archived_session_path_when_given_session_id_does_name_it_in_the_session_dir():
    result = Path(archived_session_path(ROOT, SESSION_ID))

    assert result.parent == Path(session_dir(ROOT))
    assert SESSION_ID in result.name


# ---------------------------------------------------------------------------
# lockfile_path
# ---------------------------------------------------------------------------


def test_lockfile_path_when_given_root_does_map_to_golden_name():
    path = lockfile_path(LOCKFILE_ROOT)

    assert path == str(Path(tempfile.gettempdir()) / LOCKFILE_NAME)


# ---------------------------------------------------------------------------
# log file names — single-source naming for session and supervisor logs
# ---------------------------------------------------------------------------


def test_supervisor_log_name_when_given_timestamp_does_return_filename_with_ms():
    name = supervisor_log_name(1723123456789)

    assert "1723123456789" in name
