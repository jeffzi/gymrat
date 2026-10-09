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
from gymrat.git import NotAGitRepositoryError
from gymrat.session.paths import (
    archived_session_path,
    baseline_worktree_dir,
    budget_path,
    experiment_worktree_dir,
    git_common_dir,
    lockfile_path,
    progress_path,
    repo_root,
    repository_lookup_error,
    session_dir,
    session_jsonl_path,
    supervise_lockfile_path,
    supervisor_log_name,
)
from tests._git import add_worktree, run_git
from tests.session.records._fixtures import (
    SESSION_ID,
)

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
    assert isinstance(error, GymratError)
    assert str(error) == (
        "Cannot determine the git repository at /some/dir: "
        "error: cannot open /tmp/not a git repository/config: No such file"
    )


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

    assert Path(root) == Path(repo)


def test_repo_root_when_no_directory_given_does_use_cwd(
    create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch
):
    repo = create_scratch_repo()
    nested = Path(repo) / "packages" / "core"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    root = repo_root()

    assert Path(root) == Path(repo)


@pytest.mark.parametrize(
    "lookup",
    [
        pytest.param(repo_root, id="repo-root"),
        pytest.param(git_common_dir, id="git-common-dir"),
    ],
)
def test_repository_lookup_when_directory_not_in_repo_does_raise_not_a_git_repository(
    tmp_path: Path, lookup: Callable[[str], str]
):
    with pytest.raises(NotAGitRepositoryError) as excinfo:
        lookup(str(tmp_path))

    assert str(excinfo.value) == f"Not a git repository: {tmp_path}"
    assert excinfo.value.hint == "Run gymrat from inside a git repository."


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


# Where a session command's working directory can sit inside a worktree gymrat
# owns: the worktree itself, or any directory below it.
GYMRAT_WORKTREE_PROBES = [
    pytest.param("experiment", "", id="experiment-top"),
    pytest.param("baseline", "", id="baseline-top"),
    pytest.param("experiment", "packages/core", id="below-experiment"),
]


@pytest.mark.parametrize(
    "checkout",
    [
        pytest.param(_main_checkout, id="main-checkout"),
        pytest.param(_linked_worktree, id="linked-worktree"),
    ],
)
@pytest.mark.parametrize(("worktree_name", "below"), GYMRAT_WORKTREE_PROBES)
def test_repo_root_when_probed_inside_a_gymrat_worktree_does_return_the_owning_checkout(
    create_scratch_repo: Callable[[], str],
    checkout: Callable[[str], str],
    worktree_name: str,
    below: str,
):
    owner = checkout(create_scratch_repo())
    worktree = add_worktree(owner, f".gymrat/worktrees/{worktree_name}")
    probe = Path(worktree, below)
    probe.mkdir(parents=True, exist_ok=True)

    root = repo_root(str(probe))

    assert Path(root) == Path(owner)


def test_repo_root_when_directory_above_gymrat_dir_is_not_a_repository_does_return_the_toplevel(
    tmp_path: Path,
):
    standalone = tmp_path / "plain" / ".gymrat" / "worktrees" / "standalone"
    standalone.mkdir(parents=True)
    run_git(["init"], str(standalone))

    root = repo_root(str(standalone))

    assert Path(root) == Path(standalone)


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


def test_repo_root_when_probed_in_a_worktree_outside_the_gymrat_dir_does_return_that_worktree(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    worktree = add_worktree(repo, "hand-made")

    root = repo_root(worktree)

    assert Path(root) == Path(worktree)


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


def _derive_archived(root: str) -> str:
    return archived_session_path(root, SESSION_ID)


@pytest.mark.parametrize(
    ("derive", "relative"),
    [
        pytest.param(session_dir, (".gymrat",), id="session-dir"),
        pytest.param(session_jsonl_path, (".gymrat", "session.jsonl"), id="session-log"),
        pytest.param(
            experiment_worktree_dir,
            (".gymrat", "worktrees", "experiment"),
            id="experiment-worktree",
        ),
        pytest.param(
            baseline_worktree_dir, (".gymrat", "worktrees", "baseline"), id="baseline-worktree"
        ),
        pytest.param(
            _derive_archived, (".gymrat", f"session-{SESSION_ID}.jsonl"), id="archived-session"
        ),
        pytest.param(budget_path, (".gymrat", "budget.json"), id="budget"),
        pytest.param(progress_path, (".gymrat", "progress.json"), id="progress"),
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


# The supervise lock shares the repo digest (it is keyed on the root, not the
# prefix), so its golden names are the lockfile names with the supervise prefix.
@pytest.mark.parametrize(
    ("lock_path", "prefix"),
    [
        pytest.param(lockfile_path, "gymrat-lock-", id="repo-lock"),
        pytest.param(supervise_lockfile_path, "gymrat-supervise-lock-", id="supervise-lock"),
    ],
)
@pytest.mark.parametrize(("root", "name"), LOCKFILE_NAMES)
def test_lockfile_path_when_given_root_does_map_to_golden_name(
    lock_path: Callable[[str], str], prefix: str, root: str, name: str
):
    expected = name.replace("gymrat-lock-", prefix)

    path = lock_path(root)

    assert path == str(Path(tempfile.gettempdir()) / expected)


# ---------------------------------------------------------------------------
# log file names — single-source naming for session and supervisor logs
# ---------------------------------------------------------------------------


def test_supervisor_log_name_when_given_timestamp_does_return_filename_with_ms():
    name = supervisor_log_name(1723123456789)

    assert name == "supervisor-1723123456789.jsonl"
