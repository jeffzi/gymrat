"""Repository-relative paths for session state, worktrees, and lock files.

The derivation helpers are pure string functions that never touch the
filesystem: they only join a repository root with the fixed session layout.
``repo_root`` is the one helper that shells out to git to resolve that root.
"""

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from gymrat.git import repository_lookup_error, run_git

SESSION_DIR_NAME = ".gymrat"

# First 12 hex chars of a sha256 gives a short, collision-resistant lock name.
_DIGEST_HEX_LENGTH = 12


def repo_root(cwd: str | None = None) -> str:
    """Resolve the top level of the git repository containing ``cwd``.

    Args:
        cwd: Directory to resolve the repository from. Defaults to the process
            working directory. Probing from a nested subdirectory still returns
            the repository top level, not the subdirectory.

    Returns:
        The repository's top-level path. git reports forward slashes on every
        platform, so the result is normalized to compare and hash identically
        to native paths.

    Raises:
        GymratError: When ``cwd`` is outside any repository (a
            :class:`~gymrat.git.NotAGitRepositoryError`) or git otherwise
            fails to resolve the repository.
    """
    directory = os.getcwd() if cwd is None else cwd  # noqa: PTH109 -- returns str directly, matching the str return type of this function
    try:
        toplevel = run_git(["rev-parse", "--show-toplevel"], directory).strip()
    except (subprocess.SubprocessError, OSError) as error:
        raise repository_lookup_error(directory, error) from error
    return str(Path(toplevel))


def session_dir(root: str) -> str:
    """Directory holding the session log and worktrees for ``root``."""
    return str(Path(root) / SESSION_DIR_NAME)


def session_jsonl_path(root: str) -> str:
    """Path to the active session log under ``root``."""
    return str(Path(root) / SESSION_DIR_NAME / "session.jsonl")


def archived_session_path(root: str, session_id: str) -> str:
    """Path to the archived log for a completed session under ``root``."""
    return str(Path(root) / SESSION_DIR_NAME / f"session-{session_id}.jsonl")


def experiment_worktree_dir(root: str) -> str:
    """Path to the experiment worktree under ``root``."""
    return str(Path(root) / SESSION_DIR_NAME / "worktrees" / "experiment")


def baseline_worktree_dir(root: str) -> str:
    """Path to the baseline worktree under ``root``."""
    return str(Path(root) / SESSION_DIR_NAME / "worktrees" / "baseline")


def _repo_digest(root: str) -> str:
    """Short digest of the exact root bytes, keying a lockfile to a checkout.

    The digest is taken over the root string bytes with no normalization: it is
    a cross-implementation contract, so two runs over the same checkout must
    land on the same lockfile name.
    """
    return hashlib.sha256(root.encode("utf-8")).hexdigest()[:_DIGEST_HEX_LENGTH]


def _lock_path(name_prefix: str, root: str) -> str:
    """Digest-named lockfile for ``root`` in the system temp directory."""
    return str(Path(tempfile.gettempdir()) / f"{name_prefix}-{_repo_digest(root)}.json")


def lockfile_path(root: str) -> str:
    """Single-flight lockfile guarding a gymrat run over ``root``."""
    return _lock_path("gymrat-lock", root)


def supervise_lockfile_path(root: str) -> str:
    """Lockfile guarding the supervisor for a gymrat run over ``root``."""
    return _lock_path("gymrat-supervise-lock", root)


def budget_path(root: str) -> str:
    """Path to the budget file under ``root``'s session directory."""
    return str(Path(root) / SESSION_DIR_NAME / "budget.json")


def progress_path(root: str) -> str:
    """Path to the progress sidecar file under ``root``'s session directory."""
    return str(Path(root) / SESSION_DIR_NAME / "progress.json")
