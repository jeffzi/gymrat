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
SESSION_LOG_NAME = "session.jsonl"
WORKTREES_DIR_NAME = "worktrees"

# First 12 hex chars of a sha256 gives a short, collision-resistant lock name.
_DIGEST_HEX_LENGTH = 12


def _rev_parse(flag: str, directory: str) -> str:
    """The path ``git rev-parse <flag>`` prints for ``directory``, trimmed.

    Args:
        flag: The ``rev-parse`` option naming the path to look up.
        directory: Directory the lookup runs from.

    Returns:
        The path as git printed it, without surrounding whitespace.

    Raises:
        GymratError: When ``directory`` is outside any repository (a
            :class:`~gymrat.git.NotAGitRepositoryError`) or git otherwise
            declines to answer.
    """
    try:
        return run_git(["rev-parse", flag], directory).strip()
    except (subprocess.SubprocessError, OSError) as error:
        # Bound first: ruff's DOC501 reads ``raise <call>()`` as raising the
        # callee's name and would demand it in ``Raises:``.
        err = repository_lookup_error(directory, error)
        raise err from error


def _toplevel(directory: str) -> str:
    """Top level git reports for ``directory``, normalized to native separators.

    Args:
        directory: Directory the lookup runs from.

    Returns:
        The top-level path of the worktree ``directory`` belongs to.

    Raises:
        GymratError: When ``directory`` is outside any repository (a
            :class:`~gymrat.git.NotAGitRepositoryError`) or git otherwise
            declines to answer.
    """
    # git reports forward slashes on every platform; normalizing here is what
    # lets a root compare and hash identically to a native path.
    return str(Path(_rev_parse("--show-toplevel", directory)))


def git_common_dir(root: str) -> str:
    """Absolute path of the shared git directory backing ``root``.

    Everything a repository shares across its worktrees — ``info/exclude``, the
    object store, the worktree registry — lives in the common directory, so a
    linked worktree, whose ``.git`` is a file pointing elsewhere, reaches the
    same files the main checkout uses.

    Args:
        root: Directory the lookup runs from. Any directory inside the
            repository answers with the same common directory.

    Returns:
        The resolved absolute path. git prints the path relative to the working
        directory when it sits inside it, hence the resolve against ``root``.

    Raises:
        GymratError: When ``root`` is not inside a git repository, or git
            otherwise declines to answer.
    """
    # An absolute path git prints (a linked worktree's common dir) stands on its
    # own; a relative one (``.git`` in the main checkout) resolves against root.
    return str(Path(root, _rev_parse("--git-common-dir", root)))


def _names_a_gymrat_worktree(toplevel: Path) -> bool:
    """Whether ``toplevel``'s path places it below some ``.gymrat/worktrees``.

    A pure path test, deliberately cheap: ``repo_root`` is on every command's hot
    path, so the git calls that confirm the ownership only run for a path shaped
    like one of gymrat's own worktrees.

    Args:
        toplevel: Top level git reported, as a path.

    Returns:
        ``True`` when the session directory and worktrees names appear as
        consecutive components with at least one component below them.
    """
    parts = toplevel.parts
    return any(
        parts[index] == SESSION_DIR_NAME and parts[index + 1] == WORKTREES_DIR_NAME
        for index in range(len(parts) - 2)
    )


def _gymrat_worktree_owner(toplevel: str) -> str | None:
    """The repository a gymrat-created worktree at ``toplevel`` belongs to.

    ``gymrat start`` checks its worktrees out under ``<root>/.gymrat/worktrees``,
    so a command run from inside one has to act on ``<root>``: that is where the
    session log, the budget, and the repository lock live. A linked worktree
    anywhere else is the user's own and stays its own root.

    The owner is re-resolved through git rather than taken from the common
    directory's path, so it is byte-identical to what ``repo_root`` answers when
    called at the root itself — the lock name digests those exact bytes.

    Args:
        toplevel: Top level git reported for the directory being resolved.

    Returns:
        The owning repository's top-level path, or ``None`` when ``toplevel`` is
        not a worktree gymrat created.

    Raises:
        GymratError: When git declines to resolve the common directory or the
            owning repository.
    """
    top = Path(toplevel)
    if not _names_a_gymrat_worktree(top):
        return None
    owner = Path(git_common_dir(toplevel)).parent
    if not top.is_relative_to(owner / SESSION_DIR_NAME / WORKTREES_DIR_NAME):
        return None
    return _toplevel(str(owner))


def repo_root(cwd: str | None = None) -> str:
    """Resolve the top level of the git repository containing ``cwd``.

    Args:
        cwd: Directory to resolve the repository from. Defaults to the process
            working directory. Probing from a nested subdirectory still returns
            the repository top level, not the subdirectory; probing from inside
            one of the worktrees gymrat checked out under
            ``.gymrat/worktrees`` walks out to the repository that owns it.

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
    toplevel = _toplevel(directory)
    owner = _gymrat_worktree_owner(toplevel)
    return toplevel if owner is None else owner


def _session_path(root: str, *parts: str) -> str:
    """Path under ``root``'s session directory, joining *parts* as components."""
    return str(Path(root) / SESSION_DIR_NAME / Path(*parts))


def session_dir(root: str) -> str:
    """Directory holding the session log and worktrees for ``root``."""
    return _session_path(root)


def supervisor_log_name(ms: int | str) -> str:
    """Bare filename for a supervisor log stamped with *ms*."""
    return f"supervisor-{ms}.jsonl"


def session_jsonl_path(root: str) -> str:
    """Path to the active session log under ``root``."""
    return _session_path(root, SESSION_LOG_NAME)


def archived_session_path(root: str, session_id: str) -> str:
    """Path to the archived log for a completed session under ``root``."""
    return _session_path(root, f"session-{session_id}.jsonl")


def experiment_worktree_dir(root: str) -> str:
    """Path to the worktree the session edits and commits on, under ``root``."""
    return _session_path(root, WORKTREES_DIR_NAME, "experiment")


def baseline_worktree_dir(root: str) -> str:
    """Path to the worktree detached at the session's pinned baseline commit, under ``root``."""
    return _session_path(root, WORKTREES_DIR_NAME, "baseline")


def _repo_digest(root: str) -> str:
    """Short digest of the exact root bytes, keying a lockfile to a checkout.

    The digest is taken over the root string bytes with no normalization: it is
    a cross-implementation contract, so two runs over the same checkout must
    land on the same lockfile name.

    Args:
        root: The repository root path.

    Returns:
        The first ``_DIGEST_HEX_LENGTH`` hex characters of the SHA-256 digest.
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
    return _session_path(root, "budget.json")


def progress_path(root: str) -> str:
    """Path to the progress sidecar file under ``root``'s session directory."""
    return _session_path(root, "progress.json")
