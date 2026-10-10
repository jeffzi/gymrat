"""Repository-relative paths for session state, worktrees, and lock files.

The derivation helpers are pure string functions that never touch the
filesystem: they only join a repository root with the fixed session layout.
``repo_root`` is the one helper that shells out to git to resolve that root.
"""

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

from gymrat.errors import GymratError
from gymrat.git import run_git
from gymrat.utils import stderr_text_of

SESSION_DIR_NAME = ".gymrat"
SESSION_LOG_NAME = "session.jsonl"
WORKTREES_DIR_NAME = "worktrees"
BASELINE_WORKTREE_NAME = "baseline"

# First 12 hex chars of a sha256 gives a short, collision-resistant lock name.
_DIGEST_HEX_LENGTH = 12

# Git's wording when it places a directory outside every repository.
#
# Anchored to ``^fatal:`` so a path that embeds the phrase (e.g.
# ``/tmp/not a git repository/config``) cannot skip the classification. The
# ``re.MULTILINE`` flag lets ``^`` match at line boundaries within multi-line
# stderr. ``LC_ALL=C`` in :func:`~gymrat.git.run_git` stabilizes the wording
# across locales, so a case-insensitive flag is not needed.
_NOT_A_REPOSITORY_RE = re.compile(r"^fatal: not a git repository", re.MULTILINE)


class NotAGitRepositoryError(GymratError):
    """A directory git placed outside every repository.

    Its own class because callers act on the distinction: standing outside a
    repository is a supported way to run gymrat, while a git that merely
    declined to answer says nothing about where the directory sits and must
    never be read as "no repository here".
    """


def repository_lookup_error(directory: str, cause: object) -> GymratError:
    """Classify a failed repository lookup by what git said.

    Git reporting that ``directory`` is outside every repository is an answer
    callers can act on. Every other failure — dubious ownership, an unreadable
    ``.git``, a git that cannot run at all — is git declining to answer, and
    carries git's own diagnostics so the reader sees the real reason rather than
    a wrong one.

    Args:
        directory: The directory whose repository lookup failed.
        cause: The failure to classify — typically the git exception.

    Returns:
        A :class:`NotAGitRepositoryError` when git's stderr opens
        with its not-a-repository diagnostic, otherwise a plain
        :class:`~gymrat.errors.GymratError` carrying git's diagnostics.
    """
    diagnostics = stderr_text_of(cause)

    if _NOT_A_REPOSITORY_RE.search(diagnostics):
        return NotAGitRepositoryError(
            f"Not a git repository: {directory}",
            hint="Run gymrat from inside a git repository.",
        )

    return GymratError(f"Cannot determine the git repository at {directory}: {diagnostics}")


def _rev_parse(flag: str, directory: str) -> str:
    """The path ``git rev-parse <flag>`` prints for ``directory``, trimmed.

    Args:
        flag: The ``rev-parse`` option naming the path to look up.
        directory: Directory the lookup runs from.

    Returns:
        The path as git printed it, without surrounding whitespace.

    Raises:
        GymratError: When ``directory`` is outside any repository (a
            :class:`NotAGitRepositoryError`) or git otherwise
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
            :class:`NotAGitRepositoryError`) or git otherwise
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


def _owner_candidate(toplevel: Path) -> Path | None:
    """The directory ``toplevel``'s path names as its owning checkout.

    A pure path test, deliberately cheap: ``repo_root`` is on every command's hot
    path, so the git calls that confirm the ownership only run for a path shaped
    like one of gymrat's own worktrees.

    Args:
        toplevel: Top level git reported, as a path.

    Returns:
        The directory above the innermost ``.gymrat/worktrees`` pair that has at
        least one component below it, or ``None`` when the path holds no such
        pair.
    """
    parts = toplevel.parts
    for index in range(len(parts) - 3, 0, -1):
        if parts[index] == SESSION_DIR_NAME and parts[index + 1] == WORKTREES_DIR_NAME:
            return Path(*parts[:index])
    return None


def _gymrat_worktree_owner(toplevel: str) -> str | None:
    """The checkout a gymrat-created worktree at ``toplevel`` belongs to.

    ``gymrat start`` checks its worktrees out under ``<root>/.gymrat/worktrees``,
    so a command run from inside one has to act on ``<root>``: that is where the
    session log, the budget, and the repository lock live. A linked worktree
    anywhere else is the user's own and stays its own root.

    The owner is read off the path and confirmed by a shared common directory,
    never derived from the common directory's location: that location is the
    main checkout for every linked worktree, so it cannot name an owner that is
    itself a linked worktree.

    The owner is re-resolved through git, so it is byte-identical to what
    ``repo_root`` answers when called at the root itself — the lock name digests
    those exact bytes.

    Args:
        toplevel: Top level git reported for the directory being resolved.

    Returns:
        The owning checkout's top-level path, or ``None`` when ``toplevel`` is
        not a worktree gymrat created: its path is not shaped like one, the
        directory above ``.gymrat/worktrees`` is not a checkout's top level, or
        that checkout belongs to another repository.

    Raises:
        GymratError: When git declines to resolve the common directory of
            ``toplevel``.
    """
    candidate = _owner_candidate(Path(toplevel))
    if candidate is None:
        return None
    try:
        candidate_common = git_common_dir(str(candidate))
        owner = _toplevel(str(candidate))
    except GymratError:
        return None
    if Path(owner) != candidate:
        return None
    if Path(candidate_common).resolve() != Path(git_common_dir(toplevel)).resolve():
        return None
    return owner


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
            :class:`NotAGitRepositoryError`) or git otherwise
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
    return _session_path(root, WORKTREES_DIR_NAME, BASELINE_WORKTREE_NAME)


def baseline_worktree_label() -> str:
    """The baseline worktree's path relative to the repository root, with ``/`` separators."""
    return f"{SESSION_DIR_NAME}/{WORKTREES_DIR_NAME}/{BASELINE_WORKTREE_NAME}"


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
