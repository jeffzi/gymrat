"""Benchmark targets: the two things a run can compare against.

A target names *what* to benchmark. Either the working tree as it stands, or a
committed ref materialized into its own worktree. The variant class is the
discriminant: ``isinstance`` checks distinguish them, so no separate tag field is
carried.
"""

import errno
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from gymrat.errors import GymratError
from gymrat.git import run_git
from gymrat.utils import stderr_text_of

# Hint attached to every unresolvable target, naming the two readings gymrat
# accepts for the input.
_RESOLVE_TARGET_HINT = "Pass an existing directory, or a git ref that resolves to a commit."


@dataclass(frozen=True, slots=True)
class InPlaceTarget:
    """The working tree exactly as it is on disk.

    Attributes:
        dir: The directory the benchmark runs in.
    """

    dir: str


@dataclass(frozen=True, slots=True)
class RefTarget:
    """A committed ref, benchmarked from a worktree checked out at its commit.

    Attributes:
        ref: The ref the user named (branch, tag, or revision expression).
        resolved_sha: The commit the ref resolved to when the run began.
    """

    ref: str
    resolved_sha: str


type Target = InPlaceTarget | RefTarget
"""Either the working tree in place or a committed ref in its own worktree."""


@dataclass(frozen=True, slots=True)
class WorktreeRemovalFailure:
    """A worktree cleanup could not remove, with the reason git gave.

    Attributes:
        dir: The worktree directory that could not be removed.
        error: The reason git reported for the failed removal.
    """

    dir: str
    error: str


def _try_resolve_directory(target_input: str) -> InPlaceTarget | None:
    """Attempt directory resolution before the input is tried as a ref.

    A symlink loop or an unsearchable parent says nothing about whether the
    input is a ref, so it is reported rather than silently retried as one.

    Args:
        target_input: The target as the user typed it.

    Returns:
        An :class:`InPlaceTarget` when the input names an existing directory,
        ``None`` otherwise so the caller falls through to ref resolution.

    Raises:
        GymratError: When the probe fails for a reason other than an absent
            path.
    """
    # ``absolute`` mirrors path resolution against the process cwd without
    # touching the filesystem, so a symlink loop surfaces from the stat probe
    # below (where it is classified) rather than here.
    absolute_path = Path(target_input).absolute()
    try:
        stats = absolute_path.stat()
    except OSError as error:
        # ENOENT is a missing path; ENOTDIR is a ref name carrying a slash that
        # resolves underneath one of its own components (``fix/typo`` with a
        # file named ``fix`` present). Both leave ref resolution as the input's
        # only remaining reading; any other errno is reported instead.
        if error.errno in (errno.ENOENT, errno.ENOTDIR):
            return None
        message = f"Cannot resolve target '{target_input}': {stderr_text_of(error)}"
        raise GymratError(message, hint=_RESOLVE_TARGET_HINT) from error

    if not stat.S_ISDIR(stats.st_mode):
        return None
    # realpath collapses symlinks so a symlinked target and its destination
    # compare as the same place.
    return InPlaceTarget(dir=str(absolute_path.resolve()))


def resolve_target(target_input: str, repo_dir: str) -> Target:
    """Interpret a user-supplied target as either a directory or a git ref.

    An existing directory wins over a git ref of the same name, so a branch
    named after a sibling directory resolves to the directory.

    Args:
        target_input: The directory path or git ref the user named.
        repo_dir: The repository ref resolution runs against.

    Returns:
        An :class:`InPlaceTarget` for an existing directory, otherwise a
        :class:`RefTarget` for a ref git can verify.

    Raises:
        GymratError: When the input is neither an existing directory nor a ref
            git can verify, and when the directory probe itself fails.
    """
    directory = _try_resolve_directory(target_input)
    if directory is not None:
        return directory

    try:
        # ``--end-of-options`` stops a leading-dash input being parsed as a git
        # option. ``^{commit}`` peels the ref, so a tag resolves to the commit it
        # points at and a tree or blob sha fails instead of yielding a sha no
        # worktree can check out.
        resolved_sha = run_git(
            ["rev-parse", "--verify", "--end-of-options", f"{target_input}^{{commit}}"],
            repo_dir,
        ).strip()
    except subprocess.CalledProcessError as error:
        message = f"Cannot resolve target '{target_input}': {stderr_text_of(error)}"
        raise GymratError(message, hint=_RESOLVE_TARGET_HINT) from error

    return RefTarget(ref=target_input, resolved_sha=resolved_sha)
