"""Benchmark targets: the two things a run can compare against.

A target names *what* to benchmark. Either the working tree as it stands, or a
committed ref materialized into its own worktree. The variant class is the
discriminant: ``isinstance`` checks distinguish them, so no separate tag field is
carried.
"""

import errno
import os
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gymrat_py.errors import GymratError, stderr_text_of
from gymrat_py.git import run_git, try_git

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


@dataclass(slots=True)
class WorktreeInfo:
    """A worktree directory this process claimed for a ref, pinned to a SHA.

    The directory need not exist: :func:`plan_worktree` reserves the path before
    any git runs, and :func:`materialize_worktree` can fail after creating it.
    Cleanup treats an absent directory as nothing to do rather than as an error.

    Attributes:
        dir: The path the worktree will live at (may not yet exist on disk).
        sha: The commit the worktree is checked out at.
        created: Whether ``git worktree add`` ever put this directory on disk.
            :func:`plan_worktree` starts it ``False`` and
            :func:`materialize_worktree` raises it once the add leaves something
            behind, which is what lets cleanup tell a worktree that was never
            created from one that was created and has since vanished — only the
            latter can leave a registry entry behind to clear. It is cleared
            again on a successful removal.
    """

    dir: str
    sha: str
    created: bool


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Outcome of a worktree cleanup sweep.

    Attributes:
        removed: Worktrees this call took off disk.
        failures: Worktrees left on disk, one entry each.
        prune_error: Why the repo-wide ``git worktree prune`` sweep failed, or
            ``None`` if it succeeded or never ran. Prune runs once per call, not
            once per worktree, so it gets its own slot rather than a synthetic
            entry in ``failures``.
    """

    removed: int
    failures: tuple[WorktreeRemovalFailure, ...]
    prune_error: str | None


def _unresolvable_target_error(target_input: str, cause: object) -> GymratError:
    """The failure a target the tool cannot make sense of is reported as."""
    return GymratError(
        f"Cannot resolve target '{target_input}': {stderr_text_of(cause)}",
        hint=_RESOLVE_TARGET_HINT,
    )


def _is_absent_path_error(error: OSError) -> bool:
    """Whether a probe failure means "no directory here", not "the probe broke".

    ``ENOENT`` is a missing path; ``ENOTDIR`` is a ref name carrying a slash
    that resolves underneath one of its own components (``fix/typo`` with a file
    named ``fix`` present). Both leave ref resolution as the input's only
    remaining reading; any other errno is reported instead of retried as a ref.
    """
    return error.errno in (errno.ENOENT, errno.ENOTDIR)


def _try_resolve_directory(target_input: str) -> InPlaceTarget | None:
    """Attempt directory resolution, returning ``None`` to fall through to a ref.

    A symlink loop or an unsearchable parent says nothing about whether the
    input is a ref, so it is reported rather than silently retried as one.
    """
    # ``absolute`` mirrors path resolution against the process cwd without
    # touching the filesystem, so a symlink loop surfaces from the stat probe
    # below (where it is classified) rather than here.
    absolute_path = Path(target_input).absolute()
    try:
        stats = absolute_path.stat()
    except OSError as error:
        if _is_absent_path_error(error):
            return None
        raise _unresolvable_target_error(target_input, error) from error

    if not stat.S_ISDIR(stats.st_mode):
        return None
    # realpath collapses symlinks so a symlinked target and its destination
    # compare as the same place.
    return InPlaceTarget(dir=os.path.realpath(absolute_path))


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
        raise _unresolvable_target_error(target_input, error) from error

    return RefTarget(ref=target_input, resolved_sha=resolved_sha)


def plan_worktree(ref: RefTarget) -> WorktreeInfo:
    """Choose where a ref target's worktree will live, without touching disk.

    Deciding the path up front is what lets a caller register the directory for
    cleanup before git can create it: ``git worktree add`` can be killed once the
    worktree is on disk but before it returns, and cleanup only sweeps paths
    something already names.

    Args:
        ref: The ref target whose commit the worktree will be pinned to.

    Returns:
        A planned :class:`WorktreeInfo` whose directory does not yet exist.

    Raises:
        GymratError: When the system temp directory cannot be resolved.
    """
    tmp_base = tempfile.gettempdir()
    try:
        resolved_base = os.path.realpath(tmp_base, strict=True)
    except OSError as error:
        message = f"Cannot resolve temp directory '{tmp_base}': {stderr_text_of(error)}"
        raise GymratError(message) from error

    return WorktreeInfo(
        dir=str(Path(resolved_base) / f"gymrat-wt-{uuid.uuid4()}"),
        sha=ref.resolved_sha,
        created=False,
    )


def materialize_worktree(worktree: WorktreeInfo, repo_dir: str) -> None:
    """Check a planned worktree out into its directory, detached at its SHA.

    Records on ``worktree`` whether anything reached disk, so cleanup can tell a
    worktree that was never created from one that was.

    Args:
        worktree: The planned worktree to check out; mutated in place so
            ``created`` reflects what landed on disk.
        repo_dir: The repository the worktree is added to.

    Raises:
        GymratError: When ``git worktree add`` fails — the planned directory may
            exist anyway, so callers must still hand it to
            :func:`cleanup_worktrees`.
    """
    try:
        run_git(["worktree", "add", "--detach", worktree.dir, worktree.sha], repo_dir)
    except subprocess.CalledProcessError as error:
        message = f"git worktree add failed for {worktree.sha}: {stderr_text_of(error)}"
        raise GymratError(message) from error
    finally:
        # git registers the worktree before the command returns, and the add can
        # be killed in between, so what landed on disk — not whether git exited
        # zero — is what says a registry entry may exist.
        worktree.created = Path(worktree.dir).exists()


# What handing one worktree to git accomplished. ``deregistered`` and ``stale``
# both describe a directory that vanished behind git's back; only ``stale`` may
# leave an entry a prune must collect. ``untouched`` is a worktree git never put
# on disk. A :class:`WorktreeRemovalFailure` is a directory git refused to remove.
_RemovalStatus = Literal["removed", "deregistered", "stale", "untouched"]
_RemovalOutcome = _RemovalStatus | WorktreeRemovalFailure


def _remove_worktree(worktree: WorktreeInfo, repo_dir: str) -> _RemovalOutcome:
    """Take one worktree off disk, or clear the entry left behind if it is gone.

    The removal names the worktree's own path instead of sweeping the
    repository, because git clears the entry of a directory that vanished behind
    its back only when asked for that path — which leaves a worktree of the
    user's own that is merely temporarily absent registered.
    """
    on_disk = Path(worktree.dir).exists()
    if not on_disk and not worktree.created:
        return "untouched"

    error = try_git(["worktree", "remove", "--force", worktree.dir], repo_dir)
    if error is not None:
        # Nothing stands for the user to clear by hand when the directory is
        # already gone, so a refusal there is a reason to sweep, not to report.
        if on_disk:
            return WorktreeRemovalFailure(dir=worktree.dir, error=error)
        return "stale"

    # The entry is gone — clear the flag so a later sweep treats it as untouched
    # rather than reclassifying it as stale.
    worktree.created = False
    return "removed" if on_disk else "deregistered"


def cleanup_worktrees(worktrees: Sequence[WorktreeInfo], repo_dir: str) -> CleanupResult:
    """Remove each of ``worktrees`` git has anything for.

    Never raises: callers invoke this while already handling a failed run, so a
    throw here would replace the error the user actually needs to see.
    Everything that went wrong lands in the returned result instead.

    Args:
        worktrees: The worktrees to sweep.
        repo_dir: The repository the removals and prune run against.

    Returns:
        A :class:`CleanupResult` counting removals, listing failures, and
        carrying any prune-sweep error.
    """
    failures: list[WorktreeRemovalFailure] = []
    removed = 0
    # A worktree git may still list without a directory backing it: gone before
    # the sweep reached it, or still there because its removal failed.
    may_have_stale_entry = False

    for worktree in worktrees:
        outcome = _remove_worktree(worktree, repo_dir)
        if isinstance(outcome, WorktreeRemovalFailure):
            failures.append(outcome)
        elif outcome == "removed":
            removed += 1
        elif outcome == "stale":
            may_have_stale_entry = True

    # Prune only when a targeted removal failed and may have left an entry with
    # no directory behind it. Naming each worktree already clears its own entry,
    # so an all-success sweep — or one with nothing to ask git for, in a repo_dir
    # that may not even be a git repo — has nothing to collect. Pruning anyway
    # would deregister worktrees of the user's own that are only temporarily
    # absent: an unmounted volume, a directory moved aside.
    prune_error = try_git(["worktree", "prune"], repo_dir) if may_have_stale_entry else None
    return CleanupResult(removed=removed, failures=tuple(failures), prune_error=prune_error)
