"""Sweeping the git worktrees a comparison or measurement claimed, on every exit path.

A phase that benchmarks a ref checks it out into a worktree of its own.
:func:`run_with_worktrees` wraps such a phase so every worktree it claimed is
removed whether the phase returns, raises, or is cut short by a termination
signal, and folds whatever the sweep could not remove into the result or the
error the caller sees.
"""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from gymrat.errors import GymratError
from gymrat.exec import kill_live_process_groups
from gymrat.git import try_git
from gymrat.report.text.render import format_cleanup_failures
from gymrat.sampling import WorktreeInfo
from gymrat.signals import install_termination_cleanup, write_on_exit


@dataclass(frozen=True, slots=True)
class WorktreeRemovalFailure:
    """A worktree cleanup could not remove, with the reason git gave.

    Attributes:
        dir: The worktree directory that could not be removed.
        error: The reason git reported for the failed removal.
    """

    dir: str
    error: str


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


# What handing one worktree to git accomplished. ``removed`` took a directory off
# disk; ``stale`` is a vanished directory whose entry git would not clear, which a
# prune must collect. ``None`` leaves nothing to count or sweep: a worktree git
# never put on disk, or a vanished one whose entry git cleared. A
# :class:`WorktreeRemovalFailure` is a directory git refused to remove.
type _RemovalOutcome = Literal["removed", "stale"] | WorktreeRemovalFailure | None


def _remove_worktree(worktree: WorktreeInfo, repo_dir: str) -> _RemovalOutcome:
    """Take one worktree off disk, or clear the entry left behind if it is gone.

    The removal names the worktree's own path instead of sweeping the
    repository, because git clears the entry of a directory that vanished behind
    its back only when asked for that path — which leaves a worktree of the
    user's own that is merely temporarily absent registered.

    Args:
        worktree: The worktree to remove.
        repo_dir: The repository the worktree belongs to.

    Returns:
        The removal status or a :class:`WorktreeRemovalFailure` when git
        refused.
    """
    on_disk = Path(worktree.dir).exists()
    if not on_disk and not worktree.created:
        return None

    error = try_git(["worktree", "remove", "--force", worktree.dir], repo_dir)
    if error is not None:
        # Nothing stands for the user to clear by hand when the directory is
        # already gone, so a refusal there is a reason to sweep, not to report.
        if on_disk:
            return WorktreeRemovalFailure(dir=worktree.dir, error=error)
        return "stale"

    # The entry is gone — clear the flag so a later sweep leaves it alone rather
    # than reclassifying it as stale.
    worktree.created = False
    return "removed" if on_disk else None


@dataclass(slots=True)
class _SweepLedger:
    """The worktrees no sweep has taken yet, and what the sweeps so far have learned.

    A signal can land while one sweep is inside a git call, and the sweep the
    signal path then runs is the last thing the process does. Both sweeps write
    to one ledger, so the second reports the worktrees the first could not
    remove and collects the stale entry the first had met.

    Attributes:
        worktrees: The registry of claimed worktrees, emptied as sweeps take them.
        removed: Worktrees the sweeps took off disk.
        failures: Worktrees the sweeps left on disk, one entry each.
        prune_owed: Whether a targeted removal came back stale and no sweep has
            started the prune that collects it.
    """

    worktrees: list[WorktreeInfo]
    removed: int = 0
    failures: list[WorktreeRemovalFailure] = field(default_factory=list)
    prune_owed: bool = False

    def __iter__(self) -> Iterator[WorktreeInfo]:
        # Each worktree leaves the registry before it is yielded, so a sweep that
        # starts while another is inside a git call finds only the worktrees no
        # sweep has taken: git is asked to remove each one once, and a worktree git
        # has just removed is never re-read as a vanished one that needs a prune.
        while self.worktrees:
            yield self.worktrees.pop(0)


def cleanup_worktrees(worktrees: Iterable[WorktreeInfo], repo_dir: str) -> CleanupResult:
    """Remove each of ``worktrees`` git has anything for.

    Never raises: callers invoke this while already handling a failed run, so a
    throw here would replace the error the user actually needs to see.

    Args:
        worktrees: The worktrees to sweep, each taken just before git is asked
            to remove it. A sweep that takes over from an interrupted one
            carries on from what that sweep had already recorded.
        repo_dir: The repository the removals and prune run against.

    Returns:
        A :class:`CleanupResult` counting removals, listing failures, and
        carrying any prune-sweep error.
    """
    ledger = worktrees if isinstance(worktrees, _SweepLedger) else _SweepLedger([])

    for worktree in worktrees:
        outcome = _remove_worktree(worktree, repo_dir)
        if isinstance(outcome, WorktreeRemovalFailure):
            ledger.failures.append(outcome)
        elif outcome == "removed":
            ledger.removed += 1
        elif outcome == "stale":
            ledger.prune_owed = True

    # Prune only when a targeted removal failed and may have left an entry with
    # no directory behind it. Naming each worktree already clears its own entry,
    # so an all-success sweep — or one with nothing to ask git for, in a repo_dir
    # that may not even be a git repo — has nothing to collect. Pruning anyway
    # would deregister worktrees of the user's own that are only temporarily
    # absent: an unmounted volume, a directory moved aside.
    prune_error = None
    if ledger.prune_owed:
        # Settled before git is asked: a sweep that starts while this prune is
        # running must not run a second one.
        ledger.prune_owed = False
        prune_error = try_git(["worktree", "prune"], repo_dir)
    return CleanupResult(
        removed=ledger.removed, failures=tuple(ledger.failures), prune_error=prune_error
    )


_CLEANUP_UNFINISHED = "cleanup did not finish:"


async def run_with_worktrees[M, R](
    phase: Callable[[str, list[WorktreeInfo], asyncio.Event], Awaitable[M]],
    build_result: Callable[[M, CleanupResult], R],
) -> R:
    """Run a phase that may claim worktrees, sweeping them on every exit path.

    A termination cleanup is installed before any worktree exists, so a signal
    arriving mid-run still sweeps whatever was claimed; that cleanup aborts the
    run, sweeps, and hands what the sweep left unfinished to the exit output.
    The normal path sweeps exactly once whether the phase returns or raises. A
    signal that lands during the normal path's sweep takes over only the
    worktrees that sweep has not reached, and inherits its accounting: the exit
    output names the worktrees that sweep could not remove, and a stale entry
    it had met is pruned once.

    Args:
        phase: The work to run. It receives the repository directory, the
            registry it appends claimed worktrees to, and an abort event a
            termination signal sets.
        build_result: Combines the phase's measurement with the cleanup outcome
            into the return value.

    Returns:
        The value ``build_result`` produced from the measurement and cleanup.

    Raises:
        Exception: Whatever ``phase`` raises, propagated as-is when the worktree
            sweep succeeds. When the sweep also failed, a ``GymratError`` is
            replaced by one of the same subclass and hint, and any other error
            by a plain ``Exception`` chained to the original, each with the
            cleanup diagnostics appended to its message.
    """
    repo_dir = str(Path.cwd())
    worktrees: list[WorktreeInfo] = []
    ledger = _SweepLedger(worktrees)
    abort = asyncio.Event()

    def terminate() -> None:
        abort.set()
        # Kill any live bench group synchronously: on the signal path the loop
        # may not resume to process the abort before the sweep runs, so the
        # child must be dead before cleanup_worktrees touches the worktrees.
        kill_live_process_groups()
        cleanup = cleanup_worktrees(ledger, repo_dir)
        details = format_cleanup_failures(cleanup.failures, cleanup.prune_error)
        if details:
            # The process exits from the signal handler, so no caller is left
            # to report a worktree this sweep could not remove.
            write_on_exit("\n".join([_CLEANUP_UNFINISHED, *details]) + "\n")

    uninstall = install_termination_cleanup(terminate)
    try:
        measurement = await phase(repo_dir, worktrees, abort)
    except Exception as error:
        cleanup = cleanup_worktrees(ledger, repo_dir)
        wrapped = _with_cleanup_failures(error, cleanup)
        if wrapped is error:
            raise
        raise wrapped from error
    else:
        cleanup = cleanup_worktrees(ledger, repo_dir)
        return build_result(measurement, cleanup)
    finally:
        uninstall()


def _with_cleanup_failures(error: Exception, cleanup: CleanupResult) -> Exception:
    """Fold cleanup diagnostics into ``error``, preserving its subclass, hint and reason.

    ``type(error)(...)`` reconstructs the exact subclass — a
    :class:`~gymrat.errors.CommandError` stays a ``CommandError`` and an
    ``AdapterError`` stays an ``AdapterError`` — because every
    :class:`~gymrat.errors.GymratError` shares the ``(message, *, hint, reason)``
    signature. The reason is passed explicitly so a subclass's default reason
    never replaces the one the phase raised with. A non-gymrat error has no hint
    or reason and becomes a plain ``Exception``. The caller chains ``error`` as
    the replacement's cause.

    Args:
        error: The error the phase raised.
        cleanup: The outcome of the worktree sweep.

    Returns:
        ``error`` when the sweep left nothing behind, else a same-typed
        replacement whose message appends the cleanup diagnostics.
    """
    details = format_cleanup_failures(cleanup.failures, cleanup.prune_error)
    if not details:
        return error

    combined = "\n".join([str(error), "", _CLEANUP_UNFINISHED, *details])
    if isinstance(error, GymratError):
        return type(error)(combined, hint=error.hint, reason=error.reason)
    return Exception(combined)
