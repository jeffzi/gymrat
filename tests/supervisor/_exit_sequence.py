"""Shared drivers and session-log arrangements for the ``run_exit_sequence`` tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from gymrat.session.paths import experiment_worktree_dir, lockfile_path
from gymrat.session.workspace import worktree_fingerprint
from gymrat.supervisor.exit_sequence import run_exit_sequence
from tests._config import benchless_config
from tests.loop._settle import edit_experiment, unimproved
from tests.session.records._fixtures import append_records, iteration_record
from tests.supervisor._fixtures import collecting_observer, make_context

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.session.records import IterationRecord, SessionLogRecord
    from gymrat.supervisor.events import SessionEvent, SessionObserver
    from gymrat.supervisor.exit_sequence import ExitPhase, ExitReport
    from gymrat.supervisor.supervise import EndedBy, SupervisedSession


class ExitRun(NamedTuple):
    """A finished sequence paired with the progress, events, and warnings it produced."""

    report: ExitReport
    phases: list[ExitPhase]
    events: list[SessionEvent]
    warnings: list[str]


def session_context(
    root: str, *, timeout_seconds: int = 60, checks: str | None = None
) -> SupervisedSession:
    """A supervised session rooted at ``root``, pointed at that repository's lock."""
    return make_context(
        root=root,
        lock_path=lockfile_path(root),
        log_path=str(Path(root) / ".gymrat" / "supervisor.jsonl"),
        config=benchless_config(timeout_seconds=timeout_seconds, checks=checks),
    )


async def run_sequence(
    context: SupervisedSession,
    *,
    ended_by: EndedBy = "session",
    finalize: bool = False,
    lock_poll_ms: int = 1,
    lock_wait_ms: int | None = 0,
    is_lock_held: Callable[[], bool] | None = None,
    log: SessionObserver | None = None,
    progress: Callable[[ExitPhase], None] | None = None,
) -> ExitRun:
    """Run the exit sequence over ``context``, collecting its progress, events, and warnings.

    Args:
        context: The supervised session the sequence closes out.
        ended_by: What ended the run, as ``run_exit_sequence`` takes it.
        finalize: Whether the sequence may finalize the session.
        lock_poll_ms: How often the lock probe is retried while waiting.
        lock_wait_ms: How long the lock wait is bound to, or None for the config bound.
        is_lock_held: Lock probe override, or None to use the real one.
        log: Observer override, or None to collect events into the returned run.
        progress: Progress sink override, or None to collect phases into the returned run.

    Returns:
        The report paired with the phases, events, and warnings the run produced.
        A sink override silences the matching list, which stays empty.
    """
    phases: list[ExitPhase] = []
    warnings: list[str] = []
    events, observer = collecting_observer()
    report = await run_exit_sequence(
        context,
        ended_by=ended_by,
        finalize=finalize,
        progress=phases.append if progress is None else progress,
        log=observer if log is None else log,
        warn=warnings.append,
        lock_poll_ms=lock_poll_ms,
        lock_wait_ms=lock_wait_ms,
        is_lock_held=is_lock_held,
    )
    return ExitRun(report, phases, events, warnings)


def fingerprint(root: str) -> str:
    """The experiment worktree's fingerprint, as an iteration would have measured it."""
    experiment = experiment_worktree_dir(root)
    tree = worktree_fingerprint(Path(experiment))
    if tree is None:
        msg = f"expected a fingerprint for {experiment}"
        raise AssertionError(msg)
    return tree


def measured(root: str, record: IterationRecord, *trailing: SessionLogRecord) -> None:
    """Edit the experiment worktree, then log ``record`` fingerprinted to what it measured."""
    edit_experiment(root)
    append_records(root, record.model_copy(update={"measured_tree": fingerprint(root)}), *trailing)


def improved_iteration(root: str) -> None:
    """An improved iteration the gate lets through, so the settle step keeps it."""
    measured(root, iteration_record(seq=1))


def unimproved_iteration(root: str) -> None:
    """An iteration that read as no-signal, so the settle step discards it and keeps nothing."""
    measured(root, unimproved(1, "no-signal"))
