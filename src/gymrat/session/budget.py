"""Session time budget: write, read, and clear a deadline-bounded budget file.

The budget file is written atomically via temp-file-and-replace so a concurrent
reader never sees a partial write.  Reading checks three liveness conditions
before returning the budget: the file must parse, its deadline must be ahead of
``now_ms``, and the supervise lock for the repository root must be held.  When
any condition fails the budget is treated as absent; a supervise lock that
cannot be probed is an error, not an absent budget.
"""

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from gymrat import utils
from gymrat.session.lock import is_held
from gymrat.session.paths import budget_path, session_dir, supervise_lockfile_path
from gymrat.session.records import BaselineRecord, IterationRecord, SessionLogRecord
from gymrat.signals import install_termination_cleanup
from gymrat.utils import (
    WarnSink,
    minutes_to_ms,
    read_sidecar,
    remove_sidecar,
    warn_to_stderr,
    write_text_atomic,
)

SIDES_PER_ITERATE = 2
"""An iterate cycle measures both baseline and experiment, so it costs roughly
twice a baseline-only run."""


class Budget(BaseModel):
    """Immutable snapshot of a session time budget.

    Validation is strict: a time field must be a real number, with ``bool``
    rejected, and unknown fields are refused.

    Attributes:
        max_minutes: Total minutes the session may run.
        deadline_ms: Epoch milliseconds at which the budget expires.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    max_minutes: float
    deadline_ms: float

    def remaining_ms(self, now_ms: float) -> float:
        """Milliseconds left until the deadline, clamped at zero."""
        return max(0.0, self.deadline_ms - now_ms)


def write_budget(root: str, budget: Budget) -> None:
    """Atomically write *budget* to the budget file under *root*.

    A concurrent reader sees either the previous budget or the new one, never
    a half-written file.

    Args:
        root: Repository root under which the budget file lives.
        budget: The budget snapshot to write.
    """
    write_text_atomic(Path(budget_path(root)), budget.model_dump_json())


def read_budget(root: str, *, now_ms: float) -> Budget | None:
    """Read and validate the budget file.

    Args:
        root: Repository root under which the budget file lives.
        now_ms: Current epoch milliseconds, compared against the budget's
            deadline.

    Returns:
        The validated budget, or ``None`` when the file is absent, contains
        invalid JSON, is not an object with exactly the expected numeric
        fields, its deadline has passed, or the supervise lock for *root* is
        not held.

    Raises:
        GymratError: When the supervise lock file cannot be opened, so whether
            a supervised run holds it is unknown.
    """
    budget = read_sidecar(Path(budget_path(root)), Budget)
    if budget is None or now_ms >= budget.deadline_ms:
        return None

    if not is_held(supervise_lockfile_path(root)):
        return None

    return budget


def clear_budget(root: str, warn: WarnSink = warn_to_stderr) -> None:
    """Remove the budget file, best-effort.

    A missing file is a silent success. A removal the OS refuses is never
    raised: the budget is cleared on the way out of a supervised run, where an
    error here would replace the run's own outcome. A file left behind is
    treated as absent once the supervise lock is released.

    Args:
        root: Repository root under which the budget file lives.
        warn: Sink that receives the message when the removal fails.
    """
    remove_sidecar(Path(budget_path(root)), "budget file", warn)


def arm_budget(
    root: str, max_minutes: float, *, warn: WarnSink
) -> tuple[Budget, Callable[[], None]]:
    """Create, persist, and arm cleanup for the session time budget.

    Args:
        root: Session root directory.
        max_minutes: Maximum session duration in minutes.
        warn: Receives the warning when the OS refuses to remove the budget file.

    Returns:
        A ``(budget, release)`` pair: the budget written to disk and a callback
        that removes the budget file and uninstalls the termination hook; a
        refused removal goes to ``warn`` rather than raising. The callback is
        idempotent, so the run's own teardown and the unwind of a session whose
        setup failed can both call it without clearing twice.
    """
    budget = Budget(
        max_minutes=max_minutes, deadline_ms=utils.now_ms() + minutes_to_ms(max_minutes)
    )
    Path(session_dir(root)).mkdir(parents=True, exist_ok=True)
    write_budget(root, budget)
    release = ExitStack()
    # Callbacks unwind last-in first-out: the budget file goes, then the hook.
    release.callback(install_termination_cleanup(lambda: clear_budget(root, warn)))
    release.callback(clear_budget, root, warn)
    return budget, release.close


@dataclass(frozen=True, slots=True)
class DurationEstimate:
    """Estimated wall-clock duration for one ``iterate`` cycle.

    Attributes:
        duration_ms: Estimated iterate duration in milliseconds.
        source: Which record kind supplied the raw duration
            (``"iteration"`` or ``"baseline"``).
        source_duration_ms: The raw duration from the source record, before
            any multiplier.
    """

    duration_ms: float
    source: Literal["iteration", "baseline"]
    source_duration_ms: float


def estimate_iterate_duration(
    records: Sequence[SessionLogRecord],
) -> DurationEstimate | None:
    """Estimate the wall-clock cost of one ``iterate`` from session history.

    Scans from the end:

    1. The newest ``IterationRecord`` carrying a ``duration_ms`` is returned
       directly.
    2. Failing that, the newest ``BaselineRecord`` carrying a ``duration_ms``
       is doubled (an iterate measures both baseline and experiment).

    Args:
        records: Session log records to scan, oldest-first.

    Returns:
        The estimate, or ``None`` when no record carries a duration.
    """
    for record in reversed(records):
        if isinstance(record, IterationRecord) and record.duration_ms is not None:
            return DurationEstimate(
                duration_ms=record.duration_ms,
                source="iteration",
                source_duration_ms=record.duration_ms,
            )

    for record in reversed(records):
        if isinstance(record, BaselineRecord) and record.duration_ms is not None:
            return DurationEstimate(
                duration_ms=record.duration_ms * SIDES_PER_ITERATE,
                source="baseline",
                source_duration_ms=record.duration_ms,
            )

    return None


def iterate_headroom(
    root: str, load_records: Callable[[], Sequence[SessionLogRecord] | None]
) -> tuple[float, DurationEstimate] | None:
    """Read the live budget's time left beside the estimated cost of one ``iterate``.

    The clock is read once, before the budget, and the records are loaded
    only once a live budget is found.

    Args:
        root: Repository root whose budget and supervise lock are read.
        load_records: Returns the session log records the estimate is drawn
            from, or ``None`` when they cannot be read.

    Returns:
        The milliseconds left on the budget and the iterate estimate, or
        ``None`` when no budget is live, the records cannot be read, or no
        record carries a duration.

    Raises:
        GymratError: When the supervise lock file cannot be opened, so whether
            the budget is live is unknown.
    """
    current_ms = utils.now_ms()
    budget = read_budget(root, now_ms=current_ms)
    if budget is None:
        return None
    records = load_records()
    if records is None:
        return None
    estimate = estimate_iterate_duration(records)
    if estimate is None:
        return None
    return budget.remaining_ms(current_ms), estimate
