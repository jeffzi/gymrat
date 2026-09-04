"""Session time budget: write, read, and clear a deadline-bounded budget file.

The budget file is written atomically via temp-file-and-replace so a concurrent
reader never sees a partial write.  Reading checks three liveness conditions
before returning the budget: the file must parse, its deadline must be ahead of
``now_ms``, and the supervise lock for the repository root must be held.  When
any condition fails the budget is treated as absent.
"""

import contextlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from gymrat.session.lock import is_held
from gymrat.session.paths import budget_path, supervise_lockfile_path
from gymrat.session.records.models import BaselineRecord, IterationRecord, SessionLogRecord

_BUDGET_VERSION = 1

_BASELINE_TO_ITERATE_MULTIPLIER = 2
"""An iterate cycle measures both baseline and experiment, so it costs roughly
twice a baseline-only run."""


@dataclass(frozen=True, slots=True)
class Budget:
    """Immutable snapshot of a session time budget.

    Fields:
        started_at_ms: Epoch milliseconds when the budget was created.
        max_minutes: Total minutes the session may run.
        deadline_ms: Epoch milliseconds at which the budget expires.
        version: Schema version for forward compatibility.
    """

    started_at_ms: float
    max_minutes: float
    deadline_ms: float
    version: int = _BUDGET_VERSION

    def remaining_ms(self, now_ms: float) -> float:
        """Milliseconds left until the deadline, clamped at zero."""
        return max(0.0, self.deadline_ms - now_ms)


def write_budget(root: str, budget: Budget) -> None:
    """Atomically write *budget* to the budget file under *root*.

    Writes to a temporary file in the same directory, then renames so a
    concurrent reader never sees a half-written file.
    """
    target = Path(budget_path(root))
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=target.parent,
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            json.dump(asdict(budget), tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise
    tmp_path.replace(target)


def read_budget(root: str, *, now_ms: float) -> Budget | None:
    """Read and validate the budget file, or return ``None``.

    Returns ``None`` when the file is absent, contains invalid JSON, has an
    unrecognized version, its deadline has passed, or the supervise lock for
    *root* is not held.
    """
    path = Path(budget_path(root))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        budget = Budget(**data)
    except (json.JSONDecodeError, TypeError, KeyError, OSError, UnicodeDecodeError):
        return None

    if budget.version != _BUDGET_VERSION:
        return None

    if now_ms >= budget.deadline_ms:
        return None

    if not is_held(Path(supervise_lockfile_path(root))):
        return None

    return budget


def clear_budget(root: str) -> None:
    """Remove the budget file if it exists, silently succeed otherwise."""
    Path(budget_path(root)).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class DurationEstimate:
    """Estimated wall-clock duration for one ``iterate`` cycle.

    Fields:
        duration_ms: Estimated iterate duration in milliseconds.
        source: Which record kind supplied the raw duration
            (``"iteration"`` or ``"baseline"``).
        source_duration_ms: The raw duration from the source record, before
            any multiplier.
    """

    duration_ms: float
    source: str
    source_duration_ms: float


def estimate_iterate_duration(
    records: Sequence[SessionLogRecord],
) -> DurationEstimate | None:
    """Estimate the wall-clock cost of one ``iterate`` from session history.

    Scans *records* (oldest-first) from the end:

    1. The newest ``IterationRecord`` carrying a ``duration_ms`` is returned
       directly.
    2. Failing that, the newest ``BaselineRecord`` carrying a ``duration_ms``
       is doubled (an iterate measures both baseline and experiment).
    3. With neither, returns ``None`` (unknown).
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
                duration_ms=record.duration_ms * _BASELINE_TO_ITERATE_MULTIPLIER,
                source="baseline",
                source_duration_ms=record.duration_ms,
            )

    return None
