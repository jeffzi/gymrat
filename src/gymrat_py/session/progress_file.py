"""Atomic progress sidecar for dashboard polling.

The sidecar is a single JSON file written atomically on every progress event
so that a concurrent reader (the dashboard or supervisor) never sees a partial
write.  Staleness detection lets readers discard orphaned files left by a
crashed iteration.
"""

import contextlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from gymrat_py.progress_events import (
    PassFinished,
    PassStarted,
    ProgressCallback,
    ProgressEvent,
)
from gymrat_py.session.paths import progress_path

#: A reader discards files whose mtime is older than this many seconds.
#: 600 s (10 min) is well above the longest single benchmark pass.
STALENESS_BOUND_SECONDS: int = 600


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """Point-in-time progress state serialized to the sidecar.

    The dashboard computes ETAs from ``passes_completed`` / ``passes_total``
    and ``last_pass_duration_ms``; this snapshot carries no ETA itself.
    """

    seq: int
    phase: str
    passes_completed: int
    passes_total: int
    current_side: str | None
    current_round: int
    last_pass_duration_ms: float
    started_at: float


def write_progress(root: str, snapshot: ProgressSnapshot) -> None:
    """Atomically write *snapshot* to the progress sidecar under *root*.

    Writes to a temporary file in the same directory, then renames so a
    concurrent reader never sees a half-written file.
    """
    target = Path(progress_path(root))
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=target.parent,
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            json.dump(asdict(snapshot), tmp_file)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
            raise
    tmp_path.replace(target)


def read_progress(root: str) -> ProgressSnapshot | None:
    """Read and parse the progress sidecar, or return ``None``.

    Returns ``None`` when the file is absent, contains invalid JSON, does not
    match the snapshot schema, or is stale (mtime older than
    ``STALENESS_BOUND_SECONDS``).
    """
    path = Path(progress_path(root))
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None

    if time.time() - stat.st_mtime > STALENESS_BOUND_SECONDS:
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProgressSnapshot(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def clear_progress(root: str) -> None:
    """Remove the progress sidecar if it exists, silently succeed otherwise."""
    Path(progress_path(root)).unlink(missing_ok=True)


def create_sidecar_writer(
    root: str,
    seq: int,
    *,
    started_at: float,
) -> ProgressCallback:
    """Return a callback that writes sidecar snapshots on pass events.

    The callback tracks accumulated state from ``PassStarted`` and
    ``PassFinished`` events and writes a ``ProgressSnapshot`` on each.
    Other event types are silently ignored (no write).
    """
    passes_completed = 0
    last_start_ms: float = 0.0
    last_pass_duration_ms: float = 0.0
    phase: str = "measure"
    current_side: str | None = None
    current_round: int = 0
    passes_total: int = 0

    def _on_event(event: ProgressEvent) -> None:
        nonlocal passes_completed, last_start_ms, last_pass_duration_ms
        nonlocal phase, current_side, current_round, passes_total

        if isinstance(event, PassStarted):
            last_start_ms = event.at_ms
        elif isinstance(event, PassFinished):
            passes_completed += 1
            last_pass_duration_ms = event.at_ms - last_start_ms
        else:
            return

        phase = event.phase
        current_side = event.label
        current_round = event.round
        passes_total = event.total_rounds * event.target_count

        write_progress(
            root,
            ProgressSnapshot(
                seq=seq,
                phase=phase,
                passes_completed=passes_completed,
                passes_total=passes_total,
                current_side=current_side,
                current_round=current_round,
                last_pass_duration_ms=last_pass_duration_ms,
                started_at=started_at,
            ),
        )

    return _on_event
