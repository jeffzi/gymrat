"""Atomic progress sidecar for dashboard polling.

The sidecar is a single JSON file written atomically on every progress event
so that a concurrent reader (the dashboard or supervisor) never sees a partial
write.  Staleness detection lets readers discard orphaned files left by a
crashed iteration.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ConfigDict, TypeAdapter, ValidationError, with_config

from gymrat.atomic_write import write_text_atomic
from gymrat.progress_events import (
    PassFinished,
    PassStarted,
    ProgressCallback,
    ProgressEvent,
)
from gymrat.session.paths import progress_path

#: A reader discards files whose mtime is older than this many seconds.
#: 600 s (10 min) is well above the longest single benchmark pass.
STALENESS_BOUND_SECONDS: int = 600


@with_config(ConfigDict(strict=True, extra="forbid"))
@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """Point-in-time progress state serialized to the sidecar.

    The dashboard computes ETAs from ``passes_completed`` / ``passes_total``
    and ``last_pass_duration_ms``; this snapshot carries no ETA itself.
    Reading is strict: a count must be a JSON integer, never a boolean or a
    float, and an unknown key rejects the whole file.

    Attributes:
        passes_completed: How many passes have finished in the current phase.
        passes_total: Total passes expected across all targets and rounds.
        last_pass_duration_ms: Wall-clock time of the most recently finished
            pass, in milliseconds.
    """

    passes_completed: int
    passes_total: int
    last_pass_duration_ms: float


_SNAPSHOT_ADAPTER: TypeAdapter[ProgressSnapshot] = TypeAdapter(ProgressSnapshot)


def write_progress(root: str, snapshot: ProgressSnapshot) -> None:
    """Atomically write *snapshot* to the progress sidecar under *root*.

    A concurrent reader sees either the previous snapshot or the new one,
    never a half-written file.

    Args:
        root: Repository root under which the progress sidecar lives.
        snapshot: The progress state to write.
    """
    text = _SNAPSHOT_ADAPTER.dump_json(snapshot).decode("utf-8")
    write_text_atomic(Path(progress_path(root)), text)


def read_progress(root: str) -> ProgressSnapshot | None:
    """Read and parse the progress sidecar, or return ``None``.

    Args:
        root: Repository root under which the progress sidecar lives.

    Returns:
        The snapshot, or ``None`` when the file is absent or unreadable,
        contains invalid JSON, is not an object with exactly the snapshot's
        keys and field types, or is stale (mtime older than
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
        return _SNAPSHOT_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, OSError, UnicodeDecodeError):
        return None


def clear_progress(root: str) -> None:
    """Remove the progress sidecar if it exists, silently succeed otherwise."""
    Path(progress_path(root)).unlink(missing_ok=True)


@dataclass(slots=True)
class _SidecarWriter:
    """Pass-event state accumulator that writes a sidecar snapshot per pass event.

    A phase change resets ``passes_completed`` so each phase's progress is
    counted from zero.
    """

    root: str
    passes_completed: int = 0
    last_start_ms: float = 0.0
    last_pass_duration_ms: float = 0.0
    current_phase: str = ""

    def __call__(self, event: ProgressEvent) -> None:
        if isinstance(event, PassStarted):
            self._enter_phase(event.phase)
            self.last_start_ms = event.at_ms
        elif isinstance(event, PassFinished):
            self._enter_phase(event.phase)
            self.passes_completed += 1
            self.last_pass_duration_ms = event.at_ms - self.last_start_ms
        else:
            return

        write_progress(
            self.root,
            ProgressSnapshot(
                passes_completed=self.passes_completed,
                passes_total=event.total_rounds * event.target_count,
                last_pass_duration_ms=self.last_pass_duration_ms,
            ),
        )

    def _enter_phase(self, phase: str) -> None:
        if phase != self.current_phase:
            self.passes_completed = 0
            self.current_phase = phase


def create_sidecar_writer(root: str) -> ProgressCallback:
    """Return a callback that writes sidecar snapshots on pass events.

    The callback tracks accumulated state from ``PassStarted`` and
    ``PassFinished`` events and writes a ``ProgressSnapshot`` on each.
    Other event types are silently ignored (no write).

    Args:
        root: Repository root under which the progress sidecar is written.

    Returns:
        A callback that accumulates pass state and writes a snapshot on each
        ``PassStarted`` or ``PassFinished`` event.
    """
    return _SidecarWriter(root)
