"""Atomic progress sidecar for dashboard polling.

The sidecar is a single JSON file written atomically on every progress event
so that a concurrent reader (the dashboard or supervisor) never sees a partial
write.  Staleness detection lets readers discard orphaned files left by a
crashed iteration.
"""

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from gymrat import clock as _clock
from gymrat.eta import MS_PER_SECOND
from gymrat.progress_events import PassFinished, PassStarted, ProgressEvent
from gymrat.session.paths import progress_path
from gymrat.session.sidecar import read_sidecar
from gymrat.utils import write_text_atomic

#: A reader discards files whose mtime is older than this many seconds.
#: 600 s (10 min) is well above the longest single benchmark pass.
STALENESS_BOUND_SECONDS: int = 600


class ProgressSnapshot(BaseModel):
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

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    passes_completed: int
    passes_total: int
    last_pass_duration_ms: float


def write_progress(root: str, snapshot: ProgressSnapshot) -> None:
    """Atomically write *snapshot* to the progress sidecar under *root*.

    A concurrent reader sees either the previous snapshot or the new one,
    never a half-written file.

    Args:
        root: Repository root under which the progress sidecar lives.
        snapshot: The progress state to write.
    """
    write_text_atomic(Path(progress_path(root)), snapshot.model_dump_json())


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
    except OSError:
        return None

    age_ms = _clock.now_ms() - stat.st_mtime * MS_PER_SECOND
    if age_ms > STALENESS_BOUND_SECONDS * MS_PER_SECOND:
        return None

    return read_sidecar(path, ProgressSnapshot)


def clear_progress(root: str) -> None:
    """Remove the progress sidecar if it exists, silently succeed otherwise."""
    Path(progress_path(root)).unlink(missing_ok=True)


@dataclass(slots=True)
class SidecarWriter:
    """A progress callback that writes a sidecar snapshot on each pass event.

    It accumulates state from ``PassStarted`` and ``PassFinished`` events and
    writes a ``ProgressSnapshot`` under ``root`` on each; other event types are
    ignored, with no write. A phase change resets ``passes_completed`` so each
    phase's progress is counted from zero.
    """

    root: str
    passes_completed: int = 0
    last_start_ms: float = 0.0
    last_pass_duration_ms: float = 0.0
    current_phase: str = ""

    def __call__(self, event: ProgressEvent) -> None:
        """Fold ``event`` into the pass state and write a snapshot for a pass event.

        Args:
            event: The progress event the sampling engine emitted.
        """
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
