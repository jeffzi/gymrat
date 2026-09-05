"""The external-facing context a supervised session runs under."""

from dataclasses import dataclass

from gymrat.config import BenchlessConfig


@dataclass(frozen=True, slots=True)
class SupervisedSession:
    """Immutable snapshot of everything a supervised session needs to run.

    Built by the CLI layer (``_run_session``) and threaded into :func:`supervise`,
    which extracts the values it needs rather than accepting them as individual
    keyword arguments.

    ``lock_path`` is the repository lock (``lockfile_path(root)``), not the
    supervise lock.
    """

    root: str
    log_path: str
    lock_path: str
    config: BenchlessConfig
    deadline_ms: float
    max_minutes: float
    max_usd: float | None
