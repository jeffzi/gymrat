"""Scan the session log between supervisor events for conditions that end the run."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gymrat.errors import GymratError
from gymrat.loop.iterate.run import stop_condition
from gymrat.session.store import fold_session, read_records
from gymrat.supervisor.turns import EndCondition, detect_end_condition

if TYPE_CHECKING:
    from gymrat.config import BenchlessConfig
    from gymrat.session import SessionLogRecord
    from gymrat.session.store import SessionState


def _log_size(path: str) -> int:
    """Size in bytes of the session log; a missing log is empty.

    Args:
        path: The session log path.

    Returns:
        The log size in bytes, or ``0`` when the log does not exist.

    Raises:
        GymratError: The log exists but cannot be inspected.
    """
    try:
        return Path(path).stat().st_size
    except FileNotFoundError:
        return 0
    except OSError as error:
        message = f"Cannot inspect session log {path}: {error.strerror or error}"
        raise GymratError(message) from error


class EndConditionScan:
    """Track how far the session log has been scanned and the end it found.

    Hook records are scanned only past the cursor, so a hook failure already in
    the log when the run started never ends it. The first condition found stays
    in ``pending`` until the supervisor fires it; later scans leave it alone.
    """

    def __init__(self, config: BenchlessConfig, log_path: str) -> None:
        self._config = config
        self._path = log_path
        self._size = 0
        self._cursor: int | None = None
        self._check_stop = True
        self.pending: EndCondition | None = None

    def seed(self) -> list[SessionLogRecord] | None:
        """Read the log at launch and start the scan from its current end.

        A stop condition already met at launch disarms stop-condition detection:
        the run was forced past it. When the read fails the cursor stays unset,
        and the first clean scan sets it and arms stop-condition detection from
        the state it folds, without reporting hook failures.

        Returns:
            The launch records, or ``None`` when the log cannot be inspected, read, or folded.
        """
        try:
            size = _log_size(self._path)
            records = read_records(self._path)
            state = fold_session(records)
        except GymratError:
            return None
        self._size = size
        self._cursor = len(records)
        self._arm_stop_check(state)
        return records

    def scan_if_grown(self) -> None:
        """Read the log and detect an end condition when it grew since the last scan.

        Does nothing while an end is pending or when the log size is unchanged,
        so event delivery never re-parses an untouched log.

        Raises:
            GymratError: The grown log cannot be inspected, read, or folded.
        """
        if self.pending is not None:
            return
        size = _log_size(self._path)
        if size == self._size:
            return
        records = read_records(self._path)
        state = fold_session(records)
        self._size = size
        self.detect(records, state)

    def detect(self, records: list[SessionLogRecord], state: SessionState) -> None:
        """Record the end condition in ``records`` as pending, unless one already is.

        The first scan after a failed launch read decides whether stop-condition
        detection is armed, exactly as a clean launch read would: a stop condition
        already met in that state means the run was forced past it.

        Args:
            records: The raw session log records, command records included.
            state: The session state already folded from ``records``.
        """
        if self.pending is not None:
            return
        if self._cursor is None:
            self._arm_stop_check(state)
        self.pending, self._cursor = detect_end_condition(
            self._config,
            records,
            state,
            cursor=self._cursor,
            check_stop=self._check_stop,
        )

    def _arm_stop_check(self, state: SessionState) -> None:
        """Arm stop-condition detection unless ``state`` already satisfies one."""
        self._check_stop = stop_condition(self._config, state) is None
