"""The event-log writer: a :data:`SessionObserver` that appends events as JSON.

:func:`create_event_log_writer` returns an observer that appends each event as a
single JSON line (the shared :func:`~gymrat.supervisor.events.to_json_line`
wire form) to a log file. The parent directory is created lazily on the first
write, and a write failure surfaces as a :class:`~gymrat.errors.GymratError`
naming the log path so the user knows which file failed.
"""

from pathlib import Path

from gymrat.errors import GymratError
from gymrat.supervisor.events import SessionEvent, SessionObserver, to_json_line


def probe_event_log_path(log_path: str | Path) -> None:
    """Verify ``log_path`` is writable before a session starts.

    Attempts to create the parent directory and open the file for appending.
    Raises :class:`GymratError` naming the path when the filesystem rejects the
    operation, so the command can fail up front rather than after the session.
    """
    path = Path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8"):
            pass
    except OSError as error:
        message = f"Event log path is not writable: {path}"
        raise GymratError(message) from error


def create_event_log_writer(log_path: str | Path) -> SessionObserver:
    """Return a :data:`SessionObserver` that appends each event to ``log_path``.

    The parent directory is created (recursively) on the first write if it does
    not already exist. A write failure surfaces as a :class:`GymratError` naming
    the log path, chaining the underlying OS error as its cause.
    """
    path = Path(log_path)

    def write(event: SessionEvent) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as log:
                log.write(to_json_line(event) + "\n")
        except OSError as error:
            message = f"Failed to write event log: {path}"
            raise GymratError(message) from error

    return write
