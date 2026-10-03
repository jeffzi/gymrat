"""Shared stream test doubles."""

import io
from typing import override


class FakeStream(io.StringIO):
    """A stdout/stderr stand-in whose TTY status the test controls."""

    def __init__(self, *, tty: bool):
        super().__init__()
        self._tty = tty

    @override
    def isatty(self) -> bool:
        return self._tty


class RaisingStream(io.StringIO):
    """A stream whose every write raises ``error``, as a closed pipe would.

    Args:
        error: The exception each ``write`` raises.
        tty: What ``isatty`` reports.
    """

    def __init__(self, error: OSError, *, tty: bool = False) -> None:
        super().__init__()
        self._error = error
        self._tty = tty

    @override
    def isatty(self) -> bool:
        return self._tty

    @override
    def write(self, s: str, /) -> int:
        raise self._error


class RecordingStream(io.StringIO):
    """A stderr stand-in that keeps each ``write`` call separately."""

    def __init__(self) -> None:
        super().__init__()
        self.writes: list[str] = []

    @override
    def write(self, s: str, /) -> int:
        self.writes.append(s)
        return super().write(s)
