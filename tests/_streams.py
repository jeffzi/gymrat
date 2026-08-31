"""Shared stream test doubles."""

import io
from typing import override

__all__ = ["FakeStream"]


class FakeStream(io.StringIO):
    """A stdout/stderr stand-in whose TTY status the test controls."""

    def __init__(self, *, tty: bool):
        super().__init__()
        self._tty = tty

    @override
    def isatty(self) -> bool:
        return self._tty
