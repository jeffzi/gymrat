"""Shared pseudo-terminal capture for the hardening tests that drive the CLI on a pty.

A test hands the capture's slave to the child as its terminal, while a
background thread collects everything the child draws. The module is
POSIX-only: importing it skips the importing test module on a host without
``pty``.
"""

import contextlib
import dataclasses
import os
import struct
import threading
from collections.abc import Generator

import pytest

fcntl = pytest.importorskip("fcntl", reason="POSIX-only pty")
pty = pytest.importorskip("pty", reason="POSIX-only pty")
termios = pytest.importorskip("termios", reason="POSIX-only pty")

# Seconds the reader thread is given to see end-of-file once the capture closes.
_READER_JOIN_S = 10


@dataclasses.dataclass(slots=True)
class PtyCapture:
    """A pseudo-terminal whose output a background reader collects.

    Attributes:
        slave: The terminal end to hand the child as its stdin, stdout or stderr.
        chunks: The bytes read from the terminal so far, appended while the
            child runs.
        output: Everything the child drew, decoded; set once the capture closes.
    """

    slave: int
    chunks: list[bytes] = dataclasses.field(default_factory=list)
    output: str = ""


def _drain(fd: int, chunks: list[bytes]) -> None:
    """Read a pty master until the child closes the slave, collecting bytes."""
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            return
        if not chunk:
            return
        chunks.append(chunk)


@contextlib.contextmanager
def pty_capture(size: tuple[int, int] | None = None) -> Generator[PtyCapture]:
    """Open a pseudo-terminal and collect everything drawn on it until the block exits.

    On exit the capture closes its own copy of the slave, waits for the reader
    to reach end-of-file, closes the master, and decodes what was read into
    ``output``. A child still holding the slave keeps the reader waiting up to
    the join bound, so the block should end only once the child has exited.

    Args:
        size: The ``(rows, columns)`` the terminal reports, so a renderer in the
            child draws at a known geometry; ``None`` leaves the pty's default.

    Yields:
        The capture, whose ``slave`` the caller hands to the child.
    """
    master, slave = pty.openpty()
    if size is not None:
        rows, columns = size
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("4H", rows, columns, 0, 0))
    capture = PtyCapture(slave)
    reader = threading.Thread(target=_drain, args=(master, capture.chunks))
    reader.start()
    try:
        yield capture
    finally:
        os.close(slave)
        reader.join(timeout=_READER_JOIN_S)
        os.close(master)
        capture.output = b"".join(capture.chunks).decode("utf-8", "replace")
