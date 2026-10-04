"""Shared pty helper for the hardening test modules.

The hardening tests that drive the CLI under a pseudo-terminal read its output
through ``drain``, so each test module does not carry its own copy of the read
loop.
"""

import os


def drain(fd: int, chunks: list[bytes]) -> None:
    """Read a pty master until the child closes the slave, collecting bytes."""
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            return
        if not chunk:
            return
        chunks.append(chunk)
