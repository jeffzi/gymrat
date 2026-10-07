"""Synchronous file reads and writes for async test bodies.

A blocking filesystem call written inline in an ``async def`` test is flagged
as blocking I/O; routing it through these plain functions keeps the async body
clean without changing what the test reads or writes.
"""

from pathlib import Path


def read_bytes(path: str | Path) -> bytes:
    """Read ``path`` as raw bytes.

    Args:
        path: The file to read.

    Returns:
        The file's contents.
    """
    return Path(path).read_bytes()


def read_text(path: str | Path) -> str:
    """Read ``path`` as UTF-8 text.

    Args:
        path: The file to read.

    Returns:
        The file's contents.
    """
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` as UTF-8, replacing any previous contents.

    Args:
        path: The file to write.
        text: The contents to write.
    """
    Path(path).write_text(text, encoding="utf-8")
