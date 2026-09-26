"""Write a text file atomically so a concurrent reader never sees a partial file."""

import contextlib
import os
import tempfile
from pathlib import Path


def write_text_atomic(path: Path, text: str) -> None:
    """Replace *path* with *text*, encoded as UTF-8, in one atomic step.

    The text goes to a temporary sibling that is flushed and fsynced before it
    is renamed over *path*, so a reader sees either the previous content or the
    full new content. On failure *path* is left untouched and the temporary
    file is removed.

    Args:
        path: The file to write. Its directory must already exist.
        text: The content to write.

    Raises:
        OSError: When the temporary file cannot be created, written, synced,
            or renamed over *path*.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(text.encode("utf-8"))
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        tmp_path.replace(path)
    except BaseException:
        # The original error is what the caller needs; a failed cleanup must not mask it.
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
