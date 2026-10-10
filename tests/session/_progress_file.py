"""A failure double for the progress sidecar, shared by its unit and command tests.

This is test-support code, not a test module: ``tests/session/test_progress_file.py``
and ``tests/cli/commands/loop/test_iterate.py`` import it. It carries no test
functions of its own.
"""

import os
from typing import Any
from unittest.mock import create_autospec

import pytest


def fail_unlink_of(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make removing ``path`` fail the way a win32 sharing violation does.

    Every other path is removed as usual.

    Args:
        path: The file whose removal raises ``PermissionError``.
        monkeypatch: The fixture ``os.unlink`` is replaced through.
    """
    original_unlink = os.unlink

    def failing_unlink(target: str | os.PathLike[str], *args: Any, **kwargs: Any) -> None:
        if str(target) == path:
            raise PermissionError(13, "The process cannot access the file", path)
        original_unlink(target, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", create_autospec(os.unlink, side_effect=failing_unlink))
