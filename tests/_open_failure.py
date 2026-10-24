"""A failure double for opening a file, shared by tests that need an OSError without a strerror.

It carries no test functions of its own.
"""

import builtins
import io
from typing import Any

import pytest


def fail_open_of(path: str, message: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make opening ``path`` raise an ``OSError`` that carries only ``message``.

    The error has no ``strerror``, unlike a real permission or missing-file failure. Every other
    path opens as usual.

    Args:
        path: The file whose open raises.
        message: The sole argument of the raised ``OSError``.
        monkeypatch: The fixture ``builtins.open`` and ``io.open`` are replaced through.
    """
    real_open = io.open

    def failing_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if str(file) == path:
            raise OSError(message)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)
    monkeypatch.setattr(io, "open", failing_open)
