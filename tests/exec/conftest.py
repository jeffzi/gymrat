"""Shared fixtures for the ``exec`` tests: run options and a ``killpg`` recorder."""

import functools
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.exec import ExecOptions

# exec drives POSIX process groups (killpg) and sh-only shell syntax; neither
# works under cmd.exe, so nothing in this directory runs on win32.
collect_ignore_glob: list[str] = ["test_*.py"] if sys.platform == "win32" else []


@pytest.fixture
def make_opts(tmp_path: Path) -> Callable[..., ExecOptions]:
    """Build ``ExecOptions`` rooted at the test's ``tmp_path``, with any override."""
    return functools.partial(ExecOptions, cwd=str(tmp_path))


@pytest.fixture
def record_killpg(monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[int]]:
    """Return a switch that, once flipped, records each ``os.killpg`` target instead of signaling it."""

    def start() -> list[int]:
        targets: list[int] = []

        def record(group_pid: int, _signal_number: int) -> None:
            targets.append(group_pid)

        monkeypatch.setattr(os, "killpg", record)
        return targets

    return start
