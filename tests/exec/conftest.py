"""Shared fixtures for the ``exec`` tests: run options."""

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.exec import ExecOptions


@pytest.fixture
def make_opts(tmp_path: Path) -> Callable[..., ExecOptions]:
    """Build ``ExecOptions`` rooted at the test's ``tmp_path``, with any override."""

    def _make(
        *,
        timeout_ms: int | None = None,
        abort: asyncio.Event | None = None,
        stdin: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecOptions:
        return ExecOptions(
            cwd=str(tmp_path),
            timeout_ms=timeout_ms,
            abort=abort,
            stdin=stdin,
            env=env,
        )

    return _make
