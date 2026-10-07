"""Shared fixtures for the ``exec`` tests: run options and spawn recording."""

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from gymrat.exec import ExecOptions
from tests._exec_fixtures import recorded_spawns


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


@pytest.fixture
async def spawned_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[list[asyncio.subprocess.Process]]:
    """Record every child ``exec`` or ``exec_argv`` spawns, reaping survivors in-loop."""
    async with recorded_spawns(monkeypatch) as processes:
        yield processes
