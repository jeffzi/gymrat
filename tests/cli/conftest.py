"""Shared fixtures for the CLI tests: loop-command repositories and renderer teardown."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from gymrat.loop.start import start_session
from tests._config import resolved_config
from tests._rich import stop_tracked
from tests.cli._session import write_bench_config
from tests.loop._settle import (
    keep_iteration,
    start_with,
)


@pytest.fixture(autouse=True)
def stop_tracked_renderers() -> Iterator[None]:
    """Stop every renderer the test built, so a failing test leaks no live display."""
    yield
    stop_tracked()


@pytest.fixture
def _in_non_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run from a directory that is not a git repo, so the command benches lock-free."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def stop_repo(repo: str) -> str:
    """A repository with a settled, configured session ready for the stop command."""
    start_with(repo)
    keep_iteration(repo, 1)
    write_bench_config(repo)
    return repo


@pytest.fixture
def sync_repo(repo: str) -> str:
    """A repository with an open session, ready for sync tests."""
    start_session(repo, "main", resolved_config())
    return repo
