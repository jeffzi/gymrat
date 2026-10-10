"""Shared fixtures for the CLI tests: loop-command repositories and renderer teardown."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests._rich import stop_tracked
from tests.cli._session import (
    open_stop_ready_session,
    write_settled_session,
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
    open_stop_ready_session(repo)
    return repo


@pytest.fixture
def status_repo(repo: str) -> str:
    """A repository with a configured session and one kept iteration."""
    write_settled_session(repo)
    return repo
