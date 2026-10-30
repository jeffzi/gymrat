"""Shared fixtures for the CLI tests: loop-command repositories and renderer teardown."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest

from gymrat.cli import live_display
from gymrat.cli.live_display import ErasableLive
from tests._rich import stop_tracked
from tests.cli._session import (
    open_stop_ready_session,
)


@pytest.fixture(autouse=True)
def stop_tracked_renderers() -> Iterator[None]:
    """Stop every renderer the test built, so a failing test leaks no live display."""
    yield
    stop_tracked()


def _manual_refresh_live(**kwargs: Any) -> ErasableLive:
    """Build the display a live renderer mounts, with its refresh thread switched off."""
    kwargs["auto_refresh"] = False
    return ErasableLive(**kwargs)


@pytest.fixture
def manual_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mount every live display without a refresh thread, so only ``frame()`` renders it."""
    monkeypatch.setattr(
        live_display,
        "ErasableLive",
        create_autospec(ErasableLive, side_effect=_manual_refresh_live),
    )


@pytest.fixture
def _in_non_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run from a directory that is not a git repo, so the command benches lock-free."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def stop_repo(repo: str) -> str:
    """A repository with a settled, configured session ready for the stop command."""
    open_stop_ready_session(repo)
    return repo
