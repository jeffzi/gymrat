"""Shared fixtures for the supervise tests."""

from pathlib import Path

import pytest

from gymrat.clock import now_ms
from tests.supervisor._fixtures import SupervisorClock, seed_session_log


@pytest.fixture
def root(tmp_path: Path) -> str:
    """A scratch repository root whose session log holds only its header."""
    path = str(tmp_path / "repo")
    seed_session_log(path)
    return path


@pytest.fixture
def supervisor_clock(monkeypatch: pytest.MonkeyPatch) -> SupervisorClock:
    """The supervisor's wall clock, frozen at now with its deadline one minute ahead."""
    return SupervisorClock(monkeypatch, now_ms())
