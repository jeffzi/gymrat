"""Shared fixtures for the supervise tests."""

from pathlib import Path

import pytest

from gymrat.utils import now_ms
from tests.session.records._fixtures import append_records, session_record
from tests.supervisor._fixtures import SupervisorClock


@pytest.fixture
def root(tmp_path: Path) -> str:
    """A scratch repository root whose session log holds only its header."""
    path = str(tmp_path / "repo")
    # A minimal session header, so ``read_records`` / ``fold_session`` work.
    append_records(path, session_record())
    return path


@pytest.fixture
def supervisor_clock(monkeypatch: pytest.MonkeyPatch) -> SupervisorClock:
    """The supervisor's wall clock, frozen at now with its deadline one minute ahead."""
    return SupervisorClock(monkeypatch, now_ms())
