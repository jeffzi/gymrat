"""Shared fixtures for the supervise tests."""

from pathlib import Path

import pytest

from tests.supervisor._fixtures import seed_session_log


@pytest.fixture
def root(tmp_path: Path) -> str:
    """A scratch repository root whose session log holds only its header."""
    path = str(tmp_path / "repo")
    seed_session_log(path)
    return path
