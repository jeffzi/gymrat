"""Shared fixtures for the session sidecar tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests._lock import remove_lock_files


@pytest.fixture
def root(tmp_path: Path) -> Iterator[str]:
    """A fake repo root with the .gymrat session directory, its lock files removed at teardown."""
    session = tmp_path / ".gymrat"
    session.mkdir()
    yield str(tmp_path)
    remove_lock_files(str(tmp_path))
