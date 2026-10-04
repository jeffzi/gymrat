"""Shared fixtures for the session sidecar tests."""

from pathlib import Path

import pytest


@pytest.fixture
def root(tmp_path: Path) -> str:
    """A fake repo root with the .gymrat session directory pre-created."""
    session = tmp_path / ".gymrat"
    session.mkdir()
    return str(tmp_path)
