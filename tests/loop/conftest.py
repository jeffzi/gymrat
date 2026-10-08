"""Shared fixtures for the loop command tests."""

import pytest

from tests.loop._settle import start_with


@pytest.fixture
def session_repo(repo: str) -> str:
    """The scratch repository with an open session on ``main``."""
    start_with(repo)
    return repo
