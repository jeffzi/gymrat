"""Shared fixtures for the supervise dashboard tests."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from tests.cli.supervise._fixtures import LIVE_CLASS_PATH


@pytest.fixture
def mock_live_cls() -> Iterator[MagicMock]:
    """The ``ErasableLive`` class the live-mode reporter builds, patched with an autospec."""
    with patch(LIVE_CLASS_PATH, autospec=True) as live_cls:
        yield live_cls
