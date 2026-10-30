"""Shared fixtures for the supervise dashboard tests."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from tests.cli.supervise._fixtures import LIVE_CLASS_PATH


@pytest.fixture
def mock_live_cls() -> Iterator[MagicMock]:
    """The ``ErasableLive`` class the live-mode reporter builds, patched with an autospec.

    Tests that render frames straight from ``reporter.frame()`` request it so no
    real dashboard mounts: a live ``ErasableLive`` would start a refresh thread
    painting on the process's stderr while the test moves the clock.
    """
    with patch(LIVE_CLASS_PATH, autospec=True) as live_cls:
        # An autospec'd property is a truthy mock; a display still on screen is not erased.
        live_cls.return_value.erased = False
        yield live_cls
