"""Shared teardown for the supervise tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.cli.supervise._fixtures import stop_built_reporters

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _stop_reporters() -> Iterator[None]:
    """Stop every reporter the test built, so no live refresh thread outlives it."""
    yield
    stop_built_reporters()
