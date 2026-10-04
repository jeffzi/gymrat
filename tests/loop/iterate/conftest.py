"""Shared fixtures for the iterate run tests."""

import pytest

from tests.loop.iterate._fixtures import CollectSamplesRecorder, install_collect_samples


@pytest.fixture
def samples_mock(monkeypatch: pytest.MonkeyPatch) -> CollectSamplesRecorder:
    """A recorder installed in place of ``collect_samples``, not yet wired."""
    return install_collect_samples(monkeypatch)
