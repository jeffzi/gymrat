"""Shared fixtures for the iterate run tests."""

import pytest

from tests.loop.iterate._fixtures import (
    CollectSamplesRecorder,
    install_collect_samples,
    settled_history,
    stub_improved_samples,
    write_iterate_session,
)
from tests.loop.iterate._hooks import HookScripts


@pytest.fixture
def samples_mock(monkeypatch: pytest.MonkeyPatch) -> CollectSamplesRecorder:
    """A recorder installed in place of ``collect_samples``, not yet wired."""
    return install_collect_samples(monkeypatch)


@pytest.fixture
def open_repo(repo: str) -> str:
    """A fresh open session on disk, no history, sampling left for the test to stub."""
    write_iterate_session(repo)
    return repo


@pytest.fixture
def settled(repo: str, samples_mock: CollectSamplesRecorder) -> str:
    """A settled session on disk — one kept iteration — with sampling stubbed improved."""
    write_iterate_session(repo, settled_history())
    stub_improved_samples(samples_mock, repo)
    return repo


@pytest.fixture
def hooks_setup(settled: str) -> tuple[str, str, HookScripts]:
    """A settled session, its experiment worktree, and a hook-script builder scoped to it."""
    hooks = HookScripts.for_root(settled)
    return settled, hooks.experiment_dir, hooks
