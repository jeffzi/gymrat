"""Shared fixtures for the iterate run tests."""

from pathlib import Path

import pytest

from tests.loop.iterate._fixtures import (
    CollectSamplesRecorder,
    baseline_rounds,
    improved_rounds,
    install_collect_samples,
    iterate_session_header,
    stub_samples,
)
from tests.loop.iterate._hooks import HookScripts
from tests.session.records._fixtures import committed_keep, iteration_record, write_session_log


@pytest.fixture
def samples_mock(monkeypatch: pytest.MonkeyPatch) -> CollectSamplesRecorder:
    """A recorder installed in place of ``collect_samples``, not yet wired."""
    return install_collect_samples(monkeypatch)


@pytest.fixture
def open_repo(repo: str) -> str:
    """A fresh open session on disk, no history, sampling left for the test to stub."""
    write_session_log(repo, iterate_session_header(repo))
    return repo


@pytest.fixture
def settled(repo: str, samples_mock: CollectSamplesRecorder) -> str:
    """A settled session on disk — one kept iteration — with sampling stubbed improved."""
    write_session_log(
        repo, iterate_session_header(repo), (iteration_record(seq=1), committed_keep(1))
    )
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    return repo


@pytest.fixture
def hooks_setup(settled: str) -> tuple[str, str, HookScripts]:
    """A settled session, its experiment worktree, and a hook-script builder scoped to it."""
    experiment_dir = iterate_session_header(settled).worktrees.experiment
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    return settled, experiment_dir, HookScripts(settled, experiment_dir)
