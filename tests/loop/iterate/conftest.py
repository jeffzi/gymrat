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
def hooks_setup(repo: str, samples_mock: CollectSamplesRecorder) -> tuple[str, str, HookScripts]:
    """A settled session with hook scripts wired and sampling stubbed improved."""
    experiment_dir = iterate_session_header(repo).worktrees.experiment
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    scripts = HookScripts(repo, experiment_dir)
    write_session_log(
        repo, iterate_session_header(repo), (iteration_record(seq=1), committed_keep(1))
    )
    stub_samples(samples_mock, repo, improved_rounds(), baseline_rounds())
    return repo, experiment_dir, scripts
