"""Behavioral tests for ``probe_session``.

A probe benches the session's experiment worktree once and pairs every measured
median with the reference median from the newest recorded baseline. It is a
read-only command: nothing is appended to the session log, no hook runs, and the
progress sidecar stays untouched.

The one boundary these tests stub is the process spawn — the ``exec`` the sampler
runs the consumer's prepare and bench commands through, which no test here can
run. Everything around it (the session guard, the baseline lookup, the
measurement engine, the pairing arithmetic) runs for real against a throwaway
repository from the shared ``create_scratch_repo`` factory, so the suite is
order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.errors import GymratError
from gymrat.loop.probe import PROBE_DEFAULT_SAMPLES, ProbeOptions, probe_session
from gymrat.session.paths import experiment_worktree_dir
from tests._exec_fixtures import expected_result, install_exec
from tests.loop._probe import BASELINE_SAMPLES
from tests.loop._settle import checks_config, start_with
from tests.session.records._fixtures import baseline_record

if TYPE_CHECKING:
    from tests._exec_fixtures import ExecRecorder

#: The sample count a probe requests and the count it should end up taking:
#: unset falls back to the probe default, an explicit count always wins.
SAMPLE_COUNTS = [
    pytest.param(None, PROBE_DEFAULT_SAMPLES, id="unset-falls-back-to-the-probe-default"),
    pytest.param(3, 3, id="explicit-count-wins"),
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _install_bench(monkeypatch: pytest.MonkeyPatch) -> ExecRecorder:
    """Answer every command the sampler spawns with a zero exit printing one ``total_ms`` metric."""
    return install_exec(monkeypatch, "gymrat.sampling.exec", expected_result("METRIC total_ms=90"))


# ---------------------------------------------------------------------------
# what gets benched
# ---------------------------------------------------------------------------


async def test_probe_session_when_names_given_without_a_filter_does_refuse_before_benching(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    recorder = _install_bench(monkeypatch)

    with pytest.raises(GymratError) as excinfo:
        await probe_session(repo, checks_config(filter=None), ProbeOptions(names=("total_ms",)))

    assert excinfo.value.reason == "no-filter"
    assert recorder.calls == []


async def test_probe_session_when_no_baseline_recorded_does_refuse_before_benching(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    recorder = _install_bench(monkeypatch)

    with pytest.raises(GymratError) as excinfo:
        await probe_session(repo, checks_config(), ProbeOptions())

    assert excinfo.value.reason == "no-baseline"
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# how the run is configured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("requested", "expected"), SAMPLE_COUNTS)
async def test_probe_session_when_samples_requested_or_unset_does_take_the_count_from_options_never_from_config(
    repo: str, monkeypatch: pytest.MonkeyPatch, requested: int | None, expected: int
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    recorder = _install_bench(monkeypatch)

    result = await probe_session(repo, checks_config(samples=10), ProbeOptions(samples=requested))

    assert len(recorder.calls) == expected
    assert result.samples == expected


async def test_probe_session_when_no_names_does_bench_in_the_experiment_worktree(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    recorder = _install_bench(monkeypatch)

    await probe_session(repo, checks_config(), ProbeOptions())

    assert {options.cwd for _, options in recorder.calls} == {experiment_worktree_dir(repo)}


# ---------------------------------------------------------------------------
# pairing measured medians with the baseline
# ---------------------------------------------------------------------------


async def test_probe_session_when_run_reports_metrics_does_pair_each_with_its_baseline_median(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    _install_bench(monkeypatch)

    result = await probe_session(repo, checks_config(), ProbeOptions())

    assert result.metrics[0].reference_median == 100.0
    assert result.metrics[0].delta_pct == pytest.approx(-10.0)


async def test_probe_session_when_baseline_lacks_the_metric_does_report_no_reference_or_delta(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_record(samples=({"other_ms": 1.0},)),))
    _install_bench(monkeypatch)

    result = await probe_session(repo, checks_config(), ProbeOptions())

    assert result.metrics[0].reference_median is None
    assert result.metrics[0].delta_pct is None
