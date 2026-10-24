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

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat.config import HooksConfig, KindEntry, MetricEntry
from gymrat.errors import GymratError
from gymrat.loop.probe import PROBE_DEFAULT_SAMPLES, ProbeOptions, probe_session
from gymrat.progress_events import PassFinished, PassStarted
from gymrat.session.paths import experiment_worktree_dir, progress_path
from tests._exec_fixtures import expected_result, install_exec
from tests.adapters._inputs import malformed_line_warning
from tests.loop._probe import BASELINE_SAMPLES
from tests.loop._settle import checks_config, start_with
from tests.loop.iterate._fixtures import (
    FILTER,
)
from tests.session.records._fixtures import baseline_record, log_records

if TYPE_CHECKING:
    from gymrat.progress_events import ProgressEvent
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


def _install_bench(
    monkeypatch: pytest.MonkeyPatch, stdout: str = "METRIC total_ms=90"
) -> ExecRecorder:
    """Answer every command the sampler spawns with a zero exit printing ``stdout``."""
    return install_exec(monkeypatch, "gymrat.sampling.exec", expected_result(stdout))


# ---------------------------------------------------------------------------
# what gets benched
# ---------------------------------------------------------------------------


async def test_probe_session_when_names_given_does_bench_the_filter_scoped_command(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    config = checks_config(filter=FILTER)
    names = ("total_ms", "alloc_bytes")
    recorder = _install_bench(monkeypatch)

    result = await probe_session(repo, config, ProbeOptions(names=names))

    assert {command for command, _ in recorder.calls} == {
        "npm run bench -- --filter total_ms alloc_bytes"
    }
    assert result.names == names


async def test_probe_session_when_names_given_without_a_filter_does_refuse_before_benching(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    recorder = _install_bench(monkeypatch)

    with pytest.raises(GymratError) as excinfo:
        await probe_session(repo, checks_config(filter=None), ProbeOptions(names=("total_ms",)))

    assert excinfo.value.reason == "no-filter"
    assert str(excinfo.value) == (
        "filter is not configured — set filter in gymrat.toml to scope a probe"
    )
    assert recorder.calls == []


async def test_probe_session_when_no_baseline_recorded_does_refuse_before_benching(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo)
    recorder = _install_bench(monkeypatch)

    with pytest.raises(GymratError) as excinfo:
        await probe_session(repo, checks_config(), ProbeOptions())

    assert excinfo.value.reason == "no-baseline"
    assert excinfo.value.hint is not None
    assert "gymrat measure --record" in excinfo.value.hint
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# how the run is configured
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("requested", "expected"), SAMPLE_COUNTS)
async def test_probe_session_when_sampling_does_take_the_count_from_options_never_from_config(
    repo: str, monkeypatch: pytest.MonkeyPatch, requested: int | None, expected: int
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    recorder = _install_bench(monkeypatch)

    result = await probe_session(repo, checks_config(samples=10), ProbeOptions(samples=requested))

    assert len(recorder.calls) == expected
    assert result.samples == expected


async def test_probe_session_when_no_names_does_bench_the_whole_bench_in_the_experiment_worktree_with_options_from_config(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    config = checks_config(
        bench="npm run bench",
        filter=FILTER,
        prepare="npm ci",
        timeout_seconds=900,
        metrics={"total_ms": MetricEntry(direction="higher")},
        kinds={"memory": KindEntry(gating=False)},
    )
    recorder = _install_bench(monkeypatch, "METRIC total_ms=90\nMETRIC alloc#heap=50")

    result = await probe_session(repo, config, ProbeOptions())

    assert (result.label, result.adapter, result.names) == ("experiment", "metric-lines", ())
    assert [command for command, _ in recorder.calls] == [
        "npm ci",
        *["npm run bench"] * PROBE_DEFAULT_SAMPLES,
    ]
    assert {(options.cwd, options.timeout_ms) for _, options in recorder.calls} == {
        (experiment_worktree_dir(repo), 900_000)
    }
    assert [
        (metric.name, metric.meta.direction, metric.meta.gating) for metric in result.metrics
    ] == [
        ("total_ms", "higher", True),
        ("alloc#heap", "lower", False),
    ]


async def test_probe_session_when_callbacks_given_does_forward_them_to_the_run(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    _install_bench(monkeypatch, "METRIC banana=ripe\nMETRIC total_ms=90")
    events: list[ProgressEvent] = []
    warnings: list[str] = []

    await probe_session(
        repo,
        checks_config(),
        ProbeOptions(samples=1, on_progress=events.append, warn=warnings.append),
    )

    assert [replace(event, at_ms=0) for event in events] == [
        PassStarted(round=1, total_rounds=1, target_count=1, label="experiment", at_ms=0),
        PassFinished(round=1, total_rounds=1, target_count=1, label="experiment", at_ms=0),
    ]
    assert warnings == [malformed_line_warning("METRIC banana=ripe")]


# ---------------------------------------------------------------------------
# pairing measured medians with the baseline
# ---------------------------------------------------------------------------


async def test_probe_session_when_run_reports_metrics_does_pair_each_with_its_baseline_median(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    _install_bench(monkeypatch, "METRIC total_ms=90\nMETRIC alloc#heap=50")

    result = await probe_session(repo, checks_config(), ProbeOptions())

    assert [metric.name for metric in result.metrics] == ["total_ms", "alloc#heap"]
    assert (result.metrics[0].median, result.metrics[0].spread) == (90.0, 0.0)
    assert result.metrics[0].reference_median == 100.0
    assert result.metrics[0].delta_pct == pytest.approx(-10.0)
    assert (result.metrics[1].median, result.metrics[1].meta.kind) == (50.0, "memory")


@pytest.mark.parametrize(
    ("samples", "expected_reference"),
    [
        pytest.param(({"total_ms": 0.0},), 0.0, id="baseline-median-of-zero"),
        pytest.param(({"other_ms": 1.0},), None, id="baseline-lacks-the-metric"),
    ],
)
async def test_probe_session_when_reference_missing_or_zero_does_report_no_delta(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    samples: tuple[dict[str, float], ...],
    expected_reference: float | None,
):
    start_with(repo, (baseline_record(samples=samples),))
    _install_bench(monkeypatch)

    result = await probe_session(repo, checks_config(), ProbeOptions())

    assert result.metrics[0].reference_median == expected_reference
    assert result.metrics[0].delta_pct is None


# ---------------------------------------------------------------------------
# a probe leaves no trace
# ---------------------------------------------------------------------------


async def test_probe_session_when_run_completes_does_not_touch_the_session_log_or_sidecar(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    config = checks_config(
        hooks=HooksConfig(before="npm run warm-cache", after="npm run cool-down")
    )
    before = log_records(repo)
    _install_bench(monkeypatch)

    await probe_session(repo, config, ProbeOptions())

    assert log_records(repo) == before
    assert not Path(progress_path(repo)).exists()  # noqa: ASYNC240 -- sync check in async test
