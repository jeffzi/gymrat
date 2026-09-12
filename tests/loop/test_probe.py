"""Behavioral tests for ``probe_session``.

A probe benches the session's experiment worktree once and pairs every measured
median with the reference median from the newest recorded baseline. It is a
read-only command: nothing is appended to the session log, no hook runs, and the
progress sidecar stays untouched.

The one boundary these tests stub is the measurement engine — it shells out to
the consumer's bench script, which no test here can run. Everything around it
(the session guard, the baseline lookup, the pairing arithmetic) runs for real
against a throwaway repository from the shared ``create_scratch_repo`` factory,
so the suite is order-independent and safe under ``pytest-xdist`` /
``pytest-randomly``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat.config import HooksConfig, KindEntry, MetricEntry
from gymrat.errors import GymratError
from gymrat.loop.iterate.confirm import scoped_bench
from gymrat.loop.probe import PROBE_DEFAULT_SAMPLES, ProbeOptions, probe_session
from gymrat.sampling import TargetSpec
from gymrat.session import experiment_worktree_dir, read_records, session_jsonl_path
from gymrat.session.paths import progress_path
from tests.loop._probe import (
    BASELINE_SAMPLES,
    baseline_of,
    install_measure,
    measurement,
    only_call,
)
from tests.loop.settle._fixtures import checks_config, start_with
from tests.report._inputs import measured_metric
from tests.session.records._fixtures import finalize_record

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.session import SessionLogRecord

#: The rerun template a consumer configures when their bench can be narrowed.
FILTER = "npm run bench -- --filter {names}"

#: The sample count a probe requests and the count it should end up taking:
#: unset falls back to the probe default, an explicit count always wins.
SAMPLE_COUNTS = [
    pytest.param(None, PROBE_DEFAULT_SAMPLES, id="unset-falls-back-to-the-probe-default"),
    pytest.param(3, 3, id="explicit-count-wins"),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str]) -> str:
    """A fresh scratch git repository for one probe test."""
    return create_scratch_repo()


def records_of(root: str) -> list[SessionLogRecord]:
    """Every record the session log at ``root`` currently holds."""
    return read_records(session_jsonl_path(root))


# ---------------------------------------------------------------------------
# the session guard
# ---------------------------------------------------------------------------


async def test_probe_session_when_no_session_does_refuse_pointing_at_the_command_that_opens_one(
    repo: str,
):
    with pytest.raises(GymratError) as excinfo:
        await probe_session(repo, checks_config(), ProbeOptions())

    assert excinfo.value.reason == "no-session"
    assert excinfo.value.hint is not None
    assert "gymrat start" in excinfo.value.hint


async def test_probe_session_when_session_finalized_does_refuse(repo: str):
    start_with(repo, (baseline_of(), finalize_record()))

    with pytest.raises(GymratError) as excinfo:
        await probe_session(repo, checks_config(), ProbeOptions())

    assert excinfo.value.reason == "finalized"


# ---------------------------------------------------------------------------
# what gets benched
# ---------------------------------------------------------------------------


async def test_probe_session_when_no_names_does_bench_the_whole_bench_in_the_experiment_worktree(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_of(),))
    config = checks_config(bench="npm run bench", filter=FILTER)
    recorder = install_measure(monkeypatch, measurement())

    result = await probe_session(repo, config, ProbeOptions())

    forwarded = only_call(recorder)
    assert forwarded.target == TargetSpec(label="experiment", target=experiment_worktree_dir(repo))
    assert forwarded.bench == "npm run bench"
    assert result.scoped is False
    assert result.names == ()


async def test_probe_session_when_names_given_does_bench_the_filter_scoped_command(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_of(),))
    config = checks_config(filter=FILTER)
    names = ("total_ms", "decode large payload")
    recorder = install_measure(monkeypatch, measurement())

    result = await probe_session(repo, config, ProbeOptions(names=names))

    assert only_call(recorder).bench == scoped_bench(config, names)
    assert result.scoped is True
    assert result.names == names


async def test_probe_session_when_names_given_without_a_filter_does_refuse_before_benching(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_of(),))
    recorder = install_measure(monkeypatch, measurement())

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
    recorder = install_measure(monkeypatch, measurement())

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
    start_with(repo, (baseline_of(),))
    recorder = install_measure(monkeypatch, measurement())

    result = await probe_session(
        repo, checks_config(adapter="mitata", samples=10), ProbeOptions(samples=requested)
    )

    assert only_call(recorder).samples == expected
    assert result.samples == expected
    assert result.label == "experiment"
    assert result.adapter == "mitata"


async def test_probe_session_when_run_starts_does_take_the_rest_of_the_options_from_config(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_of(),))
    config = checks_config(
        prepare="npm ci",
        adapter="mitata",
        timeout_seconds=900,
        metrics={"total_ms": MetricEntry(direction="lower", gating=True, exact=False)},
        kinds={"memory": KindEntry(gating=False)},
    )
    recorder = install_measure(monkeypatch, measurement())

    await probe_session(repo, config, ProbeOptions())

    forwarded = only_call(recorder)
    assert forwarded.prepare == "npm ci"
    assert forwarded.adapter == "mitata"
    assert forwarded.timeout_seconds == 900
    assert forwarded.config_metrics == config.metrics
    assert forwarded.config_kinds == config.kinds


async def test_probe_session_when_callbacks_given_does_forward_them_to_the_run(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_of(),))
    events: list[object] = []
    warnings: list[str] = []
    recorder = install_measure(monkeypatch, measurement())

    await probe_session(
        repo, checks_config(), ProbeOptions(on_progress=events.append, warn=warnings.append)
    )

    forwarded = only_call(recorder)
    sentinel = object()
    assert forwarded.on_progress is not None
    forwarded.on_progress(sentinel)  # type: ignore[arg-type]
    assert events[-1] is sentinel
    assert forwarded.warn is not None
    forwarded.warn("bench output looked odd")
    assert warnings[-1] == "bench output looked odd"


# ---------------------------------------------------------------------------
# pairing measured medians with the baseline
# ---------------------------------------------------------------------------


async def test_probe_session_when_run_reports_metrics_does_pair_each_with_its_baseline_median(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_of(),))
    alloc = measured_metric(
        median=50.0, spread=1.0, short_name="alloc_bytes", kind="memory", unit="bytes", gating=False
    )
    total = measured_metric(median=90.0, spread=2.0, short_name="total_ms")
    install_measure(monkeypatch, measurement({"total_ms": total, "alloc_bytes": alloc}))

    result = await probe_session(repo, checks_config(), ProbeOptions())

    assert [metric.name for metric in result.metrics] == ["total_ms", "alloc_bytes"]
    assert (result.metrics[0].median, result.metrics[0].spread) == (90.0, 2.0)
    assert result.metrics[0].reference_median == 100.0
    assert result.metrics[0].delta_pct == pytest.approx(-10.0)
    assert result.metrics[0].meta == total.meta
    assert result.metrics[1].reference_median is None
    assert result.metrics[1].delta_pct is None
    assert result.metrics[1].meta == alloc.meta


async def test_probe_session_when_metric_slower_than_the_baseline_does_report_a_positive_delta(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_of(),))
    install_measure(
        monkeypatch, measurement({"total_ms": measured_metric(median=125.0, spread=1.0)})
    )

    result = await probe_session(repo, checks_config(), ProbeOptions())

    assert result.metrics[0].delta_pct == pytest.approx(25.0)


@pytest.mark.parametrize(
    ("samples", "median", "expected_reference"),
    [
        pytest.param(({"other_ms": 7.0},), 90.0, None, id="metric-absent-from-the-baseline"),
        pytest.param(({"total_ms": 0.0},), 90.0, 0.0, id="baseline-median-of-zero"),
        pytest.param(BASELINE_SAMPLES, None, 100.0, id="run-reported-no-median"),
    ],
)
async def test_probe_session_when_no_usable_reference_does_report_no_delta(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    samples: tuple[dict[str, float], ...],
    median: float | None,
    expected_reference: float | None,
):
    start_with(repo, (baseline_of(samples),))
    spread = None if median is None else 1.0
    install_measure(
        monkeypatch, measurement({"total_ms": measured_metric(median=median, spread=spread)})
    )

    result = await probe_session(repo, checks_config(), ProbeOptions())

    assert result.metrics[0].reference_median == expected_reference
    assert result.metrics[0].delta_pct is None


async def test_probe_session_when_several_baselines_recorded_does_reference_the_newest(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_of(({"total_ms": 400.0},)), baseline_of(({"total_ms": 200.0},))))
    install_measure(monkeypatch, measurement())

    result = await probe_session(repo, checks_config(), ProbeOptions())

    assert result.metrics[0].reference_median == 200.0


# ---------------------------------------------------------------------------
# a probe leaves no trace
# ---------------------------------------------------------------------------


async def test_probe_session_when_run_completes_does_not_touch_the_session_log_or_sidecar(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    start_with(repo, (baseline_of(),))
    config = checks_config(
        hooks=HooksConfig(before="npm run warm-cache", after="npm run cool-down")
    )
    before = records_of(repo)
    install_measure(monkeypatch, measurement())

    await probe_session(repo, config, ProbeOptions())

    assert records_of(repo) == before
    assert not Path(progress_path(repo)).exists()  # noqa: ASYNC240 -- sync check in async test
