"""How ``iterate_session`` judges an edit, and ``derive_outcome`` beneath it.

Covers gating and the confirmation rerun, metrics whose delta is undefined, the
primary and what it reads when there is no change, and the outcome and closing
report the verdicts add up to.

The one boundary these tests mock is sampling; everything downstream — verdicts,
the confirmation rerun, aggregation, the record, the report — runs against a
throwaway repository. The suite is order-independent and safe under
``pytest-xdist`` / ``pytest-randomly``.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat.config import MetricEntry, ResolvedConfig, StopConfig
from gymrat.errors import GymratError
from gymrat.loop.iterate.run import IterateOptions, derive_outcome, iterate_session
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmSkipped,
    ConfirmStarted,
    JudgeFinished,
    PassFinished,
    PassStarted,
    ProgressEvent,
)
from gymrat.report.loop import GeomeanPrimary, MetricPrimary
from gymrat.session.records import (
    Confirm,
    IterationPrimary,
    PairedSamples,
)
from tests._config import resolved_config
from tests.loop.iterate._fixtures import (
    BASELINE_BYTES,
    BASELINE_MS,
    FILTER,
    PairedRun,
    assert_permutation,
    baseline_rounds,
    improved_rounds,
    last_iteration_of,
    regressed_rounds,
    regressed_run,
    report_a_pass_per_call,
    rounds,
    scaled,
    stub_runs,
    stub_samples,
    trimmed_report_lines,
)
from tests.report._comparisons import permutation_metric
from tests.session.records._fixtures import (
    log_records,
)

if TYPE_CHECKING:
    from gymrat.model import Direction
    from gymrat.report.loop import LoopPrimary
    from gymrat.report.types import MetricComparison, MetricComparisons
    from tests.loop.iterate._fixtures import CollectSamplesRecorder

#: The events that tell the display how the judge row and the confirm row end.
_JUDGE_AND_CONFIRM_EVENTS = (JudgeFinished, ConfirmStarted, ConfirmFinished, ConfirmSkipped)

#: The smallest positive float; dividing any ordinary median by it overflows to infinity.
_SMALLEST_POSITIVE_FLOAT = 5e-324


def _jittered(values: list[float], up: float, down: float) -> list[float]:
    """Nudge alternate rounds up by ``up`` and the rest down by ``down``.

    The mixed signs leave the permutation test nothing to call, while the larger
    upward nudge still drags the median above the baseline's — a run that moved
    the wrong way without saying anything, which is what ``no-signal`` means.

    Args:
        values: The rounds to nudge.
        up: How far the even-indexed rounds move up.
        down: How far the odd-indexed rounds move down.

    Returns:
        The nudged rounds, in their original order.
    """
    return [value + up if index % 2 == 0 else value - down for index, value in enumerate(values)]


def _noisy_rounds() -> list[dict[str, float]]:
    """Ten rounds that drift half a percent the wrong way without ever settling."""
    return rounds(_jittered(BASELINE_MS, 2, 1), _jittered(BASELINE_BYTES, 20, 10))


def _filtered_rounds(name: str, values: list[float]) -> list[dict[str, float]]:
    """Rounds reporting ``name`` alone, the shape a bench filtered to that metric reports."""
    return [{name: value} for value in values]


def _total_ms_gating_config() -> ResolvedConfig:
    """A config reran through the filter with ``alloc_bytes`` excused from gating."""
    return resolved_config(filter=FILTER, metrics={"alloc_bytes": MetricEntry(gating=False)})


def _primary_line(report: str) -> str:
    """The report's ``primary:`` line, failing when there is none."""
    return next(line for line in trimmed_report_lines(report) if line.startswith("primary:"))


# ---------------------------------------------------------------------------
# a gating metric comes back regressed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filter_template", "rerun"),
    [
        pytest.param(
            FILTER,
            ("npm run bench -- --filter total_ms alloc_bytes", ("total_ms", "alloc_bytes")),
            id="through-the-filter",
        ),
        pytest.param(None, ("npm run bench", None), id="whole-bench-without-a-filter"),
    ],
)
async def test_iterate_session_when_gating_regression_does_confirm_it_on_a_rerun(
    open_repo: str,
    samples_mock: CollectSamplesRecorder,
    monkeypatch: pytest.MonkeyPatch,
    filter_template: str | None,
    rerun: tuple[str, tuple[str, ...] | None],
):
    rerun_bench, filtered_metrics = rerun
    stub_runs(samples_mock, open_repo, [regressed_run(), regressed_run()])
    report_a_pass_per_call(monkeypatch, samples_mock)
    events: list[ProgressEvent] = []

    result = await iterate_session(
        open_repo,
        resolved_config(filter=filter_template),
        options=IterateOptions(on_progress=events.append),
    )

    assert samples_mock.call_count == 2
    assert samples_mock.calls[1].targets == samples_mock.calls[0].targets
    assert samples_mock.calls[1].options.bench == rerun_bench
    assert [set(e.regressed) for e in events if isinstance(e, JudgeFinished)] == [
        {"total_ms", "alloc_bytes"}
    ]
    assert [e.filtered_metrics for e in events if isinstance(e, ConfirmStarted)] == [
        filtered_metrics
    ]
    assert [e.reproduced for e in events if isinstance(e, ConfirmFinished)] == [True]
    assert [
        type(e)
        for e in events
        if isinstance(e, (PassStarted, PassFinished)) and e.phase == "confirm"
    ] == [PassStarted, PassFinished]
    assert result.record.confirm == Confirm(
        ran=True,
        filtered=("total_ms", "alloc_bytes"),
        samples=PairedSamples(
            experiment=tuple(regressed_rounds()), baseline=tuple(baseline_rounds())
        ),
    )
    assert_permutation(
        result.record.metrics["total_ms"], delta=10, verdict="regressed", confirmed=True
    )
    assert result.record.outcome == "regressed"
    assert "total_ms: regression confirmed on rerun" in trimmed_report_lines(result.report)


@pytest.mark.parametrize(
    "experiment",
    [
        pytest.param(scaled(BASELINE_MS, 0.9), id="improved"),
        pytest.param(_jittered(BASELINE_MS, 2, 1), id="no-signal"),
    ],
)
async def test_iterate_session_when_rerun_disagrees_does_demote_to_no_signal(
    open_repo: str, samples_mock: CollectSamplesRecorder, experiment: list[float]
):
    stub_runs(
        samples_mock,
        open_repo,
        [
            regressed_run(),
            PairedRun(
                _filtered_rounds("total_ms", experiment),
                _filtered_rounds("total_ms", BASELINE_MS),
            ),
        ],
    )
    resolved = _total_ms_gating_config()

    result = await iterate_session(open_repo, resolved)

    assert_permutation(
        result.record.metrics["total_ms"], delta=10, verdict="no-signal", confirmed=False
    )
    assert result.record.primary.delta_pct == pytest.approx(10, abs=1e-6)
    assert result.record.outcome == "no-signal"
    assert "total_ms: regression not confirmed on rerun" in trimmed_report_lines(result.report)


async def test_iterate_session_when_rerun_bench_fails_does_raise_without_recording(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    stub_runs(samples_mock, open_repo, [regressed_run(), GymratError("bench command failed")])
    resolved = _total_ms_gating_config()

    with pytest.raises(GymratError) as exc:
        await iterate_session(open_repo, resolved)

    assert str(exc.value) == "bench command failed"
    assert len(log_records(open_repo)) == 1


# The filter command reaches a POSIX shell, which is what decides where one
# argument ends and the next begins; win32 is skipped for the same reason the
# exec suite is.
_ARGS_SCRIPT = '#!/bin/sh\nfor arg in "$@"; do\n  echo "$arg"\ndone\n'
_ARGS_FILTER = "sh args.sh {names}"


def _paired_with(name: str, values: list[float]) -> list[dict[str, float]]:
    """One round per entry, reporting ``name`` beside a plainly named metric."""
    return [{name: value, "total_ms": value} for value in values]


def _shell_args(directory: str, command: str) -> list[str]:
    """The arguments a POSIX shell hands the stand-in bench when it runs ``command``.

    The rerun's bench string is handed to a shell verbatim, so running it through
    one is the only assertion that speaks to what the bench is really given — a
    string comparison would pass for a command the shell refuses outright.

    Args:
        directory: The directory the shell runs in.
        command: The bench command string to hand the shell.

    Returns:
        Each non-empty line the stand-in bench printed, one per argument.
    """
    printed = subprocess.run(  # noqa: S603 -- argv is a fixed shell plus a test-built command
        ["sh", "-c", command],  # noqa: S607 -- the POSIX shell is resolved from PATH on purpose
        cwd=directory,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in printed.split("\n") if line]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell argument splitting only")
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("sort(n=1000)/time", id="parentheses-and-equals"),
        pytest.param("decode large payload", id="space"),
        pytest.param("it's/time", id="single-quote"),
    ],
)
async def test_iterate_session_when_metric_name_needs_quoting_does_pass_it_as_one_argument(
    open_repo: str, samples_mock: CollectSamplesRecorder, name: str
):
    (Path(open_repo) / "args.sh").write_text(_ARGS_SCRIPT, encoding="utf-8")
    regressed = PairedRun(
        experiment=_paired_with(name, scaled(BASELINE_MS, 1.1)),
        baseline=_paired_with(name, BASELINE_MS),
    )
    stub_runs(samples_mock, open_repo, [regressed, regressed])

    await iterate_session(open_repo, resolved_config(filter=_ARGS_FILTER))

    assert _shell_args(open_repo, samples_mock.calls[1].options.bench) == [name, "total_ms"]


# ---------------------------------------------------------------------------
# the rerun never measures some or all of the regressed metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rerun", "absent", "rerun_lines"),
    [
        pytest.param(
            PairedRun(
                experiment=_filtered_rounds("total_ms", scaled(BASELINE_MS, 1.2)),
                baseline=_filtered_rounds("total_ms", BASELINE_MS),
            ),
            ("alloc_bytes",),
            ["total_ms: regression confirmed on rerun", "alloc_bytes: not measured on rerun"],
            id="silent-on-one-metric",
        ),
        pytest.param(
            PairedRun([{} for _ in range(10)], [{} for _ in range(10)]),
            ("total_ms", "alloc_bytes"),
            ["total_ms: not measured on rerun", "alloc_bytes: not measured on rerun"],
            id="no-parsable-metrics",
        ),
    ],
)
async def test_iterate_session_when_rerun_skips_a_metric_does_keep_it_regressed_as_absent(
    open_repo: str,
    samples_mock: CollectSamplesRecorder,
    rerun: PairedRun,
    absent: tuple[str, ...],
    rerun_lines: list[str],
):
    stub_runs(samples_mock, open_repo, [regressed_run(), rerun])
    expected_confirm = Confirm(
        ran=True,
        filtered=("total_ms", "alloc_bytes"),
        samples=PairedSamples(experiment=tuple(rerun.experiment), baseline=tuple(rerun.baseline)),
        absent=absent,
    )

    result = await iterate_session(open_repo, resolved_config(filter=FILTER))

    assert result.record.confirm == expected_confirm
    assert last_iteration_of(open_repo).confirm == expected_confirm
    assert result.record.samples == PairedSamples(
        experiment=tuple(regressed_rounds()), baseline=tuple(baseline_rounds())
    )
    assert {name: metric.verdict for name, metric in result.record.metrics.items()} == {
        "total_ms": "regressed",
        "alloc_bytes": "regressed",
    }
    assert_permutation(
        result.record.metrics["alloc_bytes"], delta=10, verdict="regressed", confirmed=False
    )
    assert result.record.outcome == "regressed"
    assert [
        line for line in trimmed_report_lines(result.report) if line.endswith("on rerun")
    ] == rerun_lines


# ---------------------------------------------------------------------------
# a regressed metric a rerun cannot inform
# ---------------------------------------------------------------------------


async def test_iterate_session_when_only_exact_gating_metrics_regress_does_gate_on_the_first_run_alone(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    stub_samples(samples_mock, open_repo, regressed_rounds(), baseline_rounds())
    resolved = resolved_config(
        metrics={"total_ms": MetricEntry(exact=True), "alloc_bytes": MetricEntry(gating=False)}
    )
    events: list[ProgressEvent] = []

    result = await iterate_session(
        open_repo, resolved, options=IterateOptions(on_progress=events.append)
    )

    assert samples_mock.call_count == 1
    assert [type(e) for e in events if isinstance(e, _JUDGE_AND_CONFIRM_EVENTS)] == [
        JudgeFinished,
        ConfirmSkipped,
    ]
    assert [e.regressed for e in events if isinstance(e, JudgeFinished)] == [("total_ms",)]
    total = result.record.metrics["total_ms"]
    assert total.delta_pct == pytest.approx(10, abs=1e-6)
    assert total.verdict == "regressed"
    assert total.method == "exact"
    assert total.gating is True
    assert total.confirmed is False
    assert total.p is None
    assert total.noise_pct is None
    assert result.record.outcome == "regressed"


async def test_iterate_session_when_a_non_exact_gating_metric_regresses_does_rerun_it_alone(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    stub_runs(
        samples_mock,
        open_repo,
        [
            regressed_run(),
            PairedRun(
                _filtered_rounds("alloc_bytes", scaled(BASELINE_BYTES, 1.2)),
                _filtered_rounds("alloc_bytes", BASELINE_BYTES),
            ),
        ],
    )
    resolved = resolved_config(filter=FILTER, metrics={"total_ms": MetricEntry(exact=True)})
    events: list[ProgressEvent] = []

    await iterate_session(open_repo, resolved, options=IterateOptions(on_progress=events.append))

    assert [type(e) for e in events if isinstance(e, _JUDGE_AND_CONFIRM_EVENTS)] == [
        JudgeFinished,
        ConfirmStarted,
        ConfirmFinished,
    ]
    assert samples_mock.calls[1].options.bench == "npm run bench -- --filter alloc_bytes"


async def test_iterate_session_when_metric_is_non_gating_does_inform_without_rerunning(
    open_repo: str, samples_mock: CollectSamplesRecorder
):
    experiment = rounds(scaled(BASELINE_MS, 0.9), scaled(BASELINE_BYTES, 1.1))
    stub_samples(samples_mock, open_repo, experiment, baseline_rounds())

    result = await iterate_session(
        open_repo, resolved_config(metrics={"alloc_bytes": MetricEntry(gating=False)})
    )

    assert samples_mock.call_count == 1
    assert result.record.confirm is None
    assert result.record.metrics["alloc_bytes"].verdict == "regressed"
    assert result.record.outcome == "improved"


# ---------------------------------------------------------------------------
# a metric's delta ratio is undefined: zero baseline, or overflow past the largest float
# ---------------------------------------------------------------------------


def _flat_total_ms_baseline(value: float) -> list[dict[str, float]]:
    """The baseline's ten rounds, but with ``total_ms`` flat at ``value``."""
    return rounds([value for _ in BASELINE_MS], BASELINE_BYTES)


def _zero_baseline_run() -> PairedRun:
    """A paired run whose baseline has ``total_ms`` flat at zero."""
    return PairedRun(improved_rounds(), _flat_total_ms_baseline(0.0))


def _tiny_baseline_run(sign: float) -> PairedRun:
    """A paired run whose ``total_ms`` delta overflows: a sub-normal baseline, ``sign`` experiment."""
    return PairedRun(
        rounds(scaled(BASELINE_MS, sign), BASELINE_BYTES),
        _flat_total_ms_baseline(_SMALLEST_POSITIVE_FLOAT),
    )


@pytest.fixture(
    params=[
        pytest.param((_zero_baseline_run(), -20), id="zero-baseline"),
        pytest.param((_tiny_baseline_run(1.0), 0), id="tiny-baseline-positive-experiment"),
        pytest.param((_tiny_baseline_run(-1.0), 0), id="tiny-baseline-negative-experiment"),
    ]
)
def undefined_delta_repo(
    request: pytest.FixtureRequest, open_repo: str, samples_mock: CollectSamplesRecorder
) -> tuple[str, float]:
    """An open session with no finite ``total_ms`` delta, and its expected ``alloc_bytes`` delta."""
    run, expected_alloc_delta = request.param
    stub_samples(samples_mock, open_repo, run.experiment, run.baseline)
    return open_repo, expected_alloc_delta


async def test_iterate_session_when_delta_undefined_does_null_only_that_metrics_delta(
    undefined_delta_repo: tuple[str, float],
):
    repo, expected_alloc_delta = undefined_delta_repo

    result = await iterate_session(repo, resolved_config(primary="total_ms"))

    assert result.record.metrics["total_ms"].delta_pct is None
    assert result.record.metrics["alloc_bytes"].delta_pct == pytest.approx(
        expected_alloc_delta, abs=1e-6
    )
    assert result.record.primary == IterationPrimary(kind="metric", name="total_ms", delta_pct=None)
    assert result.record.outcome == "no-signal"
    assert _primary_line(result.report) == "primary: · verdict: NO-SIGNAL"


# ---------------------------------------------------------------------------
# the primary has no change to read: never reported, no qualifying input, or flat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("run", "config", "primary", "primary_line"),
    [
        pytest.param(
            PairedRun(improved_rounds(), baseline_rounds()),
            resolved_config(primary="startup_ms"),
            IterationPrimary(kind="metric", name="startup_ms", delta_pct=None),
            "primary: · verdict: NO-SIGNAL",
            id="named-primary-never-reported",
        ),
        pytest.param(
            PairedRun(improved_rounds(), baseline_rounds()),
            resolved_config(
                metrics={
                    "total_ms": MetricEntry(gating=False),
                    "alloc_bytes": MetricEntry(gating=False),
                }
            ),
            IterationPrimary(kind="geomean", delta_pct=None),
            "primary: · verdict: NO-SIGNAL",
            id="geomean-over-only-non-gating-metrics",
        ),
        pytest.param(
            PairedRun(improved_rounds(), baseline_rounds()),
            resolved_config(unstable_noise_pct=0.0),
            IterationPrimary(kind="geomean", delta_pct=None),
            "primary: · verdict: NO-SIGNAL",
            id="geomean-over-only-unstable-metrics",
        ),
        pytest.param(
            PairedRun(baseline_rounds(), baseline_rounds()),
            resolved_config(primary="total_ms"),
            IterationPrimary(kind="metric", name="total_ms", delta_pct=0),
            "primary: 0.0% · verdict: NO-SIGNAL",
            id="named-primary-flat-reads-zero",
        ),
    ],
)
async def test_iterate_session_when_primary_has_no_change_to_read_does_report_the_primary_without_one(
    open_repo: str,
    samples_mock: CollectSamplesRecorder,
    run: PairedRun,
    config: ResolvedConfig,
    *,
    primary: IterationPrimary,
    primary_line: str,
):
    stub_samples(samples_mock, open_repo, run.experiment, run.baseline)

    result = await iterate_session(open_repo, config)

    assert result.record.primary == primary
    assert result.record.outcome == "no-signal"
    assert _primary_line(result.report) == primary_line


# ---------------------------------------------------------------------------
# the report closes on the outcome's verdict and next step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("experiment", "config", "reached", "outcome", "report_tail"),
    [
        pytest.param(
            improved_rounds(),
            resolved_config(primary="total_ms", stop=StopConfig(target_value=85)),
            False,
            "improved",
            ["primary: -10.0% · verdict: IMPROVED", "gymrat keep"],
            id="target-ahead",
        ),
        pytest.param(
            improved_rounds(),
            resolved_config(primary="total_ms", stop=StopConfig(target_value=95)),
            True,
            "improved",
            ["primary: -10.0% · verdict: IMPROVED", "target reached — keep it", "gymrat keep"],
            id="target-met",
        ),
        pytest.param(
            improved_rounds(),
            resolved_config(
                primary="total_ms",
                stop=StopConfig(target_value=85),
                metrics={"total_ms": MetricEntry(direction="higher")},
            ),
            True,
            "regressed",
            ["primary: -10.0% · verdict: REGRESSED", "fix or run gymrat discard"],
            id="higher-is-better-reads-the-other-side",
        ),
        pytest.param(
            _noisy_rounds(),
            resolved_config(primary="total_ms", stop=StopConfig(target_value=101)),
            True,
            "no-signal",
            [
                "primary: +0.5% · verdict: NO-SIGNAL",
                "target reached — keep it",
                "gymrat keep or gymrat discard",
            ],
            id="no-signal-still-states-it",
        ),
    ],
)
async def test_iterate_session_when_target_configured_does_close_the_report_on_the_judged_outcome(
    open_repo: str,
    samples_mock: CollectSamplesRecorder,
    experiment: list[dict[str, float]],
    config: ResolvedConfig,
    *,
    reached: bool,
    outcome: str,
    report_tail: list[str],
):
    stub_samples(samples_mock, open_repo, experiment, baseline_rounds())

    result = await iterate_session(open_repo, config)

    assert result.record.target_reached is reached
    assert result.record.outcome == outcome
    assert trimmed_report_lines(result.report)[-len(report_tail) :] == report_tail


# ---------------------------------------------------------------------------
# derive_outcome
# ---------------------------------------------------------------------------


def _directed_metric(direction: Direction, *, gating: bool = True) -> MetricComparison:
    """A metric judged in ``direction`` with no signal of its own."""
    return permutation_metric(verdict="no-signal", delta=0, direction=direction, gating=gating)


def _regressed_metrics(*, gating: bool) -> MetricComparisons:
    """A run whose single metric regressed, gating or not."""
    return {"decode/time": permutation_metric(verdict="regressed", delta=4, gating=gating)}


@pytest.mark.parametrize(
    ("gating", "expected"),
    [
        pytest.param(True, "regressed", id="gating-regression-overrides-the-primary"),
        pytest.param(False, "improved", id="non-gating-regression-left-out"),
    ],
)
def test_derive_outcome_when_a_metric_regressed_does_count_it_only_when_gating(
    *, gating: bool, expected: str
):
    outcome = derive_outcome(_regressed_metrics(gating=gating), GeomeanPrimary(-9))

    assert outcome == expected


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        pytest.param(GeomeanPrimary(-3), "improved", id="geomean-negative-improves"),
        pytest.param(GeomeanPrimary(3), "no-signal", id="geomean-positive-no-signal"),
        pytest.param(GeomeanPrimary(0), "no-signal", id="geomean-zero-no-signal"),
        pytest.param(MetricPrimary("lower/time", -3), "improved", id="lower-negative-improves"),
        pytest.param(MetricPrimary("lower/time", 3), "no-signal", id="lower-positive-no-signal"),
        pytest.param(MetricPrimary("higher/time", 3), "improved", id="higher-positive-improves"),
        pytest.param(MetricPrimary("higher/time", -3), "no-signal", id="higher-negative-no-signal"),
        pytest.param(MetricPrimary("lower/time", 0), "no-signal", id="lower-zero-no-signal"),
        pytest.param(MetricPrimary("higher/time", 0), "no-signal", id="higher-zero-no-signal"),
        pytest.param(MetricPrimary("absent/time", -30), "no-signal", id="primary-never-measured"),
    ],
)
def test_derive_outcome_when_no_gating_regression_does_read_the_primary(
    primary: LoopPrimary, expected: str
):
    metrics: MetricComparisons = {
        "lower/time": _directed_metric("lower"),
        "higher/time": _directed_metric("higher"),
    }

    outcome = derive_outcome(metrics, primary)

    assert outcome == expected


def test_derive_outcome_when_gating_metric_has_no_experiment_slice_does_read_the_primary():
    unmeasured = replace(_directed_metric("lower"), candidates=())

    outcome = derive_outcome({"lower/time": unmeasured}, GeomeanPrimary(-9))

    assert outcome == "improved"
