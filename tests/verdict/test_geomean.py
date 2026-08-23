"""Behavioral tests for geomean aggregation over verdict records.

Drives behavior through the public ``compute_geomean`` API. Exclusion ordering,
ratio normalization, and noise-band propagation are all exercised here.
"""

import math
from collections.abc import Sequence

import pytest

from gymrat_py.model import (
    BandVerdict,
    Direction,
    Effect,
    ExactVerdict,
    Exclusion,
    ExclusionReason,
    GeomeanResult,
    MetricMeta,
    MetricVerdict,
    Observations,
)
from gymrat_py.verdict import compute_geomean, compute_verdicts
from tests.verdict._inputs import MetricSpec, build_inputs

METRIC_BYTES_LOWER = {
    "metric": MetricMeta(direction="lower", gating=True, exact=False, unit="bytes")
}


def _noop_warn(_message: str) -> None:
    """Swallow divergence warnings so these cases stay silent on stderr."""


def create_samples(n: int, value: float) -> list[dict[str, float]]:
    return [{"metric": value} for _ in range(n)]


def unstable_band_verdict() -> BandVerdict:
    """A band verdict too noisy to judge, regardless of its ratio."""
    return BandVerdict(
        method="band",
        verdict="unstable",
        usable_n=4,
        noise_pct=250.0,
        noise_abs=25.0,
        delta=Effect(value=-50.0, unit="percent"),
        n=4,
    )


def gating_verdicts_with_noise(
    noise: Sequence[float | None],
) -> tuple[dict[str, MetricVerdict], dict[str, MetricMeta]]:
    """Gating verdicts keyed ``metric1``…``metricN``, one per entry of ``noise``.

    A number becomes a band verdict carrying that ``noise_pct``; ``None`` becomes
    an exact verdict, which carries no noise figure at all. Every verdict can
    be judged and has a usable ratio, so the geomean includes all of them.
    """
    verdicts: dict[str, MetricVerdict] = {}
    metric_meta: dict[str, MetricMeta] = {}

    for index, noise_pct in enumerate(noise):
        key = f"metric{index + 1}"
        if noise_pct is None:
            verdicts[key] = ExactVerdict(
                method="exact",
                verdict="improved",
                delta=Effect(value=-50.0, unit="percent"),
                n=4,
            )
        else:
            verdicts[key] = BandVerdict(
                method="band",
                verdict="improved",
                usable_n=4,
                noise_pct=noise_pct,
                noise_abs=noise_pct / 2,
                delta=Effect(value=-50.0, unit="percent"),
                n=4,
            )
        metric_meta[key] = MetricMeta(
            direction="lower",
            gating=True,
            exact=noise_pct is None,
            unit=None,
        )

    return verdicts, metric_meta


# ---------------------------------------------------------------------------
# Empty and exclusion cases
# ---------------------------------------------------------------------------


def test_compute_geomean_when_no_metrics_does_return_zeroed_result():
    result = compute_geomean({}, {})

    assert result == GeomeanResult(value=0.0, n=0, band=0.0, excluded=())


def test_compute_geomean_when_metric_non_gating_does_aggregate_like_any_other():
    verdicts, metric_meta = build_inputs([MetricSpec(name="metric1", delta=-5.0, gating=False)])

    result = compute_geomean(verdicts, metric_meta)

    assert result.n == 1
    assert result.value == pytest.approx(-5.0, abs=1e-5)


def test_compute_geomean_when_metric_one_sided_does_exclude_as_no_verdict_in_scope():
    verdicts, metric_meta = build_inputs(
        [
            MetricSpec(name="metric1", delta=-5.0),
            MetricSpec(name="metric2", no_verdict=True),
        ],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result.n == 1
    assert result.excluded == (Exclusion(metric="metric2", reason="no-verdict"),)


@pytest.mark.parametrize(
    ("direction", "delta", "reason"),
    [
        pytest.param("lower", math.nan, "undefined-ratio", id="nan-delta"),
        pytest.param("lower", -150.0, "infinite-rho", id="rho-negative"),
        pytest.param("lower", -100.0, "infinite-rho", id="rho-zero"),
        pytest.param("higher", -100.0, "infinite-rho", id="rho-infinite"),
    ],
)
def test_compute_geomean_when_sole_metric_ratio_invalid_does_exclude(
    direction: Direction,
    delta: float,
    reason: ExclusionReason,
):
    verdicts, metric_meta = build_inputs(
        [MetricSpec(name="metric1", direction=direction, delta=delta)],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result.value == 0.0
    assert result.n == 0
    assert result.excluded == (Exclusion(metric="metric1", reason=reason),)


# ---------------------------------------------------------------------------
# Single gating metric
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "delta", "expected_value", "precision"),
    [
        pytest.param("lower", -5.0, -5.0, 1e-5, id="lower-improve"),
        pytest.param("higher", 5.0, -4.76, 5e-2, id="higher-improve"),
        pytest.param("lower", 0.0, 0.0, 1e-5, id="no-change"),
        pytest.param("lower", 100.0, 100.0, 5e-2, id="lower-regress"),
        pytest.param("higher", -10.0, 11.11, 5e-2, id="higher-regress"),
    ],
)
def test_compute_geomean_when_single_gating_metric_does_report_value(
    direction: Direction,
    delta: float,
    expected_value: float,
    precision: float,
):
    verdicts, metric_meta = build_inputs(
        [MetricSpec(name="metric1", direction=direction, delta=delta)],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result.n == 1
    assert result.excluded == ()
    assert result.value == pytest.approx(expected_value, abs=precision)


# ---------------------------------------------------------------------------
# Multiple gating metrics
# ---------------------------------------------------------------------------


def test_compute_geomean_when_multiple_lower_metrics_does_geomean_ratios():
    verdicts, metric_meta = build_inputs(
        [
            MetricSpec(name="metric1", delta=-10.0),
            MetricSpec(name="metric2", delta=-5.0),
        ],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result.n == 2
    assert result.excluded == ()
    assert result.value == pytest.approx(-7.54, abs=5e-2)


def test_compute_geomean_when_directions_differ_does_respect_each_metric():
    verdicts, metric_meta = build_inputs(
        [
            MetricSpec(name="metric1", direction="lower", delta=-10.0),
            MetricSpec(name="metric2", direction="higher", delta=10.0),
        ],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result.n == 2
    assert result.excluded == ()
    assert result.value == pytest.approx(-9.55, abs=0.5)


@pytest.mark.parametrize(
    ("bad_delta", "reason"),
    [
        pytest.param(math.nan, "undefined-ratio", id="nan-delta"),
        pytest.param(-150.0, "infinite-rho", id="rho-negative"),
    ],
)
def test_compute_geomean_when_one_metric_invalid_does_keep_other_ratio(
    bad_delta: float,
    reason: ExclusionReason,
):
    verdicts, metric_meta = build_inputs(
        [
            MetricSpec(name="metric1", delta=bad_delta),
            MetricSpec(name="metric2", delta=-5.0),
        ],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result.n == 1
    assert result.excluded == (Exclusion(metric="metric1", reason=reason),)
    assert result.value == pytest.approx(-5.0, abs=1e-5)


def test_compute_geomean_when_all_metrics_excluded_does_return_zeroed_with_reasons():
    verdicts, metric_meta = build_inputs(
        [
            MetricSpec(name="metric1", delta=-150.0),
            MetricSpec(name="metric2", delta=math.nan),
        ],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result.value == 0.0
    assert result.n == 0
    assert result.excluded == (
        Exclusion(metric="metric1", reason="infinite-rho"),
        Exclusion(metric="metric2", reason="undefined-ratio"),
    )


# ---------------------------------------------------------------------------
# Unstable exclusion
# ---------------------------------------------------------------------------


def test_compute_geomean_when_metric_unstable_does_exclude_despite_valid_ratio():
    verdicts, metric_meta = build_inputs(
        [
            MetricSpec(name="noisy", verdict=unstable_band_verdict()),
            MetricSpec(name="stable", delta=-5.0),
        ],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result.n == 1
    assert result.excluded == (Exclusion(metric="noisy", reason="unstable"),)
    assert result.value == pytest.approx(-5.0, abs=1e-5)


def test_compute_geomean_when_unstable_delta_nan_does_report_unstable_over_undefined():
    verdicts, metric_meta = build_inputs(
        [
            MetricSpec(
                name="noisy",
                verdict=BandVerdict(
                    method="band",
                    verdict="unstable",
                    usable_n=4,
                    noise_pct=300.0,
                    noise_abs=30.0,
                    delta=Effect(value=math.nan, unit="percent"),
                    n=4,
                ),
            ),
        ],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result == GeomeanResult(
        value=0.0,
        n=0,
        band=0.0,
        excluded=(Exclusion(metric="noisy", reason="unstable"),),
    )


# ---------------------------------------------------------------------------
# Propagated noise band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("noise", "expected"),
    [
        pytest.param([4.0], 4.0, id="single"),
        pytest.param([3.0, 4.0], 2.5, id="two"),
        pytest.param([None, 6.0], 3.0, id="exact-adds-no-noise"),
    ],
)
def test_compute_geomean_when_metrics_carry_noise_does_propagate_band(
    noise: Sequence[float | None],
    expected: float,
):
    verdicts, metric_meta = gating_verdicts_with_noise(noise)

    result = compute_geomean(verdicts, metric_meta)

    assert result.band == pytest.approx(expected, abs=1e-10)


def test_compute_geomean_when_byte_metric_does_carry_quantization_noise():
    left = Observations.from_rounds(create_samples(2, 4.0))
    right = Observations.from_rounds(create_samples(2, 3.0))
    verdicts = compute_verdicts(left, right, METRIC_BYTES_LOWER, warn=_noop_warn)

    result = compute_geomean(verdicts, METRIC_BYTES_LOWER)

    assert result.n == 1
    assert result.value == pytest.approx(-25.0, abs=1e-5)
    assert result.band == pytest.approx(100 / 3, abs=1e-5)


def test_compute_geomean_when_metric_excluded_does_leave_its_noise_out_of_band():
    verdicts, metric_meta = build_inputs(
        [
            MetricSpec(name="noisy", verdict=unstable_band_verdict()),
            MetricSpec(
                name="steady",
                verdict=BandVerdict(
                    method="band",
                    verdict="improved",
                    usable_n=4,
                    noise_pct=4.0,
                    noise_abs=2.0,
                    delta=Effect(value=-50.0, unit="percent"),
                    n=4,
                ),
            ),
        ],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result.n == 1
    assert result.band == pytest.approx(4.0, abs=1e-10)
