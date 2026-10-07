"""Behavioral tests for geomean aggregation over verdict records.

Drives behavior through the public ``compute_geomean`` API. Exclusion ordering,
ratio normalization, and noise-band propagation are all exercised here.
"""

import math

import pytest

from gymrat.model import (
    BandVerdict,
    Exclusion,
    ExclusionReason,
    GeomeanResult,
)
from gymrat.verdict import compute_geomean
from tests.report._verdicts import band_verdict, exact_verdict
from tests.verdict._inputs import (
    MetricSpec,
    build_inputs,
    unstable_band_verdict,
)

# ---------------------------------------------------------------------------
# Empty and exclusion cases
# ---------------------------------------------------------------------------


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
    assert result.value == pytest.approx((math.sqrt(0.9 * 0.95) - 1) * 100, abs=1e-6)


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
    assert result.value == pytest.approx((math.sqrt(0.9 / 1.1) - 1) * 100, abs=1e-6)


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
                    delta=math.nan,
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


def test_compute_geomean_when_exact_metric_beside_noisy_one_does_add_no_noise_to_band():
    # An exact verdict carries no noise figure, but it is judged, so the geomean
    # takes it and the band halves the noisy metric's noise across both.
    verdicts, metric_meta = build_inputs([
        MetricSpec(name="metric1", verdict=exact_verdict(delta=-50.0, n=4)),
        MetricSpec(
            name="metric2",
            verdict=band_verdict(
                verdict="improved",
                usable_n=4,
                noise_pct=6.0,
                noise_abs=3.0,
                delta=-50.0,
                n=4,
            ),
        ),
    ])

    result = compute_geomean(verdicts, metric_meta)

    assert result.band == pytest.approx(3.0, abs=1e-10)


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
                    delta=-50.0,
                    n=4,
                ),
            ),
        ],
    )

    result = compute_geomean(verdicts, metric_meta)

    assert result.n == 1
    assert result.band == pytest.approx(4.0, abs=1e-10)
