"""Behavioral tests for why the verdict engine withholds a signal.

Covers the reason a ``no-signal`` verdict with a non-zero delta carries and the
order in which the reasons are checked, through the public ``compute_verdicts``
API only.
"""

from collections.abc import Callable

import pytest

from gymrat.model import BandVerdict, MetricMeta, MetricVerdict, PermutationVerdict
from tests.verdict._engine import (
    DIRECTION_MISMATCH_A,
    DIRECTION_MISMATCH_B,
    FLAT_PERMUTATION_ROUNDS,
    METRIC_APPROX_LOWER,
    METRIC_BYTES_LOWER,
    create_samples,
    get_band,
    get_permutation,
    run,
    samples,
)

# ---------------------------------------------------------------------------
# Why a no-signal verdict was withheld
# ---------------------------------------------------------------------------

# Ten windows whose paired differences cancel out (+-10, +-5, +-4, +-3, and +2
# against -1), so the sign-flip p-value is far from significant while the
# candidate median sits 0.5% above the baseline's.
_CANCELLING_A = create_samples(10, 100.0)
_CANCELLING_B = samples(90.0, 110.0, 95.0, 105.0, 96.0, 104.0, 102.0, 99.0, 97.0, 103.0)


@pytest.mark.parametrize(
    ("narrow", "meta", "samples_a", "samples_b", "expected_verdict", "expected_reason"),
    [
        # A zero baseline adds no step floor, so the delta's missing percentage
        # is the reason.
        pytest.param(
            get_permutation,
            METRIC_BYTES_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 0.0),
            create_samples(FLAT_PERMUTATION_ROUNDS, 5.0),
            "no-signal",
            "non-finite-delta",
            id="permutation-zero-byte-baseline",
        ),
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            create_samples(2, 0.0),
            create_samples(2, 5.0),
            "no-signal",
            "non-finite-delta",
            id="band-non-finite-delta",
        ),
        # The p-value is not significant either; the step floor outranks it.
        pytest.param(
            get_permutation,
            METRIC_BYTES_LOWER,
            _CANCELLING_A,
            _CANCELLING_B,
            "no-signal",
            "resolution",
            id="permutation-resolution-over-noise",
        ),
        pytest.param(
            get_permutation,
            METRIC_APPROX_LOWER,
            _CANCELLING_A,
            _CANCELLING_B,
            "no-signal",
            "noise",
            id="permutation-not-significant",
        ),
        # The same pairs as the direction-mismatch case against a one-byte step:
        # the 2% delta sits inside the 12.5% step floor, which outranks the
        # direction mismatch.
        pytest.param(
            get_permutation,
            METRIC_BYTES_LOWER,
            DIRECTION_MISMATCH_A,
            DIRECTION_MISMATCH_B,
            "no-signal",
            "resolution",
            id="permutation-resolution-over-direction-mismatch",
        ),
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            create_samples(2, 50.0),
            create_samples(2, 100.0),
            "regressed",
            None,
            id="band-regressed",
        ),
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            create_samples(2, 100.0),
            create_samples(2, 100.0),
            "no-signal",
            None,
            id="band-zero-delta",
        ),
    ],
)
def test_compute_verdicts_when_approximate_does_report_first_withheld_reason(
    narrow: Callable[[dict[str, MetricVerdict]], BandVerdict | PermutationVerdict],
    meta: dict[str, MetricMeta],
    samples_a: list[dict[str, float]],
    samples_b: list[dict[str, float]],
    *,
    expected_verdict: str,
    expected_reason: str | None,
):
    result = run(samples_a, samples_b, meta)

    verdict = narrow(result)
    assert (verdict.verdict, verdict.reason) == (expected_verdict, expected_reason)
