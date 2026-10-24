"""Behavioral tests for why the verdict engine withholds a signal.

Covers the reason a ``no-signal`` verdict with a non-zero delta carries and the
order in which the reasons are checked, through the public ``compute_verdicts``
API only.
"""

from collections.abc import Callable
from dataclasses import fields
from typing import get_args

import pytest

from gymrat.model import (
    BandVerdict,
    MetricMeta,
    MetricVerdict,
    PermutationVerdict,
    WithheldReason,
)
from tests.verdict._engine import (
    BYTE_ONE_MOVE_A,
    BYTE_ONE_MOVE_B,
    FLAT_PERMUTATION_ROUNDS,
    METRIC_APPROX_LOWER,
    METRIC_BYTES_LOWER,
    METRIC_EXACT_LOWER,
    SEPARATED_HIGH,
    SEPARATED_LOW,
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

# Nine of ten pairs rise while the candidate median sits 2% lower.
_MISMATCHED_A = create_samples(4, 0.0) + create_samples(6, 10.0)
_MISMATCHED_B = create_samples(4, 9.0) + samples(9.5) + create_samples(5, 10.1)


@pytest.mark.parametrize(
    ("narrow", "meta", "samples_a", "samples_b", "expected_verdict", "expected_reason"),
    [
        pytest.param(
            get_permutation,
            METRIC_APPROX_LOWER,
            create_samples(FLAT_PERMUTATION_ROUNDS, 5e-324),
            create_samples(FLAT_PERMUTATION_ROUNDS, 100.0),
            "no-signal",
            "non-finite-delta",
            id="permutation-non-finite-delta",
        ),
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
        # A subnormal baseline overflows the percentage to an infinite delta.
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            create_samples(2, 5e-324),
            create_samples(2, 100.0),
            "no-signal",
            "non-finite-delta",
            id="band-overflow-positive",
        ),
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            create_samples(2, 5e-324),
            create_samples(2, -100.0),
            "no-signal",
            "non-finite-delta",
            id="band-overflow-negative",
        ),
        pytest.param(
            get_band,
            METRIC_BYTES_LOWER,
            create_samples(2, 0.0),
            create_samples(2, 5.0),
            "no-signal",
            "non-finite-delta",
            id="band-zero-byte-baseline",
        ),
        pytest.param(
            get_permutation,
            METRIC_BYTES_LOWER,
            BYTE_ONE_MOVE_A,
            BYTE_ONE_MOVE_B,
            "no-signal",
            "resolution",
            id="permutation-resolution",
        ),
        # The band rejects the delta as well; the step floor outranks it.
        pytest.param(
            get_band,
            METRIC_BYTES_LOWER,
            create_samples(2, 4.0),
            create_samples(2, 3.0),
            "no-signal",
            "resolution",
            id="band-resolution",
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
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            create_samples(2, 100.0),
            samples(99.8, 99.6),
            "no-signal",
            "noise",
            id="band-within-band",
        ),
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            samples(100.0, 100.0),
            samples(100.0, 210.0),
            "no-signal",
            "noise",
            id="band-too-few-differ",
        ),
        pytest.param(
            get_permutation,
            METRIC_APPROX_LOWER,
            _MISMATCHED_A,
            _MISMATCHED_B,
            "no-signal",
            "direction-mismatch",
            id="permutation-direction-mismatch",
        ),
        # The same pairs against a one-byte step: the 2% delta sits inside the
        # 12.5% step floor, which outranks the direction mismatch.
        pytest.param(
            get_permutation,
            METRIC_BYTES_LOWER,
            _MISMATCHED_A,
            _MISMATCHED_B,
            "no-signal",
            "resolution",
            id="permutation-resolution-over-direction-mismatch",
        ),
        pytest.param(
            get_permutation,
            METRIC_APPROX_LOWER,
            SEPARATED_HIGH,
            SEPARATED_LOW,
            "improved",
            None,
            id="permutation-improved",
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
            get_permutation,
            METRIC_APPROX_LOWER,
            samples(80.0, 90.0, 95.0, 100.0, 100.0, 105.0, 110.0, 120.0),
            samples(81.0, 91.0, 96.0, 100.0, 100.0, 106.0, 111.0, 121.0),
            "no-signal",
            None,
            id="permutation-zero-delta",
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
        # The unstable override replaces a non-finite-delta withholding.
        pytest.param(
            get_band,
            METRIC_APPROX_LOWER,
            samples(-5.0, 5.0),
            samples(10.0, 10.0),
            "unstable",
            None,
            id="unstable-over-non-finite-delta",
        ),
        # The unstable override replaces a resolution withholding.
        pytest.param(
            get_band,
            METRIC_BYTES_LOWER,
            create_samples(2, 0.6),
            create_samples(2, 0.7),
            "unstable",
            None,
            id="unstable-over-resolution",
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


def test_withheld_reason_when_listed_does_include_unconfirmed():
    assert get_args(WithheldReason) == (
        "non-finite-delta",
        "resolution",
        "noise",
        "direction-mismatch",
        "unconfirmed",
    )


def test_compute_verdicts_when_exact_does_carry_no_reason_field():
    result = run(create_samples(2, 100.0), create_samples(2, 101.0), METRIC_EXACT_LOWER)

    verdict = result["metric"]
    assert verdict.method == "exact"
    assert "reason" not in {field.name for field in fields(verdict)}
