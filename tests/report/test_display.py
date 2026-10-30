"""Tests for mapping a metric verdict to the class and glyph a report shows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.display import DisplayClass, display_class
from tests.report._verdicts import band_verdict, exact_verdict, permutation_verdict

if TYPE_CHECKING:
    from gymrat.model import MetricVerdict

# ---------------------------------------------------------------------------
# display_class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        pytest.param(band_verdict(n=1, usable_n=0), "inconclusive", id="single-pair-tie"),
        pytest.param(
            permutation_verdict(verdict="improved", delta=-10, n=5),
            "inconclusive",
            id="permutation-improved-n5",
        ),
        pytest.param(exact_verdict(n=2), "within-noise", id="exact-no-signal-n2"),
        pytest.param(
            permutation_verdict(verdict="improved", delta=-10, n=6),
            "improved",
            id="permutation-at-minimum",
        ),
    ],
)
def test_display_class_when_non_exact_pair_count_below_minimum_does_read_inconclusive(
    verdict: MetricVerdict, expected: DisplayClass
):
    assert display_class(verdict) == expected


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        pytest.param(band_verdict(n=10, usable_n=0), "identical", id="every-pair-tied"),
        pytest.param(
            band_verdict(n=6, usable_n=5), "within-noise", id="band-some-pairs-differ-at-minimum"
        ),
        pytest.param(permutation_verdict(), "within-noise", id="permutation-no-signal"),
    ],
)
def test_display_class_when_no_signal_does_read_identical_only_if_every_pair_tied(
    verdict: MetricVerdict, expected: DisplayClass
):
    assert display_class(verdict) == expected
