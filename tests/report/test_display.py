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
        pytest.param(band_verdict(n=10, usable_n=0), "identical", id="every-pair-tied"),
        pytest.param(
            band_verdict(n=6, usable_n=5), "within-noise", id="band-some-pairs-differ-at-minimum"
        ),
        pytest.param(
            band_verdict(verdict="improved", delta=-10, n=10, usable_n=3),
            "improved",
            id="band-found-improvement",
        ),
        pytest.param(permutation_verdict(), "within-noise", id="permutation-no-signal"),
        # sub-minimum non-exact verdicts are inconclusive whatever they found
        pytest.param(band_verdict(n=5, usable_n=5), "inconclusive", id="too-short-for-floor"),
        pytest.param(band_verdict(n=1, usable_n=0), "inconclusive", id="single-pair-tie"),
        pytest.param(
            band_verdict(verdict="improved", delta=-10, n=4, usable_n=4),
            "inconclusive",
            id="band-improved-n4",
        ),
        pytest.param(permutation_verdict(n=3), "inconclusive", id="permutation-n3"),
        pytest.param(
            permutation_verdict(verdict="improved", delta=-10, n=5),
            "inconclusive",
            id="permutation-improved-n5",
        ),
        # exact verdicts keep their real class at any n
        pytest.param(
            exact_verdict(n=1, verdict="improved", delta=-5), "improved", id="exact-improved-n1"
        ),
        pytest.param(exact_verdict(n=2), "within-noise", id="exact-no-signal-n2"),
        # at the minimum n, non-exact verdicts keep their real class
        pytest.param(
            permutation_verdict(verdict="improved", delta=-10, n=6),
            "improved",
            id="permutation-at-minimum",
        ),
    ],
)
def test_display_class_when_verdict_given_does_map_to_shown_class(
    verdict: MetricVerdict, expected: DisplayClass
):
    assert display_class(verdict) == expected
