"""Tests for mapping a metric verdict to the class and glyph a report shows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.display import DisplayClass, display_class, get_glyph
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
        pytest.param(band_verdict(n=10, usable_n=3), "within-noise", id="ties-short-of-floor"),
        pytest.param(band_verdict(n=10, usable_n=6), "within-noise", id="ties-just-enough"),
        pytest.param(band_verdict(n=6, usable_n=5), "within-noise", id="one-below-floor"),
        pytest.param(band_verdict(n=5, usable_n=5), "inconclusive", id="too-short-for-floor"),
        pytest.param(band_verdict(n=1, usable_n=1), "inconclusive", id="single-pair-only-floor"),
        pytest.param(band_verdict(n=1, usable_n=0), "inconclusive", id="single-pair-tie"),
        pytest.param(band_verdict(n=2, usable_n=2), "inconclusive", id="two-pairs-sub-minimum"),
        pytest.param(
            band_verdict(verdict="improved", delta=-10, n=10, usable_n=3),
            "improved",
            id="band-found-improvement",
        ),
        pytest.param(
            band_verdict(verdict="unstable", n=10, usable_n=3),
            "unstable",
            id="noise-swamped-band",
        ),
        pytest.param(permutation_verdict(), "within-noise", id="permutation-no-signal"),
        pytest.param(exact_verdict(), "within-noise", id="counted-metric-unchanged"),
    ],
)
def test_display_class_when_verdict_given_does_map_to_shown_class(
    verdict: MetricVerdict, expected: str
):
    assert display_class(verdict) == expected


@pytest.mark.parametrize(
    "verdict",
    [
        pytest.param(band_verdict(n=2, usable_n=2), id="band-n2"),
        pytest.param(band_verdict(n=3, usable_n=3), id="band-n3"),
        pytest.param(band_verdict(n=3, usable_n=0), id="band-all-tied-sub-minimum"),
        pytest.param(
            band_verdict(verdict="improved", delta=-10, n=4, usable_n=4), id="band-improved-n4"
        ),
        pytest.param(band_verdict(verdict="unstable", n=5, usable_n=5), id="band-unstable-n5"),
        pytest.param(permutation_verdict(n=3), id="permutation-n3"),
        pytest.param(
            permutation_verdict(verdict="improved", delta=-10, n=5), id="permutation-improved-n5"
        ),
        pytest.param(
            permutation_verdict(verdict="regressed", delta=10, n=5), id="permutation-regressed-n5"
        ),
    ],
)
def test_display_class_when_sub_minimum_non_exact_does_return_inconclusive(
    verdict: MetricVerdict,
):
    assert display_class(verdict) == "inconclusive"


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        pytest.param(
            exact_verdict(n=1, verdict="improved", delta=-5), "improved", id="improved-n1"
        ),
        pytest.param(exact_verdict(n=2), "within-noise", id="no-signal-n2"),
        pytest.param(
            exact_verdict(n=3, verdict="regressed", delta=5), "regressed", id="regressed-n3"
        ),
        pytest.param(exact_verdict(n=5), "within-noise", id="no-signal-n5"),
    ],
)
def test_display_class_when_exact_at_any_n_does_keep_real_class(
    verdict: MetricVerdict, expected: DisplayClass
):
    assert display_class(verdict) == expected


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        pytest.param(band_verdict(n=6, usable_n=6), "within-noise", id="band-at-minimum"),
        pytest.param(
            permutation_verdict(verdict="improved", delta=-10, n=6),
            "improved",
            id="permutation-at-minimum",
        ),
    ],
)
def test_display_class_when_at_minimum_n_does_keep_real_class(
    verdict: MetricVerdict, expected: DisplayClass
):
    assert display_class(verdict) == expected


# ---------------------------------------------------------------------------
# get_glyph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shown", "expected"),
    [
        pytest.param("improved", "✓", id="improved"),
        pytest.param("regressed", "✗", id="regressed"),
        pytest.param("unstable", "≈", id="unstable"),
        pytest.param("identical", "=", id="identical"),
        pytest.param("within-noise", "~", id="within-noise"),
        pytest.param("inconclusive", "?", id="inconclusive"),
    ],
)
def test_get_glyph_when_display_class_given_does_return_expected_mark(
    shown: DisplayClass, expected: str
):
    assert get_glyph(shown) == expected
