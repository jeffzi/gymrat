"""Tests for the scoped geomean label and the style of its value."""

from __future__ import annotations

import pytest

from gymrat.model import Exclusion, GeomeanResult
from gymrat.report.geomean_label import geomean_value_style, scoped_geomean_label
from tests.report._verdicts import geomean_of

# ---------------------------------------------------------------------------
# scoped_geomean_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "geomean", "expected"),
    [
        pytest.param("entity", geomean_of(n=1), "geomean · entity (1)", id="nothing-excluded"),
        pytest.param(
            "entity",
            geomean_of(n=1, excluded=[Exclusion(metric="entity.spawn/time", reason="unstable")]),
            "geomean · entity (1/2)",
            id="one-exclusion",
        ),
        pytest.param(
            "memory",
            geomean_of(
                n=13,
                excluded=[
                    Exclusion(metric="a/heap", reason="unstable"),
                    Exclusion(metric="b/heap", reason="undefined-ratio"),
                ],
            ),
            "geomean · memory (13/15)",
            id="several-exclusions",
        ),
    ],
)
def test_scoped_geomean_label_when_subset_given_does_count_the_subset(
    scope: str, geomean: GeomeanResult, expected: str
):
    assert scoped_geomean_label(scope, geomean) == expected


# ---------------------------------------------------------------------------
# geomean_value_style
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("geomean", "expected"),
    [
        pytest.param(geomean_of(value=-6, band=5), "bold green", id="improvement-past-band"),
        pytest.param(geomean_of(value=6, band=5), "bold red", id="regression-past-band"),
        pytest.param(geomean_of(value=-4, band=5), "bold", id="improvement-inside-band"),
        pytest.param(geomean_of(value=4, band=5), "bold", id="regression-inside-band"),
        pytest.param(geomean_of(value=-5, band=5), "bold", id="level-with-band"),
        pytest.param(geomean_of(value=-0.2, band=0), "bold green", id="improvement-no-band"),
        pytest.param(geomean_of(value=float("nan"), n=0), "bold", id="no-stable-metrics"),
    ],
)
def test_geomean_value_style_when_value_given_does_style_against_band(
    geomean: GeomeanResult, expected: str
):
    assert geomean_value_style(geomean, []) == expected
