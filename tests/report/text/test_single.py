"""Tests for the single-candidate comparison text report, flat and sectioned.

Baseline, candidate, kind, group and member names carrying square brackets
print as written. The scoped geomean label counts kept metrics against the total.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.model import Exclusion
from gymrat.report.table.cells import geomean_scope_label
from gymrat.report.text.render import render_report
from gymrat.report.text.single import scoped_geomean_label

if TYPE_CHECKING:
    from gymrat.model import GeomeanResult
    from gymrat.report.types import ComparisonResult
from tests._ansi import strip_ansi
from tests.report._assertions import (
    line_starting_with,
    stripped_cells,
    table_rows,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    gating_kind,
    kind_metric,
    mixed_methods_result,
    other_kind,
)
from tests.report._verdicts import geomean_of

# ---------------------------------------------------------------------------
# bracketed names render literally
# ---------------------------------------------------------------------------


def _bracketed_labels_result() -> ComparisonResult:
    """A single-candidate run whose baseline and candidate labels read as markup tags."""
    return create_comparison_result(
        baseline_label="[dim]main",
        candidates=[create_candidate(label="[bold]turbo")],
    )


def test_render_report_when_baseline_and_candidate_labels_carry_brackets_does_print_them_literally():
    report = strip_ansi(render_report(_bracketed_labels_result()))

    header = line_starting_with(report, "metric")
    assert stripped_cells(header)[1:] == ["[dim]main", "[bold]turbo", "vs [dim]main"]


def _sectioned_bracketed_result() -> ComparisonResult:
    """Two kinds whose kind, group and member names read as markup tags."""
    time_geomean = geomean_of(-10, 1)
    memory_geomean = geomean_of(4, 1)
    return create_comparison_result(
        metrics={
            "[bold]entity/spawn#[underline]time": kind_metric(
                kind="[underline]time",
                short_name="[bold]entity.spawn",
                verdict="improved",
                delta=-10,
            ),
            "[italic]encode#[strike]memory": kind_metric(
                kind="[strike]memory",
                short_name="[italic]encode",
                verdict="regressed",
                delta=4,
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    gating_kind("[underline]time", time_geomean, {"[bold]entity": time_geomean}),
                    gating_kind("[strike]memory", memory_geomean),
                ]
            )
        ],
    )


def test_render_report_when_sectioned_names_carry_brackets_does_print_them_literally():
    report = render_report(_sectioned_bracketed_result())

    assert [stripped_cells(row)[0] for row in table_rows(report)] == [
        "[underline]time",
        "[bold]entity",
        "spawn",
        "geomean · [bold]entity (1/1)",
        "geomean · [underline]time (1/1)",
        "[strike]memory",
        "[italic]encode",
        "geomean · [strike]memory (1/1)",
    ]


# ---------------------------------------------------------------------------
# flat geomean row
# ---------------------------------------------------------------------------


def _flat_result(geomean: GeomeanResult) -> ComparisonResult:
    """The three ``other`` metrics of a mixed-methods run, closed on ``geomean``."""
    return create_comparison_result(
        metrics=mixed_methods_result(n=10).metrics,
        candidates=[
            create_candidate(
                kinds=[
                    other_kind(
                        geomean.value, geomean.n, band=geomean.band, excluded=geomean.excluded
                    )
                ]
            )
        ],
    )


def test_render_report_when_flat_geomean_excludes_a_metric_does_label_kept_against_total():
    result = _flat_result(
        geomean_of(-5.8, 2, excluded=[Exclusion(metric="flaky#other", reason="unstable")])
    )

    report = strip_ansi(render_report(result))

    assert stripped_cells(line_starting_with(report, "geomean"))[0] == "geomean (2/3)"


def test_render_report_when_flat_geomean_covers_no_metric_does_show_the_no_stable_metrics_stand_in():
    result = _flat_result(
        geomean_of(
            0,
            0,
            excluded=[
                Exclusion(metric="latency#other", reason="unstable"),
                Exclusion(metric="heap#other", reason="unstable"),
                Exclusion(metric="flaky#other", reason="unstable"),
            ],
        )
    )

    report = strip_ansi(render_report(result))

    assert "no stable metrics" in stripped_cells(line_starting_with(report, "geomean"))[-1]


# ---------------------------------------------------------------------------
# scoped_geomean_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "geomean", "expected"),
    [
        pytest.param(
            "entity",
            geomean_of(n=1),
            f"{geomean_scope_label('entity')} (1/1)",
            id="nothing-excluded",
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
            f"{geomean_scope_label('memory')} (13/15)",
            id="several-exclusions",
        ),
    ],
)
def test_scoped_geomean_label_when_exclusions_vary_does_count_kept_metrics_against_total(
    scope: str, geomean: GeomeanResult, expected: str
):
    label = scoped_geomean_label(scope, geomean)

    assert label == expected
