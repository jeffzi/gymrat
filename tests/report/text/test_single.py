"""Tests for the single-candidate comparison text report, flat and sectioned.

A within-noise row states its detectable change last, and no other row carries
one. Baseline, candidate, kind, group and member names carrying square brackets
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
    detecting_metric,
    every_class_metrics,
    gating_kind,
    kind_metric,
)
from tests.report._verdicts import geomean_of

# ---------------------------------------------------------------------------
# detectable change on within-noise rows
# ---------------------------------------------------------------------------


def test_render_report_when_every_class_carries_a_detectable_change_does_state_it_on_within_noise_rows_only():
    metrics = {
        name: detecting_metric(metric, 2.1) for name, metric in every_class_metrics().items()
    }
    result = create_comparison_result(metrics=metrics)

    rows = [stripped_cells(row) for row in table_rows(render_report(result))]

    assert [
        (cells[0], cells[-1].rsplit("  ", 1)[-1]) for cells in rows if "detects" in cells[-1]
    ] == [("flat/time", "detects ≥2.1%")]


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
        "geomean · [bold]entity (1)",
        "geomean · [underline]time (1)",
        "[strike]memory",
        "[italic]encode",
        "geomean · [strike]memory (1)",
    ]


# ---------------------------------------------------------------------------
# scoped_geomean_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scope", "geomean", "expected"),
    [
        pytest.param(
            "entity", geomean_of(n=1), f"{geomean_scope_label('entity')} (1)", id="nothing-excluded"
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
