"""Tests for grouped rendering in single-candidate, single-section comparison reports.

A single-kind comparison renders flat (no section borders).  These tests verify
that the flat renderer shows group headers with a kind suffix, indents member
rows by the group indent, and uses case names rather than full metric names.
Names and labels carrying square brackets print as written.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat.report.text.render import render_report
from gymrat.report.types import CandidateMetric, MetricComparison
from gymrat.verdict import GroupAggregate, KindAggregate

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.report.types import ComparisonResult
from tests._ansi import strip_ansi
from tests.report._assertions import (
    cells_of,
    line_containing,
    line_starting_with,
    offsets_of,
    table_region,
    table_rows,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    exact_metric,
    kind_metric,
    metric_meta,
    permutation_metric,
    two_kind_metrics,
    two_kind_result,
)
from tests.report._verdicts import exact_verdict, geomean_of


def _grouped_flat_result() -> ComparisonResult:
    """Single ``time`` kind: ``entity`` group (2 members) + ungrouped ``warmup``."""
    geomean = geomean_of(-3.2, 3)
    return create_comparison_result(
        metrics={
            "entity/alive_check#time": kind_metric(
                kind="time",
                short_name="entity.alive_check",
                verdict="improved",
                delta=-10,
            ),
            "entity/spawn#time": kind_metric(
                kind="time",
                short_name="entity.spawn",
                verdict="regressed",
                delta=4,
            ),
            "warmup#time": kind_metric(
                kind="time",
                short_name="warmup",
                verdict="no-signal",
                delta=0.3,
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean,
                        groups=(
                            GroupAggregate(
                                group="entity",
                                geomean=geomean_of(-3.1, 2),
                            ),
                        ),
                        gated_geomean=geomean,
                    )
                ]
            )
        ],
    )


# ---------------------------------------------------------------------------
# indented member rows
# ---------------------------------------------------------------------------


def test_render_report_when_flat_grouped_does_indent_member_rows():
    report = render_report(_grouped_flat_result())

    line = line_starting_with(report, "  alive_check")

    assert cells_of(line)[0].rstrip() == "  alive_check"


# ---------------------------------------------------------------------------
# kind suffix on group header
# ---------------------------------------------------------------------------


def test_render_report_when_flat_grouped_does_show_kind_on_group_header():
    # In the flat layout there is no section header to carry the kind, so it
    # is stated on the group header instead (e.g. "entity  time").
    report = strip_ansi(render_report(_grouped_flat_result()))

    entity_header = line_starting_with(report, "entity ")

    assert "time" in cells_of(entity_header)[0]


# ---------------------------------------------------------------------------
# pair count alignment across rows with and without a band
# ---------------------------------------------------------------------------


def _mixed_band_result() -> ComparisonResult:
    """A permutation-tested metric (banded) beside an exact one (no band), both short a pair."""
    return create_comparison_result(
        metrics={
            "latency#other": permutation_metric(verdict="improved", delta=-10, n=8),
            "heap#other": exact_metric(delta=-5, n=8),
        },
    )


def _undefined_ratio_result() -> ComparisonResult:
    """A banded regression beside an undefined-ratio row (no delta, no band), both short a pair."""
    return create_comparison_result(
        metrics={
            "decode#other": permutation_metric(verdict="regressed", delta=4, n=8),
            "nan-delta#other": MetricComparison(
                baseline_median=0,
                baseline_spread=None,
                candidates=(
                    CandidateMetric(median=120, verdict=exact_verdict(delta=math.nan, n=8)),
                ),
                meta=metric_meta("nan-delta#other", exact=True),
            ),
        },
    )


def _short_pairs(metric: MetricComparison, pairs: int) -> MetricComparison:
    """``metric`` with its sole candidate's verdict over ``pairs`` pairs."""
    candidate = metric.candidates[0]
    if candidate.verdict is None:
        msg = f"no verdict to shorten on {metric!r}"
        raise AssertionError(msg)
    return replace(
        metric, candidates=(replace(candidate, verdict=replace(candidate.verdict, n=pairs)),)
    )


def _sectioned_short_pairs_result() -> ComparisonResult:
    """The two-kind layout with one row in each section short a pair."""
    metrics = two_kind_metrics()
    for name in ("entity/spawn#time", "encode#memory"):
        metrics[name] = _short_pairs(metrics[name], 8)
    return replace(two_kind_result(), metrics=metrics)


@pytest.mark.parametrize(
    ("make_result", "first_row", "second_row"),
    [
        pytest.param(_mixed_band_result, "-10.0%", "-5.0%", id="exact-row-without-band"),
        pytest.param(
            _undefined_ratio_result, "decode#other", "nan-delta#other", id="undefined-ratio-row"
        ),
        pytest.param(_sectioned_short_pairs_result, "  spawn", "encode", id="across-sections"),
    ],
)
def test_render_report_when_rows_are_short_of_pairs_does_align_their_pair_counts(
    make_result: Callable[[], ComparisonResult], first_row: str, second_row: str
):
    report = strip_ansi(render_report(make_result()))

    first_line = line_containing(report, first_row)
    second_line = line_containing(report, second_row)

    assert offsets_of(first_line, "n=") == offsets_of(second_line, "n=")


def _unstable_with_band_result() -> ComparisonResult:
    """A banded improvement beside an unstable metric, both short a pair of the run's ten."""
    return create_comparison_result(
        metrics={
            "latency#other": permutation_metric(verdict="improved", delta=-10, n=8),
            "flaky#other": permutation_metric(verdict="unstable", delta=50, n=8),
        },
    )


def test_render_report_when_a_row_is_unstable_does_reserve_the_band_slot_before_its_pair_count():
    report = strip_ansi(render_report(_unstable_with_band_result()))

    unstable_line = line_containing(report, "unstable")

    assert unstable_line.endswith("≈  unstable         n=8")


# ---------------------------------------------------------------------------
# bracketed names render literally
# ---------------------------------------------------------------------------


def _bracketed_result() -> ComparisonResult:
    """One ``time`` kind whose group, member and ungrouped names read as markup tags."""
    geomean = geomean_of(-3.2, 3)
    return create_comparison_result(
        metrics={
            "[bold]entity/[dim]spawn#time": kind_metric(
                kind="time",
                short_name="[bold]entity.[dim]spawn",
                verdict="improved",
                delta=-10,
            ),
            "[bold]entity/[underline]check#time": kind_metric(
                kind="time",
                short_name="[bold]entity.[underline]check",
                verdict="regressed",
                delta=4,
            ),
            "[italic]warmup#time": kind_metric(
                kind="time",
                short_name="[italic]warmup",
                verdict="no-signal",
                delta=0.3,
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean,
                        groups=(GroupAggregate(group="[bold]entity", geomean=geomean_of(-3.1, 2)),),
                        gated_geomean=geomean,
                    )
                ]
            )
        ],
    )


def test_render_report_when_names_carry_brackets_does_print_them_literally():
    report = strip_ansi(render_report(_bracketed_result()))

    assert table_region(report)[1:] == [
        "metric",
        "<rule>",
        "[bold]entity · time",
        "[dim]spawn",
        "[underline]check",
        "",
        "[italic]warmup",
        "<rule>",
        "geomean (3 stable metrics)",
    ]


def _bracketed_labels_result() -> ComparisonResult:
    """A single-candidate run whose baseline and candidate labels read as markup tags."""
    return create_comparison_result(
        baseline_label="[dim]main",
        candidates=[create_candidate(label="[bold]turbo")],
    )


def test_render_report_when_baseline_and_candidate_labels_carry_brackets_does_print_them_literally():
    report = strip_ansi(render_report(_bracketed_labels_result()))

    header = line_starting_with(report, "metric")

    assert cells_of(header)[1].strip() == "[dim]main"
    assert cells_of(header)[2].strip() == "[bold]turbo"
    assert cells_of(header)[3].strip() == "vs [dim]main"


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
                    KindAggregate(
                        kind="[underline]time",
                        geomean=time_geomean,
                        groups=(GroupAggregate(group="[bold]entity", geomean=time_geomean),),
                        gated_geomean=time_geomean,
                    ),
                    KindAggregate(
                        kind="[strike]memory",
                        geomean=memory_geomean,
                        groups=(),
                        gated_geomean=memory_geomean,
                    ),
                ]
            )
        ],
    )


def test_render_report_when_sectioned_names_carry_brackets_does_print_them_literally():
    report = render_report(_sectioned_bracketed_result())

    assert [cells_of(row)[0].strip() for row in table_rows(report)] == [
        "[underline]time",
        "[bold]entity",
        "spawn",
        "geomean · [bold]entity (1)",
        "geomean · [underline]time (1)",
        "[strike]memory",
        "[italic]encode",
        "geomean · [strike]memory (1)",
    ]
