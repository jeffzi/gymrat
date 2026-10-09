"""Tests for the single-candidate comparison text report, flat and sectioned.

A single-kind comparison renders flat (no section borders); several kinds render
one bordered section each. These tests verify that group headers carry a kind
suffix, member rows use case names in first-appearance order, a row paired over
fewer rounds is annotated with its pair count and those counts align across rows
and sections, and a sectioned table closes on its last geomean. Names and labels
carrying square brackets print as written.

They also pin the table's column alignment: the ``±`` offset shared across value
cells, and the glyph, delta and band laid out across verdict cells, aggregate
rows included. Alignment is asserted *within* a parsed cell, since the box chrome
is rich's rather than a hand-spliced grid.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import pytest

from gymrat.report.text.render import render_report

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.report.types import ComparisonResult, MetricComparison
from tests._ansi import strip_ansi
from tests.report._assertions import (
    cells_of,
    delta_cell,
    line_containing,
    line_starting_with,
    offsets_of,
    rule_lines,
    stripped_cells,
    table_region,
    table_rows,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    exact_metric,
    gating_kind,
    kind_metric,
    memory_kind,
    mixed_methods_result,
    other_kind,
    permutation_metric,
    time_kind,
    two_kind_metrics,
    two_kind_result,
    without_gated_geomean,
)
from tests.report._verdicts import band_metric, geomean_of

# ---------------------------------------------------------------------------
# group headers, case names and first-appearance order
# ---------------------------------------------------------------------------


def test_render_report_when_flat_body_has_groups_does_list_them_in_first_appearance_order():
    geomean = geomean_of(-2, 5)
    result = create_comparison_result(
        metrics={
            "node/get#time": kind_metric(
                kind="time", short_name="node.get", verdict="improved", delta=-5
            ),
            "entity/spawn#time": kind_metric(
                kind="time",
                short_name="entity.spawn",
                verdict="regressed",
                delta=4,
            ),
            "node/set#time": kind_metric(
                kind="time",
                short_name="node.set",
                verdict="no-signal",
                delta=0.1,
            ),
            "entity/check#time": kind_metric(
                kind="time",
                short_name="entity.check",
                verdict="improved",
                delta=-3,
            ),
            "warmup#time": kind_metric(
                kind="time", short_name="warmup", verdict="no-signal", delta=0.3
            ),
        },
        candidates=[
            create_candidate(
                kinds=[
                    gating_kind(
                        "time",
                        geomean,
                        {"node": geomean_of(-2.5, 2), "entity": geomean_of(0.5, 2)},
                    )
                ]
            )
        ],
    )

    region = table_region(render_report(result))[1:]

    assert region == [
        "metric",
        "<rule>",
        "node · time",
        "get",
        "set",
        "",
        "entity · time",
        "spawn",
        "check",
        "",
        "warmup",
        "<rule>",
        "geomean (5 stable metrics)",
    ]


# ---------------------------------------------------------------------------
# pair counts: annotated per row, aligned across rows with and without a band
# ---------------------------------------------------------------------------


def _undefined_ratio_result() -> ComparisonResult:
    """A banded regression beside an undefined-ratio row (no delta, no band), both short a pair."""
    return create_comparison_result(
        metrics={
            "decode#other": permutation_metric(verdict="regressed", delta=4, n=8),
            "nan-delta#other": exact_metric(
                delta=math.nan,
                n=8,
                unit=None,
                baseline_median=0,
                median=120,
                short_name="nan-delta#other",
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


def _unstable_beside_wide_delta_result() -> ComparisonResult:
    """A banded regression whose delta is wider than ``unstable``, beside an unstable metric."""
    return create_comparison_result(
        metrics={
            "bloat#other": permutation_metric(verdict="regressed", delta=12345.6, n=8),
            "flaky#other": permutation_metric(verdict="unstable", delta=50, n=8),
        },
    )


@pytest.mark.parametrize(
    ("make_result", "first_row", "second_row"),
    [
        pytest.param(
            partial(mixed_methods_result, n=8), "-10.0%", "-5.0%", id="exact-row-without-band"
        ),
        pytest.param(
            _undefined_ratio_result, "decode#other", "nan-delta#other", id="undefined-ratio-row"
        ),
        pytest.param(_sectioned_short_pairs_result, "  spawn", "encode", id="across-sections"),
        pytest.param(
            partial(mixed_methods_result, n=8),
            "latency#other",
            "flaky#other",
            id="word-wider-than-delta",
        ),
        pytest.param(
            _unstable_beside_wide_delta_result,
            "bloat#other",
            "flaky#other",
            id="delta-wider-than-word",
        ),
    ],
)
def test_render_report_when_rows_are_short_of_pairs_does_align_their_pair_counts(
    make_result: Callable[[], ComparisonResult], first_row: str, second_row: str
):
    report = strip_ansi(render_report(make_result()))

    first_line = line_containing(report, first_row)
    second_line = line_containing(report, second_row)
    first_offsets = offsets_of(first_line, "n=")
    assert first_offsets != []
    assert first_offsets == offsets_of(second_line, "n=")


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        pytest.param(
            permutation_metric(verdict="improved", delta=-10, n=8),
            "✓  -10.0%  ±2.5%  n=8",
            id="permutation",
        ),
        pytest.param(
            band_metric(verdict="improved", delta=-5, n=8),
            "✓  -5.0%  ±2.5%  n=8",
            id="band",
        ),
        pytest.param(
            exact_metric(delta=-7.9, n=6, unit="ns"),
            "✓  -7.9%  n=6",
            id="exact",
        ),
    ],
)
def test_render_report_when_metric_paired_fewer_rounds_does_annotate_with_pair_count(
    metric: MetricComparison, expected: str
):
    result = create_comparison_result(samples=10, metrics={"decode/time": metric})

    row = line_starting_with(render_report(result), "decode/time")

    assert stripped_cells(row)[-1] == expected


# ---------------------------------------------------------------------------
# value column alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metrics", "first", "second"),
    [
        pytest.param(
            {
                "first/metric": permutation_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=162000,
                    baseline_spread=9,
                    unit="ns",
                ),
                "second/metric": permutation_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=29200,
                    baseline_spread=12,
                    unit="ns",
                ),
            },
            "162.0µs ±  9%",
            " 29.2µs ± 12%",
            id="percentage-spreads",
        ),
        pytest.param(
            {
                "first/metric": permutation_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=5,
                    baseline_spread=7620,
                    unit="bytes",
                ),
                "second/metric": permutation_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=49152,
                    baseline_spread=1,
                    unit="bytes",
                ),
            },
            "    5B ± 381B",
            "49.2KB ±   1%",
            id="absolute-beside-percentage",
        ),
    ],
)
def test_render_report_when_aligning_value_columns_does_stack_magnitude_and_spread(
    metrics: dict[str, MetricComparison], first: str, second: str
):
    report = render_report(create_comparison_result(metrics=metrics))
    first_cell = cells_of(line_starting_with(report, "first/metric"))[1]
    second_cell = cells_of(line_starting_with(report, "second/metric"))[1]

    assert first in first_cell
    assert second in second_cell
    assert first_cell.index("±") == second_cell.index("±")


def test_render_report_when_a_magnitude_has_no_spread_does_keep_it_in_the_magnitude_field():
    report = render_report(
        create_comparison_result(
            metrics={
                "first/metric": permutation_metric(
                    verdict="improved",
                    delta=-10,
                    baseline_median=2048,
                    baseline_spread=2,
                    unit="ns",
                ),
                "second/metric": exact_metric(
                    delta=0, unit=None, baseline_median=120, short_name="second/metric"
                ),
            }
        )
    )
    first_cell = cells_of(line_starting_with(report, "first/metric"))[1]
    second_cell = cells_of(line_starting_with(report, "second/metric"))[1]

    assert first_cell.index("2.0µs") + len("2.0µs") == second_cell.index("120") + len("120")


# ---------------------------------------------------------------------------
# verdict column alignment
# ---------------------------------------------------------------------------


def test_render_report_when_aligning_the_verdict_column_does_lay_out_glyph_delta_and_band():
    result = create_comparison_result(
        metrics={
            "regressed/time": permutation_metric(
                verdict="regressed", delta=0.4, noise_pct=2.5, unit="ns"
            ),
            "flat/time": permutation_metric(verdict="no-signal", delta=0, noise_pct=100, unit="ns"),
            "improved/time": permutation_metric(
                verdict="improved", delta=-12.4, noise_pct=30, unit="ns"
            ),
        }
    )

    report = render_report(result)

    assert stripped_cells(line_starting_with(report, "regressed/time"))[-1] == "✗   +0.4%  ±  2.5%"
    assert stripped_cells(line_starting_with(report, "flat/time"))[-1] == "~    0.0%  ±100.0%"
    assert stripped_cells(line_starting_with(report, "improved/time"))[-1] == "✓  -12.4%  ± 30.0%"


# ---------------------------------------------------------------------------
# aggregate noise band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        pytest.param(
            create_comparison_result(
                samples=1,
                metrics={"decode/time": band_metric(delta=-0.4, noise_pct=0.5, n=1, unit="ns")},
                candidates=[create_candidate(kinds=[other_kind(-0.1, 1, band=0.5)])],
            ),
            "-0.1%  ±0.5%",
            id="aggregate-only-band",
        ),
        pytest.param(
            create_comparison_result(
                metrics={"faster/time": permutation_metric(verdict="improved", delta=-17.5)},
                candidates=[create_candidate(kinds=[other_kind(-5.8, 1, band=1.2)])],
            ),
            "-5.8%  ±1.2%",
            id="row-and-aggregate-band",
        ),
    ],
)
def test_render_report_when_the_aggregate_carries_a_band_does_state_it_within_the_verdict_column(
    result: ComparisonResult, expected: str
):
    report = render_report(result)
    row = line_starting_with(report, "geomean")
    rule = rule_lines(report)[0]

    assert delta_cell(row).strip() == expected
    assert len(strip_ansi(row).rstrip()) <= len(rule)


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
                kinds=[gating_kind("time", geomean, {"[bold]entity": geomean_of(-3.1, 2)})]
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
# sectioned aggregates
# ---------------------------------------------------------------------------


def _several_kinds_gate() -> ComparisonResult:
    """The two-kind layout with the ``memory`` kind gating too."""
    metrics = dict(two_kind_metrics())
    encode = metrics["encode#memory"]
    metrics["encode#memory"] = replace(encode, meta=replace(encode.meta, gating=True))
    return create_comparison_result(
        metrics=metrics,
        candidates=[
            create_candidate(
                kinds=[time_kind(), replace(memory_kind(), gated_geomean=geomean_of(6.1, 1))]
            )
        ],
    )


def _no_kind_gates() -> ComparisonResult:
    """The two-kind layout with no ``time`` metric gating, so no kind gates."""
    metrics = dict(two_kind_metrics())
    for name in ("entity/alive_check#time", "entity/spawn#time", "warmup#time"):
        entry = metrics[name]
        metrics[name] = replace(entry, meta=replace(entry.meta, gating=False))
    return replace(
        two_kind_result(kinds=[without_gated_geomean(time_kind()), memory_kind()]),
        metrics=metrics,
    )


@pytest.mark.parametrize(
    "make_result",
    [
        pytest.param(_several_kinds_gate, id="several-kinds-gate"),
        pytest.param(_no_kind_gates, id="no-kind-gates"),
    ],
)
def test_render_report_when_closing_a_sectioned_table_does_end_on_the_last_geomean(
    make_result: Callable[[], ComparisonResult],
):
    report = render_report(make_result())

    assert table_region(report)[-1] == "geomean · memory (1)"
