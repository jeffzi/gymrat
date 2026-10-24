"""Tests for the single-candidate comparison text report, flat and sectioned.

A single-kind comparison renders flat (no section borders); several kinds render
one bordered section each. These tests verify that group headers carry a kind
suffix, member rows use case names, a row paired over
fewer rounds is annotated with its pair count and those counts align across rows
and sections, and a sectioned table closes on its last geomean. Names and labels
carrying square brackets print as written.

They also pin the table's column alignment: the metric column sized in terminal
cells for CJK names, the ``±`` offset shared across value cells, and the glyph,
delta, band, pair count and detectable change laid out across verdict cells,
aggregate rows included. A within-noise row states its detectable change last,
and no other row moves to make room for it. Alignment
is asserted *within* a parsed cell, since the box chrome is rich's rather than a
hand-spliced grid.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import pytest
from rich.cells import cell_len

from gymrat.model import Exclusion
from gymrat.report.text.render import render_report
from gymrat.report.text.single import scoped_geomean_label
from gymrat.report.types import ReportOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.model import GeomeanResult
    from gymrat.report.types import ComparisonResult, MetricComparison
from tests._ansi import strip_ansi
from tests.report._assertions import (
    cells_of,
    delta_cell,
    line_containing,
    line_starting_with,
    offsets_of,
    rule_lines,
    sgr_for,
    stripped_cells,
    styles_at,
    table_region,
    table_rows,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    every_class_metrics,
    exact_metric,
    gating_kind,
    kind_metric,
    memory_kind,
    mixed_methods_result,
    other_kind,
    permutation_metric,
    single_sample_result,
    time_kind,
    two_kind_metrics,
    two_kind_result,
    without_gated_geomean,
)
from tests.report._verdicts import band_metric, geomean_of

# ---------------------------------------------------------------------------
# group headers and case names
# ---------------------------------------------------------------------------


def test_render_report_when_flat_body_has_groups_does_head_each_group_with_its_kind_and_list_case_names():
    geomean = geomean_of(-2, 5)
    result = create_comparison_result(
        metrics={
            "node/get#time": kind_metric(
                kind="time", short_name="node.get", verdict="improved", delta=-5
            ),
            "node/set#time": kind_metric(
                kind="time",
                short_name="node.set",
                verdict="no-signal",
                delta=0.1,
            ),
            "entity/spawn#time": kind_metric(
                kind="time",
                short_name="entity.spawn",
                verdict="regressed",
                delta=4,
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
# pair counts: annotated per row, aligned across rows and sections
# ---------------------------------------------------------------------------


def _detecting(metric: MetricComparison, pct: float) -> MetricComparison:
    """``metric`` with its sole candidate's verdict carrying a ``pct`` detectable change."""
    candidate = metric.candidates[0]
    if candidate.verdict is None or candidate.verdict.method == "exact":
        msg = f"no approximate verdict to carry a detectable change on {metric!r}"
        raise AssertionError(msg)
    return replace(
        metric,
        candidates=(replace(candidate, verdict=replace(candidate.verdict, detectable_pct=pct)),),
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


def _detecting_beside_regression_result() -> ComparisonResult:
    """A within-noise row carrying a detectable change beside a regression, both short a pair."""
    return create_comparison_result(
        metrics={
            "flat/time": _detecting(permutation_metric(verdict="no-signal", delta=0.3, n=8), 2.1),
            "slower/time": permutation_metric(verdict="regressed", delta=12.4, n=8),
        },
    )


@pytest.mark.parametrize(
    ("make_result", "first_row", "second_row"),
    [
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
        pytest.param(
            _detecting_beside_regression_result,
            "flat/time",
            "slower/time",
            id="detectable-change-beside-regression",
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
# CJK metric names — column widths measured in terminal cells
# ---------------------------------------------------------------------------


def test_render_report_when_metric_name_has_cjk_does_align_separator_columns():
    result = create_comparison_result(
        metrics={
            "ascii-name": permutation_metric(
                verdict="improved", delta=-10, baseline_median=1000, unit="ns"
            ),
            "测试指标": permutation_metric(
                verdict="regressed", delta=5, baseline_median=2000, unit="ns"
            ),
        }
    )

    report = render_report(result)
    ascii_row = line_starting_with(report, "ascii-name")
    cjk_row = line_starting_with(report, "测试指标")

    # Column separators must sit at the same terminal cell position in both rows,
    # meaning the label column accounted for double-width CJK cells.
    ascii_sep = cell_len(ascii_row[: ascii_row.index("│")])
    cjk_sep = cell_len(cjk_row[: cjk_row.index("│")])
    assert ascii_sep == cjk_sep


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
# detectable change on within-noise rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        pytest.param(
            permutation_metric(verdict="no-signal", delta=0.3),
            "~  +0.3%  ±2.5%  detects ≥2.1%",
            id="permutation",
        ),
        pytest.param(
            band_metric(verdict="no-signal", delta=-1.0, n=10),
            "~  -1.0%  ±2.5%  detects ≥2.1%",
            id="band",
        ),
        pytest.param(
            permutation_metric(verdict="no-signal", delta=0.3, n=8),
            "~  +0.3%  ±2.5%  n=8  detects ≥2.1%",
            id="with-pair-count",
        ),
    ],
)
def test_render_report_when_within_noise_row_carries_a_detectable_change_does_state_it_last(
    metric: MetricComparison, expected: str
):
    result = create_comparison_result(metrics={"flat/time": _detecting(metric, 2.1)})

    row = line_starting_with(render_report(result), "flat/time")

    assert stripped_cells(row)[-1] == expected


def _every_class_short_of_pairs() -> dict[str, MetricComparison]:
    """One metric per display class, each showing a pair count."""
    return {
        name: metric if name == "single-pair/time" else _short_pairs(metric, 8)
        for name, metric in every_class_metrics().items()
    }


def _verdict_cells(result: ComparisonResult) -> dict[str, str]:
    """Each table row's verdict cell, padding stripped, keyed by its first cell."""
    rows = [stripped_cells(row) for row in table_rows(render_report(result))]
    return {cells[0]: cells[-1] for cells in rows}


def test_render_report_when_one_row_carries_a_detectable_change_does_leave_every_other_row_as_before():
    metrics = _every_class_short_of_pairs()
    before = _verdict_cells(create_comparison_result(metrics=metrics))
    detecting = {**metrics, "flat/time": _detecting(metrics["flat/time"], 2.1)}

    after = _verdict_cells(create_comparison_result(metrics=detecting))

    assert sum("n=" in cell for cell in before.values()) >= 5
    del before["flat/time"], after["flat/time"]
    assert after == before


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            _detecting(permutation_metric(verdict="no-signal", delta=0.3), 2.1),
            _detecting(permutation_metric(verdict="no-signal", delta=-1.0), 12.5),
            id="figures-differ-in-width",
        ),
        pytest.param(
            _detecting(permutation_metric(verdict="no-signal", delta=0.3, n=8), 2.1),
            _detecting(permutation_metric(verdict="no-signal", delta=-1.0), 2.1),
            id="one-shows-a-pair-count",
        ),
    ],
)
def test_render_report_when_within_noise_rows_carry_detectable_changes_does_align_them(
    first: MetricComparison, second: MetricComparison
):
    result = create_comparison_result(metrics={"first/time": first, "second/time": second})

    report = strip_ansi(render_report(result))

    first_line = line_starting_with(report, "first/time")
    second_line = line_starting_with(report, "second/time")
    first_offsets = offsets_of(first_line, "detects ≥")
    assert first_offsets != []
    assert first_offsets == offsets_of(second_line, "detects ≥")
    assert first_line.rindex("%") == second_line.rindex("%")


def test_render_report_when_every_class_carries_a_detectable_change_does_state_it_on_within_noise_rows_only():
    metrics = {name: _detecting(metric, 2.1) for name, metric in every_class_metrics().items()}
    result = create_comparison_result(metrics=metrics)

    rows = [stripped_cells(row) for row in table_rows(render_report(result))]

    assert [cells[0] for cells in rows if "detects" in cells[-1]] == ["flat/time"]


def test_render_report_when_colored_does_dim_the_detectable_change():
    result = create_comparison_result(
        metrics={"flat/time": _detecting(permutation_metric(verdict="no-signal", delta=0.3), 2.1)}
    )

    row = line_containing(render_report(result, ReportOptions(color=True)), "flat/time")

    assert strip_ansi(stripped_cells(row)[-1]) == "~  +0.3%  ±2.5%  detects ≥2.1%"
    assert styles_at(row, "detects") == sgr_for("dim")


# ---------------------------------------------------------------------------
# aggregate noise band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        pytest.param(
            replace(
                single_sample_result(),
                candidates=(create_candidate(kinds=[other_kind(-0.1, 2, band=0.5)]),),
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
def test_scoped_geomean_label_when_exclusions_vary_does_count_kept_metrics_against_total(
    scope: str, geomean: GeomeanResult, expected: str
):
    assert scoped_geomean_label(scope, geomean) == expected
