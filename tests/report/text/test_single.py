"""Tests for the single-candidate comparison text report, flat and sectioned.

A single-kind comparison renders flat (no section borders); several kinds render
one bordered section each. These tests verify that group headers carry a kind
suffix, member rows use case names in first-appearance order, pair counts align
across rows and sections, excluded metrics are counted into an aggregate's
provenance, and a sectioned table closes on its last geomean. Names and labels
carrying square brackets print as written.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat.model import Exclusion
from gymrat.report.text.render import render_report
from gymrat.report.types import CandidateMetric, MetricComparison

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.report.types import ComparisonResult
from tests._ansi import strip_ansi
from tests.report._assertions import (
    line_containing,
    line_starting_with,
    offsets_of,
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
    metric_meta,
    permutation_metric,
    time_kind,
    two_kind_metrics,
    two_kind_result,
    without_gated_geomean,
)
from tests.report._verdicts import exact_verdict, geomean_of

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


def _unstable_with_band_result() -> ComparisonResult:
    """A banded improvement beside an unstable metric, both short a pair of the run's ten."""
    return create_comparison_result(
        metrics={
            "latency#other": permutation_metric(verdict="improved", delta=-10, n=8),
            "flaky#other": permutation_metric(verdict="unstable", delta=50, n=8),
        },
    )


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
        pytest.param(_mixed_band_result, "-10.0%", "-5.0%", id="exact-row-without-band"),
        pytest.param(
            _undefined_ratio_result, "decode#other", "nan-delta#other", id="undefined-ratio-row"
        ),
        pytest.param(_sectioned_short_pairs_result, "  spawn", "encode", id="across-sections"),
        pytest.param(
            _unstable_with_band_result, "latency#other", "flaky#other", id="word-wider-than-delta"
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


def test_render_report_when_metrics_are_excluded_does_count_them_into_the_provenance():
    result = replace(
        two_kind_result(),
        candidates=(
            create_candidate(
                kinds=[
                    replace(
                        time_kind(),
                        geomean=geomean_of(
                            -3.2,
                            2,
                            excluded=[Exclusion(metric="warmup#time", reason="unstable")],
                        ),
                    ),
                    memory_kind(),
                ]
            ),
        ),
    )

    row = line_starting_with(render_report(result), "geomean · time")

    assert stripped_cells(row)[0] == "geomean · time (2/3)"


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
        two_kind_result(),
        metrics=metrics,
        candidates=(create_candidate(kinds=[without_gated_geomean(time_kind()), memory_kind()]),),
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
