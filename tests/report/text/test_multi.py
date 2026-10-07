"""Tests for the multi-candidate comparison text report.

These cover the candidate-per-column table and its per-candidate aggregate
cells, how its cells are colored, the sectioned layout, the per-candidate
highlights, and the verbose method footer.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat.model import Exclusion
from gymrat.report.text.render import render_report
from gymrat.report.types import ReportOptions
from gymrat.verdict import GroupAggregate, KindAggregate
from tests._ansi import (
    TRAILING_SGR_RUN,
    strip_ansi,
)
from tests.report._assertions import (
    cells_of,
    highlight_lines,
    line_containing,
    line_starting_with,
    styles_at,
    table_region,
    table_rows,
)
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    create_comparison_result,
    memory_kind,
    multi_candidate_result,
    n_way_kind_metric,
    n_way_metric,
    other_kind,
    permutation_metric,
    time_kind,
    two_kind_metrics,
    two_kind_result,
    without_gated_geomean,
)
from tests.report._verdicts import geomean_of

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.report.types import ComparisonResult


def _stripped_cells(line: str) -> list[str]:
    """The cells of `line`, stripped of their padding."""
    return [cell.strip() for cell in cells_of(line)]


# ---------------------------------------------------------------------------
# candidate columns
# ---------------------------------------------------------------------------


def test_render_report_when_many_candidates_does_pair_each_figure_with_its_own_verdict():
    row = line_starting_with(render_report(multi_candidate_result()), "decode/time")

    assert _stripped_cells(row) == [
        "decode/time",
        "100ns ± 1%",
        "90ns ± 1%  ✓  -10.0%",
        "104ns ± 1%  ✗  +4.0%",
        "150ns ± 3%  ≈  unstable",
    ]


def test_render_report_when_many_candidates_does_size_the_last_column_to_fit_its_aggregate():
    bare = strip_ansi(render_report(multi_candidate_result(2)))
    rules = [line for line in bare.split("\n") if re.match(r"^─+┼", line)]
    geomean_line = line_starting_with(bare, "geomean")

    assert rules
    for rule in rules:
        assert len(rule) >= len(geomean_line)


def _bracketed_result() -> ComparisonResult:
    """Two candidates whose labels, like the metric names, read as markup tags."""
    return create_comparison_result(
        candidates=[
            create_candidate(label="[bold]fast"),
            create_candidate(label="[dim]slow"),
        ],
        metrics={
            "[italic]decode/time": n_way_metric([
                NWayCandidate(verdict="improved", delta=-10, median=90),
                NWayCandidate(verdict="regressed", delta=4, median=104),
            ]),
        },
    )


def test_render_report_when_names_carry_brackets_does_print_them_literally():
    report = strip_ansi(render_report(_bracketed_result()))

    assert _stripped_cells(line_starting_with(report, "metric")) == [
        "metric",
        "main",
        "[bold]fast",
        "[dim]slow",
    ]
    assert _stripped_cells(line_starting_with(report, "[italic]decode/time")) == [
        "[italic]decode/time",
        "100ns ± 1%",
        "90ns ± 1%  ✓  -10.0%",
        "104ns ± 1%  ✗  +4.0%",
    ]


# ---------------------------------------------------------------------------
# candidate column color
# ---------------------------------------------------------------------------


def _dimming_result() -> ComparisonResult:
    """A two-candidate run with a quiet row and a row where one candidate moved."""
    return create_comparison_result(
        candidates=[
            create_candidate(label="candidate-a"),
            create_candidate(label="candidate-b"),
        ],
        metrics={
            "flat/time": n_way_metric([
                NWayCandidate(verdict="no-signal", delta=0.3, median=100),
                NWayCandidate(verdict="unstable", delta=-50, median=50),
            ]),
            "mixed/time": n_way_metric([
                NWayCandidate(verdict="no-signal", delta=0.3, median=100),
                NWayCandidate(verdict="improved", delta=-17.5, median=83),
            ]),
        },
    )


@pytest.mark.parametrize(
    ("make_result", "row", "token", "code"),
    [
        pytest.param(_dimming_result, "flat/time", "~", "2", id="quiet-glyph-dim"),
        pytest.param(_dimming_result, "flat/time", "≈", "33", id="unstable-glyph-amber"),
        pytest.param(
            multi_candidate_result, "decode/time", "unstable", "33", id="unstable-word-amber"
        ),
        pytest.param(_dimming_result, "mixed/time", "+0.3%", "2", id="quiet-delta-on-bright-row"),
    ],
)
def test_render_report_when_colored_does_style_each_cell_verdict_on_its_own(
    make_result: Callable[[], ComparisonResult], row: str, token: str, code: str
):
    line = line_containing(render_report(make_result(), ReportOptions(color=True)), row)

    assert code in styles_at(line, token)


def test_render_report_when_colored_does_leave_name_and_values_plain_on_a_quiet_row():
    cells = cells_of(
        line_containing(render_report(_dimming_result(), ReportOptions(color=True)), "flat/time")
    )

    assert "\x1b[" not in "│".join(cells[:2])
    assert "\x1b[" not in TRAILING_SGR_RUN.sub("", cells[2][: cells[2].index("~")])
    assert "\x1b[" not in TRAILING_SGR_RUN.sub("", cells[3][: cells[3].index("≈")])


# ---------------------------------------------------------------------------
# sectioned layout
# ---------------------------------------------------------------------------


def test_render_report_when_kind_is_informational_by_metric_overrides_does_say_so():
    report = render_report(replace(two_kind_result(), config_kinds=None))

    assert line_containing(report, "informational") == "informational — gating off"


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

    assert cells_of(row)[0].strip() == "geomean · time (2/3)"


def _several_kinds_gate() -> ComparisonResult:
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


def _sectioned_bracketed_result() -> ComparisonResult:
    """Two kinds, one grouped, whose baseline, kinds, group and short names read as markup tags."""
    return create_comparison_result(
        baseline_label="[dim]main",
        metrics={
            "[bold]entity/spawn#other": n_way_kind_metric(
                kind="[underline]time",
                short_name="[bold]entity.spawn",
                candidates=[
                    NWayCandidate(verdict="improved", delta=-10, median=90),
                    NWayCandidate(verdict="regressed", delta=4, median=104),
                ],
            ),
            "encode#other": n_way_kind_metric(
                kind="[strike]memory",
                short_name="[italic]encode",
                candidates=[
                    NWayCandidate(verdict="improved", delta=-7, median=93),
                    NWayCandidate(verdict="improved", delta=-2, median=98),
                ],
            ),
        },
        candidates=[
            create_candidate(
                label=label,
                kinds=[
                    KindAggregate(
                        kind="[underline]time",
                        geomean=(time_geomean := geomean_of(time_delta, 1)),
                        groups=(GroupAggregate(group="[bold]entity", geomean=time_geomean),),
                        gated_geomean=time_geomean,
                    ),
                    KindAggregate(
                        kind="[strike]memory",
                        geomean=(memory_geomean := geomean_of(memory_delta, 1)),
                        groups=(),
                        gated_geomean=memory_geomean,
                    ),
                ],
            )
            for label, time_delta, memory_delta in (
                ("candidate-a", -10, -7),
                ("candidate-b", 4, -2),
            )
        ],
    )


def test_render_report_when_sectioned_names_carry_brackets_does_print_them_literally():
    report = render_report(_sectioned_bracketed_result())

    assert [_stripped_cells(row)[:2] for row in table_rows(report)] == [
        ["[underline]time", "[dim]main"],
        ["[bold]entity", ""],
        ["spawn", "100ns ± 1%"],
        ["geomean · [bold]entity", ""],
        ["geomean · [underline]time", ""],
        ["[strike]memory", "[dim]main"],
        ["[italic]encode", "100ns ± 1%"],
        ["geomean · [strike]memory", ""],
    ]


# ---------------------------------------------------------------------------
# sectioned layout color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "value"),
    [
        pytest.param("geomean · entity", "-3.1%", id="sub-geomean"),
        pytest.param("geomean · time", "-3.2%", id="kind-geomean"),
    ],
)
def test_render_report_when_colored_does_paint_an_improving_aggregate_green(label: str, value: str):

    line = line_containing(render_report(two_kind_result(), ReportOptions(color=True)), label)

    assert styles_at(line, value) == ["1", "32"]


# ---------------------------------------------------------------------------
# section ordering: table, summary, highlights, method block
# ---------------------------------------------------------------------------


def _ordered_result() -> ComparisonResult:
    """A two-metric run whose only footer content is the permutation method line."""
    return create_comparison_result(
        baseline_label="main",
        metrics={
            "metric1/time": permutation_metric(verdict="improved", delta=-10, unit="ns"),
            "metric2/time": permutation_metric(
                verdict="no-signal", delta=2, gating=False, unit="ns"
            ),
        },
        candidates=[create_candidate(label="faster", kinds=[other_kind(-5, 1)])],
    )


def test_render_report_when_verbose_does_add_the_method_block_below_a_blank_line():
    lines = render_report(_ordered_result(), ReportOptions(verbose=True)).split("\n")

    method = next(i for i, line in enumerate(lines) if "sign-flip permutation test" in line)
    assert (lines[method - 1], method) == ("", len(lines) - 1)


# ---------------------------------------------------------------------------
# per-candidate summary and highlights
# ---------------------------------------------------------------------------


def test_render_report_when_many_candidates_does_group_highlights_per_candidate():
    highlights = highlight_lines(render_report(multi_candidate_result()))

    assert highlights == [
        "  candidate-a",
        "    ✓ decode/time  -10.0%",
        "  candidate-b",
        "    ✗ decode/time   +4.0%",
        "  candidate-c",
        "    ≈ decode/time  unstable  noise ±30.0%",
        "  unstable metrics won't stabilize with more samples",
    ]


# ---------------------------------------------------------------------------
# single-kind grouping with multiple candidates
# ---------------------------------------------------------------------------


def _flat_grouped_bracketed_result() -> ComparisonResult:
    """Two candidates, single kind, whose one group reads as a markup tag."""
    return create_comparison_result(
        metrics={
            "[bold]entity/spawn#other": n_way_kind_metric(
                kind="other",
                short_name="[bold]entity.spawn",
                candidates=[
                    NWayCandidate(verdict="improved", delta=-10, median=90),
                    NWayCandidate(verdict="regressed", delta=4, median=104),
                ],
            ),
            "[bold]entity/alive#other": n_way_kind_metric(
                kind="other",
                short_name="[bold]entity.alive",
                candidates=[
                    NWayCandidate(verdict="no-signal", delta=0.3, median=100),
                    NWayCandidate(verdict="improved", delta=-5, median=95),
                ],
            ),
        },
        candidates=[
            create_candidate(
                label=label,
                kinds=[
                    KindAggregate(
                        kind="other",
                        geomean=(geomean := geomean_of(delta, 1)),
                        groups=(GroupAggregate(group="[bold]entity", geomean=geomean),),
                        gated_geomean=geomean,
                    )
                ],
            )
            for label, delta in (("candidate-a", -10), ("candidate-b", 4))
        ],
    )


def test_render_report_when_flat_grouped_many_candidates_does_print_group_brackets_literally():
    report = render_report(_flat_grouped_bracketed_result())

    assert [_stripped_cells(row)[0] for row in table_rows(report)] == [
        "metric",
        "[bold]entity · other",
        "spawn",
        "alive",
        "geomean",
    ]
