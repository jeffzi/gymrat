"""Tests for the multi-candidate comparison text report.

These cover the table-focused cases: the candidate-per-column table, its
per-candidate aggregate cells, the sectioned layout shared with the
single-candidate table, and how all of it is colored. The summary, highlight,
and footer blocks belong to the report-assembly task and are left out here.

Column alignment is asserted by byte offset here, since the multi-candidate
columns are laid out on a fixed-width grid whose separators must stack across
sections.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat.config import KindEntry
from gymrat.model import Exclusion
from gymrat.report.text import render_report
from gymrat.report.types import ReportOptions
from gymrat.verdict import KindAggregate
from tests.report._inputs import (
    DIMMED_LINE,
    NWayCandidate,
    cells_of,
    create_candidate,
    create_comparison_result,
    geomean_of,
    grouped_comparison,
    highlight_lines,
    last_table_row,
    line_containing,
    line_starting_with,
    memory_kind,
    multi_candidate_result,
    n_way_kind_metric,
    n_way_metric,
    offsets_of,
    other_kind,
    permutation_metric,
    separator_offsets,
    separator_styles,
    strip_ansi,
    styles_at,
    table_region,
    time_kind,
    two_kind_metrics,
    two_kind_result,
    without_gated_geomean,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.report.types import ComparisonResult


def _value_part_of(cell: str, glyph: str) -> str:
    """The part of ``cell`` ahead of its verdict ``glyph`` — the value and its padding.

    A candidate column packs a value and a verdict into one cell, so this is the
    half that carries no verdict of its own and must stay unstyled. The escape
    sequences opening the verdict's own style sit right in front of the glyph, so
    they are trimmed off the tail rather than counted against the value.
    """
    index = cell.find(glyph)
    if index == -1:
        msg = f"no {glyph!r} in cell: {cell!r}"
        raise AssertionError(msg)
    return re.sub(r"(?:\x1b\[\d+m)*$", "", cell[:index])


# ---------------------------------------------------------------------------
# candidate columns
# ---------------------------------------------------------------------------


def test_render_report_when_many_candidates_does_head_one_column_per_candidate():
    header_line = line_starting_with(render_report(multi_candidate_result()), "metric")

    assert [cell.strip() for cell in cells_of(header_line)] == [
        "metric",
        "main",
        "candidate-a",
        "candidate-b",
        "candidate-c",
    ]


def test_render_report_when_many_candidates_does_pair_each_figure_with_its_own_verdict():
    row = line_starting_with(render_report(multi_candidate_result()), "decode/time")

    assert [cell.strip() for cell in cells_of(row)] == [
        "decode/time",
        "100ns ± 1%",
        "90ns ± 1%  ✓  -10.0%",
        "104ns ± 1%  ✗  +4.0%",
        "150ns ± 3%  ≈  unstable",
    ]


def test_render_report_when_many_candidates_does_carry_one_geomean_per_column():
    row = line_starting_with(render_report(multi_candidate_result()), "geomean")

    assert [cell.strip() for cell in cells_of(row)] == [
        "geomean",
        "",
        "-10.0% · 1 stable metric",
        "+4.0% · 1 stable metric",
        "0.0% · 1 stable metric",
    ]


def test_render_report_when_many_candidates_does_close_the_table_on_the_geomean_row():
    row = last_table_row(render_report(multi_candidate_result()))

    assert cells_of(row)[0].strip() == "geomean"


def test_render_report_when_many_candidates_does_stack_separators_across_the_table():
    report = render_report(multi_candidate_result())
    header_offsets = separator_offsets(line_starting_with(report, "metric"))

    assert separator_offsets(line_starting_with(report, "decode/time")) == header_offsets
    assert separator_offsets(line_starting_with(report, "geomean")) == header_offsets


def test_render_report_when_many_candidates_does_size_the_last_column_to_fit_its_aggregate():
    bare = strip_ansi(render_report(multi_candidate_result(2)))
    rules = [line for line in bare.split("\n") if re.match(r"^─+┼", line)]
    geomean_line = line_starting_with(bare, "geomean")

    assert rules
    for rule in rules:
        assert len(rule) >= len(geomean_line)


# ---------------------------------------------------------------------------
# candidate column color
# ---------------------------------------------------------------------------


def _dimming_result() -> ComparisonResult:
    """A two-candidate run whose two rows differ in whether any candidate moved.

    ``mixed/time`` moved for one candidate and stayed flat for the other, which
    is exactly the row a per-candidate dimming rule would wrongly recede.
    """
    return create_comparison_result(
        candidates=[
            create_candidate(label="candidate-a"),
            create_candidate(label="candidate-b"),
        ],
        metrics={
            "flat/time": n_way_metric(
                [
                    NWayCandidate(verdict="no-signal", delta=0.3, median=100),
                    NWayCandidate(verdict="unstable", delta=-50, median=50),
                ]
            ),
            "mixed/time": n_way_metric(
                [
                    NWayCandidate(verdict="no-signal", delta=0.3, median=100),
                    NWayCandidate(verdict="improved", delta=-17.5, median=83),
                ]
            ),
        },
    )


@pytest.mark.parametrize("row", ["flat/time", "mixed/time"])
def test_render_report_when_colored_does_leave_a_mixed_row_without_end_to_end_dim(
    monkeypatch: pytest.MonkeyPatch, row: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    line = line_containing(render_report(_dimming_result()), row)

    assert not DIMMED_LINE.match(line)


def test_render_report_when_colored_does_style_each_cell_verdict_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    row = line_containing(render_report(_dimming_result()), "flat/time")

    assert "2" in styles_at(row, "~")
    assert "33" in styles_at(row, "≈")


def test_render_report_when_colored_does_leave_name_and_values_plain_on_a_quiet_row(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    cells = cells_of(line_containing(render_report(_dimming_result()), "flat/time"))

    assert "\x1b[" not in "│".join(cells[:2])
    assert "\x1b[" not in _value_part_of(cells[2], "~")
    assert "\x1b[" not in _value_part_of(cells[3], "≈")


@pytest.mark.parametrize(
    ("column", "glyph"),
    [
        pytest.param(2, "✓", id="improved"),
        pytest.param(3, "✗", id="regressed"),
        pytest.param(4, "≈", id="unstable"),
    ],
)
def test_render_report_when_colored_does_leave_a_cell_value_plain(
    monkeypatch: pytest.MonkeyPatch, column: int, glyph: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    row = line_containing(render_report(multi_candidate_result()), "decode/time")

    assert "\x1b[" not in _value_part_of(cells_of(row)[column], glyph)


def test_render_report_when_colored_does_paint_an_unstable_cell_verdict_amber(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    row = line_containing(render_report(multi_candidate_result()), "decode/time")

    assert "33" in styles_at(row, "≈")
    assert "33" in styles_at(row, "unstable")


def test_render_report_when_colored_does_pad_on_plain_text_so_columns_line_up(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    bare = strip_ansi(render_report(multi_candidate_result()))
    header_offsets = separator_offsets(line_starting_with(bare, "metric"))

    assert separator_offsets(line_starting_with(bare, "decode/time")) == header_offsets
    assert separator_offsets(line_starting_with(bare, "geomean")) == header_offsets


def test_render_report_when_colored_does_color_glyph_and_delta_together(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    row = line_containing(render_report(multi_candidate_result()), "decode/time")

    assert "32" in styles_at(row, "✓")
    assert "32" in styles_at(row, "-10.0%")
    assert "31" in styles_at(row, "✗")
    assert "31" in styles_at(row, "+4.0%")


def test_render_report_when_colored_does_dim_the_quiet_segment_on_a_bright_row(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    row = line_containing(render_report(_dimming_result()), "mixed/time")

    assert "2" in styles_at(row, "~")
    assert "2" in styles_at(row, "+0.3%")


def test_render_report_when_colored_does_dim_the_provenance_in_geomean_cells(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    geomean = line_containing(render_report(multi_candidate_result()), "geomean")

    assert "2" in styles_at(geomean, "1 stable metric")


# ---------------------------------------------------------------------------
# sectioned layout
# ---------------------------------------------------------------------------


def test_render_report_when_many_kinds_does_give_each_a_titled_section():
    assert table_region(render_report(two_kind_result())) == [
        "gymrat compare · baseline main ↔ perf/faster-decode · 10 paired samples · adapter: mitata",
        "",
        "<border>",
        "time",
        "<rule>",
        "entity",
        "alive_check",
        "spawn",
        "geomean · entity (2)",
        "",
        "warmup",
        "<rule>",
        "geomean · time (3)",
        "",
        "informational — gating off (config: kinds.memory.gating = false)",
        "<border>",
        "memory",
        "<rule>",
        "encode",
        "<rule>",
        "geomean · memory (1)",
    ]


def test_render_report_when_many_kinds_does_span_the_top_border_across_the_full_width():
    report = strip_ansi(render_report(two_kind_result()))
    lines = report.split("\n")
    header = line_starting_with(report, "time")
    header_index = lines.index(header)
    border = lines[header_index - 1]
    rule = lines[header_index + 1]

    assert "┼" not in border
    assert len(border) == len(rule)


def test_render_report_when_many_kinds_does_join_the_top_border_to_the_columns():
    report = strip_ansi(render_report(two_kind_result()))
    lines = report.split("\n")
    header = line_starting_with(report, "time")
    border = lines[lines.index(header) - 1]

    assert offsets_of(border, "┬") == separator_offsets(header)


def test_render_report_when_many_kinds_does_line_every_section_up():
    report = render_report(two_kind_result())
    bare = strip_ansi(report)
    offsets = separator_offsets(line_starting_with(bare, "time"))

    assert separator_offsets(line_starting_with(bare, "memory")) == offsets
    assert separator_offsets(line_starting_with(report, "  alive_check")) == offsets
    assert separator_offsets(line_starting_with(report, "entity ")) == offsets
    assert separator_offsets(line_starting_with(report, "geomean · memory")) == offsets


@pytest.mark.parametrize(
    ("make_result", "expected"),
    [
        pytest.param(
            two_kind_result,
            "informational — gating off (config: kinds.memory.gating = false)",
            id="kind-level-config",
        ),
        pytest.param(
            lambda: replace(two_kind_result(), config_kinds=None),
            "informational — gating off",
            id="per-metric-overrides",
        ),
    ],
)
def test_render_report_when_kind_is_informational_does_credit_the_config_source(
    make_result: Callable[[], ComparisonResult], expected: str
):
    report = render_report(make_result())

    assert line_containing(report, "informational") == expected


@pytest.mark.parametrize(
    "row",
    [
        pytest.param("  alive_check", id="grouped-indented"),
        pytest.param("warmup", id="bare-short-name"),
    ],
)
def test_render_report_when_many_kinds_does_name_a_metric_row(row: str):
    line = line_starting_with(render_report(two_kind_result()), row)

    assert cells_of(line)[0].rstrip() == row


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
        pytest.param(two_kind_result, id="one-kind-gates"),
        pytest.param(_several_kinds_gate, id="several-kinds-gate"),
        pytest.param(_no_kind_gates, id="no-kind-gates"),
    ],
)
def test_render_report_when_closing_a_sectioned_table_does_end_on_the_last_geomean(
    make_result: Callable[[], ComparisonResult],
):
    report = render_report(make_result())

    assert table_region(report)[-1] == "geomean · memory (1)"
    assert "geomean · gated" not in report


def test_render_report_when_many_kinds_and_candidates_does_carry_one_figure_per_column():
    report = render_report(grouped_comparison())

    def cells_at(label: str) -> list[str]:
        return [cell.strip() for cell in cells_of(line_starting_with(report, label))]

    assert cells_at("geomean · entity") == [
        "geomean · entity",
        "",
        "-10.0% · 1 stable metric",
        "+4.0% · 1 stable metric",
    ]
    assert cells_at("geomean · time") == [
        "geomean · time",
        "",
        "-10.0% · 1 stable metric",
        "+4.0% · 1 stable metric",
    ]
    assert table_region(report)[-1] == "geomean · memory"


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
def test_render_report_when_colored_does_paint_an_improving_aggregate_green(
    monkeypatch: pytest.MonkeyPatch, label: str, value: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    line = line_containing(render_report(two_kind_result()), label)

    assert styles_at(line, value) == ["1", "32"]


def test_render_report_when_colored_does_embolden_the_kind_and_dim_the_tag(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    report = render_report(two_kind_result())
    header = next(
        line
        for line in report.split("\n")
        if "│" in line and strip_ansi(line).lstrip().startswith("memory")
    )

    assert styles_at(header, "memory") == ["1"]
    assert styles_at(line_containing(report, "informational"), "informational") == ["2"]


def test_render_report_when_colored_does_leave_separators_in_the_default_color(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    rows = [line for line in render_report(two_kind_result()).split("\n") if "│" in line]
    inherited = [row for row in rows if any(styles for styles in separator_styles(row))]

    assert inherited == []


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


def test_render_report_when_ordering_does_emit_table_summary_highlights_and_close():
    lines = render_report(_ordered_result()).split("\n")

    assert "gymrat compare · baseline main ↔ faster" in lines[0]
    assert re.match(r"^metric\s+│", lines[1])
    assert re.match(r"^─+┼", lines[2])
    assert "metric1/time" in lines[3]
    assert "metric2/time" in lines[4]
    assert re.match(r"^─+┼", lines[5])
    assert "geomean" in lines[6]
    assert lines[7] == ""
    assert "✓ 1 improved" in lines[8]
    assert lines[9] == ""
    assert lines[10] == "highlights"
    assert "metric1/time" in lines[11]
    assert len(lines) == 12


def test_render_report_when_verbose_does_add_the_method_block_below_a_blank_line():
    lines = render_report(_ordered_result(), ReportOptions(verbose=True)).split("\n")

    assert "metric1/time" in lines[11]
    assert lines[12] == ""
    assert "sign-flip permutation test" in lines[13]
    assert len(lines) == 14


# ---------------------------------------------------------------------------
# per-candidate summary and highlights
# ---------------------------------------------------------------------------


def test_render_report_when_many_candidates_does_summarize_each_on_its_own_line():
    summaries = [
        line
        for line in render_report(multi_candidate_result()).split("\n")
        if re.search(r"✓ \d+ improved", line)
    ]

    assert summaries == [
        (
            "candidate-a  ✓ 1 improved   ✗ 0 regressed   ≈ 0 unstable   "
            "= 0 identical   ~ 0 within noise   ? 0 inconclusive"
        ),
        (
            "candidate-b  ✓ 0 improved   ✗ 1 regressed   ≈ 0 unstable   "
            "= 0 identical   ~ 0 within noise   ? 0 inconclusive"
        ),
        (
            "candidate-c  ✓ 0 improved   ✗ 0 regressed   ≈ 1 unstable   "
            "= 0 identical   ~ 0 within noise   ? 0 inconclusive"
        ),
    ]


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


def test_render_report_when_no_candidate_has_a_highlight_does_drop_the_section():
    result = create_comparison_result(
        candidates=[
            create_candidate(label="candidate-a"),
            create_candidate(label="candidate-b"),
        ],
        metrics={
            "decode/time": n_way_metric(
                [
                    NWayCandidate(verdict="no-signal", delta=0.4, median=100),
                    NWayCandidate(verdict="no-signal", delta=-0.3, median=100),
                ]
            ),
        },
    )

    report = render_report(result)

    assert "highlights" not in report
    assert "candidate-a  ✓ 0 improved" in report


@pytest.mark.parametrize(
    "label",
    [
        pytest.param("candidate-a", id="a"),
        pytest.param("candidate-b", id="b"),
        pytest.param("candidate-c", id="c"),
    ],
)
def test_render_report_when_colored_does_embolden_the_candidate_summary_label(
    monkeypatch: pytest.MonkeyPatch, label: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    summary = next(
        line
        for line in render_report(multi_candidate_result()).split("\n")
        if re.search(r"✓ \d+ improved", strip_ansi(line)) and label in strip_ansi(line)
    )

    assert "1" in styles_at(summary, label)


def test_render_report_when_colored_does_embolden_the_candidate_highlight_sub_labels(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    highlights = highlight_lines(render_report(multi_candidate_result()))
    sub_labels = [
        line
        for line in highlights
        if strip_ansi(line).strip() in {"candidate-a", "candidate-b", "candidate-c"}
    ]

    assert len(sub_labels) == 3
    assert all("\x1b[1m" in sub_label for sub_label in sub_labels)


# ---------------------------------------------------------------------------
# sectioned highlights: kind-prefixed names
# ---------------------------------------------------------------------------


def test_render_report_when_sectioned_does_name_each_highlight_by_kind_and_short_metric():
    highlights = [line.strip() for line in highlight_lines(render_report(two_kind_result()))]

    assert highlights == [
        "✗ time · entity.spawn         +4.0%",
        "✓ time · entity.alive_check  -10.0%",
        "✓ memory · encode             -7.0%",
    ]


def test_render_report_when_sectioned_and_many_candidates_does_prefix_the_kind_per_subsection():
    result = create_comparison_result(
        metrics={
            "entity/alive_check#time": n_way_kind_metric(
                kind="time",
                short_name="entity.alive_check",
                candidates=[
                    NWayCandidate(verdict="improved", delta=-10, median=90),
                    NWayCandidate(verdict="regressed", delta=4, median=104),
                ],
            ),
            "encode#memory": n_way_kind_metric(
                kind="memory",
                short_name="encode",
                gating=False,
                candidates=[
                    NWayCandidate(verdict="improved", delta=-7, median=93),
                    NWayCandidate(verdict="improved", delta=-2, median=98),
                ],
            ),
        },
        candidates=[
            create_candidate(
                label="candidate-a",
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean_of(-10, 1),
                        groups=(),
                        gated_geomean=geomean_of(-10, 1),
                    ),
                    memory_kind(),
                ],
            ),
            create_candidate(
                label="candidate-b",
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=geomean_of(4, 1),
                        groups=(),
                        gated_geomean=geomean_of(4, 1),
                    ),
                    KindAggregate(
                        kind="memory", geomean=geomean_of(-2, 1), groups=(), gated_geomean=None
                    ),
                ],
            ),
        ],
        config_kinds={"memory": KindEntry(gating=False)},
    )

    assert highlight_lines(render_report(result)) == [
        "  candidate-a",
        "    ✓ time · entity.alive_check  -10.0%",
        "    ✓ memory · encode             -7.0%",
        "  candidate-b",
        "    ✗ time · entity.alive_check   +4.0%",
        "    ✓ memory · encode             -2.0%",
    ]
