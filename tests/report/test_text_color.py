"""Tests for the colored comparison and measurement text reports.

These cover the colored comparison and measurement reports. The comparison
block pins how the assembled report paints its verdict rows, run and column
headers, the verdict summary, the highlights block, and the verbose method
footer and hint. The measure block pins the measure report's header and tables.
Column alignment is checked by byte offset only where that proves cross-section
alignment; content is pinned by parsed cell.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat.model import Exclusion
from gymrat.report.text import render_measure_report, render_report
from gymrat.report.types import MeasurementResult, ReportOptions
from gymrat.targets import WorktreeRemovalFailure
from tests.report._inputs import (
    DIMMED_LINE,
    band_metric,
    cells_of,
    create_candidate,
    create_comparison_result,
    create_measurement_result,
    delta_cell,
    exact_metric,
    highlight_lines,
    line_containing,
    line_starting_with,
    measured_metric,
    other_kind,
    permutation_metric,
    separator_offsets,
    separator_styles,
    strip_ansi,
    styles_at,
    table_region,
    table_rows,
    two_kind_measurement,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.model import ApproximateVerdict
    from gymrat.report.types import ComparisonResult


def _summary_segment(summary: str, label: str) -> str:
    """The triple-space-delimited summary segment whose plain text contains *label*."""
    for seg in summary.split("   "):
        if label in strip_ansi(seg):
            return seg
    msg = f"no segment with {label!r} in summary: {strip_ansi(summary)!r}"
    raise AssertionError(msg)


def _colorful_result() -> ComparisonResult:
    """A run whose rows cover every verdict class, plus a geomean figure."""
    return create_comparison_result(
        metrics={
            "faster/time": permutation_metric(verdict="improved", delta=-17.5, unit="ns"),
            "slower/time": permutation_metric(verdict="regressed", delta=2.4, unit="ns"),
            "flat/time": permutation_metric(verdict="no-signal", delta=0.3, unit="ns"),
            "tied/heap": band_metric(verdict="no-signal", delta=-0.5, n=10, usable_n=0),
            "single-pair/time": band_metric(delta=-0.4, noise_pct=0.5, n=1, unit="ns"),
            "jittery/time": permutation_metric(verdict="unstable", delta=-50, noise_pct=30),
        },
        candidates=[
            create_candidate(
                kinds=[
                    other_kind(
                        -5.8,
                        3,
                        excluded=[Exclusion(metric="jittery/time", reason="unstable")],
                    )
                ]
            )
        ],
        worktrees_removed=1,
        worktrees_left_behind=[WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="is locked")],
        worktree_prune_error="fatal: not a git repository",
    )


def _flat_measurement() -> MeasurementResult:
    """A flat single-kind run of two metrics measured in nanoseconds."""
    return create_measurement_result(
        metrics={
            "decode/time": measured_metric(median=100, spread=1, unit="ns"),
            "encode/time": measured_metric(median=2048, spread=2, unit="ns"),
        }
    )


# ---------------------------------------------------------------------------
# run header
# ---------------------------------------------------------------------------


def test_render_measure_report_when_rendering_header_does_name_target_samples_adapter():
    result = create_measurement_result(label="experiment", samples=10, adapter="mitata")

    output = render_measure_report(result)

    assert "gymrat measure · experiment · 10 samples · adapter: mitata" in output


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        pytest.param(1, "· 1 sample ·", id="one"),
        pytest.param(2, "· 2 samples ·", id="many"),
    ],
)
def test_render_measure_report_when_header_counts_samples_does_match_noun(
    samples: int, expected: str
):
    output = render_measure_report(create_measurement_result(samples=samples))

    assert expected in output


# ---------------------------------------------------------------------------
# flat single-kind table
# ---------------------------------------------------------------------------


def test_render_measure_report_when_one_kind_does_draw_one_flat_table():
    assert table_region(render_measure_report(_flat_measurement())) == [
        "gymrat measure · main · 10 samples · adapter: mitata",
        "metric",
        "<rule>",
        "decode/time",
        "encode/time",
    ]


def test_render_measure_report_when_one_kind_does_label_value_column_with_the_target():
    header_line = line_starting_with(render_measure_report(_flat_measurement()), "metric")

    assert [cell.strip() for cell in cells_of(header_line)] == ["metric", "main"]


def test_render_measure_report_when_one_kind_does_state_each_median_in_its_unit():
    rows = table_rows(render_measure_report(_flat_measurement()))[1:]

    assert [[cell.strip() for cell in cells_of(row)] for row in rows] == [
        ["decode/time", "100ns ± 1%"],
        ["encode/time", "2.0µs ± 2%"],
    ]


# ---------------------------------------------------------------------------
# metric row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spread", "expected"),
    [
        pytest.param(1, "100ns ± 1%", id="with-spread"),
        pytest.param(None, "100ns", id="no-spread"),
    ],
)
def test_render_measure_report_when_rendering_a_metric_row_does_state_the_median(
    spread: float | None, expected: str
):
    result = create_measurement_result(
        metrics={"decode/time": measured_metric(median=100, spread=spread, unit="ns")}
    )

    row = line_starting_with(render_measure_report(result), "decode/time")

    assert cells_of(row)[-1].strip() == expected


# ---------------------------------------------------------------------------
# nothing to compare against
# ---------------------------------------------------------------------------


def test_render_measure_report_when_nothing_to_compare_does_carry_no_verdicts_or_aggregates():
    output = strip_ansi(render_measure_report(two_kind_measurement()))

    assert "geomean" not in output
    assert "highlights" not in output
    assert "vs " not in output
    assert not any(glyph in output for glyph in "✓✗≈~?")


# ---------------------------------------------------------------------------
# multi-kind sections
# ---------------------------------------------------------------------------


def test_render_measure_report_when_many_kinds_does_give_each_a_titled_section():
    assert table_region(render_measure_report(two_kind_measurement())) == [
        "gymrat measure · main · 10 samples · adapter: mitata",
        "",
        "<border>",
        "time",
        "<rule>",
        "entity",
        "alive_check",
        "spawn",
        "",
        "warmup",
        "",
        "informational — gating off (config: kinds.memory.gating = false)",
        "<border>",
        "memory",
        "<rule>",
        "encode",
    ]


@pytest.mark.parametrize(
    "row",
    [
        pytest.param("  alive_check", id="grouped-indented"),
        pytest.param("warmup", id="bare-short-name"),
    ],
)
def test_render_measure_report_when_many_kinds_does_name_a_metric_row(row: str):
    line = line_starting_with(render_measure_report(two_kind_measurement()), row)

    assert cells_of(line)[0].rstrip() == row


@pytest.mark.parametrize(
    ("make_result", "expected"),
    [
        pytest.param(
            two_kind_measurement,
            "informational — gating off (config: kinds.memory.gating = false)",
            id="kind-level-config",
        ),
        pytest.param(
            lambda: replace(two_kind_measurement(), config_kinds=None),
            "informational — gating off",
            id="per-metric-overrides",
        ),
    ],
)
def test_render_measure_report_when_kind_is_informational_does_credit_the_config_source(
    make_result: Callable[[], MeasurementResult], expected: str
):
    report = render_measure_report(make_result())

    assert line_containing(report, "informational") == expected


def test_render_measure_report_when_many_kinds_does_line_every_section_up_with_the_first():
    report = render_measure_report(two_kind_measurement())
    offsets = separator_offsets(line_starting_with(report, "time"))

    assert separator_offsets(line_starting_with(report, "memory")) == offsets
    assert separator_offsets(line_starting_with(report, "entity ")) == offsets
    assert separator_offsets(line_starting_with(report, "  alive_check")) == offsets


def test_render_measure_report_when_many_kinds_does_state_each_kind_median_in_its_unit():
    report = render_measure_report(two_kind_measurement())

    assert cells_of(line_starting_with(report, "  spawn"))[-1].strip() == "104ns ± 1%"
    assert cells_of(line_starting_with(report, "encode"))[-1].strip() == "93B ± 1%"


# ---------------------------------------------------------------------------
# color
# ---------------------------------------------------------------------------


def test_render_measure_report_when_colored_does_style_the_target_like_a_variant(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    report = render_measure_report(two_kind_measurement())

    assert styles_at(line_containing(report, "gymrat measure"), "main") == ["1", "4"]


def test_render_measure_report_when_colored_does_embolden_kind_and_dim_informational(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    report = render_measure_report(two_kind_measurement())
    header = next(
        line
        for line in report.split("\n")
        if "│" in line and strip_ansi(line).lstrip().startswith("memory")
    )

    assert styles_at(header, "memory") == ["1"]
    assert styles_at(line_containing(report, "informational"), "informational") == ["2"]


def test_render_measure_report_when_colored_does_leave_separators_in_the_default_color(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    rows = [
        line for line in render_measure_report(two_kind_measurement()).split("\n") if "│" in line
    ]
    inherited = [row for row in rows if any(styles for styles in separator_styles(row))]

    assert inherited == []


def test_render_measure_report_when_no_color_is_set_does_leave_the_report_unstyled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    assert "\x1b[" not in render_measure_report(two_kind_measurement())


@pytest.mark.parametrize(
    ("color", "styled"),
    [
        pytest.param(False, False, id="off"),
        pytest.param(True, True, id="on"),
    ],
)
def test_render_measure_report_when_color_option_set_does_override_the_environment(
    monkeypatch: pytest.MonkeyPatch, color: bool, styled: bool
):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    output = render_measure_report(two_kind_measurement(), ReportOptions(color=color))

    assert ("\x1b[" in output) is styled


# ===========================================================================
# render_report — colored comparison report
# ===========================================================================


def test_render_report_when_no_color_is_set_does_leave_the_report_unstyled():
    assert "\x1b[" not in render_report(_colorful_result())


def test_render_report_when_colored_does_pad_on_plain_text_so_columns_line_up(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    bare = strip_ansi(render_report(_colorful_result()))
    header_offsets = separator_offsets(line_starting_with(bare, "metric"))

    assert separator_offsets(line_starting_with(bare, "faster/time")) == header_offsets
    assert separator_offsets(line_starting_with(bare, "geomean")) == header_offsets


def test_render_report_when_colored_does_measure_verdict_subfields_on_plain_text(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={
            "improved/time": permutation_metric(
                verdict="improved", delta=-12.4, noise_pct=2.5, unit="ns"
            ),
            "flat/time": permutation_metric(
                verdict="no-signal", delta=0.4, noise_pct=100, unit="ns"
            ),
        }
    )

    bare = strip_ansi(render_report(result))

    assert cells_of(line_starting_with(bare, "improved/time"))[-1].strip() == "✓  -12.4%  ±  2.5%"
    assert cells_of(line_starting_with(bare, "flat/time"))[-1].strip() == "~   +0.4%  ±100.0%"


@pytest.mark.parametrize(
    ("metric", "glyph", "code"),
    [
        pytest.param("faster/time", "✓", "32", id="improved-green"),
        pytest.param("slower/time", "✗", "31", id="regressed-red"),
        pytest.param("jittery/time", "≈", "33", id="unstable-yellow"),
        pytest.param("tied/heap", "=", "36", id="identical-cyan"),
        pytest.param("flat/time", "~", "2", id="within-noise-dim"),
        pytest.param("single-pair/time", "?", "2", id="inconclusive-dim"),
    ],
)
def test_render_report_when_colored_does_paint_each_verdict_on_its_row(
    monkeypatch: pytest.MonkeyPatch, metric: str, glyph: str, code: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    row = line_containing(render_report(_colorful_result()), metric)

    assert code in styles_at(row, glyph)


@pytest.mark.parametrize(
    "metric",
    [
        pytest.param("flat/time", id="within-noise"),
        pytest.param("tied/heap", id="identical"),
        pytest.param("single-pair/time", id="inconclusive"),
        pytest.param("jittery/time", id="unstable"),
    ],
)
def test_render_report_when_colored_does_leave_name_and_value_cells_unstyled(
    monkeypatch: pytest.MonkeyPatch, metric: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    row = line_containing(render_report(_colorful_result()), metric)

    assert "\x1b[" not in "│".join(cells_of(row)[:-1])


@pytest.mark.parametrize(
    "metric",
    [
        pytest.param("faster/time", id="improved"),
        pytest.param("slower/time", id="regressed"),
    ],
)
def test_render_report_when_colored_does_leave_a_bright_row_without_end_to_end_dim(
    monkeypatch: pytest.MonkeyPatch, metric: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    row = line_containing(render_report(_colorful_result()), metric)

    assert not DIMMED_LINE.match(row)


def test_render_report_when_colored_does_embolden_the_geomean_figure(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    report = render_report(_colorful_result())

    assert "1" in styles_at(line_containing(report, "geomean"), "-5.8%")


@pytest.mark.parametrize(
    "anchor",
    [
        pytest.param("gymrat compare", id="run-header"),
        pytest.param("metric  ", id="column-header"),
    ],
)
def test_render_report_when_colored_does_embolden_and_underline_variant_names(
    monkeypatch: pytest.MonkeyPatch, anchor: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    line = line_containing(render_report(_colorful_result()), anchor)

    assert styles_at(line, "main") == ["1", "4"]
    assert styles_at(line, "perf/faster-decode") == ["1", "4"]


def test_render_report_when_colored_does_embolden_the_baseline_in_the_delta_header(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    header = line_containing(render_report(_colorful_result()), "metric  ")

    assert styles_at(delta_cell(header), "main") == ["1", "4"]


@pytest.mark.parametrize(
    "baseline",
    [
        pytest.param("v", id="v"),
        pytest.param("s", id="s"),
        pytest.param("vs", id="vs"),
    ],
)
def test_render_report_when_colored_does_embolden_the_baseline_after_the_vs_prefix(
    monkeypatch: pytest.MonkeyPatch, baseline: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        baseline_label=baseline,
        metrics={"a/time": permutation_metric(verdict="improved", delta=-10, unit="ns")},
    )

    cell = delta_cell(line_containing(render_report(result), "metric  "))

    assert f"vs \x1b[1;4m{baseline}\x1b[0m" in cell


def test_render_report_when_colored_does_leave_the_rest_of_the_column_header_unstyled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    header = line_containing(render_report(_colorful_result()), "metric  ")

    assert not re.match(r"^\x1b\[1m", header)
    assert styles_at(header, "metric") == []


def test_render_report_when_colored_does_embolden_gymrat_compare_in_the_header(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    header = line_containing(render_report(_colorful_result()), "gymrat compare")

    assert "1" in styles_at(header, "gymrat compare")


def test_render_report_when_colored_does_dim_each_separator_in_the_header(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    header = line_containing(render_report(_colorful_result()), "gymrat compare")

    assert "2" in styles_at(header, "·")


def test_render_report_when_colored_does_leave_a_dotted_variant_name_out_of_dimming(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        baseline_label="main·1",  # cspell:disable-line
        candidates=[create_candidate(label="perf·2")],  # cspell:disable-line
    )

    header = line_containing(render_report(result), "gymrat compare")

    assert "\x1b[1;4mmain·1\x1b[0m" in header  # cspell:disable-line
    assert "\x1b[1;4mperf·2\x1b[0m" in header  # cspell:disable-line


def test_render_report_when_colored_does_leave_a_dotted_adapter_name_out_of_dimming(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(adapter="metric·lines")  # cspell:disable-line

    header = line_containing(render_report(result), "gymrat compare")

    assert "adapter: metric·lines" in header  # cspell:disable-line


# ---------------------------------------------------------------------------
# verdict summary color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "code"),
    [
        pytest.param("improved", "32", id="improved-green"),
        pytest.param("regressed", "31", id="regressed-red"),
        pytest.param("unstable", "33", id="unstable-yellow"),
    ],
)
def test_render_report_when_colored_does_style_the_non_zero_tally_in_the_summary(
    monkeypatch: pytest.MonkeyPatch, label: str, code: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    summary = line_containing(render_report(_colorful_result()), "improved")

    assert f"\x1b[{code}m" in _summary_segment(summary, label)


@pytest.mark.parametrize(
    "glyph", [pytest.param("✗", id="regressed"), pytest.param("=", id="identical")]
)
def test_render_report_when_colored_does_dim_a_zero_count_segment_in_the_summary(
    monkeypatch: pytest.MonkeyPatch, glyph: str
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={"faster/time": permutation_metric(verdict="improved", delta=-10, unit="ns")}
    )
    summary = line_containing(render_report(result), "improved")

    assert "2" in styles_at(summary, glyph)


def test_render_report_when_colored_does_paint_the_non_zero_identical_tally_cyan(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={"tied/heap": band_metric(verdict="no-signal", delta=-0.5, n=10, usable_n=0)}
    )
    summary = line_containing(render_report(result), "identical")

    assert "\x1b[36m" in _summary_segment(summary, "identical")


def test_render_report_when_colored_does_dim_the_within_noise_segment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    summary = line_containing(render_report(_colorful_result()), "within noise")

    assert "\x1b[2m" in _summary_segment(summary, "within noise")


# ---------------------------------------------------------------------------
# highlights color
# ---------------------------------------------------------------------------


def test_render_report_when_colored_does_embolden_the_highlights_heading(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    heading = next(
        line
        for line in render_report(_colorful_result()).split("\n")
        if strip_ansi(line) == "highlights"
    )

    assert re.match(r"^\x1b\[1m", heading)


# The glyph and SGR color each highlight verdict class carries in a colored entry.
_HIGHLIGHT_GLYPH_COLOR: dict[ApproximateVerdict, tuple[str, str]] = {
    "improved": ("✓", "32"),
    "regressed": ("✗", "31"),
}


@pytest.mark.parametrize(
    ("verdict", "metric", "delta"),
    [
        pytest.param("improved", "faster/time", -17.5, id="improved-green"),
        pytest.param("regressed", "slower/time", 2.2, id="regressed-red"),
    ],
)
def test_render_report_when_colored_does_style_a_highlight_glyph_and_delta(
    monkeypatch: pytest.MonkeyPatch,
    verdict: ApproximateVerdict,
    metric: str,
    delta: float,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={metric: permutation_metric(verdict=verdict, delta=delta, unit="ns")}
    )
    entry = highlight_lines(render_report(result))[0]

    glyph, code = _HIGHLIGHT_GLYPH_COLOR[verdict]
    delta_text = f"{'+' if delta > 0 else ''}{delta:.1f}%"
    assert code in styles_at(entry, glyph)
    assert code in styles_at(entry, delta_text)


def test_render_report_when_colored_does_style_the_unstable_highlight_glyph_and_word(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={"jittery/time": band_metric(verdict="unstable", delta=5, noise_pct=30)}
    )
    entry = highlight_lines(render_report(result))[0]

    assert "33" in styles_at(entry, "≈")
    assert "33" in styles_at(entry, "unstable")


def test_render_report_when_colored_does_style_the_verdict_word_not_a_matching_name(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={"unstable-parse/time": band_metric(verdict="unstable", delta=5, noise_pct=30)}
    )
    entry = highlight_lines(render_report(result))[0]

    assert strip_ansi(entry).strip() == "≈ unstable-parse/time  unstable  noise ±30.0%"
    assert "33" in styles_at(entry, "unstable", last=True)


def test_render_report_when_colored_does_dim_the_evidence_suffixes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={
            "cheaper/heap": exact_metric(delta=-7.9),
            "jittery/time": band_metric(verdict="unstable", delta=5, noise_pct=30),
        }
    )
    highlights = highlight_lines(render_report(result))
    exact_entry = next(line for line in highlights if "cheaper/heap" in line)
    unstable_entry = next(line for line in highlights if "jittery/time" in line)

    assert "2" in styles_at(exact_entry, "(exact)")
    assert "2" in styles_at(unstable_entry, "noise")


def test_render_report_when_colored_does_dim_the_futility_note(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={"jittery/time": band_metric(verdict="unstable", delta=5, noise_pct=30)}
    )
    note = line_containing(render_report(result), "won't stabilize")

    assert "2" in styles_at(note, "unstable metrics")


# ---------------------------------------------------------------------------
# method footer and hint color
# ---------------------------------------------------------------------------


def test_render_report_when_colored_does_dim_the_verdict_method_description(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={"a/time": permutation_metric(verdict="improved", delta=-10, unit="ns")}
    )
    method = line_containing(render_report(result, ReportOptions(verbose=True)), "permutation")

    assert DIMMED_LINE.match(method)


def _band_fallback_result() -> ComparisonResult:
    """A single no-signal metric that fell back to the band, so the hint and footer render."""
    return create_comparison_result(metrics={"a/time": band_metric(verdict="no-signal", delta=-5)})


def test_render_report_when_colored_does_dim_the_noise_band_description(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    band = line_containing(
        render_report(_band_fallback_result(), ReportOptions(verbose=True)), "noise band"
    )

    assert DIMMED_LINE.match(band)


def test_render_report_when_colored_does_dim_the_hint_line(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORCE_COLOR", "1")

    hint = line_containing(render_report(_band_fallback_result()), "some rounds were dropped")

    assert DIMMED_LINE.match(hint)


def test_render_report_when_color_off_does_render_the_hint_line_plain():
    hint = line_containing(render_report(_band_fallback_result()), "some rounds were dropped")

    assert "\x1b[" not in hint


def test_render_report_when_colored_does_dim_the_band_annotation_on_bright_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    report = render_report(_colorful_result())

    assert "2" in styles_at(line_containing(report, "faster/time"), "±2.5%")
    assert "2" in styles_at(line_containing(report, "slower/time"), "±2.5%")


def test_render_report_when_colored_does_color_the_delta_not_the_band_on_a_shared_digits_row(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    result = create_comparison_result(
        metrics={"collision/time": band_metric(delta=0, noise_pct=10, n=10, usable_n=0)}
    )

    row = line_containing(render_report(result), "collision/time")

    assert "36" in styles_at(row, "0.0%")
    assert "2" in styles_at(row, "±10.0%")


def test_render_report_when_colored_does_leave_the_floor_band_off_an_inconclusive_row(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    row = line_containing(render_report(_colorful_result()), "single-pair/time")

    assert "±" not in strip_ansi(cells_of(row)[-1])


def test_render_report_when_color_option_false_does_override_force_color(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    output = render_report(_colorful_result(), ReportOptions(color=False))

    assert "\x1b[" not in output


def test_render_report_when_color_option_true_does_override_no_color():
    output = render_report(_colorful_result(), ReportOptions(color=True))

    assert "\x1b[" in output
