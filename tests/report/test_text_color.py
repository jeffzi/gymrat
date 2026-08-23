"""Tests for the single-target measurement text report.

These port the ``renderMeasureReport`` cases from ``text-color.test.ts`` — the
report's header, its flat and sectioned tables, and how it styles them. The
byte-exact golden snapshots and the worktree-cleanup footer are out of this
task's scope (the footer arrives with the comparison footers), so those cases are
left for a later task. Where the TypeScript suite pinned column alignment by byte
offset, the port keeps the offset check only where it proves cross-section
alignment; content is pinned by parsed cell.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat_py.report.text import render_measure_report
from gymrat_py.report.types import MeasurementResult, ReportOptions
from tests.report._inputs import (
    cells_of,
    create_measurement_result,
    line_containing,
    line_starting_with,
    measured_metric,
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


@pytest.fixture(autouse=True)
def _no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)


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
