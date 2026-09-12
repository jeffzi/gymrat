"""Tests for the probe text report.

A probe is a mid-session spot check: it benches the experiment worktree once and
lines every measured median up against the newest recorded baseline. The report
is deliberately thinner than a compare — the measure table plus a reference
column and a delta column, with no verdict, no geomean, and no significance
test, because a single unpaired run cannot support any of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.report.text.probe import render_probe_report
from gymrat.report.types import ReportOptions
from tests.report._inputs import (
    cells_of,
    delta_cell,
    line_containing,
    line_starting_with,
    probe_metric,
    probe_result,
    strip_ansi,
    styles_at,
    table_rows,
)

if TYPE_CHECKING:
    from gymrat.model import Direction

# ---------------------------------------------------------------------------
# run header
# ---------------------------------------------------------------------------


def test_render_probe_report_when_rendering_header_does_name_target_samples_adapter():
    result = probe_result(label="experiment", samples=6, adapter="mitata")

    output = render_probe_report(result)

    assert "gymrat probe · experiment · 6 samples · adapter: mitata" in strip_ansi(output)


def test_render_probe_report_when_scoped_does_name_the_scoped_metrics_in_the_header():
    result = probe_result(scoped=True, names=("total_ms", "decode large payload"))

    header = line_containing(render_probe_report(result), "gymrat probe")

    assert strip_ansi(header).endswith("· scoped: total_ms, decode large payload")


def test_render_probe_report_when_not_scoped_does_leave_the_scope_out_of_the_header():
    header = line_containing(render_probe_report(probe_result()), "gymrat probe")

    assert "scoped" not in strip_ansi(header)


# ---------------------------------------------------------------------------
# metric rows
# ---------------------------------------------------------------------------


def test_render_probe_report_when_metric_paired_does_state_median_reference_and_delta():
    result = probe_result(
        metrics=[
            probe_metric(
                "decode/time",
                median=90.0,
                spread=2.0,
                reference_median=100.0,
                delta_pct=-10.0,
                unit="ns",
            )
        ]
    )

    row = line_starting_with(strip_ansi(render_probe_report(result)), "decode/time")

    assert [cell.strip() for cell in cells_of(row)] == [
        "decode/time",
        "90ns ± 2%",
        "100ns",
        "-10.0%",
    ]


def test_render_probe_report_when_metric_slower_than_the_baseline_does_sign_the_delta():
    result = probe_result(
        metrics=[probe_metric("decode/time", median=125.0, reference_median=100.0, delta_pct=25.0)]
    )

    row = line_starting_with(strip_ansi(render_probe_report(result)), "decode/time")

    assert delta_cell(row).strip() == "+25.0%"


def test_render_probe_report_when_metric_has_no_reference_does_say_so_in_the_delta_column():
    result = probe_result(
        metrics=[
            probe_metric(
                "alloc_bytes",
                median=50.0,
                spread=1.0,
                reference_median=None,
                delta_pct=None,
                unit="bytes",
            )
        ]
    )

    row = line_starting_with(strip_ansi(render_probe_report(result)), "alloc_bytes")

    assert [cell.strip() for cell in cells_of(row)] == [
        "alloc_bytes",
        "50B ± 1%",
        "",
        "no reference",
    ]


def test_render_probe_report_when_several_metrics_does_keep_them_in_result_order():
    result = probe_result(
        metrics=[
            probe_metric("total_ms", unit="ns"),
            probe_metric("alloc_bytes", unit="bytes", reference_median=None, delta_pct=None),
        ]
    )

    rows = table_rows(render_probe_report(result))[1:]

    assert [cells_of(row)[0].strip() for row in rows] == ["total_ms", "alloc_bytes"]


# ---------------------------------------------------------------------------
# nothing a single unpaired run can conclude
# ---------------------------------------------------------------------------


def test_render_probe_report_when_rendering_does_carry_no_verdict_geomean_or_significance():
    result = probe_result(
        metrics=[
            probe_metric("total_ms", unit="ns"),
            probe_metric("alloc_bytes", unit="bytes", delta_pct=4.0, reference_median=96.0),
        ]
    )

    output = strip_ansi(render_probe_report(result))

    assert "geomean" not in output
    assert "highlights" not in output
    assert "permutation" not in output
    assert "noise" not in output
    for glyph in "✓✗≈~?":
        assert glyph not in output, f"verdict glyph {glyph!r} should not appear"


# ---------------------------------------------------------------------------
# color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "delta_pct", "rendered", "code"),
    [
        pytest.param("lower", -10.0, "-10.0%", "32", id="lower-faster-is-improved"),
        pytest.param("lower", 8.0, "+8.0%", "31", id="lower-slower-is-regressed"),
        pytest.param("higher", 8.0, "+8.0%", "32", id="higher-more-is-improved"),
        pytest.param("higher", -10.0, "-10.0%", "31", id="higher-less-is-regressed"),
    ],
)
def test_render_probe_report_when_colored_does_paint_the_delta_by_its_direction(
    direction: Direction, delta_pct: float, rendered: str, code: str
):
    result = probe_result(
        metrics=[probe_metric("decode/time", delta_pct=delta_pct, direction=direction, unit="ns")]
    )

    row = line_containing(render_probe_report(result, ReportOptions(color=True)), "decode/time")

    assert code in styles_at(row, rendered)


def test_render_probe_report_when_color_off_does_carry_the_same_text_unstyled():
    result = probe_result(metrics=[probe_metric("decode/time", delta_pct=-10.0, unit="ns")])

    output = render_probe_report(result, ReportOptions(color=False))

    assert "\x1b[" not in output
    assert output == strip_ansi(render_probe_report(result, ReportOptions(color=True)))
