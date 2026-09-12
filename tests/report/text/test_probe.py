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
    line_containing,
    line_starting_with,
    probe_metric,
    probe_result,
    strip_ansi,
    styles_at,
    table_rows,
)

if TYPE_CHECKING:
    from gymrat.loop.probe import ProbeMetric
    from gymrat.model import Direction

# ---------------------------------------------------------------------------
# run header
# ---------------------------------------------------------------------------


def test_render_probe_report_when_rendering_header_does_name_target_samples_adapter():
    result = probe_result(label="experiment", samples=6, adapter="mitata")

    output = render_probe_report(result)

    assert "gymrat probe · experiment · 6 samples · adapter: mitata" in strip_ansi(output)


@pytest.mark.parametrize(
    ("scoped", "names", "suffix"),
    [
        pytest.param(
            True,
            ("total_ms", "decode large payload"),
            "· scoped: total_ms, decode large payload",
            id="scoped",
        ),
        pytest.param(False, (), None, id="not-scoped"),
    ],
)
def test_render_probe_report_when_scope_varies_does_reflect_scope_in_the_header(
    scoped: bool, names: tuple[str, ...], suffix: str | None
):
    result = probe_result(scoped=scoped, names=names)

    header = line_containing(render_probe_report(result), "gymrat probe")

    if suffix is None:
        assert "scoped" not in strip_ansi(header)
    else:
        assert strip_ansi(header).endswith(suffix)


# ---------------------------------------------------------------------------
# metric rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "cells"),
    [
        pytest.param(
            probe_metric(
                "decode/time",
                median=90.0,
                spread=2.0,
                reference_median=100.0,
                delta_pct=-10.0,
                unit="ns",
            ),
            ["decode/time", "90ns ± 2%", "100ns", "-10.0%"],
            id="faster-than-the-baseline",
        ),
        pytest.param(
            probe_metric(
                "decode/time",
                median=125.0,
                spread=2.0,
                reference_median=100.0,
                delta_pct=25.0,
                unit="ns",
            ),
            ["decode/time", "125ns ± 2%", "100ns", "+25.0%"],
            id="slower-than-the-baseline-signs-the-delta",
        ),
        pytest.param(
            probe_metric(
                "alloc_bytes",
                median=50.0,
                spread=1.0,
                reference_median=None,
                delta_pct=None,
                unit="bytes",
            ),
            ["alloc_bytes", "50B ± 1%", "", "no reference"],
            id="no-reference-says-so-in-the-delta-column",
        ),
    ],
)
def test_render_probe_report_when_metric_rendered_does_state_median_reference_and_delta(
    metric: ProbeMetric, cells: list[str]
):
    result = probe_result(metrics=[metric])

    row = line_starting_with(strip_ansi(render_probe_report(result)), metric.name)

    assert [cell.strip() for cell in cells_of(row)] == cells


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
    present = [glyph for glyph in "✓✗≈~?" if glyph in output]
    assert not present, f"verdict glyphs should not appear: {present!r}"


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
