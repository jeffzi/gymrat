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
from tests._ansi import strip_ansi
from tests.report._assertions import (
    cells_of,
    delta_cell,
    line_containing,
    styles_at,
    table_rows,
)
from tests.report._probes import probe_metric, probe_result

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion

    from gymrat.loop.probe import ProbeResult
    from gymrat.model import Direction

# ---------------------------------------------------------------------------
# run header
# ---------------------------------------------------------------------------


def test_render_probe_report_when_names_given_does_reflect_scope_in_the_header():
    result = probe_result(names=("total_ms", "decode large payload"))

    header = line_containing(render_probe_report(result), "gymrat probe")

    assert strip_ansi(header).endswith("· scoped: total_ms, decode large payload")


# ---------------------------------------------------------------------------
# metric rows
# ---------------------------------------------------------------------------


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
# color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "delta_pct", "rendered", "code"),
    [
        pytest.param("lower", 8.0, "+8.0%", "31", id="lower-slower-is-regressed"),
        pytest.param("higher", -10.0, "-10.0%", "31", id="higher-less-is-regressed"),
        pytest.param("lower", -10.0, "-10.0%", "32", id="lower-faster-is-improved"),
        pytest.param("higher", 20.0, "+20.0%", "32", id="higher-more-is-improved"),
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


@pytest.mark.parametrize(
    ("median", "delta_pct", "rendered"),
    [
        pytest.param(90.0, None, "", id="zero-reference-leaves-the-delta-blank"),
        pytest.param(0.0, 0.0, "0.0%", id="both-zero-is-no-change"),
    ],
)
def test_render_probe_report_when_reference_zero_does_leave_the_delta_unstyled(
    median: float, delta_pct: float | None, rendered: str
):
    result = probe_result(
        metrics=[
            probe_metric(
                "decode/time", median=median, reference_median=0.0, delta_pct=delta_pct, unit="ns"
            )
        ]
    )

    row = line_containing(render_probe_report(result, ReportOptions(color=True)), "decode/time")

    assert strip_ansi(delta_cell(row)).strip() == rendered
    assert "\x1b[" not in delta_cell(row)


# ---------------------------------------------------------------------------
# whole report — golden
# ---------------------------------------------------------------------------


def _golden_probe() -> ProbeResult:
    """A probe with a paired metric, a higher-is-better metric, and one the baseline lacks."""
    return probe_result(
        metrics=[
            probe_metric("total_ns", unit="ns", kind="time"),
            probe_metric(
                "ops_per_sec",
                median=1200.0,
                reference_median=1000.0,
                delta_pct=20.0,
                direction="higher",
                kind="throughput",
            ),
            probe_metric("cold_start_ns", reference_median=None, delta_pct=None, unit="ns"),
        ]
    )


def test_render_probe_report_when_colored_does_style_every_heading():
    report = render_probe_report(_golden_probe(), ReportOptions(color=True))

    header = line_containing(report, "gymrat probe")
    heading = line_containing(report, "baseline")
    assert (
        styles_at(header, "gymrat probe"),
        styles_at(header, "·"),
        styles_at(header, "experiment"),
        styles_at(heading, "time"),
        styles_at(heading, "experiment"),
    ) == (["1"], ["2"], ["1", "4"], ["1"], ["1", "4"])


def test_render_probe_report_when_rendered_does_match_its_golden(snapshot: SnapshotAssertion):
    report = render_probe_report(_golden_probe(), ReportOptions(color=False))

    assert report.split("\n") == snapshot
