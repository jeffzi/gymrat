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

from gymrat.report.style import VERDICT_STYLES
from gymrat.report.text.probe import render_probe_report
from gymrat.report.types import ReportOptions
from tests._ansi import strip_ansi
from tests.report._assertions import (
    delta_cell,
    line_containing,
    sgr_for,
    styles_at,
)
from tests.report._probes import golden_probe, probe_metric, probe_result

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

    assert strip_ansi(header).endswith("total_ms, decode large payload")


# ---------------------------------------------------------------------------
# color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "delta_pct", "rendered", "codes"),
    [
        pytest.param(
            "higher",
            -10.0,
            "-10.0%",
            sgr_for(VERDICT_STYLES["regressed"]),
            id="higher-less-is-regressed",
        ),
        pytest.param(
            "lower",
            -10.0,
            "-10.0%",
            sgr_for(VERDICT_STYLES["improved"]),
            id="lower-faster-is-improved",
        ),
    ],
)
def test_render_probe_report_when_colored_does_paint_the_delta_by_its_direction(
    direction: Direction, delta_pct: float, rendered: str, codes: list[str]
):
    result = probe_result(
        metrics=[probe_metric("decode/time", delta_pct=delta_pct, direction=direction, unit="ns")]
    )

    row = line_containing(render_probe_report(result, ReportOptions(color=True)), "decode/time")

    assert styles_at(row, rendered) == codes


def _zero_reference_result(median: float, delta_pct: float | None) -> ProbeResult:
    """A probe of ``decode/time`` against a zero reference median."""
    return probe_result(
        metrics=[
            probe_metric(
                "decode/time", median=median, reference_median=0.0, delta_pct=delta_pct, unit="ns"
            )
        ]
    )


@pytest.mark.parametrize(
    ("median", "delta_pct", "rendered"),
    [
        pytest.param(90.0, None, "", id="zero-reference-leaves-the-delta-blank"),
        pytest.param(0.0, 0.0, "0.0%", id="both-zero-is-no-change"),
    ],
)
def test_render_probe_report_when_reference_zero_does_render_a_blank_or_zero_delta(
    median: float, delta_pct: float | None, rendered: str
):
    result = _zero_reference_result(median, delta_pct)

    row = line_containing(render_probe_report(result, ReportOptions(color=True)), "decode/time")

    assert strip_ansi(delta_cell(row)).strip() == rendered


@pytest.mark.parametrize(
    ("median", "delta_pct"),
    [
        pytest.param(90.0, None, id="zero-reference"),
        pytest.param(0.0, 0.0, id="both-zero"),
    ],
)
def test_render_probe_report_when_reference_zero_does_leave_the_delta_unstyled(
    median: float, delta_pct: float | None
):
    result = _zero_reference_result(median, delta_pct)

    row = line_containing(render_probe_report(result, ReportOptions(color=True)), "decode/time")

    assert "\x1b[" not in delta_cell(row)


# ---------------------------------------------------------------------------
# whole report — golden
# ---------------------------------------------------------------------------


def test_render_probe_report_when_rendered_does_match_its_golden(snapshot: SnapshotAssertion):
    report = render_probe_report(golden_probe(), ReportOptions(color=False))

    assert report.split("\n") == snapshot
