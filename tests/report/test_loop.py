"""Tests for the loop report fragments.

These cover the loop header, the verdict block, and outcome derivation. The
header block pins the iteration wording and sample pluralization; the verdict
block pins the line shape and the outcome word; outcome derivation pins the full
truth table, including the gating-regression override, direction-aware metric
primaries, the exactly-zero cases, and the unmeasured-primary case.

Color is pinned by intent rather than by raw byte sequences: the loop fragments
return rich markup, and the color cases resolve that markup with
``render_lines(..., color=True)`` and read the SGR codes off it with
``styles_at``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat_py.report.loop import (
    GeomeanPrimary,
    LoopPrimary,
    MetricPrimary,
    derive_outcome,
    format_loop_header,
    format_verdict_block,
)
from gymrat_py.report.style import render_lines
from tests.report._inputs import signed_rank_metric, styles_at

if TYPE_CHECKING:
    from gymrat_py.model import Direction
    from gymrat_py.report.loop import LoopOutcome
    from gymrat_py.report.types import MetricComparison, MetricComparisons

# A width wide enough that no loop fragment ever soft-wraps.
_WIDTH = 200


def _plain(*markup: str) -> str:
    """The markup rendered without color, as a terminal's visible text would read."""
    return render_lines(*markup, color=False, width=_WIDTH)


def _colored(*markup: str) -> str:
    """The markup rendered with color forced on, ANSI escapes and all."""
    return render_lines(*markup, color=True, width=_WIDTH)


def _geomean_primary(delta_pct: float = -4.2) -> GeomeanPrimary:
    """The geomean primary, improving by default."""
    return GeomeanPrimary(delta_pct=delta_pct)


def _directed_metric(direction: Direction, *, gating: bool = True) -> MetricComparison:
    """A metric judged in ``direction`` with no signal of its own."""
    return signed_rank_metric(verdict="no-signal", delta=0, direction=direction, gating=gating)


def _regressed_metrics(*, gating: bool) -> MetricComparisons:
    """A run whose single metric regressed, gating or not."""
    return {"decode/time": signed_rank_metric(verdict="regressed", delta=4, gating=gating)}


# ---------------------------------------------------------------------------
# format_loop_header
# ---------------------------------------------------------------------------


def test_format_loop_header_when_given_seq_and_samples_does_name_iteration_comparison_and_count():
    header = _plain(format_loop_header(7, 6))

    assert header == "iteration 7 · experiment vs baseline · 6 paired samples"


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        pytest.param(1, "· 1 paired sample", id="one"),
        pytest.param(2, "· 2 paired samples", id="many"),
    ],
)
def test_format_loop_header_when_counting_samples_does_match_noun(samples: int, expected: str):
    header = _plain(format_loop_header(1, samples))

    assert expected in header


def test_format_loop_header_when_colored_does_embolden_the_iteration_label():
    header = _colored(format_loop_header(7, 6))

    assert "1" in styles_at(header, "iteration 7")


def test_format_loop_header_when_colored_does_dim_each_separator():
    header = _colored(format_loop_header(7, 6))

    assert "2" in styles_at(header, "·")


# ---------------------------------------------------------------------------
# format_verdict_block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "word"),
    [
        pytest.param("improved", "IMPROVED", id="improved"),
        pytest.param("regressed", "REGRESSED", id="regressed"),
        pytest.param("no-signal", "NO-SIGNAL", id="no-signal"),
    ],
)
def test_format_verdict_block_when_given_outcome_does_state_primary_delta_and_verdict(
    outcome: LoopOutcome, word: str
):
    block = format_verdict_block(
        outcome=outcome, primary=_geomean_primary(), next_step="gymrat keep"
    )

    assert _plain(block[0]) == f"primary: -4.2% · verdict: {word}"


def test_format_verdict_block_when_given_next_step_does_close_the_block_with_it():
    block = format_verdict_block(
        outcome="regressed", primary=_geomean_primary(3.1), next_step="fix or gymrat discard"
    )

    assert len(block) == 2
    assert _plain(block[1]) == "Hint: fix or gymrat discard"


@pytest.mark.parametrize(
    ("outcome", "word", "color_code"),
    [
        pytest.param("improved", "IMPROVED", "32", id="improved-green"),
        pytest.param("regressed", "REGRESSED", "31", id="regressed-red"),
        pytest.param("no-signal", "NO-SIGNAL", None, id="no-signal-uncolored"),
    ],
)
def test_format_verdict_block_when_colored_does_paint_the_verdict_word(
    outcome: LoopOutcome, word: str, color_code: str | None
):
    block = format_verdict_block(
        outcome=outcome, primary=_geomean_primary(), next_step="gymrat keep"
    )

    codes = styles_at(_colored(block[0]), word)

    assert "1" in codes
    if color_code is None:
        assert "31" not in codes
        assert "32" not in codes
    else:
        assert color_code in codes


# ---------------------------------------------------------------------------
# derive_outcome
# ---------------------------------------------------------------------------


def test_derive_outcome_when_gating_metric_regressed_does_report_regressed_over_the_primary():
    outcome = derive_outcome(_regressed_metrics(gating=True), _geomean_primary(-9))

    assert outcome == "regressed"


def test_derive_outcome_when_non_gating_metric_regressed_does_leave_it_out_of_the_outcome():
    outcome = derive_outcome(_regressed_metrics(gating=False), _geomean_primary(-9))

    assert outcome == "improved"


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        pytest.param(GeomeanPrimary(-3), "improved", id="geomean-negative-improves"),
        pytest.param(GeomeanPrimary(3), "no-signal", id="geomean-positive-no-signal"),
        pytest.param(GeomeanPrimary(0), "no-signal", id="geomean-zero-no-signal"),
        pytest.param(MetricPrimary("lower/time", -3), "improved", id="lower-negative-improves"),
        pytest.param(MetricPrimary("lower/time", 3), "no-signal", id="lower-positive-no-signal"),
        pytest.param(MetricPrimary("higher/time", 3), "improved", id="higher-positive-improves"),
        pytest.param(MetricPrimary("higher/time", -3), "no-signal", id="higher-negative-no-signal"),
    ],
)
def test_derive_outcome_when_no_gating_regression_does_read_the_primary(
    primary: LoopPrimary, expected: str
):
    metrics: MetricComparisons = {
        "lower/time": _directed_metric("lower"),
        "higher/time": _directed_metric("higher"),
    }

    assert derive_outcome(metrics, primary) == expected


def test_derive_outcome_when_primary_metric_never_measured_does_report_no_signal():
    primary = MetricPrimary("absent/time", -30)

    outcome = derive_outcome({"lower/time": _directed_metric("lower")}, primary)

    assert outcome == "no-signal"
