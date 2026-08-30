"""Tests for the loop report fragments.

These cover the loop header, the verdict block, outcome derivation, and the
status-report formatters. The header block pins the iteration wording and sample
pluralization; the verdict block pins the line shape and the outcome word;
outcome derivation pins the full truth table, including the gating-regression
override, direction-aware metric primaries, the exactly-zero cases, and the
unmeasured-primary case. The status formatters pin the header block, the
per-iteration line (glyph, delta, and settle state), the baseline medians, the
totals-and-stop footer, and the finalized closer.

Color is pinned by intent rather than by raw byte sequences: the loop fragments
return rich markup, and the color cases resolve that markup with
``render_lines(..., color=True)`` and read the SGR codes off it with
``styles_at``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

from gymrat.config import StopConfig
from gymrat.report.loop import (
    GeomeanPrimary,
    LoopPrimary,
    MetricPrimary,
    SettleDiscarded,
    SettleKeepBlocked,
    SettleKept,
    SettleUnsettled,
    StatusIteration,
    StatusSummary,
    derive_outcome,
    format_loop_header,
    format_status_baseline,
    format_status_finalized,
    format_status_footer,
    format_status_header,
    format_status_iteration,
    format_verdict_block,
)
from gymrat.report.style import render_lines
from gymrat.session import BaselineRecord, BaselineRef, Worktrees
from tests.report._inputs import permutation_metric, styles_at
from tests.session.records._fixtures import SESSION_ID, finalize_record, session_record

if TYPE_CHECKING:
    from gymrat.model import Direction
    from gymrat.report.loop import LoopOutcome, SettleState
    from gymrat.report.types import MetricComparison, MetricComparisons

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
    return permutation_metric(verdict="no-signal", delta=0, direction=direction, gating=gating)


def _regressed_metrics(*, gating: bool) -> MetricComparisons:
    """A run whose single metric regressed, gating or not."""
    return {"decode/time": permutation_metric(verdict="regressed", delta=4, gating=gating)}


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
    assert _plain(block[1]) == "fix or gymrat discard"


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


def test_format_verdict_block_when_target_reached_but_regressed_does_omit_target_hint():
    block = format_verdict_block(
        outcome="regressed",
        primary=_geomean_primary(3.1),
        next_step="fix or run gymrat discard",
        target_reached=True,
    )

    plain = "\n".join(_plain(line) for line in block)
    assert "target reached" not in plain


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


# ---------------------------------------------------------------------------
# status formatters
# ---------------------------------------------------------------------------

# A 40-hex baseline sha whose first seven characters are recognizable on their own.
_BASELINE_SHA = "a1b2c3d" + "e" * 33
# A 40-hex keep-commit sha whose first seven characters are recognizable on their own.
_KEEP_COMMIT = "b1b2b3b" + "c" * 33


def _status_iteration(settle: SettleState) -> StatusIteration:
    """An improved iteration numbered 1, settled the way ``settle`` says."""
    return StatusIteration(seq=1, delta_pct=-7.2, outcome="improved", settle=settle)


def _status_summary(**overrides: Any) -> StatusSummary:
    """A session four iterations in, one kept and one thrown away."""
    default = StatusSummary(iteration_count=4, keep_count=1, discard_count=1, target_reached=False)
    return replace(default, **overrides) if overrides else default


# ---------------------------------------------------------------------------
# format_status_header
# ---------------------------------------------------------------------------


def test_format_status_header_when_given_session_does_name_session_baseline_branch_worktrees_adapter():
    session = session_record(baseline=BaselineRef(ref="main", sha=_BASELINE_SHA))

    lines = format_status_header(session)

    assert [_plain(line) for line in lines] == [
        f"session {SESSION_ID} · baseline main@a1b2c3d · adapter metric-lines",
        f"branch gymrat/{SESSION_ID}",
        "experiment worktree /repo/.gymrat/worktrees/experiment",
        "baseline worktree /repo/.gymrat/worktrees/baseline",
    ]


def test_format_status_header_when_colored_does_embolden_the_session():
    session = session_record(baseline=BaselineRef(ref="main", sha=_BASELINE_SHA))

    header = _colored(format_status_header(session)[0])

    assert "1" in styles_at(header, f"session {SESSION_ID}")


# ---------------------------------------------------------------------------
# format_status_iteration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("settle", "expected"),
    [
        pytest.param(
            SettleKept(commit=_KEEP_COMMIT),
            "iteration 1 · ✓ -7.2% · kept b1b2b3b",
            id="kept-with-commit",
        ),
        pytest.param(SettleKept(), "iteration 1 · ✓ -7.2% · kept", id="kept-pending"),
        pytest.param(SettleDiscarded(), "iteration 1 · ✓ -7.2% · discarded", id="discarded"),
        pytest.param(SettleUnsettled(), "iteration 1 · ✓ -7.2% · unsettled", id="unsettled"),
        pytest.param(
            SettleKeepBlocked(reason="checks-failed"),
            "iteration 1 · ✓ -7.2% · keep-blocked (checks-failed)",
            id="blocked-with-reason",
        ),
        pytest.param(
            SettleKeepBlocked(), "iteration 1 · ✓ -7.2% · keep-blocked", id="blocked-no-reason"
        ),
    ],
)
def test_format_status_iteration_when_given_settle_does_state_it(
    settle: SettleState, expected: str
):
    assert _plain(format_status_iteration(_status_iteration(settle))) == expected


@pytest.mark.parametrize(
    ("outcome", "glyph"),
    [
        pytest.param("improved", "✓", id="improved"),
        pytest.param("regressed", "✗", id="regressed"),
        pytest.param("no-signal", "~", id="no-signal"),
    ],
)
def test_format_status_iteration_when_given_outcome_does_mark_it_with_glyph(
    outcome: LoopOutcome, glyph: str
):
    entry = replace(_status_iteration(SettleUnsettled()), outcome=outcome)

    assert _plain(format_status_iteration(entry)) == f"iteration 1 · {glyph} -7.2% · unsettled"


def test_format_status_iteration_when_delta_unmeasured_does_state_no_percentage():
    entry = replace(_status_iteration(SettleUnsettled()), delta_pct=None, outcome="no-signal")

    assert _plain(format_status_iteration(entry)) == "iteration 1 · ~ · unsettled"


@pytest.mark.parametrize(
    ("outcome", "glyph", "color_code"),
    [
        pytest.param("improved", "✓", "32", id="improved-green"),
        pytest.param("regressed", "✗", "31", id="regressed-red"),
    ],
)
def test_format_status_iteration_when_colored_does_paint_the_glyph(
    outcome: LoopOutcome, glyph: str, color_code: str
):
    entry = replace(_status_iteration(SettleUnsettled()), outcome=outcome)

    assert color_code in styles_at(_colored(format_status_iteration(entry)), glyph)


# ---------------------------------------------------------------------------
# format_status_baseline
# ---------------------------------------------------------------------------


def test_format_status_baseline_when_given_samples_does_state_label_and_median_per_metric():
    record = BaselineRecord(
        type="baseline",
        at="2026-08-08T14:15:30.000Z",
        label="main",
        samples=(
            {"total_ms": 15200, "alloc_bytes": 1500},
            {"total_ms": 15184, "alloc_bytes": 1540},
        ),
    )

    line = _plain(format_status_baseline(record))

    assert line == "baseline main · total_ms 15192 · alloc_bytes 1520"


def test_format_status_baseline_when_a_round_omits_a_metric_does_median_over_rounds_that_reported_it():
    record = BaselineRecord(
        type="baseline",
        at="2026-08-08T14:15:30.000Z",
        label="main",
        samples=({"total_ms": 100, "alloc_bytes": 40}, {"total_ms": 300}),
    )

    line = _plain(format_status_baseline(record))

    assert line == "baseline main · total_ms 200 · alloc_bytes 40"


# ---------------------------------------------------------------------------
# format_status_footer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        pytest.param(
            _status_summary(iteration_count=1, keep_count=1, discard_count=0),
            "1 iteration · 1 kept · 0 discarded",
            id="one-iteration",
        ),
        pytest.param(_status_summary(), "4 iterations · 1 kept · 1 discarded", id="several"),
    ],
)
def test_format_status_footer_when_given_summary_does_total_the_settles(
    summary: StatusSummary, expected: str
):
    assert _plain(format_status_footer(summary)[0]) == expected


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        pytest.param(
            _status_summary(stop=StopConfig(max_iterations=30)),
            "stop: 4 of 30 iterations",
            id="max-iterations",
        ),
        pytest.param(
            _status_summary(stop=StopConfig(target_value=95)),
            "stop: target pending",
            id="target-pending",
        ),
        pytest.param(
            _status_summary(stop=StopConfig(target_value=95), target_reached=True),
            "stop: target reached",
            id="target-reached",
        ),
        pytest.param(
            _status_summary(stop=StopConfig(target_value=95, max_iterations=30)),
            "stop: 4 of 30 iterations · target pending",
            id="both-conditions",
        ),
        pytest.param(_status_summary(), None, id="no-stop"),
        pytest.param(_status_summary(stop=StopConfig()), None, id="empty-stop"),
    ],
)
def test_format_status_footer_when_stop_configured_does_state_the_conditions(
    summary: StatusSummary, expected: str | None
):
    lines = [_plain(line) for line in format_status_footer(summary)]

    stop_line = next((line for line in lines if line.startswith("stop:")), None)

    assert stop_line == expected


# ---------------------------------------------------------------------------
# format_status_finalized
# ---------------------------------------------------------------------------


def test_format_status_finalized_when_given_record_does_name_the_branch_and_commit():
    line = _plain(format_status_finalized(finalize_record()))

    assert line == f"finalized · branch gymrat/{SESSION_ID}-final · commit ccccccc"


def test_format_status_finalized_when_colored_does_embolden_finalized():
    line = _colored(format_status_finalized(finalize_record()))

    assert "1" in styles_at(line, "finalized")


# ---------------------------------------------------------------------------
# Rich markup escape in status rendering (B22)
# ---------------------------------------------------------------------------


def test_format_status_header_when_worktree_path_contains_brackets_does_render_them_literally():
    """A worktree path with brackets must render literally, not as Rich markup."""
    session = session_record(
        baseline=BaselineRef(ref="main", sha=_BASELINE_SHA),
        worktrees=Worktrees(
            experiment="/repo/.gymrat/worktrees/[experiment]",
            baseline="/repo/.gymrat/worktrees/[baseline]",
        ),
    )

    lines = format_status_header(session)

    experiment_line = _plain(lines[2])
    baseline_line = _plain(lines[3])

    assert "[experiment]" in experiment_line
    assert "[baseline]" in baseline_line


def test_format_status_baseline_when_metric_name_contains_brackets_does_render_them_literally():
    """A metric name with brackets must render literally in the baseline line."""
    record = BaselineRecord(
        type="baseline",
        at="2026-08-08T14:15:30.000Z",
        label="main",
        samples=({"total[ms]": 15200},),
    )

    line = _plain(format_status_baseline(record))

    assert "total[ms]" in line
