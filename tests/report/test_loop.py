"""Tests for the loop report fragments.

These cover the loop header, the verdict block, and the status-report
formatters. The header block pins the iteration wording and sample
pluralization; the verdict block pins the line shape, the outcome word, and the
color of each rerun phrase. The status formatters pin the header block, the
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
    RerunConfirmation,
    SettleDiscarded,
    SettleKeepBlocked,
    SettleKept,
    SettleUnsettled,
    StatusIteration,
    StatusSummary,
    baseline_medians,
    format_loop_header,
    format_status_baseline,
    format_status_finalized,
    format_status_footer,
    format_status_header,
    format_status_iteration,
    format_status_stop,
    format_verdict_block,
)
from gymrat.session.workspace import BaselineRef, Worktrees
from tests.report._assertions import render_colored, render_plain, styles_at
from tests.session.records._fixtures import (
    RECOGNIZABLE_BASELINE_SHA,
    RECOGNIZABLE_KEEP_COMMIT,
    SESSION_ID,
    baseline_record,
    finalize_record,
    session_record,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gymrat.report.loop import RerunAnswer, SettleState
    from gymrat.session.schema import Outcome


# ---------------------------------------------------------------------------
# format_loop_header
# ---------------------------------------------------------------------------


def test_format_loop_header_when_given_seq_and_samples_does_render_the_header_line():
    header = format_loop_header(7, 6)

    assert render_plain(header) == "iteration 7 · experiment vs baseline · 6 paired samples"
    assert "1" in styles_at(render_colored(header), "iteration 7")


# ---------------------------------------------------------------------------
# format_verdict_block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "word", "codes"),
    [
        pytest.param("improved", "IMPROVED", ["1", "32"], id="improved-green"),
        pytest.param("regressed", "REGRESSED", ["1", "31"], id="regressed-red"),
        pytest.param("no-signal", "NO-SIGNAL", ["1"], id="no-signal-uncolored"),
    ],
)
def test_format_verdict_block_when_given_outcome_does_render_the_verdict_line(
    outcome: Outcome, word: str, codes: list[str]
):
    block = format_verdict_block(
        outcome=outcome, primary=GeomeanPrimary(delta_pct=-4.2), next_step="gymrat keep"
    )

    assert render_plain(block[0]) == f"primary: -4.2% · verdict: {word}"
    assert styles_at(render_colored(block[0]), word) == codes


def test_format_verdict_block_when_target_reached_but_regressed_does_omit_target_hint():
    block = format_verdict_block(
        outcome="regressed",
        primary=GeomeanPrimary(delta_pct=3.1),
        next_step="fix or run gymrat discard",
        target_reached=True,
    )

    assert [render_plain(line) for line in block] == [
        "primary: +3.1% · verdict: REGRESSED",
        "fix or run gymrat discard",
    ]


@pytest.mark.parametrize(
    ("answer", "phrase", "color_code"),
    [
        pytest.param("confirmed", "regression confirmed on rerun", "31", id="confirmed"),
        pytest.param("disagreed", "regression not confirmed on rerun", "2", id="disagreed"),
        pytest.param("absent", "not measured on rerun", "33", id="absent"),
    ],
)
def test_format_verdict_block_when_rerun_does_color_phrase_by_answer(
    answer: RerunAnswer, phrase: str, color_code: str
):
    rerun = RerunConfirmation(metric="entity/alive_check#time", answer=answer)

    block = format_verdict_block(
        outcome="regressed",
        primary=GeomeanPrimary(delta_pct=3.1),
        next_step="gymrat discard",
        reruns=[rerun],
    )

    assert render_plain(block[0]) == f"entity/alive_check#time: {phrase}"
    assert styles_at(render_colored(block[0]), phrase) == [color_code]


# ---------------------------------------------------------------------------
# status formatters
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    ("worktrees", "expected_worktree_lines"),
    [
        pytest.param(
            Worktrees(
                experiment="/repo/.gymrat/worktrees/experiment",
                baseline="/repo/.gymrat/worktrees/baseline",
            ),
            [
                "experiment worktree /repo/.gymrat/worktrees/experiment",
                "baseline worktree /repo/.gymrat/worktrees/baseline",
            ],
            id="plain-paths",
        ),
        pytest.param(
            Worktrees(
                experiment="/repo/.gymrat/worktrees/[experiment]",
                baseline="/repo/.gymrat/worktrees/[baseline]",
            ),
            [
                "experiment worktree /repo/.gymrat/worktrees/[experiment]",
                "baseline worktree /repo/.gymrat/worktrees/[baseline]",
            ],
            id="bracketed-paths-render-literally",
        ),
    ],
)
def test_format_status_header_when_given_session_does_name_bold_session_baseline_branch_worktrees_adapter(
    worktrees: Worktrees, expected_worktree_lines: list[str]
):
    session = session_record(
        baseline=BaselineRef(ref="main", sha=RECOGNIZABLE_BASELINE_SHA), worktrees=worktrees
    )

    lines = format_status_header(session)

    assert [render_plain(line) for line in lines] == [
        f"session {SESSION_ID} · baseline main@a1b2c3d · adapter metric-lines",
        f"branch gymrat/{SESSION_ID}",
        *expected_worktree_lines,
    ]
    assert "1" in styles_at(render_colored(lines[0]), f"session {SESSION_ID}")


# ---------------------------------------------------------------------------
# format_status_iteration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "expected", "glyph", "glyph_codes"),
    [
        pytest.param(
            "improved", "iteration 1 · ✓ -7.2% · unsettled", "✓", ["1", "32"], id="improved-green"
        ),
        pytest.param(
            "regressed", "iteration 1 · ✗ -7.2% · unsettled", "✗", ["1", "31"], id="regressed-red"
        ),
        pytest.param(
            "no-signal", "iteration 1 · ~ -7.2% · unsettled", "~", ["1"], id="no-signal-uncolored"
        ),
    ],
)
def test_format_status_iteration_when_outcome_varies_does_paint_the_outcome_glyph(
    outcome: Outcome, expected: str, glyph: str, glyph_codes: list[str]
):
    entry = replace(_status_iteration(SettleUnsettled()), outcome=outcome)

    line = format_status_iteration(entry)

    assert render_plain(line) == expected
    assert styles_at(render_colored(line), glyph) == glyph_codes


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(
            _status_iteration(SettleKept(commit=RECOGNIZABLE_KEEP_COMMIT)),
            "iteration 1 · ✓ -7.2% · kept b1b2b3b",
            id="kept-with-commit",
        ),
        pytest.param(
            _status_iteration(SettleKept()), "iteration 1 · ✓ -7.2% · kept", id="kept-pending"
        ),
        pytest.param(
            _status_iteration(SettleDiscarded()),
            "iteration 1 · ✓ -7.2% · discarded",
            id="discarded",
        ),
        pytest.param(
            _status_iteration(SettleUnsettled()),
            "iteration 1 · ✓ -7.2% · unsettled",
            id="unsettled",
        ),
        pytest.param(
            _status_iteration(SettleKeepBlocked(reason="checks-failed")),
            "iteration 1 · ✓ -7.2% · keep-blocked (checks-failed)",
            id="blocked-with-reason",
        ),
        pytest.param(
            _status_iteration(SettleKeepBlocked()),
            "iteration 1 · ✓ -7.2% · keep-blocked",
            id="blocked-no-reason",
        ),
        pytest.param(
            replace(_status_iteration(SettleUnsettled()), delta_pct=None, outcome="no-signal"),
            "iteration 1 · ~ · unsettled",
            id="delta-unmeasured-states-no-percentage",
        ),
    ],
)
def test_format_status_iteration_when_settle_varies_does_state_the_settle(
    entry: StatusIteration, expected: str
):
    line = format_status_iteration(entry)

    assert render_plain(line) == expected


# ---------------------------------------------------------------------------
# baseline_medians
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        pytest.param(({"total_ms": 15200},), {"total_ms": 15200}, id="single-round"),
        pytest.param(
            ({"total_ms": 15200}, {"total_ms": 15184}),
            {"total_ms": 15192},
            id="even-rounds-average-the-middle-pair",
        ),
        pytest.param(
            ({"total_ms": 100}, {"total_ms": 300}, {"total_ms": 260}),
            {"total_ms": 260},
            id="odd-rounds-take-the-middle",
        ),
        pytest.param(
            ({"total_ms": 100, "alloc_bytes": 40}, {"total_ms": 300}),
            {"total_ms": 200, "alloc_bytes": 40},
            id="metric-absent-from-a-round-medians-the-rounds-that-have-it",
        ),
    ],
)
def test_baseline_medians_when_given_record_does_median_each_metric_over_its_rounds(
    samples: tuple[Mapping[str, float], ...], expected: dict[str, float]
):
    medians = baseline_medians(baseline_record(samples=samples))

    assert medians == expected


# ---------------------------------------------------------------------------
# format_status_baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        pytest.param(
            (
                {"total_ms": 15200, "alloc_bytes": 1500},
                {"total_ms": 15184, "alloc_bytes": 1540},
            ),
            "baseline main · total_ms 15192 · alloc_bytes 1520",
            id="two-metrics",
        ),
        pytest.param(
            ({"total[ms]": 15200},),
            "baseline main · total[ms] 15200",
            id="bracketed-name-renders-literally",
        ),
    ],
)
def test_format_status_baseline_when_given_samples_does_state_label_and_median_per_metric(
    samples: tuple[Mapping[str, float], ...], expected: str
):
    record = baseline_record(samples=samples)

    line = render_plain(format_status_baseline(record))

    assert line == expected


# ---------------------------------------------------------------------------
# format_status_footer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("summary", "stop_lines"),
    [
        pytest.param(
            _status_summary(stop=StopConfig(max_iterations=30)),
            ("stop: 4 of 30 iterations",),
            id="max-iterations",
        ),
        pytest.param(
            _status_summary(stop=StopConfig(target_value=95)),
            ("stop: target pending",),
            id="target-pending",
        ),
        pytest.param(
            _status_summary(stop=StopConfig(target_value=95), target_reached=True),
            ("stop: target reached",),
            id="target-reached",
        ),
        pytest.param(
            _status_summary(stop=StopConfig(target_value=95, max_iterations=30)),
            ("stop: 4 of 30 iterations · target pending",),
            id="both-conditions",
        ),
        pytest.param(_status_summary(), (), id="no-stop"),
        pytest.param(_status_summary(stop=StopConfig()), (), id="empty-stop"),
    ],
)
def test_format_status_footer_when_stop_configured_does_state_the_conditions(
    summary: StatusSummary, stop_lines: tuple[str, ...]
):
    lines = format_status_footer(summary)

    assert [render_plain(line) for line in lines] == [
        "4 iterations · 1 kept · 1 discarded",
        *stop_lines,
    ]


# ---------------------------------------------------------------------------
# format_status_finalized
# ---------------------------------------------------------------------------


def test_format_status_finalized_when_given_record_does_render_the_finalized_line():
    line = format_status_finalized(finalize_record())

    assert render_plain(line) == f"finalized · branch gymrat/{SESSION_ID}-final · commit ccccccc"
    assert "1" in styles_at(render_colored(line), "finalized")


# ---------------------------------------------------------------------------
# format_status_stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        pytest.param("user requested stop", "stopped · user requested stop", id="single-line"),
        pytest.param(
            "target reached\ncleaning up\nfinal notes",
            "stopped · target reached",
            id="multiline-keeps-the-first-line",
        ),
    ],
)
def test_format_status_stop_when_given_message_does_render_the_stopped_line(
    message: str, expected: str
):
    line = format_status_stop(message)

    assert render_plain(line) == expected
    assert "1" in styles_at(render_colored(line), "stopped")
