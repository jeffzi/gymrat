"""Tests for the loop report fragments.

These cover the loop header, the verdict block, and the status-report
formatters. The header block pins the iteration wording and sample
pluralization; the verdict block pins the line shape, the outcome word, and the
dimmed metric name on rerun lines. The status formatters pin the header block, the
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

from gymrat.config.types import StopConfig
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
    SESSION_ID,
    baseline_record,
    finalize_record,
    session_record,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gymrat.report.loop import LoopOutcome, RerunAnswer, SettleState


def _geomean_primary(delta_pct: float = -4.2) -> GeomeanPrimary:
    """The geomean primary, improving by default."""
    return GeomeanPrimary(delta_pct=delta_pct)


# ---------------------------------------------------------------------------
# format_loop_header
# ---------------------------------------------------------------------------


def test_format_loop_header_when_given_seq_and_samples_does_name_iteration_comparison_and_count():
    header = render_plain(format_loop_header(7, 6))

    assert header == "iteration 7 · experiment vs baseline · 6 paired samples"


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        pytest.param(1, "· 1 paired sample", id="one"),
        pytest.param(2, "· 2 paired samples", id="many"),
    ],
)
def test_format_loop_header_when_counting_samples_does_match_noun(samples: int, expected: str):
    header = render_plain(format_loop_header(1, samples))

    assert expected in header


def test_format_loop_header_when_colored_does_embolden_the_iteration_label():
    header = render_colored(format_loop_header(7, 6))

    assert "1" in styles_at(header, "iteration 7")


def test_format_loop_header_when_colored_does_dim_each_separator():
    header = render_colored(format_loop_header(7, 6))

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

    assert render_plain(block[0]) == f"primary: -4.2% · verdict: {word}"


def test_format_verdict_block_when_given_next_step_does_close_the_block_with_it():
    block = format_verdict_block(
        outcome="regressed", primary=_geomean_primary(3.1), next_step="fix or gymrat discard"
    )

    assert len(block) == 2
    assert render_plain(block[1]) == "fix or gymrat discard"


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

    codes = styles_at(render_colored(block[0]), word)

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

    plain = "\n".join(render_plain(line) for line in block)
    assert "target reached" not in plain


@pytest.mark.parametrize(
    ("answer", "phrase"),
    [
        pytest.param("confirmed", "regression confirmed on rerun", id="confirmed"),
        pytest.param("disagreed", "regression not confirmed on rerun", id="disagreed"),
        pytest.param("absent", "not measured on rerun", id="absent"),
    ],
)
def test_format_verdict_block_when_rerun_does_dim_group_and_kind_in_metric_name(
    answer: RerunAnswer, phrase: str
):
    rerun = RerunConfirmation(metric="entity/alive_check#time", answer=answer)

    block = format_verdict_block(
        outcome="regressed",
        primary=GeomeanPrimary(delta_pct=3.1),
        next_step="gymrat discard",
        reruns=[rerun],
    )

    rerun_line = next(line for line in block if phrase in render_plain(line))
    colored = render_colored(rerun_line)
    assert "2" in styles_at(colored, "entity/")
    assert "2" in styles_at(colored, "#time")


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

    assert [render_plain(line) for line in lines] == [
        f"session {SESSION_ID} · baseline main@a1b2c3d · adapter metric-lines",
        f"branch gymrat/{SESSION_ID}",
        "experiment worktree /repo/.gymrat/worktrees/experiment",
        "baseline worktree /repo/.gymrat/worktrees/baseline",
    ]


def test_format_status_header_when_colored_does_embolden_the_session():
    session = session_record(baseline=BaselineRef(ref="main", sha=_BASELINE_SHA))

    header = render_colored(format_status_header(session)[0])

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
    assert render_plain(format_status_iteration(_status_iteration(settle))) == expected


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

    assert (
        render_plain(format_status_iteration(entry)) == f"iteration 1 · {glyph} -7.2% · unsettled"
    )


def test_format_status_iteration_when_delta_unmeasured_does_state_no_percentage():
    entry = replace(_status_iteration(SettleUnsettled()), delta_pct=None, outcome="no-signal")

    assert render_plain(format_status_iteration(entry)) == "iteration 1 · ~ · unsettled"


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

    assert color_code in styles_at(render_colored(format_status_iteration(entry)), glyph)


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


def test_format_status_baseline_when_given_samples_does_state_label_and_median_per_metric():
    record = baseline_record(
        samples=(
            {"total_ms": 15200, "alloc_bytes": 1500},
            {"total_ms": 15184, "alloc_bytes": 1540},
        )
    )

    line = render_plain(format_status_baseline(record))

    assert line == "baseline main · total_ms 15192 · alloc_bytes 1520"


def test_format_status_baseline_when_a_round_omits_a_metric_does_median_over_rounds_that_reported_it():
    record = baseline_record(samples=({"total_ms": 100, "alloc_bytes": 40}, {"total_ms": 300}))

    line = render_plain(format_status_baseline(record))

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
    assert render_plain(format_status_footer(summary)[0]) == expected


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
    lines = [render_plain(line) for line in format_status_footer(summary)]

    stop_line = next((line for line in lines if line.startswith("stop:")), None)

    assert stop_line == expected


# ---------------------------------------------------------------------------
# format_status_finalized
# ---------------------------------------------------------------------------


def test_format_status_finalized_when_given_record_does_name_the_branch_and_commit():
    line = render_plain(format_status_finalized(finalize_record()))

    assert line == f"finalized · branch gymrat/{SESSION_ID}-final · commit ccccccc"


def test_format_status_finalized_when_colored_does_embolden_finalized():
    line = render_colored(format_status_finalized(finalize_record()))

    assert "1" in styles_at(line, "finalized")


# ---------------------------------------------------------------------------
# Rich markup escape in status rendering (B22)
# ---------------------------------------------------------------------------


def test_format_status_header_when_worktree_path_contains_brackets_does_render_them_literally():
    session = session_record(
        baseline=BaselineRef(ref="main", sha=_BASELINE_SHA),
        worktrees=Worktrees(
            experiment="/repo/.gymrat/worktrees/[experiment]",
            baseline="/repo/.gymrat/worktrees/[baseline]",
        ),
    )

    lines = format_status_header(session)

    experiment_line = render_plain(lines[2])
    baseline_line = render_plain(lines[3])

    assert "[experiment]" in experiment_line
    assert "[baseline]" in baseline_line


def test_format_status_baseline_when_metric_name_contains_brackets_does_render_them_literally():
    record = baseline_record(samples=({"total[ms]": 15200},))

    line = render_plain(format_status_baseline(record))

    assert "total[ms]" in line


# ---------------------------------------------------------------------------
# format_status_stop
# ---------------------------------------------------------------------------


def test_format_status_stop_when_given_single_line_message_does_render_stopped_and_message():
    line = render_plain(format_status_stop("user requested stop"))

    assert line == "stopped · user requested stop"


def test_format_status_stop_when_given_multiline_message_does_render_only_the_first_line():
    line = render_plain(format_status_stop("target reached\ncleaning up\nfinal notes"))

    assert line == "stopped · target reached"


def test_format_status_stop_when_colored_does_embolden_stopped():
    line = render_colored(format_status_stop("user requested stop"))

    assert "1" in styles_at(line, "stopped")


def test_format_status_stop_when_colored_does_dim_the_separator():
    line = render_colored(format_status_stop("user requested stop"))

    assert "2" in styles_at(line, "·")
