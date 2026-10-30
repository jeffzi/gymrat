"""Tests for the loop report fragments.

These cover the loop header, the verdict block, the start summary, and the
status-report formatters. The header block pins the iteration wording and sample
pluralization; the verdict block pins the line shape, the target hint, and where
rerun lines sit. The status formatters pin the header block, the per-iteration
line (glyph, delta, and settle state), the baseline medians, the
totals-and-stop footer, and the finalized closer.
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
    format_loop_header,
    format_start_summary,
    format_status_baseline,
    format_status_finalized,
    format_status_footer,
    format_status_header,
    format_status_iteration,
    format_status_stop,
    format_verdict_block,
)
from gymrat.session.workspace import BaselineRef, Worktrees
from tests.report._assertions import render_plain
from tests.session.records._fixtures import (
    ARCHIVED_SESSION_ID,
    ARCHIVED_SESSION_PATH,
    RECOGNIZABLE_BASELINE_SHA,
    RECOGNIZABLE_KEEP_COMMIT,
    SESSION_ID,
    baseline_record,
    finalize_record,
    session_record,
    start_result,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gymrat.loop.start import StartResult
    from gymrat.report.loop import SettleState


# ---------------------------------------------------------------------------
# format_loop_header
# ---------------------------------------------------------------------------


def test_format_loop_header_when_given_seq_and_samples_does_render_the_header_line():
    header = format_loop_header(7, 6)

    assert render_plain(header) == "iteration 7 · experiment vs baseline · 6 paired samples"


# ---------------------------------------------------------------------------
# format_verdict_block
# ---------------------------------------------------------------------------


def test_format_verdict_block_when_target_reached_does_state_the_target_above_the_next_step():
    block = format_verdict_block(
        outcome="improved",
        primary=GeomeanPrimary(delta_pct=-4.2),
        next_step="gymrat keep",
        target_reached=True,
    )

    assert [render_plain(line) for line in block] == [
        "primary: -4.2% · verdict: IMPROVED",
        "target reached — keep it",
        "gymrat keep",
    ]


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


def test_format_verdict_block_when_rerun_does_lead_with_the_metric_and_its_answer():
    rerun = RerunConfirmation(metric="entity/alive_check#time", answer="confirmed")

    block = format_verdict_block(
        outcome="regressed",
        primary=GeomeanPrimary(delta_pct=3.1),
        next_step="gymrat discard",
        reruns=[rerun],
    )

    assert render_plain(block[0]) == "entity/alive_check#time: regression confirmed on rerun"


# ---------------------------------------------------------------------------
# format_start_summary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "runbook", "extra_rows"),
    [
        pytest.param(
            start_result(),
            None,
            [],
            id="neither",
        ),
        pytest.param(
            start_result(),
            "RUNBOOK.md",
            ["  runbook: RUNBOOK.md — read it before your first edit"],
            id="runbook",
        ),
        pytest.param(
            start_result(archived=ARCHIVED_SESSION_ID, archived_path=ARCHIVED_SESSION_PATH),
            None,
            [f"  archived the finalized session {ARCHIVED_SESSION_ID} to {ARCHIVED_SESSION_PATH}"],
            id="archived",
        ),
    ],
)
def test_format_start_summary_when_runbook_or_archive_given_does_add_its_row(
    result: StartResult, runbook: str | None, extra_rows: list[str]
):
    lines = format_start_summary(result, runbook).split("\n")

    assert lines[5:-1] == extra_rows


def test_format_start_summary_when_started_does_name_the_branch():
    session = session_record()

    lines = format_start_summary(start_result(session=session), None).split("\n")

    assert session.branch in lines[1]


def test_format_start_summary_when_started_does_close_on_the_experiment_worktree():
    session = session_record()

    lines = format_start_summary(start_result(session=session), None).split("\n")

    assert session.worktrees.experiment in lines[-1]


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


def test_format_status_header_when_given_session_does_name_session_baseline_branch_worktrees_adapter():
    worktrees = Worktrees(
        experiment="/repo/.gymrat/worktrees/[experiment]",
        baseline="/repo/.gymrat/worktrees/[baseline]",
    )
    session = session_record(
        baseline=BaselineRef(ref="main", sha=RECOGNIZABLE_BASELINE_SHA), worktrees=worktrees
    )

    lines = format_status_header(session)

    assert [render_plain(line) for line in lines] == [
        f"session {SESSION_ID} · baseline main@a1b2c3d · adapter metric-lines",
        f"branch gymrat/{SESSION_ID}",
        "experiment worktree /repo/.gymrat/worktrees/[experiment]",
        "baseline worktree /repo/.gymrat/worktrees/[baseline]",
    ]


# ---------------------------------------------------------------------------
# format_status_iteration
# ---------------------------------------------------------------------------


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
            _status_iteration(SettleKept(outcome="regressed")),
            "iteration 1 · ✓ -7.2% · kept (regressed)",
            id="kept-overriding-the-outcome",
        ),
        pytest.param(
            _status_iteration(SettleDiscarded()),
            "iteration 1 · ✓ -7.2% · discarded",
            id="discarded",
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
    ],
)
def test_format_status_iteration_when_settle_varies_does_state_the_settle(
    entry: StatusIteration, expected: str
):
    line = format_status_iteration(entry)

    assert render_plain(line) == expected


def test_format_status_iteration_when_delta_unmeasured_does_omit_the_percentage():
    entry = replace(_status_iteration(SettleUnsettled()), delta_pct=None, outcome="no-signal")

    line = format_status_iteration(entry)

    assert render_plain(line) == "iteration 1 · ~ · unsettled"


# ---------------------------------------------------------------------------
# format_status_baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        pytest.param(
            ({"total_ms": 15200, "alloc_bytes": 1500},),
            "baseline main · total_ms 15200 · alloc_bytes 1500",
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


# ---------------------------------------------------------------------------
# format_status_stop
# ---------------------------------------------------------------------------


def test_format_status_stop_when_given_message_does_render_the_stopped_line():
    line = format_status_stop("user requested stop")

    assert render_plain(line) == "stopped · user requested stop"
