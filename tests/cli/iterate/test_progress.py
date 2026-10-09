"""Tests for the iterate progress renderer (live checklist + plain modes).

Tests inject a deterministic ``Clock`` from ``tests._rich`` and capture
output through ``sealed_console``.  Frame content is pinned with syrupy
golden snapshots; plain-mode milestones use exact-line equality; live wiring
assertions check ``Live`` attributes directly.
"""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING

import pytest

from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmSkipped,
    ConfirmStarted,
    HookFinished,
    HookStarted,
    IterationRecorded,
    JudgeFinished,
    JudgeStarted,
    PrepareFinished,
    PrepareStarted,
)
from tests._rich import (
    Clock,
    console_output,
    frame_text,
    sealed_console,
)
from tests.cli._progress_helpers import iterate_renderer, report_full_pass
from tests.cli._progress_helpers import ms_from_clock as _ms
from tests.cli._progress_helpers import pass_started as _pass_started

if TYPE_CHECKING:
    from rich.console import Console, RenderableType
    from rich.segment import Segment
    from syrupy.assertion import SnapshotAssertion

    from gymrat.cli.iterate.progress import IterateRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: A metric name holding rich emoji-code syntax (``:fire:``), which must print literally.
_EMOJI_LIKE_METRIC = "cpu:fire:total"

#: A regressed metric name holding emoji-code and fragment syntax, which must print literally.
_EMOJI_LIKE_REGRESSED = "lat:100:p99#time"


def _last_line(console: Console) -> str:
    lines = [ln for ln in console_output(console).splitlines() if ln.strip()]
    return lines[-1]


def _sample_one_round(renderer: IterateRenderer, clock: Clock[float]) -> None:
    for label in ("baseline", "candidate"):
        report_full_pass(renderer, clock, 1, 1, target_count=2, label=label, duration_s=2.5)


def _frame(renderer: IterateRenderer, clock: Clock[float], *, width: int = 80) -> str:
    return frame_text(renderer.frame(), width=width, get_time=clock)


def _style_name(segment: Segment) -> str:
    return str(segment.style) if segment.style else ""


def _judge_detail_style_runs(renderable: RenderableType) -> list[tuple[str, str]]:
    """Render *renderable* in color and return its judge row's detail.

    Args:
        renderable: The iterate frame to render.

    Returns:
        The detail as style-merged ``(text, style)`` runs. The done glyph and
        the ``judged`` label that open the row are dropped, so only the
        verdict's own styling remains.
    """
    styled = sealed_console(width=120, no_color=False, color_system="truecolor")
    lines = styled.render_lines(renderable, pad=False)
    judge_row = next(line for line in lines if "judged" in "".join(seg.text for seg in line))
    runs = [
        ("".join(seg.text for seg in segments), style)
        for style, segments in itertools.groupby(judge_row, key=_style_name)
    ]
    label_end = next(i for i, (text, _style) in enumerate(runs) if text.endswith("judged "))
    return runs[label_end + 1 :]


_live = functools.partial(iterate_renderer, "live")
_plain = functools.partial(iterate_renderer, "plain")


# ---------------------------------------------------------------------------
# Initial frame, hooks and prepare
# ---------------------------------------------------------------------------


def test_frame_when_initial_does_show_all_nodes_pending(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(seq=3, session_id="abc-123", metric_count=4)

    result = _frame(renderer, clock)

    assert result == snapshot


def test_frame_when_before_hook_running_does_show_spinner(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(has_before_hook=True)

    renderer.report(HookStarted(stage="before", at_ms=0))

    assert _frame(renderer, clock) == snapshot


def test_frame_when_worktree_preparing_does_name_its_target(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live()

    renderer.report(PrepareStarted(label="baseline", at_ms=0))

    assert _frame(renderer, clock) == snapshot


def test_frame_when_both_worktrees_prepared_does_show_elapsed(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live()
    renderer.report(HookFinished(stage="before", at_ms=0))
    renderer.report(PrepareStarted(label="baseline", at_ms=0))
    clock.tick(3)
    renderer.report(PrepareFinished(label="baseline", at_ms=_ms(clock)))
    renderer.report(PrepareStarted(label="candidate", at_ms=_ms(clock)))
    clock.tick(2)

    renderer.report(PrepareFinished(label="candidate", at_ms=_ms(clock)))

    assert _frame(renderer, clock) == snapshot


# ---------------------------------------------------------------------------
# Sampling passes
# ---------------------------------------------------------------------------


def test_frame_when_passes_mid_run_does_show_pass_progress(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=5)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    report_full_pass(renderer, clock, 1, 5, target_count=2, label="baseline", duration_s=10)
    clock.tick(1)
    report_full_pass(renderer, clock, 1, 5, target_count=2, label="candidate", duration_s=10)
    clock.tick(1)
    renderer.report(
        _pass_started(
            2,
            5,
            target_count=2,
            label="baseline",
            at_ms=_ms(clock),
        )
    )
    clock.tick(41)

    result = _frame(renderer, clock)

    assert result == snapshot


def test_frame_when_header_before_first_pass_completes_does_show_elapsed_without_eta(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live()
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(7)
    renderer.report(
        _pass_started(1, 5, label="baseline", at_ms=_ms(clock)),
    )
    clock.tick(3)

    result = _frame(renderer, clock)

    assert result == snapshot


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def test_frame_when_judge_started_does_show_running_with_elapsed(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=1)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    _sample_one_round(renderer, clock)
    clock.tick(1)
    renderer.report(JudgeStarted(at_ms=_ms(clock)))
    clock.tick(3)

    result = _frame(renderer, clock)

    assert result == snapshot


def test_frame_when_judge_finished_after_started_does_show_elapsed(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=1)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    _sample_one_round(renderer, clock)
    clock.tick(1)
    renderer.report(JudgeStarted(at_ms=_ms(clock)))
    clock.tick(4)

    renderer.report(JudgeFinished(primary_delta_pct=-3.2, regressed=("latency",), at_ms=_ms(clock)))

    assert _frame(renderer, clock) == snapshot


def test_frame_when_metric_names_look_like_emoji_codes_does_print_them_literally(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=1, primary_metric=_EMOJI_LIKE_METRIC)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    _sample_one_round(renderer, clock)
    clock.tick(1)

    renderer.report(
        JudgeFinished(primary_delta_pct=-3.2, regressed=(_EMOJI_LIKE_REGRESSED,), at_ms=_ms(clock))
    )

    assert _frame(renderer, clock) == snapshot


def test_frame_when_judge_finished_no_regressions_does_drop_the_confirm_row(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=1, metric_count=4)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    _sample_one_round(renderer, clock)
    clock.tick(1)

    renderer.report(JudgeFinished(primary_delta_pct=-2.0, regressed=(), at_ms=_ms(clock)))

    assert _frame(renderer, clock) == snapshot


def test_frame_when_judge_finished_does_dim_wording_and_style_regressed_names_inline():
    _console, clock, renderer = _live(sample_count=1, metric_count=5)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    _sample_one_round(renderer, clock)
    clock.tick(1)
    regressed = ("node/access#time", "parse[json]", "throughput", "alloc")

    renderer.report(JudgeFinished(primary_delta_pct=-3.2, regressed=regressed, at_ms=_ms(clock)))

    assert _judge_detail_style_runs(renderer.frame()) == [
        ("-3.2% on geomean · 4 regressed: node/", "dim"),
        ("access", ""),
        ("#time, ", "dim"),
        ("parse[json]", ""),
        (", ", "dim"),
        ("throughput", ""),
        (", …", "dim"),
    ]


# ---------------------------------------------------------------------------
# Confirm
# ---------------------------------------------------------------------------


def test_frame_when_confirm_runs_after_an_alerting_judge_does_show_its_progress_row(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=5)
    clock.tick(5)
    renderer.report(JudgeFinished(primary_delta_pct=2.5, regressed=("latency",), at_ms=_ms(clock)))
    clock.tick(0.1)
    renderer.report(ConfirmStarted(filtered_metrics=("latency",), at_ms=_ms(clock)))
    clock.tick(0.9)

    renderer.report(
        _pass_started(
            1,
            5,
            target_count=2,
            label="baseline",
            at_ms=_ms(clock),
            phase="confirm",
        )
    )

    assert _frame(renderer, clock) == snapshot


def test_frame_when_confirm_skipped_after_a_regression_does_keep_a_skipped_confirm_row(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(sample_count=1)
    clock.tick(5)
    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("latency",), at_ms=_ms(clock)))
    renderer.report(ConfirmSkipped(at_ms=_ms(clock)))
    clock.tick(1)
    renderer.report(IterationRecorded(seq=1, outcome="regressed", at_ms=_ms(clock)))

    result = _frame(renderer, clock)

    assert result == snapshot


def test_frame_when_confirm_finished_does_show_summary_on_node_line(snapshot: SnapshotAssertion):
    _console, clock, renderer = _live(sample_count=2)
    clock.tick(5)
    renderer.report(JudgeFinished(primary_delta_pct=2.0, regressed=("x",), at_ms=_ms(clock)))
    clock.tick(0.1)
    renderer.report(ConfirmStarted(filtered_metrics=("x",), at_ms=_ms(clock)))
    for round_num, label in itertools.product((1, 2), ("baseline", "experiment")):
        clock.tick(0.5)
        report_full_pass(
            renderer,
            clock,
            round_num,
            2,
            duration_s=0.5,
            target_count=2,
            label=label,
            phase="confirm",
        )
    clock.tick(10.9)

    renderer.report(ConfirmFinished(reproduced=True, at_ms=_ms(clock)))

    assert _frame(renderer, clock) == snapshot


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


def test_frame_when_recorded_with_checks_cmd_does_show_gymrat_keep(
    snapshot: SnapshotAssertion,
):
    _console, clock, renderer = _live(
        seq=3,
        checks_cmd="npm run check && npm test",
    )
    clock.tick(15)

    renderer.report(IterationRecorded(seq=3, outcome="unsettled", at_ms=_ms(clock)))

    assert _frame(renderer, clock, width=100) == snapshot


# ---------------------------------------------------------------------------
# Compact layout -- one progress row on a short terminal
# ---------------------------------------------------------------------------


def test_frame_when_compact_layout_does_show_single_row():
    _console, clock, renderer = _live(height=10, sample_count=5)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    clock.tick(1)
    renderer.report(
        _pass_started(
            1,
            5,
            target_count=2,
            label="A",
            at_ms=_ms(clock),
        )
    )

    result = _frame(renderer, clock)

    assert result == "⠋ sampling                                            0% · A · 00:00/--:--"


def test_frame_when_compact_pass_finished_does_show_progress_with_an_eta():
    _console, clock, renderer = _live(height=10, sample_count=5)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    report_full_pass(renderer, clock, 1, 5, label="A", duration_s=2)

    renderer.report(_pass_started(2, 5, target_count=1, label="B", at_ms=_ms(clock)))

    assert _frame(renderer, clock) == (
        "⠴ sampling ━━━━                                      10% · B · 00:02/00:20"
    )


def test_frame_when_compact_confirm_started_does_restart_progress_for_confirm():
    _console, clock, renderer = _live(height=10, sample_count=1, metric_count=3)
    renderer.report(PrepareFinished(label="bench", at_ms=0))
    report_full_pass(renderer, clock, 1, 1, duration_s=5)
    renderer.report(JudgeFinished(primary_delta_pct=-2.5, regressed=("latency",), at_ms=_ms(clock)))

    renderer.report(ConfirmStarted(filtered_metrics=("latency",), at_ms=_ms(clock)))

    assert _frame(renderer, clock) == (
        "⠹ confirming                                            0%  00:00/00:00"
    )


# ---------------------------------------------------------------------------
# Plain mode -- exact timestamped milestone lines
# ---------------------------------------------------------------------------


def test_report_when_plain_judge_names_look_like_emoji_codes_does_print_them_literally():
    console, _clock, renderer = _plain(width=120, metric_count=5, primary_metric=_EMOJI_LIKE_METRIC)

    renderer.report(
        JudgeFinished(primary_delta_pct=2.0, regressed=(_EMOJI_LIKE_REGRESSED,), at_ms=0)
    )

    assert _last_line(console) == (
        "[00:00:00] judge +2.0% on cpu:fire:total · 1 regressed: lat:100:p99#time"
    )


# ---------------------------------------------------------------------------
# Live wiring -- Live attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verbose", "expected_transient"),
    [
        pytest.param(False, True, id="verbose-off"),
        pytest.param(True, False, id="verbose-on"),
    ],
)
def test_iterate_renderer_when_created_does_set_live_transient_from_verbose(
    verbose: bool,
    expected_transient: bool,
):
    _console, _clock, renderer = _live(verbose=verbose)

    assert renderer.live is not None
    assert renderer.live.transient is expected_transient
