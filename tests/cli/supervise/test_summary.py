"""Tests for the closing summary ``gymrat supervise`` prints when a run ends.

They pin the text and styling of the headline, agent, model, effort, best, loop,
and log rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from gymrat.cli.supervise.reducer import loop_plain_text
from gymrat.cli.supervise.summary import build_summary
from gymrat.cli.supervise.text import NO_SESSION_TEXT
from gymrat.session.store import BestIteration
from gymrat.supervisor.events import SUMMARY_MAX_CHARS
from gymrat.supervisor.exit_sequence import ExitReport, ExitStep
from tests._rich import frame_text
from tests.cli.supervise._fixtures import (
    FRAME_WIDTH,
    make_supervision_result,
    read_result,
    session_state_three_iterations,
)
from tests.session.records._fixtures import session_state

if TYPE_CHECKING:
    from rich.console import RenderableType

    from gymrat.config import Effort
    from gymrat.session.store import ReadSessionResult
    from gymrat.supervisor.driver import SessionEndReason
    from gymrat.supervisor.supervise import EndedBy, SupervisionResult


# ---------------------------------------------------------------------------
# closing summary
# ---------------------------------------------------------------------------

_LOG_PATH = "/repo/.gymrat/supervisor-1.jsonl"
_LOG_ROW = f"  log     {_LOG_PATH}"
_NO_SESSION_ROW = f"  loop    {NO_SESSION_TEXT}"

#: An exit sequence that took no step, so the summary shows no ``exit`` row.
_NO_EXIT_STEPS = ExitReport(steps=())


def _summary(supervision: SupervisionResult | None = None, **overrides: Any) -> RenderableType:
    """Build the closing summary with the standard log path, no session and no exit step.

    Args:
        supervision: The run's result; a completed session run by default.
        **overrides: ``build_summary`` keywords that replace the defaults.

    Returns:
        The closing summary renderable.
    """
    options: dict[str, Any] = {
        "log_path": _LOG_PATH,
        "session_result": None,
        "exit_report": _NO_EXIT_STEPS,
        **overrides,
    }
    return build_summary(
        supervision if supervision is not None else make_supervision_result(), **options
    )


def _session_result() -> ReadSessionResult:
    """A three-iteration session whose third iteration is the best."""
    return read_result(
        session_state_three_iterations(-4.2, "improved", seq=3),
        has_baseline=True,
        best=BestIteration(
            delta_pct=-4.2,
            seq=3,
            label="wall_time",
            baseline_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        ),
    )


def test_summary_when_run_has_a_best_iteration_does_render_the_best_row():
    summary = _summary(session_result=_session_result())

    assert frame_text(summary, width=FRAME_WIDTH) == (
        "✓ completed · 1m 0s · $0.05\n"
        "  best    -4.2% wall_time vs baseline a1b2c3d (iteration 3)\n"
        f"  loop    {loop_plain_text(_session_result(), None)}\n"
        f"{_LOG_ROW}"
    )


_CAPPED_OR_ERRORED = [
    pytest.param(
        "interrupted",
        "wall-clock",
        "! interrupted by wall-clock cap · 1m 0s · $0.05",
        id="wall-clock-cap",
    ),
    pytest.param(
        "interrupted",
        "spend-cap",
        "! interrupted by spend cap · 1m 0s · $0.05",
        id="spend-cap",
    ),
    pytest.param("error", "session", "✗ error · 1m 0s · $0.05", id="error"),
]


@pytest.mark.parametrize(("reason", "ended_by", "headline"), _CAPPED_OR_ERRORED)
def test_summary_when_run_capped_or_errored_does_hide_the_agent_row(
    reason: SessionEndReason, ended_by: EndedBy, headline: str
) -> None:
    summary = _summary(
        make_supervision_result(reason=reason, ended_by=ended_by), final_text="Some final text."
    )

    assert frame_text(summary, width=FRAME_WIDTH) == (f"{headline}\n{_NO_SESSION_ROW}\n{_LOG_ROW}")


# ---------------------------------------------------------------------------
# closing summary — agent row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("final_text", "expected_lines"),
    [
        pytest.param(
            "Line one.\nLine two.",
            [
                "✓ completed · 1m 0s · $0.05",
                "  agent   Line one.",
                "          Line two.",
                _NO_SESSION_ROW,
                _LOG_ROW,
            ],
            id="single-newline",
        ),
    ],
)
def test_summary_agent_row_when_final_text_given_does_indent_continuation_lines_under_content(
    final_text: str, expected_lines: list[str]
) -> None:
    summary = _summary(
        make_supervision_result(reason="completed", ended_by="session"), final_text=final_text
    )

    text = frame_text(summary, width=FRAME_WIDTH)

    assert text.splitlines() == expected_lines


# ---------------------------------------------------------------------------
# closing summary — agent row clipping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("final_text", "expected_agent_row"),
    [
        pytest.param(
            "x" * SUMMARY_MAX_CHARS, f"  agent   {'x' * SUMMARY_MAX_CHARS}", id="at-threshold"
        ),
        pytest.param(
            "a" * (SUMMARY_MAX_CHARS + 50),
            f"  agent   {'a' * SUMMARY_MAX_CHARS}… (full message in log)",
            id="over-threshold",
        ),
    ],
)
def test_summary_agent_row_when_text_length_varies_does_clip_only_past_the_threshold(
    final_text: str, expected_agent_row: str
):
    summary = _summary(
        make_supervision_result(reason="completed", ended_by="session"), final_text=final_text
    )

    text = frame_text(summary, width=FRAME_WIDTH + 200)

    assert text.splitlines()[1] == expected_agent_row


def test_summary_agent_row_when_clipped_text_is_multiline_does_indent_every_continuation_line():
    lead = "Line one.\nLine two.\n"
    long_text = lead + "c" * (SUMMARY_MAX_CHARS + 50)

    summary = _summary(
        make_supervision_result(reason="completed", ended_by="session"), final_text=long_text
    )

    agent_rows = frame_text(summary, width=FRAME_WIDTH + 200).splitlines()[1:-2]

    assert agent_rows == [
        "  agent   Line one.",
        "          Line two.",
        f"          {'c' * (SUMMARY_MAX_CHARS - len(lead))}… (full message in log)",
    ]


# ---------------------------------------------------------------------------
# closing summary — model and effort display
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "effort", "expected_line"),
    [
        pytest.param("opus", None, "  model   opus", id="model-only"),
        pytest.param(None, "high", "  effort  high", id="effort-only"),
    ],
)
def test_summary_when_model_or_effort_in_force_does_show_labelled_row(
    model: str | None, effort: Effort | None, expected_line: str
) -> None:
    summary = _summary(model=model, effort=effort)

    text = frame_text(summary, width=FRAME_WIDTH)

    assert text.splitlines() == [
        "✓ completed · 1m 0s · $0.05",
        expected_line,
        _NO_SESSION_ROW,
        _LOG_ROW,
    ]


# ---------------------------------------------------------------------------
# closing summary — stop-condition, guard and hook-failure endings
# ---------------------------------------------------------------------------

_STOP_CONDITION_REASON = "max iterations (2 of 2)"
_HOOK_FAILURE_REASON = "after hook failed on iteration 2: exit 1 (stdout 80 B, stderr 5 B)"
_EMPTY_LOOP_ROW = (
    f"  loop    {loop_plain_text(read_result(session_state(), has_baseline=True), None)}"
)


@pytest.mark.parametrize(
    ("ended_by", "end_reason", "expected_lines"),
    [
        pytest.param(
            "stop-condition",
            _STOP_CONDITION_REASON,
            [
                f"✓ stopped: {_STOP_CONDITION_REASON} · 1m 0s · $0.05",
                "  agent   Some final text.",
                _EMPTY_LOOP_ROW,
                _LOG_ROW,
            ],
            id="stop-condition-agent-text",
        ),
        pytest.param(
            "hook-failure",
            _HOOK_FAILURE_REASON,
            [f"! stopped: {_HOOK_FAILURE_REASON} · 1m 0s · $0.05", _EMPTY_LOOP_ROW, _LOG_ROW],
            id="hook-failure",
        ),
        pytest.param(
            "guard",
            "safety limit reached",
            [
                "! stopped by guard: safety limit reached · 1m 0s · $0.05",
                "  agent   Some final text.",
                _EMPTY_LOOP_ROW,
                _LOG_ROW,
            ],
            id="guard-shows-the-agent-row",
        ),
    ],
)
def test_summary_when_a_stop_guard_or_hook_ended_the_run_does_render_its_ending(
    ended_by: EndedBy, end_reason: str, expected_lines: list[str]
) -> None:
    session_result = read_result(session_state(), has_baseline=True)

    summary = _summary(
        make_supervision_result(reason="interrupted", ended_by=ended_by, end_reason=end_reason),
        session_result=session_result,
        final_text="Some final text.",
    )

    assert frame_text(summary, width=FRAME_WIDTH).splitlines() == expected_lines


def test_summary_agent_row_when_stop_message_recorded_does_show_it_over_the_final_text() -> None:
    session_result = read_result(
        session_state(), has_baseline=True, stop_message="Target reached, stopping."
    )

    summary = _summary(
        make_supervision_result(
            reason="interrupted", ended_by="stop-condition", end_reason=_STOP_CONDITION_REASON
        ),
        session_result=session_result,
        final_text="Some final text.",
    )

    assert frame_text(summary, width=FRAME_WIDTH).splitlines() == [
        f"✓ stopped: {_STOP_CONDITION_REASON} · 1m 0s · $0.05",
        "  agent   Target reached, stopping.",
        _EMPTY_LOOP_ROW,
        _LOG_ROW,
    ]


# ---------------------------------------------------------------------------
# closing summary — exit rows
# ---------------------------------------------------------------------------

_EXIT_STEPS = (
    ExitStep(kind="settled", text="kept iteration 3 (-4.2% wall_time)"),
    ExitStep(kind="nothing", text="session already finalized"),
)
_EXIT_ROWS = [
    "  exit    kept iteration 3 (-4.2% wall_time)",
    "  exit    session already finalized",
]


@pytest.mark.parametrize(
    ("error", "expected_exit_rows"),
    [
        pytest.param(None, _EXIT_ROWS, id="steps-only"),
        pytest.param(
            "finalize failed: disk full",
            [*_EXIT_ROWS, "  exit    error: finalize failed: disk full"],
            id="steps-then-error",
        ),
    ],
)
def test_summary_when_exit_report_given_does_render_exit_rows_between_loop_and_log(
    error: str | None, expected_exit_rows: list[str]
) -> None:
    summary = _summary(exit_report=ExitReport(steps=_EXIT_STEPS, error=error))

    assert frame_text(summary, width=FRAME_WIDTH).splitlines() == [
        "✓ completed · 1m 0s · $0.05",
        _NO_SESSION_ROW,
        *expected_exit_rows,
        _LOG_ROW,
    ]
