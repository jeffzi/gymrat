"""Tests for the closing summary ``gymrat supervise`` prints when a run ends.

They pin the text and styling of the headline, agent, model, effort, best, loop,
and log rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from gymrat.cli.supervise.summary import build_summary
from gymrat.cli.supervise.types import BestIteration
from gymrat.supervisor.events import SUMMARY_MAX_CHARS
from gymrat.supervisor.exit_sequence import ExitReport, ExitStep
from tests._ansi import SGR_GREEN, SGR_RED, SGR_YELLOW, assert_has_sgr
from tests._rich import frame_text
from tests.cli.supervise._fixtures import (
    FRAME_WIDTH,
    make_read_session,
    make_supervision_result,
    render_colored,
    session_state_three_iterations,
)
from tests.session.records._fixtures import session_state

if TYPE_CHECKING:
    from rich.console import RenderableType

    from gymrat.cli.supervise.types import ReadSessionResult
    from gymrat.config import Effort
    from gymrat.supervisor.driver import SessionEndReason
    from gymrat.supervisor.supervise import EndedBy, SupervisionResult


# ---------------------------------------------------------------------------
# closing summary
# ---------------------------------------------------------------------------

_LOG_PATH = "/repo/.gymrat/supervisor-1.jsonl"
_LOG_ROW = f"  log     {_LOG_PATH}"

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


def _session_result(*, with_best: bool) -> ReadSessionResult:
    """A three-iteration session, with or without a best-iteration record."""
    state = session_state_three_iterations(-4.2, "improved", seq=3)
    if not with_best:
        return make_read_session(state, has_baseline=True)()
    return make_read_session(
        state,
        has_baseline=True,
        best=BestIteration(
            delta_pct=-4.2,
            seq=3,
            label="wall_time",
            baseline_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        ),
    )()


def test_summary_when_run_has_a_best_iteration_does_render_the_best_row():
    summary = _summary(session_result=_session_result(with_best=True))

    assert frame_text(summary, width=FRAME_WIDTH) == (
        "✓ completed · 1m 0s · $0.05\n"
        "  best    -4.2% wall_time vs baseline a1b2c3d (iteration 3)\n"
        "  loop    3 iterations · 2 kept · 1 discarded · last -4.2% improved\n"
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


def test_summary_headline_when_session_completes_does_state_completed() -> None:
    summary = _summary(make_supervision_result(reason="completed", ended_by="session"))

    assert frame_text(summary, width=FRAME_WIDTH) == (
        f"✓ completed · 1m 0s · $0.05\n  loop    no session yet\n{_LOG_ROW}"
    )


@pytest.mark.parametrize(("reason", "ended_by", "headline"), _CAPPED_OR_ERRORED)
def test_summary_when_run_capped_or_errored_does_hide_the_agent_row(
    reason: SessionEndReason, ended_by: EndedBy, headline: str
) -> None:
    summary = _summary(
        make_supervision_result(reason=reason, ended_by=ended_by), final_text="Some final text."
    )

    assert frame_text(summary, width=FRAME_WIDTH) == (
        f"{headline}\n  loop    no session yet\n{_LOG_ROW}"
    )


def test_summary_when_log_lives_under_home_does_abbreviate_the_prefix_with_a_tilde():
    log_path = str(Path.home() / ".gymrat" / "supervisor-1.jsonl")

    summary = _summary(log_path=log_path)

    assert (
        frame_text(summary, width=FRAME_WIDTH).splitlines()[-1]
        == "  log     ~/.gymrat/supervisor-1.jsonl"
    )


@pytest.mark.parametrize(
    ("reason", "ended_by", "expected_sgr"),
    [
        pytest.param("completed", "session", SGR_GREEN, id="completed-green"),
        pytest.param("interrupted", "wall-clock", SGR_YELLOW, id="capped-yellow"),
        pytest.param("error", "session", SGR_RED, id="error-red"),
        pytest.param("interrupted", "stop-condition", SGR_GREEN, id="stop-condition-green"),
        pytest.param("interrupted", "hook-failure", SGR_YELLOW, id="hook-failure-yellow"),
    ],
)
def test_summary_headline_when_rendered_with_color_does_emit_outcome_styling(
    reason: SessionEndReason, ended_by: EndedBy, expected_sgr: int
) -> None:
    summary = _summary(make_supervision_result(reason=reason, ended_by=ended_by))

    colored = render_colored(summary)

    assert_has_sgr(colored.splitlines()[:1], expected_sgr)


def test_summary_log_row_when_rendered_with_color_does_leave_the_path_unstyled():
    summary = _summary()

    colored = render_colored(summary)

    assert "\x1b[" not in colored.splitlines()[-1]


# ---------------------------------------------------------------------------
# closing summary — agent row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("final_text", "expected_lines"),
    [
        pytest.param(
            "The task is complete.",
            [
                "✓ completed · 1m 0s · $0.05",
                "  agent   The task is complete.",
                "  loop    no session yet",
                _LOG_ROW,
            ],
            id="single-line",
        ),
        pytest.param(
            "First paragraph.\n\nSecond paragraph.",
            [
                "✓ completed · 1m 0s · $0.05",
                "  agent   First paragraph.",
                "",
                "          Second paragraph.",
                "  loop    no session yet",
                _LOG_ROW,
            ],
            id="paragraph-break",
        ),
        pytest.param(
            "Line one.\nLine two.",
            [
                "✓ completed · 1m 0s · $0.05",
                "  agent   Line one.",
                "          Line two.",
                "  loop    no session yet",
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
        "  loop    no session yet",
        _LOG_ROW,
    ]


# ---------------------------------------------------------------------------
# closing summary — stop-condition, guard and hook-failure endings
# ---------------------------------------------------------------------------

_STOP_CONDITION_REASON = "max iterations (2 of 2)"
_HOOK_FAILURE_REASON = "after hook failed on iteration 2: exit 1 (stdout 80 B, stderr 5 B)"
_EMPTY_LOOP_ROW = "  loop    baseline recorded · no iterations yet"


@pytest.mark.parametrize(
    ("ended_by", "end_reason", "stop_message", "expected_lines"),
    [
        pytest.param(
            "stop-condition",
            _STOP_CONDITION_REASON,
            None,
            [
                f"✓ stopped: {_STOP_CONDITION_REASON} · 1m 0s · $0.05",
                "  agent   Some final text.",
                _EMPTY_LOOP_ROW,
                _LOG_ROW,
            ],
            id="stop-condition-agent-text",
        ),
        pytest.param(
            "stop-condition",
            _STOP_CONDITION_REASON,
            "Target reached, stopping.",
            [
                f"✓ stopped: {_STOP_CONDITION_REASON} · 1m 0s · $0.05",
                "  agent   Target reached, stopping.",
                _EMPTY_LOOP_ROW,
                _LOG_ROW,
            ],
            id="stop-condition-stop-message",
        ),
        pytest.param(
            "hook-failure",
            _HOOK_FAILURE_REASON,
            None,
            [f"! stopped: {_HOOK_FAILURE_REASON} · 1m 0s · $0.05", _EMPTY_LOOP_ROW, _LOG_ROW],
            id="hook-failure",
        ),
        pytest.param(
            "guard",
            "safety limit reached",
            None,
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
def test_summary_when_a_stop_guard_or_hook_ended_the_run_does_render_the_stopped_headline(
    ended_by: EndedBy, end_reason: str, stop_message: str | None, expected_lines: list[str]
) -> None:
    session_result = make_read_session(
        session_state(), has_baseline=True, stop_message=stop_message
    )()

    summary = _summary(
        make_supervision_result(reason="interrupted", ended_by=ended_by, end_reason=end_reason),
        session_result=session_result,
        final_text="Some final text.",
    )

    assert frame_text(summary, width=FRAME_WIDTH).splitlines() == expected_lines


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
        "  loop    no session yet",
        *expected_exit_rows,
        _LOG_ROW,
    ]


def test_summary_exit_error_row_when_rendered_with_color_does_emit_alert_styling() -> None:
    summary = _summary(
        exit_report=ExitReport(steps=_EXIT_STEPS, error="finalize failed: disk full")
    )

    colored = render_colored(summary)

    assert_has_sgr(colored.splitlines()[-2:-1], SGR_YELLOW)
