"""Tests for the closing summary ``gymrat supervise`` prints when a run ends.

They pin the text and styling of the headline, agent, model, effort, best, loop,
and log rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat.cli.supervise.summary import SessionLabels, build_summary
from gymrat.supervisor.events import SUMMARY_MAX_CHARS
from gymrat.supervisor.exit_sequence import ExitReport, ExitStep
from tests._ansi import SGR_GREEN, SGR_RED, SGR_YELLOW, assert_has_sgr
from tests._rich import frame_text
from tests.cli.supervise._fixtures import (
    FRAME_WIDTH,
    make_iteration,
    make_read_session,
    make_supervision_result,
    render_colored,
    session_state,
    session_state_three_iterations,
)

if TYPE_CHECKING:
    from gymrat.cli.supervise.progress import ReadSessionResult
    from gymrat.supervisor import SessionEndReason
    from gymrat.supervisor.supervise import EndedBy


# ---------------------------------------------------------------------------
# closing summary
# ---------------------------------------------------------------------------

_LOG_PATH = "/repo/.gymrat/supervisor-1.jsonl"
_LOG_ROW = f"  log     {_LOG_PATH}"


def _session_result(*, with_best: bool) -> ReadSessionResult:
    """A three-iteration session, with or without a best-iteration record."""
    state = session_state_three_iterations(-4.2, "improved", seq=3)
    if not with_best:
        return make_read_session(state, has_baseline=True)()
    return make_read_session(
        state,
        has_baseline=True,
        best_delta_pct=-4.2,
        best_seq=3,
        primary_label="wall_time",
        baseline_sha="a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
    )()


def test_summary_when_run_has_a_best_iteration_does_render_headline_best_loop_and_log():
    summary = build_summary(
        make_supervision_result(reason="interrupted", ended_by="wall-clock", cost_usd=0.16),
        log_path=_LOG_PATH,
        session_result=_session_result(with_best=True),
    )

    assert frame_text(summary, width=FRAME_WIDTH) == (
        "! interrupted by wall-clock cap · 1m 0s · $0.16\n"
        "  best    -4.2% wall_time vs baseline a1b2c3d (iteration 3)\n"
        "  loop    3 iterations · 2 kept · 1 discarded · last -4.2% improved\n"
        f"{_LOG_ROW}"
    )


@pytest.mark.parametrize(
    ("reason", "ended_by", "expected"),
    [
        pytest.param("completed", "session", "✓ completed · 1m 0s · $0.05", id="session-end"),
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
    ],
)
def test_summary_headline_when_run_ends_does_name_the_outcome_duration_and_cost(
    reason: str, ended_by: str, expected: str
) -> None:
    summary = build_summary(
        make_supervision_result(reason=reason, ended_by=ended_by),  # type: ignore[arg-type]
        log_path=_LOG_PATH,
        session_result=None,
    )

    assert frame_text(summary, width=FRAME_WIDTH).splitlines()[0] == expected


@pytest.mark.parametrize(
    ("session_result", "expected_loop"),
    [
        pytest.param(None, "  loop    no session yet", id="no-session"),
        pytest.param(
            _session_result(with_best=False),
            "  loop    3 iterations · 2 kept · 1 discarded · last -4.2% improved",
            id="no-best-iteration",
        ),
    ],
)
def test_summary_when_no_best_delta_does_omit_the_best_row(
    session_result: ReadSessionResult | None, expected_loop: str
) -> None:
    summary = build_summary(
        make_supervision_result(), log_path=_LOG_PATH, session_result=session_result
    )

    assert frame_text(summary, width=FRAME_WIDTH) == (
        f"✓ completed · 1m 0s · $0.05\n{expected_loop}\n{_LOG_ROW}"
    )


@pytest.mark.parametrize(
    ("session_result", "expected_loop"),
    [
        pytest.param(
            make_read_session(session_state(), has_baseline=True)(),
            "  loop    baseline recorded · no iterations yet",
            id="zero-iterations",
        ),
        pytest.param(
            make_read_session(
                session_state(
                    iteration_count=1,
                    keep_count=1,
                    discard_count=0,
                    last_iteration=make_iteration(-4.2, "improved"),
                ),
                has_baseline=True,
            )(),
            "  loop    1 iteration · 1 kept · 0 discarded · last -4.2% improved",
            id="one-iteration",
        ),
    ],
)
def test_summary_loop_row_when_iteration_count_varies_does_name_the_count_as_a_noun(
    session_result: ReadSessionResult, expected_loop: str
) -> None:
    summary = build_summary(
        make_supervision_result(), log_path=_LOG_PATH, session_result=session_result
    )

    assert frame_text(summary, width=FRAME_WIDTH).splitlines()[1] == expected_loop


def test_summary_when_log_lives_under_home_does_abbreviate_the_prefix_with_a_tilde():
    log_path = str(Path.home() / ".gymrat" / "supervisor-1.jsonl")

    summary = build_summary(make_supervision_result(), log_path=log_path, session_result=None)

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
    reason: str, ended_by: str, expected_sgr: int
) -> None:
    summary = build_summary(
        make_supervision_result(reason=reason, ended_by=ended_by),  # type: ignore[arg-type]
        log_path=_LOG_PATH,
        session_result=None,
    )

    colored = render_colored(summary)

    assert_has_sgr(colored.splitlines()[:1], expected_sgr)


def test_summary_log_row_when_rendered_with_color_does_leave_the_path_unstyled():
    summary = build_summary(make_supervision_result(), log_path=_LOG_PATH, session_result=None)

    colored = render_colored(summary)

    assert "\x1b[" not in colored.splitlines()[-1]


# ---------------------------------------------------------------------------
# closing summary — agent row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "ended_by", "expected_headline"),
    [
        pytest.param(
            "interrupted",
            "wall-clock",
            "! interrupted by wall-clock cap · 1m 0s · $0.05",
            id="wall-clock-cap",
        ),
        pytest.param("error", "session", "✗ error · 1m 0s · $0.05", id="error"),
    ],
)
def test_summary_when_cap_or_error_ended_does_not_show_agent_row(
    reason: SessionEndReason, ended_by: EndedBy, expected_headline: str
) -> None:
    summary = build_summary(
        make_supervision_result(reason=reason, ended_by=ended_by),
        log_path=_LOG_PATH,
        session_result=None,
        final_text="Some final text.",
    )

    text = frame_text(summary, width=FRAME_WIDTH)

    assert "  agent" not in text
    assert text == f"{expected_headline}\n  loop    no session yet\n{_LOG_ROW}"


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
def test_summary_when_final_text_multiline_does_indent_continuation_under_content(
    final_text: str, expected_lines: list[str]
) -> None:
    summary = build_summary(
        make_supervision_result(reason="completed", ended_by="session"),
        log_path=_LOG_PATH,
        session_result=None,
        final_text=final_text,
    )

    text = frame_text(summary, width=FRAME_WIDTH)

    assert text.splitlines() == expected_lines


# ---------------------------------------------------------------------------
# closing summary — agent row clipping
# ---------------------------------------------------------------------------


def test_summary_agent_row_when_text_at_threshold_does_render_unchanged():
    short_text = "x" * SUMMARY_MAX_CHARS

    summary = build_summary(
        make_supervision_result(reason="completed", ended_by="session"),
        log_path=_LOG_PATH,
        session_result=None,
        final_text=short_text,
    )

    text = frame_text(summary, width=FRAME_WIDTH + 200)
    agent_line = next(line for line in text.splitlines() if "agent" in line)

    assert agent_line == f"  agent   {short_text}"
    assert "(full message in log)" not in text


def test_summary_agent_row_when_text_exceeds_threshold_does_clip_with_ellipsis_and_log_note():
    long_text = "a" * (SUMMARY_MAX_CHARS + 50)

    summary = build_summary(
        make_supervision_result(reason="completed", ended_by="session"),
        log_path=_LOG_PATH,
        session_result=None,
        final_text=long_text,
    )

    text = frame_text(summary, width=FRAME_WIDTH + 200)
    agent_lines = [line for line in text.splitlines() if "agent" in line]

    assert len(agent_lines) == 1
    agent_line = agent_lines[0]
    assert "…" in agent_line
    assert "(full message in log)" in agent_line
    assert len(agent_line) < len(f"  agent   {long_text}")


def test_summary_agent_row_when_text_below_threshold_does_not_append_log_note():
    short_text = "Short status message."

    summary = build_summary(
        make_supervision_result(reason="completed", ended_by="session"),
        log_path=_LOG_PATH,
        session_result=None,
        final_text=short_text,
    )

    text = frame_text(summary, width=FRAME_WIDTH)

    assert "(full message in log)" not in text
    assert any("Short status message." in line for line in text.splitlines())


# ---------------------------------------------------------------------------
# closing summary — agent row stop-message preference
# ---------------------------------------------------------------------------


def test_summary_agent_row_when_stop_message_present_does_show_stop_message() -> None:
    session_result = make_read_session(
        session_state(),
        has_baseline=True,
        stop_message="Target reached, stopping.",
    )()

    summary = build_summary(
        make_supervision_result(reason="completed", ended_by="session"),
        log_path=_LOG_PATH,
        session_result=session_result,
        final_text="Some other final text.",
    )

    text = frame_text(summary, width=FRAME_WIDTH)
    agent_line = next(line for line in text.splitlines() if "agent" in line)

    assert agent_line == "  agent   Target reached, stopping."
    assert "Some other final text." not in text


# ---------------------------------------------------------------------------
# closing summary — model and effort display
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "expected_line"),
    [
        pytest.param(SessionLabels(model="opus"), "  model   opus", id="model-only"),
        pytest.param(SessionLabels(effort="high"), "  effort  high", id="effort-only"),
    ],
)
def test_summary_when_model_or_effort_in_force_does_show_labelled_row(
    labels: SessionLabels, expected_line: str
) -> None:
    summary = build_summary(
        make_supervision_result(), log_path=_LOG_PATH, session_result=None, labels=labels
    )

    text = frame_text(summary, width=FRAME_WIDTH)

    assert expected_line in text.splitlines()


def test_summary_when_no_labels_does_omit_model_and_effort_rows() -> None:
    summary = build_summary(
        make_supervision_result(), log_path=_LOG_PATH, session_result=None, labels=SessionLabels()
    )

    text = frame_text(summary, width=FRAME_WIDTH)

    assert "model" not in text.lower()
    assert "effort" not in text.lower()


# ---------------------------------------------------------------------------
# closing summary — guard-ended session
# ---------------------------------------------------------------------------


def test_summary_headline_when_guard_ended_does_show_stopped_by_guard_with_reason() -> None:
    summary = build_summary(
        make_supervision_result(
            reason="interrupted",
            ended_by="guard",
            end_reason="safety limit reached",
        ),
        log_path=_LOG_PATH,
        session_result=None,
    )

    headline = frame_text(summary, width=FRAME_WIDTH).splitlines()[0]

    assert "stopped by guard" in headline
    assert "safety limit reached" in headline


def test_summary_when_guard_ended_does_show_agent_row() -> None:
    summary = build_summary(
        make_supervision_result(
            reason="interrupted",
            ended_by="guard",
            end_reason="safety limit reached",
        ),
        log_path=_LOG_PATH,
        session_result=None,
        final_text="I was working on the optimization.",
    )

    text = frame_text(summary, width=FRAME_WIDTH)

    assert "  agent" in text
    assert "I was working on the optimization." in text


# ---------------------------------------------------------------------------
# closing summary — stop-condition and hook-failure endings
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
    ],
)
def test_summary_when_stop_condition_or_hook_failure_ended_does_render_stopped_headline(
    ended_by: EndedBy, end_reason: str, stop_message: str | None, expected_lines: list[str]
) -> None:
    session_result = make_read_session(
        session_state(), has_baseline=True, stop_message=stop_message
    )()

    summary = build_summary(
        make_supervision_result(reason="interrupted", ended_by=ended_by, end_reason=end_reason),
        log_path=_LOG_PATH,
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
    summary = build_summary(
        make_supervision_result(),
        log_path=_LOG_PATH,
        session_result=None,
        exit_report=ExitReport(steps=_EXIT_STEPS, error=error),
    )

    assert frame_text(summary, width=FRAME_WIDTH).splitlines() == [
        "✓ completed · 1m 0s · $0.05",
        "  loop    no session yet",
        *expected_exit_rows,
        _LOG_ROW,
    ]


def test_summary_exit_error_row_when_rendered_with_color_does_emit_alert_styling() -> None:
    summary = build_summary(
        make_supervision_result(),
        log_path=_LOG_PATH,
        session_result=None,
        exit_report=ExitReport(steps=_EXIT_STEPS, error="finalize failed: disk full"),
    )

    colored = render_colored(summary)

    assert_has_sgr(colored.splitlines()[-2:-1], SGR_YELLOW)
