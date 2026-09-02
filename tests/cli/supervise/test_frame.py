"""Tests for the Rich renderables the supervise command builds.

The dashboard-styling tests verify that the TUI renders with appropriate colors
and styles instead of plain white text: they render ``reporter.frame()`` through
a color-enabled console and check for ANSI escape codes on specific content
lines.  The closing-summary tests pin the text and styling of the four-line
block ``gymrat supervise`` prints when a run ends.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from rich.console import Console, RenderableType
from rich.panel import Panel

from gymrat.cli.style import CLI_THEME
from gymrat.cli.supervise.frame import build_summary
from tests._ansi import SGR_RE, strip_sgr
from tests._rich import frame_text
from tests.cli.supervise._fixtures import (
    FRAME_WIDTH,
    IDLE_WARN_MS,
    ReporterKit,
    fire_cap,
    fire_launch,
    fire_launch_and_bash_cycle,
    fire_launch_and_bash_start,
    fire_model_phase,
    fire_thinking_update,
    fire_tool_end,
    fire_tool_start,
    fire_usage_update,
    make_iteration,
    make_read_session,
    make_reporter,
    make_supervision_result,
    render_frame,
    session_state,
    session_state_three_iterations,
)

if TYPE_CHECKING:
    from gymrat.cli.supervise.progress import ReadSessionResult, SuperviseReporter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Standard ANSI SGR parameter codes
_SGR_BOLD = 1
_SGR_DIM = 2
_SGR_RED = 31
_SGR_GREEN = 32
_SGR_YELLOW = 33
_SGR_BLUE = 34
_SGR_CYAN = 36


def _render_colored(renderable: RenderableType, *, width: int = FRAME_WIDTH) -> str:
    """Render *renderable* through a sealed console with standard color."""
    buf = StringIO()
    console = Console(
        file=buf,
        width=width,
        force_terminal=True,
        no_color=False,
        color_system="standard",
        legacy_windows=False,
        _environ={},
        theme=CLI_THEME,
    )
    console.print(renderable)
    return buf.getvalue()


def _render_content_colored(reporter: SuperviseReporter, *, width: int = FRAME_WIDTH) -> str:
    """Render the panel's inner content with standard color.

    Extracts ``panel.renderable`` so Panel border styling does not leak
    into content-styling assertions.
    """
    panel = reporter.frame()
    assert isinstance(panel, Panel)
    return _render_colored(panel.renderable, width=width)


def _lines_containing(output: str, needle: str) -> list[str]:
    """Return raw (styled) lines whose plain-text content contains *needle*."""
    return [line for line in output.splitlines() if needle in strip_sgr(line)]


def _has_sgr(text: str, code: int) -> bool:
    """True when *text* contains an ANSI SGR sequence with parameter *code*."""
    target = str(code)
    return any(target in match.group(1).split(";") for match in SGR_RE.finditer(text))


def _assert_has_sgr(lines: list[str], code: int) -> None:
    """Assert *lines* is non-empty and at least one line carries SGR *code*."""
    assert lines
    assert any(_has_sgr(line, code) for line in lines)


def _line_after(frame: str, needle: str) -> str:
    """Return the line immediately following the first line containing *needle*."""
    lines = frame.splitlines()
    idx = next(i for i, line in enumerate(lines) if needle in line)
    return lines[idx + 1]


def _assert_is_nested_line(line: str) -> None:
    """A nested subagent line is marked with an arrow (``↳`` or its ASCII fallback)."""
    assert "↳" in line or "->" in line


def _fire_waiting_bash_cycle(kit: ReporterKit, *, above_threshold: bool = False) -> None:
    """Launch, then run a Bash start/end cycle, optionally idling past the warn threshold."""
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000)
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Bash", "bash-1", 3000)
    if above_threshold:
        kit.clock.now = 3000 + IDLE_WARN_MS + 1


# ---------------------------------------------------------------------------
# panel border
# ---------------------------------------------------------------------------


def test_build_frame_panel_when_launched_does_have_nondefault_border_style():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    panel = kit.reporter.frame()

    assert isinstance(panel, Panel)
    assert panel.border_style != "none"


# ---------------------------------------------------------------------------
# panel title styling
# ---------------------------------------------------------------------------


def test_panel_title_when_label_present_does_style_supervise_and_label_with_label_style():
    """STYLE_LABEL is ``bold blue`` — the title line must carry bold SGR."""
    kit = make_reporter(label="ecstatic-ts", session_id="", branch="")
    fire_launch(kit.reporter.observer, 1000)

    panel = kit.reporter.frame()
    assert isinstance(panel, Panel)
    colored = _render_colored(panel)
    title_line = colored.splitlines()[0]

    _assert_has_sgr([title_line], _SGR_BOLD)
    _assert_has_sgr([title_line], _SGR_BLUE)


def test_panel_title_when_connector_present_does_dim_the_connector_word():
    """Connector words (``session``, ``branch``) render dim in the title."""
    kit = make_reporter(
        label="",
        session_id="20260813-125044-34ec",
        branch="gymrat/20260813-125044-34ec",
    )
    fire_launch(kit.reporter.observer, 1000)

    panel = kit.reporter.frame()
    assert isinstance(panel, Panel)
    colored = _render_colored(panel)
    title_line = colored.splitlines()[0]

    _assert_has_sgr([title_line], _SGR_DIM)


# ---------------------------------------------------------------------------
# cost row styling
# ---------------------------------------------------------------------------


def test_cost_when_rendered_with_color_does_emit_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_usage_update(kit.reporter.observer, 4.12, 2000)

    colored = _render_content_colored(kit.reporter)
    cost_lines = _lines_containing(colored, "cost")

    assert cost_lines
    assert any("\x1b[" in line for line in cost_lines)


# ---------------------------------------------------------------------------
# loop row styling
# ---------------------------------------------------------------------------


def test_loop_iter_count_when_rendered_with_color_does_emit_bold_styling():
    state = session_state_three_iterations(-3.2, "improved")
    kit = make_reporter(
        max_iterations=20,
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    colored = _render_content_colored(kit.reporter)
    loop_lines = _lines_containing(colored, "iter")

    _assert_has_sgr(loop_lines, _SGR_BOLD)


@pytest.mark.parametrize(
    ("outcome", "delta_pct", "expected_sgr"),
    [
        pytest.param("regressed", 3.2, _SGR_RED, id="regressed-red"),
        pytest.param("improved", -3.2, _SGR_GREEN, id="improved-green"),
    ],
)
def test_loop_outcome_when_rendered_with_color_does_emit_expected_styling(
    outcome: str, delta_pct: float, expected_sgr: int
) -> None:
    state = session_state(
        iteration_count=1,
        last_iteration=make_iteration(delta_pct, outcome),
    )
    kit = make_reporter(
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    colored = _render_content_colored(kit.reporter)
    outcome_lines = _lines_containing(colored, outcome)

    _assert_has_sgr(outcome_lines, expected_sgr)


# ---------------------------------------------------------------------------
# best row styling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta_pct", "expected_sgr"),
    [
        pytest.param(-6.8, _SGR_GREEN, id="negative-delta-green"),
        pytest.param(3.5, _SGR_RED, id="positive-delta-red"),
    ],
)
def test_best_delta_when_rendered_with_color_does_emit_sign_dependent_styling(
    delta_pct: float, expected_sgr: int
) -> None:
    outcome = "improved" if delta_pct < 0 else "regressed"
    state = session_state(
        iteration_count=3,
        keep_count=1,
        discard_count=2,
        last_iteration=make_iteration(delta_pct, outcome, seq=3),
    )
    kit = make_reporter(
        read_session=make_read_session(
            state,
            has_baseline=True,
            best_delta_pct=delta_pct,
            best_seq=3,
        ),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    colored = _render_content_colored(kit.reporter)
    best_lines = _lines_containing(colored, "best")

    _assert_has_sgr(best_lines, expected_sgr)


# ---------------------------------------------------------------------------
# liveness styling
# ---------------------------------------------------------------------------


def test_liveness_starting_when_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    colored = _render_content_colored(kit.reporter)
    starting_lines = _lines_containing(colored, "starting")

    _assert_has_sgr(starting_lines, _SGR_DIM)


def test_liveness_inflight_when_rendered_with_color_does_not_emit_special_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000, input_summary="gymrat iterate")
    kit.clock.now = 7000

    colored = _render_content_colored(kit.reporter)
    bash_lines = _lines_containing(colored, "Bash")

    assert bash_lines
    assert not any(_has_sgr(line, _SGR_BOLD) for line in bash_lines)
    assert not any(_has_sgr(line, _SGR_DIM) for line in bash_lines)
    assert not any(_has_sgr(line, _SGR_CYAN) for line in bash_lines)


def test_liveness_inflight_when_rendered_does_match_finished_tool_column_layout():
    """An in-flight tool row should contain wall-clock, tool name, summary, and elapsed.

    The layout must match the column alignment of a finished tool line.
    """
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000, input_summary="gymrat iterate")
    kit.clock.now = 7000

    plain = render_frame(kit.reporter)
    bash_lines = _lines_containing(plain, "Bash")

    assert bash_lines, "expected at least one line containing 'Bash'"
    inflight_line = bash_lines[0]
    expected_wall = "00:00:02"
    assert expected_wall in inflight_line, f"wall-clock {expected_wall!r} not in {inflight_line!r}"
    assert "Bash" in inflight_line
    assert "gymrat iterate" in inflight_line
    assert "5s" in inflight_line


def test_liveness_responding_when_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_model_phase(kit.reporter.observer, 2000, "responding")

    colored = _render_content_colored(kit.reporter)
    responding_lines = _lines_containing(colored, "responding")

    _assert_has_sgr(responding_lines, _SGR_DIM)


def test_liveness_composing_when_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_model_phase(kit.reporter.observer, 2000, "tool_input", tool_name="Edit")

    colored = _render_content_colored(kit.reporter)
    preparing_lines = _lines_containing(colored, "preparing")

    _assert_has_sgr(preparing_lines, _SGR_DIM)


def test_liveness_waiting_when_below_threshold_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    _fire_waiting_bash_cycle(kit)

    colored = _render_content_colored(kit.reporter)
    waiting_lines = _lines_containing(colored, "waiting")

    _assert_has_sgr(waiting_lines, _SGR_DIM)


def test_liveness_waiting_when_above_threshold_rendered_with_color_does_emit_yellow_styling():
    kit = make_reporter()
    _fire_waiting_bash_cycle(kit, above_threshold=True)

    colored = _render_content_colored(kit.reporter)
    no_output_lines = _lines_containing(colored, "no output")

    _assert_has_sgr(no_output_lines, _SGR_YELLOW)


@pytest.mark.parametrize(
    ("phase", "needle"),
    [
        pytest.param("responding", "responding", id="responding"),
        pytest.param("tool_input", "preparing", id="composing-as-preparing"),
    ],
)
def test_liveness_phase_when_color_off_does_not_emit_sgr(phase: str, needle: str):
    kit = make_reporter(color=False)
    fire_launch(kit.reporter.observer, 1000)
    tool_name = "Edit" if phase == "tool_input" else None
    fire_model_phase(kit.reporter.observer, 2000, phase, tool_name=tool_name)

    colored = _render_content_colored(kit.reporter)
    phase_lines = _lines_containing(colored, needle)

    assert phase_lines
    assert not any("\x1b[" in line for line in phase_lines)


def test_liveness_waiting_when_below_threshold_color_off_does_not_emit_sgr():
    kit = make_reporter(color=False)
    _fire_waiting_bash_cycle(kit)

    colored = _render_content_colored(kit.reporter)
    waiting_lines = _lines_containing(colored, "waiting")

    assert waiting_lines
    assert not any("\x1b[" in line for line in waiting_lines)


def test_liveness_waiting_when_above_threshold_color_off_does_not_emit_sgr():
    kit = make_reporter(color=False)
    _fire_waiting_bash_cycle(kit, above_threshold=True)

    colored = _render_content_colored(kit.reporter)
    no_output_lines = _lines_containing(colored, "no output")

    assert no_output_lines
    assert not any("\x1b[" in line for line in no_output_lines)


def test_liveness_capped_when_rendered_with_color_does_emit_yellow_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_cap(kit.reporter.observer, "wall-clock")

    colored = _render_content_colored(kit.reporter)
    cap_lines = _lines_containing(colored, "interrupting")

    _assert_has_sgr(cap_lines, _SGR_YELLOW)


# ---------------------------------------------------------------------------
# finished tool lines ordering
# ---------------------------------------------------------------------------


def test_finished_tools_when_three_completed_does_render_newest_first():
    """Finished tool history lines appear newest-first (closest to the in-flight row)."""
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Read", "read-1", 2000, input_summary="oldest.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Read", "read-1", 3000)

    kit.clock.now = 4000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 4000, input_summary="middle.ts")
    kit.clock.now = 5000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 5000)

    kit.clock.now = 6000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 6000, input_summary="newest.ts")
    kit.clock.now = 7000
    fire_tool_end(kit.reporter.observer, "Bash", "bash-1", 7000)

    plain = render_frame(kit.reporter)
    tool_lines = [
        line
        for line in plain.splitlines()
        if any(name in line for name in ("oldest.ts", "middle.ts", "newest.ts"))
    ]

    assert len(tool_lines) == 3, (
        f"expected 3 tool history lines, got {len(tool_lines)}: {tool_lines}"
    )
    assert "newest.ts" in tool_lines[0], f"first line should be newest: {tool_lines[0]}"
    assert "oldest.ts" in tool_lines[-1], f"last line should be oldest: {tool_lines[-1]}"


# ---------------------------------------------------------------------------
# finished tool lines styling
# ---------------------------------------------------------------------------


def test_finished_tool_line_does_emit_dim_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 3000)

    colored = _render_content_colored(kit.reporter)
    finished_lines = _lines_containing(colored, "archetype")

    _assert_has_sgr(finished_lines, _SGR_DIM)


def test_finished_tool_line_when_failed_does_emit_dim_red_styling():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 3000, result="error")

    colored = _render_content_colored(kit.reporter)
    edit_lines = _lines_containing(colored, "Edit")

    _assert_has_sgr(edit_lines, _SGR_RED)
    _assert_has_sgr(edit_lines, _SGR_DIM)


# ---------------------------------------------------------------------------
# nested subagent line
# ---------------------------------------------------------------------------


def test_nested_tool_when_in_flight_does_render_arrow_line_under_parent():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    fire_tool_start(
        observer,
        "Read",
        "nested-read-1",
        2000,
        parent_tool_use_id="bash-1",
        input_summary="src/config.ts",
    )
    kit.clock.now = 5000

    nested_line = _line_after(render_frame(kit.reporter), "Bash")

    _assert_is_nested_line(nested_line)
    assert "Read" in nested_line
    assert "src/config.ts" in nested_line
    assert "3s" in nested_line


def test_nested_phase_when_thinking_does_render_arrow_line_with_thinking():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    fire_model_phase(observer, 2000, "thinking", parent_tool_use_id="bash-1")
    kit.clock.now = 4000

    nested_line = _line_after(render_frame(kit.reporter), "Bash")

    _assert_is_nested_line(nested_line)
    assert "thinking" in nested_line
    assert "2s" in nested_line


def test_nested_phase_when_responding_does_render_arrow_line_with_responding():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    fire_model_phase(observer, 2000, "responding", parent_tool_use_id="bash-1")
    kit.clock.now = 3000

    nested_line = _line_after(render_frame(kit.reporter), "Bash")

    _assert_is_nested_line(nested_line)
    assert "responding" in nested_line


def test_nested_phase_when_composing_does_render_arrow_line_with_preparing():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    fire_model_phase(observer, 2000, "tool_input", tool_name="Edit", parent_tool_use_id="bash-1")
    kit.clock.now = 3000

    nested_line = _line_after(render_frame(kit.reporter), "Bash")

    _assert_is_nested_line(nested_line)
    assert "preparing" in nested_line
    assert "Edit" in nested_line


def test_nested_when_no_activity_does_not_render_arrow_line():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(observer, "Bash", "bash-1", 2000, input_summary="gymrat iterate")
    kit.clock.now = 5000

    frame = render_frame(kit.reporter)

    assert "↳" not in frame
    assert "->" not in frame or "gymrat" in frame


def test_nested_tool_when_rendered_with_color_does_emit_dim_styling():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    fire_tool_start(
        observer,
        "Read",
        "nested-read-1",
        2000,
        parent_tool_use_id="bash-1",
        input_summary="src/config.ts",
    )
    kit.clock.now = 5000

    colored = _render_content_colored(kit.reporter)
    nested_lines = _lines_containing(colored, "config.ts")

    _assert_has_sgr(nested_lines, _SGR_DIM)


# ---------------------------------------------------------------------------
# tool-name column width — nested tools excluded
# ---------------------------------------------------------------------------


def test_tool_name_column_width_when_nested_tool_present_does_ignore_nested_width():
    """Nested tool names do not widen the tool-name column."""
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)

    kit.clock.now = 2000
    fire_tool_start(observer, "Bash", "bash-1", 2000, input_summary="run tests")
    kit.clock.now = 2500
    fire_tool_start(
        observer,
        "LongNestedToolName",
        "nested-1",
        2500,
        parent_tool_use_id="bash-1",
        input_summary="something",
    )
    kit.clock.now = 3000

    frame = render_frame(kit.reporter)
    bash_lines = _lines_containing(frame, "Bash")

    assert bash_lines
    assert "LongNestedToolName" not in bash_lines[0]


# ---------------------------------------------------------------------------
# no token count on non-Thinking states
# ---------------------------------------------------------------------------


def test_liveness_responding_when_rendered_does_not_show_token_count():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    fire_thinking_update(observer, 1500, estimated_tokens=500)
    fire_model_phase(observer, 2000, "responding")

    frame = render_frame(kit.reporter)

    assert "responding" in frame
    assert "500" not in frame


def test_liveness_composing_when_rendered_does_not_show_token_count():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    fire_thinking_update(observer, 1500, estimated_tokens=500)
    fire_model_phase(observer, 2000, "tool_input", tool_name="Edit")

    frame = render_frame(kit.reporter)

    assert "preparing" in frame
    assert "500" not in frame


# ---------------------------------------------------------------------------
# no "idle" anywhere in frame
# ---------------------------------------------------------------------------


def test_frame_when_any_state_does_never_contain_idle():
    """The word 'idle' must not appear in the frame for any liveness state."""
    kit = make_reporter()
    _fire_waiting_bash_cycle(kit, above_threshold=True)

    frame = render_frame(kit.reporter)

    assert "idle" not in frame


# ---------------------------------------------------------------------------
# closing summary
# ---------------------------------------------------------------------------

_LOG_PATH = "/repo/.gymrat/supervisor-1.jsonl"
_LOG_ROW = f"  log   {_LOG_PATH}"


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
        "  best  -4.2% wall_time vs baseline a1b2c3d (iter 3)\n"
        "  loop  iter 3 · 2 kept · 1 discarded · last -4.2% improved\n"
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
        pytest.param(None, "  loop  no session yet", id="no-session"),
        pytest.param(
            _session_result(with_best=False),
            "  loop  iter 3 · 2 kept · 1 discarded · last -4.2% improved",
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


def test_summary_when_log_lives_under_home_does_abbreviate_the_prefix_with_a_tilde():
    log_path = str(Path.home() / ".gymrat" / "supervisor-1.jsonl")

    summary = build_summary(make_supervision_result(), log_path=log_path, session_result=None)

    assert (
        frame_text(summary, width=FRAME_WIDTH).splitlines()[-1]
        == "  log   ~/.gymrat/supervisor-1.jsonl"
    )


@pytest.mark.parametrize(
    ("reason", "ended_by", "expected_sgr"),
    [
        pytest.param("completed", "session", _SGR_GREEN, id="completed-green"),
        pytest.param("interrupted", "wall-clock", _SGR_YELLOW, id="capped-yellow"),
        pytest.param("error", "session", _SGR_RED, id="error-red"),
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

    colored = _render_colored(summary)

    _assert_has_sgr(colored.splitlines()[:1], expected_sgr)


def test_summary_log_row_when_rendered_with_color_does_leave_the_path_unstyled():
    summary = build_summary(make_supervision_result(), log_path=_LOG_PATH, session_result=None)

    colored = _render_colored(summary)

    assert "\x1b[" not in colored.splitlines()[-1]
