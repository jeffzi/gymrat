"""Behavioral tests for the supervise dashboard's liveness line.

These cover what the liveness line shows as the session moves: the starting
state, in-flight and finished tools (with their marks, truncation and
wall-clock times), the waiting state and its idle threshold, model phase
transitions, nested subagent activity, the iterate sidecar and MCP tool
detection, turn-end and follow-up transitions, and events the line ignores.
"""

from __future__ import annotations

from datetime import UTC, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from gymrat.cli.supervise.progress import IDLE_WARN_MS
from gymrat.session.progress_file import ProgressSnapshot
from gymrat.supervisor.events import TextDeltaEvent, ThinkingUpdateEvent
from tests.cli.supervise._fixtures import (
    ReporterKit,
    _epoch_ms_to_local_hms,
    cap_event,
    fire_launch_and_bash_start,
    follow_up_event,
    launch_event,
    line_after,
    make_reporter,
    model_phase_event,
    render_frame,
    thinking_event,
    tool_end_event,
    tool_start_event,
    turn_end_event,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import tzinfo

    from syrupy.assertion import SnapshotAssertion

    from gymrat.supervisor.events import SessionObserver


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _lines_containing(frame: str, needle: str) -> list[str]:
    """Return lines from *frame* whose content contains *needle*."""
    return [line for line in frame.splitlines() if needle in line]


def _content_line(frame: str, needle: str) -> str:
    """The sole frame row containing *needle*, with panel border and padding stripped."""
    lines = _lines_containing(frame, needle)
    assert len(lines) == 1, f"expected exactly one line containing {needle!r}, got {lines}"
    return lines[0].split("│")[1].strip()


# ---------------------------------------------------------------------------
# liveness — starting state
# ---------------------------------------------------------------------------


def test_liveness_when_no_tool_has_started_does_show_starting():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))

    frame = render_frame(kit.reporter)

    assert "starting" in frame


# ---------------------------------------------------------------------------
# liveness — ended tools
# ---------------------------------------------------------------------------


def test_liveness_when_four_tools_finish_does_show_only_last_three(snapshot: SnapshotAssertion):
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))

    for i, (name, summary) in enumerate(
        [("Read", "src/a.ts"), ("Edit", "src/b.ts"), ("Bash", "npm test"), ("Read", "src/c.ts")],
        start=1,
    ):
        ts = 1000 + i * 1000
        kit.clock.now = ts
        kit.reporter.observer(tool_start_event(name, f"t-{i}", ts, input_summary=summary))
        kit.clock.now = ts + 500
        kit.reporter.observer(tool_end_event(name, f"t-{i}", ts + 500))

    frame = render_frame(kit.reporter)

    assert frame == snapshot


def test_liveness_when_one_of_several_tools_ends_does_fall_back_to_remaining():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Read", "read-1", 2000))
    kit.clock.now = 2500
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2500))
    kit.clock.now = 3000
    kit.reporter.observer(tool_end_event("Read", "read-1", 3000))

    frame = render_frame(kit.reporter)

    assert "Bash" in frame


def test_liveness_when_untracked_tool_ends_does_not_change():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.reporter.observer(tool_end_event("Bash", "never-started", 3000))

    frame = render_frame(kit.reporter)

    assert "starting" in frame


# ---------------------------------------------------------------------------
# finished tool marks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expect_error_mark"),
    [
        pytest.param("ok", False, id="ok"),
        pytest.param("error", True, id="error"),
    ],
)
def test_finished_tool_when_ended_does_show_expected_mark(result: str, expect_error_mark: bool):
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event("Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    )
    kit.clock.now = 3000
    kit.reporter.observer(tool_end_event("Edit", "edit-1", 3000, result=result))

    frame = render_frame(kit.reporter)

    assert "Edit" in frame
    assert ("✗" in frame) == expect_error_mark


# ---------------------------------------------------------------------------
# sub-second finished tool
# ---------------------------------------------------------------------------


def test_finished_tool_when_under_one_second_does_show_less_than_one_second():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Edit", "edit-1", 2000, input_summary="src/a.ts"))
    kit.clock.now = 2500
    kit.reporter.observer(tool_end_event("Edit", "edit-1", 2500))

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "Edit") == "00:00:02  Edit   src/a.ts  <1s"


# ---------------------------------------------------------------------------
# in-flight tool truncation
# ---------------------------------------------------------------------------


def test_liveness_when_in_flight_summary_exceeds_width_does_truncate_to_one_line():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    long_summary = "src/" + "/".join(f"level{i}" for i in range(20)) + "/file.ts"
    kit.reporter.observer(tool_start_event("Edit", "edit-1", 2000, input_summary=long_summary))
    kit.clock.now = 7000

    frame = render_frame(kit.reporter)
    liveness_lines = [line for line in frame.splitlines() if "Edit" in line]

    assert len(liveness_lines) == 1, f"expected one liveness line, got {liveness_lines}"
    assert "…" in liveness_lines[0]


# ---------------------------------------------------------------------------
# wall-clock finished-tool lines
# ---------------------------------------------------------------------------


# 2023-11-14 22:13:20 UTC.
_WALL_CLOCK_EPOCH_MS = 1_700_000_000_000


@pytest.mark.parametrize(
    ("tz", "expected_clock"),
    [
        pytest.param(UTC, "22:13:20", id="utc"),
        pytest.param(timezone(timedelta(hours=5, minutes=30)), "03:43:20", id="half-hour-ahead"),
        pytest.param(timezone(timedelta(hours=-5)), "17:13:20", id="five-hours-behind"),
    ],
)
def test_finished_tool_when_ended_does_show_wall_clock_in_the_given_tz(
    tz: tzinfo, expected_clock: str
):
    started_at = _WALL_CLOCK_EPOCH_MS - 1000
    kit = make_reporter(tz=tz, clock_start=started_at)
    kit.reporter.observer(launch_event(started_at))
    kit.reporter.observer(
        tool_start_event("Edit", "edit-1", started_at, input_summary="src/archetype.ts")
    )
    kit.clock.now = _WALL_CLOCK_EPOCH_MS
    kit.reporter.observer(tool_end_event("Edit", "edit-1", _WALL_CLOCK_EPOCH_MS))

    frame = render_frame(kit.reporter)

    assert f"{expected_clock}  Edit   src/archetype.ts" in frame


# ---------------------------------------------------------------------------
# wall-clock — local-timezone default
# ---------------------------------------------------------------------------


def test_finished_tool_when_no_explicit_tz_does_use_system_local_time():
    kit = make_reporter(tz=None)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event("Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    )
    kit.clock.now = 3000
    kit.reporter.observer(tool_end_event("Edit", "edit-1", 3000))

    frame = render_frame(kit.reporter)

    assert _epoch_ms_to_local_hms(3000) in frame


# ---------------------------------------------------------------------------
# above-threshold waiting — last tool context
# ---------------------------------------------------------------------------

# The clock reading right after the Bash end that both the threshold and the
# custom idle_warn_ms tests below freeze on.
_BASH_END_MS = 3000


def _make_reporter_past_bash_end(
    *, idle_warn_ms: int = IDLE_WARN_MS, result: str = "ok"
) -> ReporterKit:
    """A reporter with the given idle-warn threshold, clock frozen right after a Bash end."""
    kit = make_reporter(idle_warn_ms=idle_warn_ms)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2000))
    kit.clock.now = _BASH_END_MS
    kit.reporter.observer(tool_end_event("Bash", "bash-1", _BASH_END_MS, result=result))
    return kit


@pytest.mark.parametrize(
    ("result", "expected_fragment"),
    [
        pytest.param("ok", "(last tool: Bash at 00:00:03)", id="ok-tool"),
        pytest.param("error", "(last tool: Bash ✗ at 00:00:03)", id="errored-tool"),
    ],
)
def test_liveness_when_waiting_past_threshold_does_show_last_tool_context(
    result: str, expected_fragment: str
):
    kit = _make_reporter_past_bash_end(result=result)
    kit.clock.now = _BASH_END_MS + IDLE_WARN_MS + 1

    frame = render_frame(kit.reporter)

    assert "no output" in frame
    assert expected_fragment in frame


def test_liveness_when_waiting_past_threshold_no_tool_does_omit_parenthetical():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.reporter.observer(model_phase_event(2000, "turn_end"))
    kit.clock.now = 2000 + IDLE_WARN_MS + 1

    frame = render_frame(kit.reporter)

    assert "no output" in frame
    assert "(last tool" not in frame


# ---------------------------------------------------------------------------
# custom idle_warn_ms — configurable threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "expected_fragment", "unexpected_fragment"),
    [
        pytest.param(1, "no output", "waiting", id="past-custom-idle-warn"),
        pytest.param(-1, "waiting", "no output", id="below-custom-idle-warn"),
    ],
)
def test_liveness_when_waiting_around_custom_idle_warn_does_show_expected_state(
    offset: int, expected_fragment: str, unexpected_fragment: str
):
    custom_ms = 100
    kit = _make_reporter_past_bash_end(idle_warn_ms=custom_ms)
    kit.clock.now = _BASH_END_MS + custom_ms + offset

    frame = render_frame(kit.reporter)

    assert expected_fragment in frame
    assert unexpected_fragment not in frame


# ---------------------------------------------------------------------------
# liveness — model phase transitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        pytest.param("thinking", "thinking", id="thinking"),
        pytest.param("responding", "responding", id="responding"),
        pytest.param("turn_end", "waiting", id="turn_end"),
    ],
)
def test_liveness_when_model_phase_does_show_expected_state(phase: str, expected: str):
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(model_phase_event(2000, phase))

    frame = render_frame(kit.reporter)

    assert expected in frame


def test_liveness_when_model_phase_thinking_after_thinking_update_does_preserve_token_count():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(thinking_event(1500, estimated_tokens=200))
    observer(model_phase_event(2000, "thinking"))

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "thinking") == "thinking  ~200 tokens  0s"


@pytest.mark.parametrize(
    ("tool_name", "expected_in_frame"),
    [
        pytest.param("Edit", "Edit", id="with-tool-name"),
        pytest.param(None, "unknown", id="without-tool-name"),
    ],
)
def test_liveness_when_model_phase_tool_input_does_show_preparing(
    tool_name: str | None, expected_in_frame: str
):
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(model_phase_event(2000, "tool_input", tool_name=tool_name))

    frame = render_frame(kit.reporter)

    assert "preparing" in frame
    assert expected_in_frame in frame


# ---------------------------------------------------------------------------
# liveness — model phase ignored when capped or in-flight
# ---------------------------------------------------------------------------


def _capped(observer: SessionObserver) -> None:
    """Launch, then fire a wall-clock cap, leaving the liveness line capped."""
    observer(launch_event(1000))
    observer(cap_event("wall-clock"))


def _fire_model_phase_while_capped(observer: SessionObserver) -> None:
    """Fire a responding model-phase event after the liveness line is capped."""
    observer(model_phase_event(6000, "responding"))


def _fire_thinking_update_while_capped(observer: SessionObserver) -> None:
    """Fire a thinking-update event after the liveness line is capped."""
    observer(thinking_event(6000, estimated_tokens=100))


def _fire_model_phase_while_in_flight(observer: SessionObserver) -> None:
    """Fire a responding model-phase event while a tool is in flight."""
    observer(model_phase_event(2000, "responding"))


def _fire_thinking_update_while_in_flight(observer: SessionObserver) -> None:
    """Fire a thinking-update event while a tool is in flight."""
    observer(thinking_event(2000, estimated_tokens=100))


@pytest.mark.parametrize(
    ("state_setup", "fire_event", "kept_text", "displaced_text"),
    [
        pytest.param(
            _capped,
            _fire_model_phase_while_capped,
            "interrupting",
            "responding",
            id="capped-ignores-model-phase",
        ),
        pytest.param(
            _capped,
            _fire_thinking_update_while_capped,
            "interrupting",
            "100",
            id="capped-ignores-thinking-update",
        ),
        pytest.param(
            fire_launch_and_bash_start,
            _fire_model_phase_while_in_flight,
            "Bash",
            "responding",
            id="in-flight-ignores-model-phase",
        ),
        pytest.param(
            fire_launch_and_bash_start,
            _fire_thinking_update_while_in_flight,
            "Bash",
            "100",
            id="in-flight-ignores-thinking-update",
        ),
    ],
)
def test_liveness_when_capped_or_in_flight_does_ignore_model_phase_and_thinking_events(
    state_setup: Callable[[SessionObserver], None],
    fire_event: Callable[[SessionObserver], None],
    kept_text: str,
    displaced_text: str,
):
    kit = make_reporter()
    observer = kit.reporter.observer
    state_setup(observer)
    fire_event(observer)

    frame = render_frame(kit.reporter)

    assert kept_text in frame
    assert displaced_text not in frame


# ---------------------------------------------------------------------------
# liveness — nested subagent activity
# ---------------------------------------------------------------------------


def _liveness_lines(frame: str, needle: str) -> list[str]:
    """Top-level liveness lines containing *needle*, excluding nested (``↳``) lines."""
    return [line for line in frame.splitlines() if needle in line and "↳" not in line]


def _fire_nested_tool_start(observer: SessionObserver) -> None:
    """Fire a nested Read tool-start event under a parent Bash tool."""
    observer(tool_start_event("Read", "nested-read-1", 2000, parent_tool_use_id="bash-1"))


def _fire_nested_thinking_update(observer: SessionObserver) -> None:
    """Fire a nested thinking-update event under a parent Bash tool."""
    observer(thinking_event(2000, estimated_tokens=999, parent_tool_use_id="bash-1"))


def _fire_nested_model_phase(observer: SessionObserver) -> None:
    """Fire a nested responding model-phase event under a parent Bash tool."""
    observer(model_phase_event(2000, "responding", parent_tool_use_id="bash-1"))


@pytest.mark.parametrize(
    "fire_nested_event",
    [
        pytest.param(_fire_nested_tool_start, id="nested-tool-starts"),
        pytest.param(_fire_nested_thinking_update, id="nested-thinking-update"),
        pytest.param(_fire_nested_model_phase, id="nested-model-phase"),
    ],
)
def test_liveness_when_nested_event_arrives_does_not_change_top_level_liveness(
    fire_nested_event: Callable[[SessionObserver], None],
):
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    fire_nested_event(observer)

    frame = render_frame(kit.reporter)

    assert len(_liveness_lines(frame, "Bash")) == 1


def test_liveness_when_nested_tool_ends_does_not_appear_in_finished_tools():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    observer(tool_start_event("Read", "nested-read-1", 2000, parent_tool_use_id="bash-1"))
    observer(tool_end_event("Read", "nested-read-1", 2500, parent_tool_use_id="bash-1"))

    frame = render_frame(kit.reporter)

    assert "Bash" in frame
    assert "Read" not in frame


def test_liveness_when_nested_event_has_no_matching_parent_does_ignore():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(tool_start_event("Read", "nested-read-1", 2000, parent_tool_use_id="nonexistent"))

    frame = render_frame(kit.reporter)

    assert "starting" in frame
    assert "Read" not in frame


# ---------------------------------------------------------------------------
# liveness — iterate sidecar
# ---------------------------------------------------------------------------


def test_liveness_when_iterate_sidecar_has_no_pass_left_does_show_passes_without_an_estimate():
    sidecar = ProgressSnapshot(
        passes_completed=10,
        passes_total=10,
        last_pass_duration_ms=225_000.0,
    )
    kit = make_reporter(read_progress=lambda _root: sidecar)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate"))
    kit.clock.now = 2000 + 31 * 60 * 1000

    frame = render_frame(kit.reporter)

    passes_row = next(line for line in frame.splitlines() if "passes" in line)
    assert passes_row.strip("│ ") == "passes 10/10 · 31m 0s"


def test_liveness_when_iterate_tool_has_sidecar_does_show_passes_bar(
    snapshot: SnapshotAssertion,
):
    sidecar = ProgressSnapshot(
        passes_completed=7,
        passes_total=10,
        last_pass_duration_ms=225_000.0,
    )

    def fake_read_progress(_root: str) -> ProgressSnapshot | None:
        return sidecar

    kit = make_reporter(read_progress=fake_read_progress)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate"))
    kit.clock.now = 2000 + 31 * 60 * 1000

    frame = render_frame(kit.reporter)

    assert frame == snapshot


def test_liveness_when_iterate_tool_has_no_sidecar_does_show_plain_elapsed():
    def fake_read_progress(_root: str) -> ProgressSnapshot | None:
        return None

    kit = make_reporter(read_progress=fake_read_progress)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate"))
    kit.clock.now = 7000

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "Bash") == "00:00:02  Bash   gymrat iterate  5s"


# ---------------------------------------------------------------------------
# MCP iterate tool detection
# ---------------------------------------------------------------------------


def test_liveness_when_iterate_tool_has_sidecar_does_show_passes_nest():
    sidecar = ProgressSnapshot(
        passes_completed=4,
        passes_total=8,
        last_pass_duration_ms=120_000.0,
    )

    def fake_read_progress(_root: str) -> ProgressSnapshot | None:
        return sidecar

    kit = make_reporter(read_progress=fake_read_progress)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event("mcp__gymrat__iterate", "mcp-1", 2000, input_summary="gymrat iterate")
    )
    kit.clock.now = 2000 + 10 * 60 * 1000

    frame = render_frame(kit.reporter)

    assert "4/8" in frame
    passes_line = line_after(frame, "mcp__gymrat__iterate")
    assert "passes" in passes_line


def test_liveness_when_mcp_probe_tool_in_flight_does_show_summary_not_nest():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event("mcp__gymrat__probe", "mcp-2", 2000, input_summary="gymrat probe a b")
    )
    kit.clock.now = 5000

    frame = render_frame(kit.reporter)

    assert "mcp__gymrat__probe" in frame
    assert "gymrat probe a b" in frame
    assert "↳" not in frame


def test_liveness_when_non_iterate_mcp_tool_in_flight_does_not_show_sidecar():
    sidecar = ProgressSnapshot(
        passes_completed=4,
        passes_total=8,
        last_pass_duration_ms=120_000.0,
    )

    def fake_read_progress(_root: str) -> ProgressSnapshot | None:
        return sidecar

    kit = make_reporter(read_progress=fake_read_progress)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event("mcp__gymrat__probe", "mcp-2", 2000, input_summary="gymrat probe a b")
    )
    kit.clock.now = 5000

    frame = render_frame(kit.reporter)

    assert "4/8" not in frame


# ---------------------------------------------------------------------------
# no token count on non-Thinking states
# ---------------------------------------------------------------------------


def test_liveness_responding_when_rendered_does_not_show_token_count():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(thinking_event(1500, estimated_tokens=500))
    observer(model_phase_event(2000, "responding"))

    frame = render_frame(kit.reporter)

    assert "responding" in frame
    assert "500" not in frame


def test_liveness_composing_when_rendered_does_not_show_token_count():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(thinking_event(1500, estimated_tokens=500))
    observer(model_phase_event(2000, "tool_input", tool_name="Edit"))

    frame = render_frame(kit.reporter)

    assert "preparing" in frame
    assert "500" not in frame


# ---------------------------------------------------------------------------
# events that do not render
# ---------------------------------------------------------------------------


def test_liveness_when_text_delta_after_launch_does_stay_starting():
    kit = make_reporter()
    observer = kit.reporter.observer

    observer(launch_event(1000))
    observer(TextDeltaEvent(at=2_500_000_000, chunk="hello"))

    frame = render_frame(kit.reporter)

    assert "starting" in frame


def test_liveness_when_thinking_after_launch_does_show_thinking():
    kit = make_reporter()
    observer = kit.reporter.observer

    observer(launch_event(1000))
    observer(ThinkingUpdateEvent(at=2_500_000_000, estimated_tokens=100, delta=10))

    frame = render_frame(kit.reporter)

    assert "thinking" in frame
    assert "100" in frame


# ---------------------------------------------------------------------------
# liveness — turn end and follow-up transitions
# ---------------------------------------------------------------------------


def test_liveness_when_turn_ends_does_show_waiting():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(model_phase_event(1500, "responding"))
    observer(turn_end_event(2000, text="done"))

    frame = render_frame(kit.reporter)

    assert "waiting" in frame


def test_liveness_when_follow_up_does_not_change_liveness():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(model_phase_event(1500, "responding"))
    observer(turn_end_event(2000, text="done"))
    observer(follow_up_event(3000, action="replied"))

    frame = render_frame(kit.reporter)

    assert "waiting" in frame
