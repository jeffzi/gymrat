"""Tests for the pure supervise dashboard reducer.

The reducer owns every state transition the dashboard makes: it reads no clock,
touches no terminal, and performs no session I/O.  These tests therefore build
events and states directly and never go through the reporter shell, so nothing
here imports the rendering layer.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

import pytest

from gymrat.cli.supervise.reducer import (
    ReporterState,
    advance,
    exit_phase,
    loop_plain_text,
    plain_line,
    wants_session_refresh,
)
from gymrat.cli.supervise.types import (
    Capped,
    Composing,
    Exiting,
    FinishedTool,
    InFlight,
    NestedPhase,
    ReadSessionResult,
    Responding,
    RunningTool,
    Starting,
    Thinking,
    Waiting,
)
from gymrat.supervisor.events import (
    CompactionEvent,
    TextDeltaEvent,
)
from gymrat.supervisor.exit_sequence import ExitPhase
from gymrat.utils import NS_PER_MS
from tests._imports import modules_imported_by
from tests.cli.supervise._fixtures import (
    TOOL_START_MS,
    cap_event,
    follow_up_event,
    launch_event,
    model_phase_event,
    read_result,
    session_state_three_iterations,
    thinking_event,
    tool_end_event,
    tool_start_event,
    turn_end_event,
    usage_event,
)
from tests.session.records._fixtures import (
    SUPERVISED_SESSION_ID,
    empty_session_state,
    finalize_record,
    make_iteration,
    session_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.session.store import SessionState
    from gymrat.supervisor.events import ModelPhase, SessionEvent

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def make_state(**changes: Any) -> ReporterState:
    """A fresh reporter state carrying the standard run facts, fields overridden."""
    base = ReporterState(
        root="/tmp/repo",
        max_minutes=60,
        max_usd=None,
        max_iterations=20,
        session_id=SUPERVISED_SESSION_ID,
        branch=f"gymrat/{SUPERVISED_SESSION_ID}",
        model=None,
        effort=None,
        log_path="/tmp/repo/.gymrat/supervisor.jsonl",
    )
    return replace(base, **changes)


def loop_session() -> SessionState:
    """A session with two iterations, one kept and one discarded, last regressed."""
    return session_state(
        iteration_count=2,
        keep_count=1,
        discard_count=1,
        last_iteration=make_iteration(3.2, "regressed"),
    )


def capped_state() -> ReporterState:
    """A state whose liveness is frozen by a wall-clock cap."""
    return make_state(liveness=Capped("wall-clock", "interrupting"))


def started(
    tool_name: str,
    tool_use_id: str,
    at_ms: int = TOOL_START_MS,
    *,
    base: ReporterState | None = None,
) -> ReporterState:
    """*base* (or a fresh state) advanced through a tool-start event for *tool_name*."""
    return advance(
        base if base is not None else make_state(),
        tool_start_event(tool_name, tool_use_id, at_ms),
        None,
    )


def bash_in_flight_state() -> ReporterState:
    """A state with a top-level Bash call in flight since 1500 ms."""
    return started("Bash", "bash-1", 1500)


def nested_read_state() -> ReporterState:
    """A Bash call in flight with a nested Read running since 2000 ms."""
    return advance(
        bash_in_flight_state(),
        tool_start_event("Read", "read-1", 2000, parent_tool_use_id="bash-1"),
        None,
    )


def exiting_state() -> ReporterState:
    """A state whose run-end exit sequence has been settling since 7000 ms."""
    return exit_phase(make_state(), ExitPhase(kind="settling", pid=None), 7000)


def waiting_state() -> ReporterState:
    """A state waiting on the agent since 2000 ms."""
    return make_state(liveness=Waiting(since=2000))


def emit(
    state: ReporterState,
    event: SessionEvent,
    session: ReadSessionResult | None = None,
) -> tuple[ReporterState, str | None]:
    """Advance *state* over *event* and pair the result with its plain-mode line."""
    after = advance(state, event, session)
    return after, plain_line(state, after, event)


# ---------------------------------------------------------------------------
# advance — caps, launch, usage
# ---------------------------------------------------------------------------


def test_advance_when_cap_fires_does_freeze_liveness_without_touching_the_last_decision():
    before = make_state(last_decision="turn 1 ended · replied")

    after = advance(before, cap_event("wall-clock", action="ending"), None)

    assert after.liveness == Capped("wall-clock", "ending")
    assert after.last_decision == "turn 1 ended · replied"


def test_advance_when_launch_does_start_the_run_on_the_passed_session():
    session = read_result()

    after = advance(make_state(), launch_event(1000), session)

    assert after.launch_timestamp == 1000
    assert after.session_result is session


def test_advance_when_usage_update_does_record_the_cost():
    after = advance(make_state(), usage_event(1.42), None)

    assert after.cost_usd == 1.42


# ---------------------------------------------------------------------------
# advance — tool start
# ---------------------------------------------------------------------------


def test_advance_when_top_level_tool_starts_does_go_in_flight():
    after = advance(
        make_state(), tool_start_event("Bash", "bash-1", 2000, input_summary="npm test"), None
    )

    assert dict(after.in_flight_tools) == {
        "bash-1": RunningTool(tool_name="Bash", input_summary="npm test", since=2000)
    }
    assert after.liveness == InFlight(
        tool_use_id="bash-1", tool_name="Bash", since=2000, input_summary="npm test"
    )


def test_advance_when_tool_starts_while_capped_does_track_it_but_stay_capped():
    before = capped_state()

    after = started("Bash", "bash-1", base=before)

    assert dict(after.in_flight_tools).keys() == {"bash-1"}
    assert after.liveness == before.liveness


def test_advance_when_nested_tool_starts_under_an_in_flight_parent_does_record_nested_activity():
    before = bash_in_flight_state()

    after = advance(
        before, tool_start_event("Read", "read-1", 2000, parent_tool_use_id="bash-1"), None
    )

    assert dict(after.nested) == {
        "bash-1": RunningTool(tool_name="Read", input_summary="...", since=2000)
    }
    assert after.liveness == before.liveness
    assert dict(after.in_flight_tools).keys() == {"bash-1"}


def test_advance_when_nested_tool_has_no_in_flight_parent_does_not_change_state():
    before = make_state()

    after = advance(
        before, tool_start_event("Read", "read-1", 2000, parent_tool_use_id="ghost"), None
    )

    assert after == before


# ---------------------------------------------------------------------------
# advance — tool end
# ---------------------------------------------------------------------------


_EARLIER_READ = read_result()
_LATER_READ = read_result(loop_session(), has_baseline=True)


@pytest.mark.parametrize(
    ("passed", "expected_session"),
    [
        pytest.param(_LATER_READ, _LATER_READ, id="session-passed"),
        pytest.param(None, _EARLIER_READ, id="no-session-passed-keeps-the-earlier-read"),
    ],
)
def test_advance_when_top_level_tool_ends_does_wait_on_it_with_the_resolved_session(
    passed: ReadSessionResult | None, expected_session: ReadSessionResult
):
    before = started("Read", "read-1", base=make_state(session_result=_EARLIER_READ))

    after = advance(before, tool_end_event("Read", "read-1", 3000), passed)

    finished = FinishedTool(
        tool_name="Read", input_summary="...", duration_ms=1000, result="ok", ended_at=3000
    )
    assert after.finished_tools == (finished,)
    assert dict(after.in_flight_tools) == {}
    assert after.liveness == Waiting(since=3000, last_tool=finished)
    assert after.session_result is expected_session


def test_advance_when_one_of_two_tools_ends_does_fall_back_to_the_remaining_tool():
    before = started("Read", "read-1")
    before = started("Bash", "bash-1", 2500, base=before)

    after = advance(before, tool_end_event("Read", "read-1", 3000), None)

    assert after.liveness == InFlight(
        tool_use_id="bash-1", tool_name="Bash", since=2500, input_summary="..."
    )


def test_advance_when_tool_ends_while_capped_does_stay_capped():
    before = started("Bash", "bash-1", base=capped_state())

    after = advance(before, tool_end_event("Bash", "bash-1", 3000), None)

    assert after.liveness == Capped("wall-clock", "interrupting")
    assert len(after.finished_tools) == 1


def test_advance_when_untracked_tool_ends_does_log_it_without_summary_or_liveness_change():
    after = advance(make_state(), tool_end_event("Bash", "never-started", 3000), None)

    assert after.finished_tools == (
        FinishedTool(
            tool_name="Bash", input_summary="", duration_ms=1000, result="ok", ended_at=3000
        ),
    )
    assert after.liveness == Starting()


def test_advance_when_more_tools_finish_than_the_bound_does_keep_only_the_most_recent():
    state = make_state()
    for index in range(1, 5):
        at_ms = 1000 + index * 1000
        state = advance(
            state,
            tool_start_event("Read", f"t-{index}", at_ms, input_summary=f"src/{index}.ts"),
            None,
        )
        state = advance(
            state,
            tool_end_event("Read", f"t-{index}", at_ms + 500, started_at_ms=at_ms),
            None,
        )

    assert [tool.input_summary for tool in state.finished_tools] == [
        "src/2.ts",
        "src/3.ts",
        "src/4.ts",
    ]


def test_advance_when_tracked_nested_tool_ends_does_retire_it():
    before = nested_read_state()
    session = read_result()

    after = advance(
        before, tool_end_event("Read", "read-1", 2500, parent_tool_use_id="bash-1"), session
    )

    assert after.finished_tools == ()
    assert after.liveness == before.liveness
    assert dict(after.nested) == {}
    assert after.session_result is session


@pytest.mark.parametrize(
    "line_setter",
    [
        pytest.param(
            lambda: tool_start_event("Grep", "grep-1", 2100, parent_tool_use_id="bash-1"),
            id="running-tool",
        ),
        pytest.param(
            lambda: model_phase_event(2100, "responding", parent_tool_use_id="bash-1"),
            id="model-phase",
        ),
    ],
)
def test_advance_when_untracked_nested_tool_ends_does_leave_the_nested_line_alone(
    line_setter: Callable[[], SessionEvent],
):
    before = advance(bash_in_flight_state(), line_setter(), None)

    after = advance(
        before, tool_end_event("Read", "read-1", 2500, parent_tool_use_id="bash-1"), None
    )

    assert (set(dict(before.nested)), after.nested) == ({"bash-1"}, before.nested)


def two_nested_tools_state() -> ReporterState:
    """A Bash call in flight with a nested Read (2000 ms) and Grep (2100 ms) both running."""
    state = advance(
        bash_in_flight_state(),
        tool_start_event("Read", "read-1", 2000, parent_tool_use_id="bash-1"),
        None,
    )
    return advance(
        state, tool_start_event("Grep", "grep-1", 2100, parent_tool_use_id="bash-1"), None
    )


@pytest.mark.parametrize(
    ("ended", "still_running"),
    [
        pytest.param(
            ("Read", "read-1"),
            RunningTool(tool_name="Grep", input_summary="...", since=2100),
            id="first-started-ends",
        ),
        pytest.param(
            ("Grep", "grep-1"),
            RunningTool(tool_name="Read", input_summary="...", since=2000),
            id="last-started-ends",
        ),
    ],
)
def test_advance_when_one_of_two_nested_tools_ends_does_show_the_one_still_running(
    ended: tuple[str, str], still_running: RunningTool
):
    before = two_nested_tools_state()

    after = advance(before, tool_end_event(*ended, 2500, parent_tool_use_id="bash-1"), None)

    assert dict(after.nested) == {"bash-1": still_running}


def test_advance_when_nested_model_phase_arrives_after_a_sibling_ends_does_keep_the_running_tool():
    before = advance(
        two_nested_tools_state(),
        tool_end_event("Read", "read-1", 2500, parent_tool_use_id="bash-1"),
        None,
    )

    after = advance(
        before, model_phase_event(2600, "responding", parent_tool_use_id="bash-1"), None
    )

    assert dict(after.nested) == {
        "bash-1": RunningTool(tool_name="Grep", input_summary="...", since=2100)
    }


def test_advance_when_parent_tool_ends_does_drop_only_its_own_nested_tools():
    before = started("Task", "task-1", 1600, base=two_nested_tools_state())
    before = advance(
        before, tool_start_event("Glob", "glob-1", 2200, parent_tool_use_id="task-1"), None
    )

    after = advance(before, tool_end_event("Bash", "bash-1", 3000, started_at_ms=1500), None)

    assert dict(after.nested) == {
        "task-1": RunningTool(tool_name="Glob", input_summary="...", since=2200)
    }


def test_advance_when_nested_tool_outlives_another_parent_does_clear_its_line_when_it_ends():
    before = started("Task", "task-1", 1600, base=two_nested_tools_state())
    before = advance(
        before, tool_start_event("Glob", "glob-1", 2200, parent_tool_use_id="task-1"), None
    )
    before = advance(before, tool_end_event("Bash", "bash-1", 3000, started_at_ms=1500), None)

    after = advance(
        before, tool_end_event("Glob", "glob-1", 3500, parent_tool_use_id="task-1"), None
    )

    assert dict(after.nested) == {}


@pytest.mark.parametrize(
    "make_event",
    [
        pytest.param(lambda: turn_end_event(4000), id="turn-end"),
        pytest.param(lambda: model_phase_event(4000, "turn_end"), id="turn-end-phase"),
    ],
)
def test_advance_when_turn_ends_after_a_tool_finished_does_wait_on_that_tool(
    make_event: Callable[[], SessionEvent],
):
    before = advance(started("Read", "read-1"), tool_end_event("Read", "read-1", 3000), None)

    after = advance(before, make_event(), None)

    assert after.liveness == Waiting(since=4000, last_tool=before.finished_tools[-1])


# ---------------------------------------------------------------------------
# advance — thinking and model phase
# ---------------------------------------------------------------------------


def test_advance_when_thinking_update_does_show_thinking_with_the_token_estimate():
    after = advance(make_state(), thinking_event(1500, estimated_tokens=200), None)

    assert after.liveness == Thinking(since=1500, estimated_tokens=200)


@pytest.mark.parametrize(
    ("phase", "tool_name", "expected"),
    [
        pytest.param("responding", None, Responding(since=2000), id="responding"),
        pytest.param(
            "tool_input", "Edit", Composing(tool_name="Edit", since=2000), id="tool-input"
        ),
        pytest.param("turn_end", None, Waiting(since=2000), id="turn-end"),
    ],
)
def test_advance_when_model_phase_arrives_does_set_the_matching_liveness(
    phase: ModelPhase, tool_name: str | None, expected: object
):
    after = advance(make_state(), model_phase_event(2000, phase, tool_name=tool_name), None)

    assert after.liveness == expected


def test_advance_when_model_phase_thinking_follows_a_thinking_update_does_keep_the_estimate():
    before = advance(make_state(), thinking_event(1500, estimated_tokens=200), None)

    after = advance(before, model_phase_event(2000, "thinking"), None)

    assert isinstance(after.liveness, Thinking)
    assert after.liveness.estimated_tokens == 200


@pytest.mark.parametrize(
    "blocking", [capped_state, bash_in_flight_state], ids=["capped", "in-flight"]
)
@pytest.mark.parametrize(
    "make_event",
    [
        pytest.param(lambda: thinking_event(2000, estimated_tokens=999), id="thinking-update"),
        pytest.param(lambda: model_phase_event(2000, "responding"), id="model-phase"),
    ],
)
def test_advance_when_liveness_is_blocking_does_ignore_thinking_and_phase_events(
    blocking: Callable[[], ReporterState], make_event: Callable[[], SessionEvent]
):
    before = blocking()

    after = advance(before, make_event(), None)

    assert after.liveness == before.liveness


_NESTED_MODEL_PHASES = [
    pytest.param("thinking", None, id="thinking"),
    pytest.param("responding", None, id="responding"),
    pytest.param("tool_input", "Edit", id="tool-input"),
]


@pytest.mark.parametrize(("phase", "tool_name"), _NESTED_MODEL_PHASES)
def test_advance_when_nested_model_phase_arrives_does_record_it_under_the_parent(
    phase: ModelPhase, tool_name: str | None
):
    before = bash_in_flight_state()

    after = advance(
        before,
        model_phase_event(2000, phase, tool_name=tool_name, parent_tool_use_id="bash-1"),
        None,
    )

    assert after.liveness == before.liveness
    assert dict(after.nested) == {
        "bash-1": NestedPhase(phase=phase, since=2000, tool_name=tool_name)
    }


@pytest.mark.parametrize(
    ("phase", "tool_name"),
    [*_NESTED_MODEL_PHASES, pytest.param("turn_end", None, id="turn-end")],
)
def test_advance_when_nested_model_phase_arrives_during_a_nested_tool_does_keep_the_tool(
    phase: ModelPhase, tool_name: str | None
):
    before = nested_read_state()

    after = advance(
        before,
        model_phase_event(2500, phase, tool_name=tool_name, parent_tool_use_id="bash-1"),
        None,
    )

    assert after == before


@pytest.mark.parametrize(("phase", "tool_name"), _NESTED_MODEL_PHASES)
def test_advance_when_nested_turn_ends_does_clear_the_nested_model_phase(
    phase: ModelPhase, tool_name: str | None
):
    before = advance(
        bash_in_flight_state(),
        model_phase_event(2000, phase, tool_name=tool_name, parent_tool_use_id="bash-1"),
        None,
    )

    after = advance(before, model_phase_event(2500, "turn_end", parent_tool_use_id="bash-1"), None)

    assert dict(after.nested) == {}


def test_advance_when_nested_thinking_update_arrives_does_not_change_state():
    before = bash_in_flight_state()

    after = advance(
        before, thinking_event(2000, estimated_tokens=999, parent_tool_use_id="bash-1"), None
    )

    assert after == before


# ---------------------------------------------------------------------------
# advance — turn end, follow-up, compaction, text delta
# ---------------------------------------------------------------------------


def test_advance_when_agent_turn_ends_does_wait_after_recording_the_turn():
    after = advance(make_state(), turn_end_event(2000, text="done"), None)

    assert after.turn_count == 1
    assert after.last_agent_text == "done"
    assert after.liveness == Waiting(since=2000)


def test_advance_when_injected_turn_ends_does_count_it_without_replacing_the_agent_text():
    before = advance(make_state(), turn_end_event(2000, text="done"), None)

    after = advance(before, turn_end_event(3000, text="injected", origin="injected"), None)

    assert after.turn_count == 2
    assert after.last_agent_text == "done"


def test_advance_when_turn_ends_while_capped_does_count_it_but_stay_capped():
    before = capped_state()

    after = advance(before, turn_end_event(6000), None)

    assert after.turn_count == 1
    assert after.liveness == before.liveness


@pytest.mark.parametrize(
    ("action", "reason", "expected"),
    [
        pytest.param("replied", None, "turn 1 ended · replied", id="replied"),
        pytest.param("waiting", None, "turn 1 ended · waiting for gymrat", id="waiting"),
        pytest.param(
            "ended", "budget exhausted", "turn 1 ended · ended budget exhausted", id="ended"
        ),
        pytest.param("ended", None, "turn 1 ended · ended", id="ended-without-reason"),
    ],
)
def test_advance_when_follow_up_arrives_does_record_the_turn_decision(
    action: Literal["replied", "waiting", "ended"], reason: str | None, expected: str
):
    before = advance(make_state(), turn_end_event(2000), None)

    after = advance(before, follow_up_event(3000, action=action, reason=reason), None)

    assert after.last_decision == expected


def test_advance_when_compaction_arrives_does_record_the_context_decision():
    after = advance(make_state(), CompactionEvent(at=3000 * NS_PER_MS), None)

    assert after.last_decision == "context compacted"


def test_advance_when_text_delta_arrives_does_not_change_state():
    before = make_state()

    after = advance(before, TextDeltaEvent(at=2000 * NS_PER_MS, chunk="hi"), None)

    assert after == before


# ---------------------------------------------------------------------------
# exit_phase — run-end exit sequence
# ---------------------------------------------------------------------------


def test_exit_phase_when_same_phase_repeats_does_return_the_state_unchanged():
    before = exiting_state()

    after = exit_phase(before, ExitPhase(kind="settling", pid=None), 9000)

    assert after is before


def test_exit_phase_when_lock_holder_changes_does_restart_the_timestamp():
    before = exit_phase(make_state(), ExitPhase(kind="waiting-lock", pid=4242), 7000)

    after = exit_phase(before, ExitPhase(kind="waiting-lock", pid=5151), 9000)

    assert after.liveness == Exiting(kind="waiting-lock", since=9000, pid=5151)


@pytest.mark.parametrize(
    ("reason", "passed", "expected_decision", "expected_session"),
    [
        pytest.param(
            "discard", _LATER_READ, "exit · discard", _LATER_READ, id="reason-with-session-passed"
        ),
        pytest.param("", None, "exit", _EARLIER_READ, id="empty-reason-keeps-the-earlier-read"),
    ],
)
def test_advance_when_ended_follow_up_arrives_while_exiting_does_record_the_exit_step(
    reason: str,
    passed: ReadSessionResult | None,
    expected_decision: str,
    expected_session: ReadSessionResult,
):
    before = replace(exiting_state(), session_result=_EARLIER_READ)

    after = advance(before, follow_up_event(8000, action="ended", reason=reason), passed)

    assert (after.last_decision, after.session_result) == (expected_decision, expected_session)


# ---------------------------------------------------------------------------
# wants_session_refresh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        pytest.param(launch_event(1000), True, id="launch"),
        pytest.param(tool_end_event("Read", "read-1", 3000), True, id="tool-end"),
        pytest.param(
            tool_end_event("Read", "nested-1", 4200, parent_tool_use_id="bash-1"),
            True,
            id="nested-tool-end",
        ),
        pytest.param(usage_event(1.0), False, id="usage-update"),
        pytest.param(tool_start_event("Bash", "bash-2", 2000), False, id="tool-start"),
    ],
)
def test_wants_session_refresh_when_event_arrives_does_match_the_reread_contract(
    event: SessionEvent, expected: bool
):
    state = make_state()

    refresh = wants_session_refresh(state, event)

    assert refresh is expected


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        pytest.param(exiting_state, True, id="exiting"),
        pytest.param(waiting_state, False, id="waiting"),
        pytest.param(make_state, False, id="starting"),
        pytest.param(lambda: started("Bash", "bash-1"), False, id="in-flight"),
        pytest.param(capped_state, False, id="capped"),
    ],
)
def test_wants_session_refresh_when_ended_follow_up_arrives_does_reread_only_while_exiting(
    make: Callable[[], ReporterState], expected: bool
):
    state = make()
    event = follow_up_event(8000, action="ended", reason="keep")

    refresh = wants_session_refresh(state, event)

    assert refresh is expected


# ---------------------------------------------------------------------------
# plain_line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_minutes", "max_usd", "expected"),
    [
        pytest.param(60, None, "caps 60m", id="no-spend-cap"),
        pytest.param(60, 5.0, "caps 60m, $5.00", id="spend-cap"),
        pytest.param(5.5, None, "caps 5.5m", id="fractional-minutes-keep-the-decimal"),
        pytest.param(10.0, None, "caps 10m", id="whole-float-minutes-drop-the-decimal"),
    ],
)
def test_plain_line_when_launch_does_return_the_caps_line(
    max_minutes: float, max_usd: float | None, expected: str
):
    state = make_state(max_minutes=max_minutes, max_usd=max_usd)

    _, line = emit(
        state, launch_event(1000, max_minutes=max_minutes, max_usd=max_usd), read_result()
    )

    assert line == expected


@pytest.mark.parametrize(
    ("state", "event", "expected"),
    [
        pytest.param(make_state(), usage_event(1.42), "cost $1.42", id="usage-update"),
        pytest.param(
            make_state(), cap_event("wall-clock"), "cap wall-clock — interrupting", id="cap"
        ),
        pytest.param(
            make_state(),
            cap_event("wall-clock", action="ending"),
            "cap wall-clock — ending",
            id="cap-ending",
        ),
        pytest.param(
            advance(make_state(), turn_end_event(2000), None),
            follow_up_event(3000),
            "turn 1 ended · replied",
            id="follow-up",
        ),
        pytest.param(
            make_state(), CompactionEvent(at=3000 * NS_PER_MS), "context compacted", id="compaction"
        ),
        pytest.param(make_state(), tool_start_event("Bash", "bash-1", 2000), None, id="tool-start"),
    ],
)
def test_plain_line_when_event_arrives_does_return_its_line_or_nothing(
    state: ReporterState, event: SessionEvent, expected: str | None
):
    _, line = emit(state, event)

    assert line == expected


def test_plain_line_when_the_loop_text_changes_does_return_the_loop_line():
    state = started("Bash", "bash-1")
    session = read_result(loop_session(), has_baseline=True)

    _, line = emit(state, tool_end_event("Bash", "bash-1", 3000), session)

    assert line == "2/20 iterations · 1 kept · 1 discarded · last +3.2% regressed"


def test_plain_line_when_the_loop_text_is_unchanged_does_return_nothing():
    session = read_result(loop_session(), has_baseline=True)
    state = started("Bash", "bash-1")
    state = advance(state, tool_end_event("Bash", "bash-1", 3000), session)
    state = advance(state, tool_start_event("Bash", "bash-2", 4000), None)

    _, line = emit(state, tool_end_event("Bash", "bash-2", 5000, started_at_ms=4000), session)

    assert line is None


def test_plain_line_when_no_session_has_been_read_does_not_return_the_loop_line():
    state = started("Bash", "bash-1")

    _, line = emit(state, tool_end_event("Bash", "bash-1", 3000), None)

    assert line is None


# ---------------------------------------------------------------------------
# loop_plain_text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "has_baseline", "max_iterations", "expected"),
    [
        pytest.param(None, False, 20, "no session yet", id="no-session"),
        pytest.param(
            session_state(iteration_count=1),
            True,
            None,
            "1 iteration · 0 kept · 0 discarded",
            id="one-uncapped",
        ),
        pytest.param(
            session_state(iteration_count=2),
            True,
            None,
            "2 iterations · 0 kept · 0 discarded",
            id="two-uncapped",
        ),
        pytest.param(
            session_state(iteration_count=1),
            True,
            20,
            "1/20 iterations · 0 kept · 0 discarded",
            id="one-capped",
        ),
        pytest.param(
            empty_session_state(),
            True,
            20,
            "baseline recorded · no iterations yet",
            id="baseline-recorded",
        ),
        pytest.param(
            session_state_three_iterations(-3.2, "improved"),
            True,
            20,
            "3/20 iterations · 2 kept · 1 discarded · last -3.2% improved",
            id="iterations-present",
        ),
        pytest.param(
            session_state(iteration_count=1, last_iteration=make_iteration(2.2, "regressed")),
            True,
            None,
            "1 iteration · 0 kept · 0 discarded · last +2.2% regressed",
            id="finite-delta",
        ),
        pytest.param(
            session_state(iteration_count=1, last_iteration=make_iteration(None, "no-signal")),
            True,
            None,
            "1 iteration · 0 kept · 0 discarded · last — no-signal",
            id="missing-delta-renders-em-dash",
        ),
        pytest.param(
            session_state(
                iteration_count=1, unsettled=True, last_iteration=make_iteration(-2.0, "improved")
            ),
            True,
            None,
            "1 iteration · 0 kept · 0 discarded · last -2.0% improved, unsettled",
            id="unsettled",
        ),
        pytest.param(
            replace(session_state_three_iterations(-5.0, "improved"), finalized=finalize_record()),
            True,
            None,
            "3 iterations · finalized",
            id="finalized",
        ),
    ],
)
def test_loop_plain_text_when_session_varies_does_describe_the_loop(
    state: SessionState | None, has_baseline: bool, max_iterations: int | None, expected: str
):
    session = None if state is None else read_result(state, has_baseline=has_baseline)

    text = loop_plain_text(session, max_iterations)

    assert text == expected


# ---------------------------------------------------------------------------
# view-layer seam
# ---------------------------------------------------------------------------


def test_importing_reducer_when_loaded_does_not_load_the_frame_module():
    loaded = modules_imported_by("gymrat.cli.supervise.reducer")

    assert "gymrat.cli.supervise.frame" not in loaded, (
        "importing the reducer pulled in the Rich frame builder"
    )
