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
    plain_line,
    wants_session_refresh,
)
from gymrat.cli.supervise.state import (
    Capped,
    Composing,
    FinishedTool,
    InFlight,
    NestedTool,
    ReadSessionResult,
    Responding,
    Starting,
    Thinking,
    TrackedTool,
    Waiting,
)
from gymrat.session import IterationPrimary
from gymrat.supervisor.events import (
    CompactionEvent,
    TextDeltaEvent,
)
from tests.cli.supervise._fixtures import (
    _NS_PER_MS,
    cap_event,
    empty_session_state,
    follow_up_event,
    launch_event,
    model_phase_event,
    thinking_event,
    tool_end_event,
    tool_start_event,
    turn_end_event,
    usage_event,
)
from tests.event_docs._imports import modules_imported_by
from tests.session.records._fixtures import iteration_record

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.session.store import SessionState
    from gymrat.supervisor.events import SessionEvent

# Milliseconds the shared tool-start builders stamp, so the matching end
# builder can derive a duration against it.
_TOOL_START_MS = 2000


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
        label="ecstatic-ts",
        session_id="20260813-125044-34ec",
        branch="gymrat/20260813-125044-34ec",
        model=None,
        effort=None,
        log_path="/tmp/repo/.gymrat/supervisor.jsonl",
    )
    return replace(base, **changes)


def loop_session() -> SessionState:
    """A session with two iterations, one kept and one discarded, last regressed."""
    return replace(
        empty_session_state(),
        iteration_count=2,
        keep_count=1,
        discard_count=1,
        last_iteration=iteration_record(
            seq=1,
            primary=IterationPrimary(kind="geomean", delta_pct=3.2),
            outcome="regressed",
        ),
    )


def read_result(
    state: SessionState | None = None, *, has_baseline: bool = False
) -> ReadSessionResult:
    """A session read result wrapping *state*, defaulting to the empty session."""
    return ReadSessionResult(
        state=state if state is not None else empty_session_state(), has_baseline=has_baseline
    )


def capped_state() -> ReporterState:
    """A state whose liveness is frozen by a wall-clock cap."""
    return make_state(liveness=Capped("wall-clock", "interrupting"))


def started(
    tool_name: str,
    tool_use_id: str,
    at_ms: int = _TOOL_START_MS,
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


def emit(
    state: ReporterState,
    event: SessionEvent,
    session: ReadSessionResult | None = None,
) -> tuple[ReporterState, str | None]:
    """Advance *state* over *event* and pair the result with its plain-mode line."""
    after = advance(state, event, session)
    return after, plain_line(state, after, event)


# ---------------------------------------------------------------------------
# purity
# ---------------------------------------------------------------------------


def test_advance_when_applied_twice_to_same_input_does_return_equal_states():
    before = make_state()
    event = tool_start_event("Bash", "bash-1", 2000)

    assert advance(before, event, None) == advance(before, event, None)


def test_advance_when_applied_does_leave_input_state_unchanged():
    before = make_state()

    advance(before, tool_start_event("Bash", "bash-1", 2000), None)

    assert before == make_state()


# ---------------------------------------------------------------------------
# advance — caps, launch, usage
# ---------------------------------------------------------------------------


def test_advance_when_cap_fires_does_freeze_liveness_without_touching_the_last_decision():
    before = make_state(last_decision="turn 1 ended · replied")

    after = advance(before, cap_event("wall-clock", action="ending"), None)

    assert after.liveness == Capped("wall-clock", "ending")
    assert after.last_decision == "turn 1 ended · replied"


def test_advance_when_launch_does_record_the_timestamp_and_the_passed_session():
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


def test_advance_when_top_level_tool_starts_does_track_it_and_go_in_flight():
    after = advance(
        make_state(), tool_start_event("Bash", "bash-1", 2000, input_summary="npm test"), None
    )

    assert dict(after.in_flight_tools) == {
        "bash-1": TrackedTool(tool_name="Bash", started_at=2000, input_summary="npm test")
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
        "bash-1": NestedTool(tool_name="Read", input_summary="...", since=2000)
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


def test_advance_when_top_level_tool_ends_does_log_the_finished_tool_and_wait():
    before = started("Read", "read-1")

    after = advance(before, tool_end_event("Read", "read-1", 3000), None)

    assert after.finished_tools == (
        FinishedTool(
            tool_name="Read", input_summary="...", duration_ms=1000, result="ok", ended_at=3000
        ),
    )
    assert dict(after.in_flight_tools) == {}
    assert after.liveness == Waiting(since=3000, tool_name="Read", tool_ended_at=3000, result="ok")


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


def test_advance_when_bash_tool_ends_does_take_the_passed_session_result():
    before = started("Bash", "bash-1")
    session = read_result()

    after = advance(before, tool_end_event("Bash", "bash-1", 3000), session)

    assert after.session_result is session


def test_advance_when_tracked_non_bash_tool_ends_does_keep_the_previous_session_result():
    session = read_result()
    before = started("Read", "read-1", base=make_state(session_result=session))

    after = advance(before, tool_end_event("Read", "read-1", 3000), None)

    assert after.session_result is session


def test_advance_when_nested_tool_ends_does_not_log_it_as_a_finished_tool():
    before = bash_in_flight_state()
    before = advance(
        before, tool_start_event("Read", "read-1", 2000, parent_tool_use_id="bash-1"), None
    )

    after = advance(
        before, tool_end_event("Read", "read-1", 2500, parent_tool_use_id="bash-1"), None
    )

    assert after.finished_tools == ()
    assert after.liveness == before.liveness


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
    phase: str, tool_name: str | None, expected: object
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


@pytest.mark.parametrize(
    "make_event",
    [
        pytest.param(
            lambda: model_phase_event(2000, "responding", parent_tool_use_id="bash-1"),
            id="model-phase",
        ),
        pytest.param(
            lambda: model_phase_event(
                2000, "tool_input", tool_name="Edit", parent_tool_use_id="bash-1"
            ),
            id="tool-input",
        ),
    ],
)
def test_advance_when_nested_model_phase_arrives_does_record_it_under_the_parent(
    make_event: Callable[[], SessionEvent],
):
    before = bash_in_flight_state()

    after = advance(before, make_event(), None)

    assert after.liveness == before.liveness
    assert set(dict(after.nested)) == {"bash-1"}


def test_advance_when_nested_thinking_update_arrives_does_not_change_state():
    before = bash_in_flight_state()

    after = advance(
        before, thinking_event(2000, estimated_tokens=999, parent_tool_use_id="bash-1"), None
    )

    assert after == before


# ---------------------------------------------------------------------------
# advance — turn end, follow-up, compaction, text delta
# ---------------------------------------------------------------------------


def test_advance_when_agent_turn_ends_does_count_it_keep_the_text_and_wait():
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
    ],
)
def test_advance_when_follow_up_arrives_does_record_the_turn_decision(
    action: Literal["replied", "waiting", "ended"], reason: str | None, expected: str
):
    before = advance(make_state(), turn_end_event(2000), None)

    after = advance(before, follow_up_event(3000, action=action, reason=reason), None)

    assert after.last_decision == expected


def test_advance_when_compaction_arrives_does_record_the_context_decision():
    after = advance(make_state(), CompactionEvent(at=3000 * _NS_PER_MS), None)

    assert after.last_decision == "context compacted"


def test_advance_when_text_delta_arrives_does_not_change_state():
    before = make_state()

    after = advance(before, TextDeltaEvent(at=2000 * _NS_PER_MS, chunk="hi"), None)

    assert after == before


# ---------------------------------------------------------------------------
# wants_session_refresh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        pytest.param(launch_event(1000), True, id="launch"),
        pytest.param(tool_end_event("Bash", "bash-1", 3000), True, id="top-level-bash"),
        pytest.param(tool_end_event("Read", "untracked", 3000), True, id="untracked-tool"),
        pytest.param(tool_end_event("Read", "read-1", 3000), False, id="tracked-non-bash"),
        pytest.param(
            tool_end_event("Bash", "bash-1", 3000, parent_tool_use_id="bash-1"),
            False,
            id="nested-bash",
        ),
        pytest.param(usage_event(1.0), False, id="usage-update"),
        pytest.param(tool_start_event("Bash", "bash-2", 2000), False, id="tool-start"),
    ],
)
def test_wants_session_refresh_when_event_arrives_does_match_the_reread_contract(
    event: SessionEvent, expected: bool
):
    state = started("Bash", "bash-1")
    state = started("Read", "read-1", base=state)

    assert wants_session_refresh(state, event) is expected


# ---------------------------------------------------------------------------
# plain_line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_usd", "expected"),
    [
        pytest.param(None, "caps 60m", id="no-spend-cap"),
        pytest.param(5.0, "caps 60m, $5.00", id="spend-cap"),
    ],
)
def test_plain_line_when_launch_does_return_the_caps_line(max_usd: float | None, expected: str):
    state = make_state(max_usd=max_usd)

    _, line = emit(state, launch_event(1000, max_usd=max_usd), read_result())

    assert line == expected


def test_plain_line_when_usage_update_does_return_the_cost_line():
    _, line = emit(make_state(), usage_event(1.42))

    assert line == "cost $1.42"


def test_plain_line_when_cap_fires_does_return_the_cap_line():
    _, line = emit(make_state(), cap_event("wall-clock"))

    assert line == "cap wall-clock — interrupting"


def test_plain_line_when_follow_up_arrives_does_return_the_decision_line():
    state = advance(make_state(), turn_end_event(2000), None)

    _, line = emit(state, follow_up_event(3000))

    assert line == "turn 1 ended · replied"


def test_plain_line_when_compaction_arrives_does_return_the_context_line():
    _, line = emit(make_state(), CompactionEvent(at=3000 * _NS_PER_MS))

    assert line == "context compacted"


def test_plain_line_when_tool_starts_does_return_nothing():
    _, line = emit(make_state(), tool_start_event("Bash", "bash-1", 2000))

    assert line is None


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
# view-layer seam
# ---------------------------------------------------------------------------


def test_importing_reducer_when_loaded_does_not_load_the_frame_module():
    loaded = modules_imported_by("gymrat.cli.supervise.reducer")

    assert "gymrat.cli.supervise.frame" not in loaded, (
        "importing the reducer pulled in the Rich frame builder"
    )
