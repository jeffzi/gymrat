"""Tests for the Claude driver turn protocol.

A result message emits a ``TurnEndEvent`` and the stream continues.
Session settlement happens via ``end()``, ``interrupt()``, or natural stream
termination — never from a result message alone.
"""

import math
from collections.abc import Awaitable, Callable, Sequence
from operator import methodcaller
from typing import override

import pytest
from claude_agent_sdk import MessageOrigin, ResultMessage, TextBlock

from gymrat.supervisor.driver import DriverSession
from gymrat.supervisor.events import (
    SessionEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)
from tests.supervisor._fixtures import (
    FakeClient,
    FiniteClient,
    assistant,
    collecting_observer,
    end_and_settle,
    events_of,
    make_prompt,
    result_message,
    run_outcome,
    settled_outcome,
    start_claude_session,
    start_past_turns,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FailingFollowUpClient(FakeClient):
    """A client whose every ``query`` after the kickoff raises."""

    @override
    async def query(self, prompt: str) -> None:
        if self.query_prompts:
            message = "connection lost"
            raise RuntimeError(message)
        await super().query(prompt)


async def _run_turns(messages: Sequence[object]) -> list[SessionEvent]:
    """Run a session until every scripted result closed its turn, end it, and return its events."""
    turns = sum(isinstance(message, ResultMessage) for message in messages)
    session, events = await start_past_turns(FakeClient(messages), turns)
    await end_and_settle(session)
    return events


# ---------------------------------------------------------------------------
# A result message emits a TurnEndEvent without ending the stream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("messages", "texts"),
    [
        pytest.param(
            [assistant(TextBlock(text="first answer")), result_message(total_cost_usd=0.01)],
            ["first answer"],
            id="top-level-text",
        ),
        pytest.param([result_message(total_cost_usd=0.01)], [""], id="no-text-is-empty"),
        pytest.param(
            [
                assistant(TextBlock(text="turn one text")),
                result_message(total_cost_usd=0.01),
                assistant(TextBlock(text="turn two text")),
                result_message(total_cost_usd=0.02),
            ],
            ["turn one text", "turn two text"],
            id="second-turn-resets-the-text",
        ),
        pytest.param(
            [
                assistant(TextBlock(text="turn one text")),
                result_message(total_cost_usd=0.01),
                result_message(total_cost_usd=0.02),
            ],
            ["turn one text", ""],
            id="second-turn-without-text-carries-none-over",
        ),
        pytest.param(
            [
                assistant(TextBlock(text="top level")),
                assistant(TextBlock(text="subagent output"), parent_tool_use_id="tu_sub"),
                result_message(total_cost_usd=0.01),
            ],
            ["top level"],
            id="subagent-text-ignored",
        ),
    ],
)
async def test_start_when_result_ends_a_turn_does_carry_that_turns_top_level_text(
    messages: list[object], texts: list[str]
):
    events = await _run_turns(messages)

    assert [turn_end.text for turn_end in events_of(events, TurnEndEvent)] == texts


# ---------------------------------------------------------------------------
# Turn origin detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        pytest.param(None, "agent", id="origin-absent"),
        pytest.param({"kind": "human"}, "agent", id="origin-human"),
        pytest.param({"kind": "task-notification"}, "injected", id="origin-task-notification"),
    ],
)
async def test_start_when_result_carries_origin_does_report_turn_origin(
    origin: MessageOrigin | None, expected: str
):
    events = await _run_turns([result_message(total_cost_usd=0.01, origin=origin)])

    turn_ends = events_of(events, TurnEndEvent)
    assert [turn_end.origin for turn_end in turn_ends] == [expected]


# ---------------------------------------------------------------------------
# Cost tracking across turns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("costs", "usage_costs", "turn_end_costs"),
    [
        pytest.param([0.05, 0.15], [0.05, 0.15, 0.15], [0.05, 0.15], id="rising-usable-costs"),
        pytest.param([0.05, math.nan], [0.05, 0.05], [0.05, 0.05], id="nan-after-a-usable-cost"),
        pytest.param([None], [0.0], [0.0], id="none-before-any-cost"),
    ],
)
async def test_start_when_results_report_costs_does_track_the_running_cost(
    costs: list[float | None], usage_costs: list[float], turn_end_costs: list[float]
):
    messages = [result_message(total_cost_usd=cost) for cost in costs]

    events = await _run_turns(messages)

    # The last usage update is the one ``end()`` commits at the running cost.
    assert [update.cost_usd for update in events_of(events, UsageUpdateEvent)] == usage_costs
    assert [turn_end.cost_usd for turn_end in events_of(events, TurnEndEvent)] == turn_end_costs


# ---------------------------------------------------------------------------
# is_error handling and budget exhaustion
# ---------------------------------------------------------------------------


async def test_start_when_budget_exhausted_does_flag_the_turn_end_without_settling_error():
    result = result_message(
        subtype="error_max_budget_usd",
        is_error=True,
        total_cost_usd=1.50,
        result="Budget exceeded",
    )

    session, events = await start_past_turns(FakeClient([result]), turns=1)
    outcome = await end_and_settle(session)

    turn_ends = events_of(events, TurnEndEvent)
    assert [(t.budget_exhausted, t.cost_usd) for t in turn_ends] == [(True, 1.50)]
    assert outcome.reason == "completed"


# ---------------------------------------------------------------------------
# send(text) forwards to client.query on the same connection until the session settles
# ---------------------------------------------------------------------------


async def test_send_when_called_does_forward_text_to_client_query():
    client = FakeClient([result_message(total_cost_usd=0.01)])
    session, _ = await start_past_turns(client, turns=1, prompt=make_prompt(kickoff="initial"))

    await session.send("follow up message")
    await end_and_settle(session)

    assert client.query_prompts == ["initial", "follow up message"]


async def test_send_when_client_query_fails_does_settle_error_with_its_message():
    session, _ = await start_past_turns(
        _FailingFollowUpClient([result_message(total_cost_usd=0.01)]), turns=1
    )

    await session.send("follow up message")
    outcome = await settled_outcome(session)

    assert (outcome.reason, outcome.message, outcome.cost_usd) == (
        "error",
        "connection lost",
        0.01,
    )


# ---------------------------------------------------------------------------
# end() settles the session completed unless it already settled or was interrupted
# ---------------------------------------------------------------------------


async def test_end_when_called_does_settle_completed_at_the_running_cost():
    session, _ = await start_past_turns(FakeClient([result_message(total_cost_usd=0.20)]), turns=1)

    outcome = await end_and_settle(session)

    assert (outcome.reason, outcome.cost_usd) == ("completed", 0.20)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(methodcaller("send", "should be ignored"), id="send"),
        pytest.param(methodcaller("end"), id="end"),
    ],
)
async def test_start_when_settled_by_an_error_result_does_ignore_later_session_calls(
    call: Callable[[DriverSession], Awaitable[None]],
):
    result = result_message(subtype="error", is_error=True, result="fatal", total_cost_usd=0.10)
    client = FakeClient([result])
    probe = collecting_observer()
    session = start_claude_session(client, probe.observer)
    await settled_outcome(session)
    queries_before = list(client.query_prompts)
    events_before = list(probe.events)

    await call(session)

    assert client.query_prompts == queries_before
    assert probe.events == events_before


# ---------------------------------------------------------------------------
# Natural stream termination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        pytest.param(
            [assistant(TextBlock(text="hello"))],
            ("error", "Agent stream ended without a result message", 0.0),
            id="no-turn-closed",
        ),
        pytest.param(
            [result_message(total_cost_usd=0.05)], ("completed", None, 0.05), id="one-turn-closed"
        ),
    ],
)
async def test_start_when_stream_ends_on_its_own_does_settle_by_whether_a_turn_closed(
    messages: list[object], expected: tuple[str, str | None, float]
):
    outcome = await run_outcome(FiniteClient(messages))

    assert (outcome.reason, outcome.message, outcome.cost_usd) == expected
