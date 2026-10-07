"""Tests for the Claude driver turn protocol.

A result message emits a ``TurnEndEvent`` and the stream continues.
Session settlement happens via ``end()``, ``interrupt()``, or natural stream
termination — never from a result message alone.
"""

import asyncio
import math
from collections.abc import Sequence

import pytest
from claude_agent_sdk import MessageOrigin, ResultMessage, TextBlock

from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.driver import DriverSession, SessionOutcome, SessionPrompt
from gymrat.supervisor.events import (
    SessionEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)
from tests.supervisor._fixtures import (
    FactoryProbe,
    FakeClient,
    FiniteClient,
    assistant,
    collecting_observer,
    events_of,
    make_prompt,
    noop_observer,
    result_message,
    run_outcome,
    wait_for_event_or_task,
)

_TEST_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def _outcome(session: DriverSession) -> SessionOutcome:
    """Await the session's outcome within the test timeout."""
    return await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)


async def _settle(session: DriverSession) -> SessionOutcome:
    """End the session and await its settled outcome."""
    await session.end()
    return await _outcome(session)


async def _start_past_turns(
    client: FakeClient, turns: int, prompt: SessionPrompt | None = None
) -> tuple[DriverSession, list[SessionEvent]]:
    """Start a session over ``client`` and wait until it has closed ``turns`` turns.

    Waiting on the turn ends themselves, not on a count of event-loop yields,
    keeps ``end()`` from landing before the stream has drawn every result.

    Args:
        client: The fake client the session streams from.
        turns: How many ``TurnEndEvent`` the session must emit before returning.
        prompt: The prompt to start with; ``make_prompt()`` when omitted.

    Returns:
        The running session and the list its observer keeps appending events to.
    """
    events: list[SessionEvent] = []
    turns_closed = asyncio.Event()

    def observer(event: SessionEvent) -> None:
        events.append(event)
        if len(events_of(events, TurnEndEvent)) == turns:
            turns_closed.set()

    driver = create_claude_driver(client_factory=FactoryProbe(client))
    session = driver.start(prompt or make_prompt(), observer, asyncio.Event())
    await wait_for_event_or_task(turns_closed, session.outcome)
    return session, events


async def _run_turns(messages: Sequence[object]) -> list[SessionEvent]:
    """Run a session until every scripted result closed its turn, end it, and return its events."""
    turns = sum(isinstance(message, ResultMessage) for message in messages)
    session, events = await _start_past_turns(FakeClient(messages), turns)
    await _settle(session)
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


async def test_start_when_two_results_with_rising_cost_does_record_each_results_cost():
    messages = [
        result_message(total_cost_usd=0.05),
        result_message(total_cost_usd=0.15),
    ]

    events = await _run_turns(messages)

    usage_updates = events_of(events, UsageUpdateEvent)
    assert [u.cost_usd for u in usage_updates] == [0.05, 0.15, 0.15]
    turn_ends = events_of(events, TurnEndEvent)
    assert [turn_end.cost_usd for turn_end in turn_ends] == [0.05, 0.15]


@pytest.mark.parametrize(
    "bogus_cost",
    [
        pytest.param(math.nan, id="nan"),
    ],
)
async def test_start_when_result_cost_is_unusable_does_keep_running_cost(bogus_cost: float):
    messages = [
        result_message(total_cost_usd=0.05),
        result_message(total_cost_usd=bogus_cost),
    ]

    events = await _run_turns(messages)

    usage_costs = {update.cost_usd for update in events_of(events, UsageUpdateEvent)}
    turn_end_costs = [turn_end.cost_usd for turn_end in events_of(events, TurnEndEvent)]
    assert usage_costs == {0.05}
    assert turn_end_costs == [0.05, 0.05]


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
    session, events = await _start_past_turns(FakeClient([result]), turns=1)

    outcome = await _settle(session)

    turn_ends = events_of(events, TurnEndEvent)
    assert [(t.budget_exhausted, t.cost_usd) for t in turn_ends] == [(True, 1.50)]
    assert outcome.reason == "completed"


# ---------------------------------------------------------------------------
# send(text) forwards to client.query on the same connection until the session settles
# ---------------------------------------------------------------------------


async def test_send_when_called_does_forward_text_to_client_query():
    client = FakeClient([result_message(total_cost_usd=0.01)])
    session, _ = await _start_past_turns(client, turns=1, prompt=make_prompt(kickoff="initial"))

    await session.send("follow up message")
    await _settle(session)

    assert client.query_prompts == ["initial", "follow up message"]


async def test_send_when_session_settled_does_noop():
    result = result_message(subtype="error", is_error=True, result="fatal")
    client = FakeClient([result])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    session = driver.start(make_prompt(), noop_observer(), asyncio.Event())
    outcome = await _outcome(session)
    queries_before = list(client.query_prompts)

    await session.send("should be ignored")

    assert outcome.reason == "error"
    assert client.query_prompts == queries_before


# ---------------------------------------------------------------------------
# end() settles the session completed unless it already settled or was interrupted
# ---------------------------------------------------------------------------


async def test_end_when_called_does_settle_completed_at_the_running_cost():
    session, events = await _start_past_turns(
        FakeClient([result_message(total_cost_usd=0.20)]), turns=1
    )

    outcome = await _settle(session)

    assert (outcome.reason, outcome.cost_usd) == ("completed", 0.20)
    assert [u.cost_usd for u in events_of(events, UsageUpdateEvent)] == [0.20, 0.20]


async def test_end_when_called_after_interrupt_does_preserve_interrupted():
    client = FakeClient([result_message(total_cost_usd=0.10)])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer, asyncio.Event())
    await session.interrupt()
    outcome = await _settle(session)

    assert outcome.reason == "interrupted"
    assert events_of(probe.events, UsageUpdateEvent) == []


async def test_end_when_called_on_settled_session_does_noop():
    result = result_message(subtype="error", is_error=True, result="fatal", total_cost_usd=0.10)
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    session = driver.start(make_prompt(), probe.observer, asyncio.Event())
    await _outcome(session)
    events_before_end = list(probe.events)

    await session.end()

    assert probe.events == events_before_end


# ---------------------------------------------------------------------------
# Natural stream termination
# ---------------------------------------------------------------------------


async def test_start_when_stream_ends_without_a_turn_end_does_settle_error():
    client = FiniteClient([assistant(TextBlock(text="hello"))])

    outcome = await run_outcome(client)

    assert outcome.reason == "error"
    assert outcome.message == "Agent stream ended without a result message"
