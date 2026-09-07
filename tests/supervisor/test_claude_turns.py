"""Tests for the Claude driver turn protocol.

A result message emits a ``TurnEndEvent`` and the stream continues.
Session settlement happens via ``end()``, ``interrupt()``, or natural stream
termination — never from a result message alone.
"""

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import override

from gymrat.supervisor import create_claude_driver
from gymrat.supervisor.driver import Driver, SessionOutcome, SessionPrompt
from gymrat.supervisor.events import (
    SessionEvent,
    SessionObserver,
    TurnEndEvent,
    UsageUpdateEvent,
)
from tests.supervisor._fixtures import (
    FactoryProbe,
    FakeClient,
    FiniteClient,
    collecting_observer,
    make_prompt,
    noop_observer,
    result_message,
)

_TEST_TIMEOUT_S = 5.0

# ---------------------------------------------------------------------------
# test-local client variants
# ---------------------------------------------------------------------------


class QueryRaisingClient(FakeClient):
    """A client whose ``query`` raises on the *n*-th call."""

    def __init__(
        self,
        messages: Sequence[object],
        *,
        raise_on_query: int = 1,
        error: Exception | None = None,
    ) -> None:
        super().__init__(messages)
        self._raise_on = raise_on_query
        self._query_count = 0
        self._error = error or RuntimeError("query failed")

    @override
    async def query(self, prompt: str) -> None:
        self._query_count += 1
        if self._query_count >= self._raise_on:
            raise self._error


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


async def run_session(
    driver: Driver,
    observer: SessionObserver,
    prompt: SessionPrompt | None = None,
    abort: asyncio.Event | None = None,
    *,
    max_wait: float = 30.0,
) -> SessionOutcome:
    """Start a session and await its settled outcome."""
    session = driver.start(prompt or make_prompt(), observer, abort)
    return await asyncio.wait_for(session.outcome, max_wait)


async def run_outcome(
    client: FakeClient, observer: SessionObserver | None = None
) -> SessionOutcome:
    """Drive ``client`` through a session and return its settled outcome."""
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    return await run_session(driver, observer or noop_observer())


def events_of[T: SessionEvent](events: Sequence[SessionEvent], event_type: type[T]) -> list[T]:
    return [e for e in events if isinstance(e, event_type)]


async def _drain(n: int = 4) -> None:
    """Let the event loop process n pending callbacks."""
    for _ in range(n):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# behavior 1: result message emits TurnEndEvent, does not end stream
# ---------------------------------------------------------------------------


async def test_turn_when_result_message_received_does_emit_turn_end_event():
    """A result message produces a TurnEndEvent with its text."""
    result = result_message(total_cost_usd=0.05)
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)

    # Give messages time to process, then end the session
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert len(turn_ends) == 1


async def test_turn_when_result_has_top_level_text_does_carry_text_on_turn_end():
    """TurnEndEvent.text is the last top-level text block of the turn."""
    messages = [
        SimpleNamespace(content=[SimpleNamespace(text="first answer")]),
        result_message(total_cost_usd=0.01),
    ]
    client = FakeClient(messages)
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain(5)
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert len(turn_ends) == 1
    assert turn_ends[0].text == "first answer"


async def test_turn_when_result_has_no_text_does_carry_empty_string():
    """TurnEndEvent.text is '' when the turn had no top-level text blocks."""
    result = result_message(total_cost_usd=0.01)
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert len(turn_ends) == 1
    assert turn_ends[0].text == ""


async def test_turn_when_second_turn_does_reset_text_accumulator():
    """Each turn's TurnEndEvent carries only that turn's text."""
    messages = [
        SimpleNamespace(content=[SimpleNamespace(text="turn one text")]),
        result_message(total_cost_usd=0.01),
        SimpleNamespace(content=[SimpleNamespace(text="turn two text")]),
        result_message(total_cost_usd=0.02),
    ]
    client = FakeClient(messages)
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain(7)
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert len(turn_ends) == 2
    assert turn_ends[0].text == "turn one text"
    assert turn_ends[1].text == "turn two text"


async def test_turn_when_subagent_text_present_does_ignore_subagent_text_in_turn_end():
    """Only top-level text (parent_tool_use_id is None) counts for TurnEndEvent.text."""
    top_level_msg = SimpleNamespace(content=[SimpleNamespace(text="top level")])
    subagent_msg = SimpleNamespace(
        content=[SimpleNamespace(text="subagent output")],
        parent_tool_use_id="tu_sub",
    )
    messages = [top_level_msg, subagent_msg, result_message(total_cost_usd=0.01)]
    client = FakeClient(messages)
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain(6)
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert len(turn_ends) == 1
    assert turn_ends[0].text == "top level"


# ---------------------------------------------------------------------------
# behavior 2: origin detection
# ---------------------------------------------------------------------------


async def test_turn_when_origin_absent_does_report_agent():
    """No origin attribute → origin is 'agent'."""
    result = result_message(total_cost_usd=0.01)
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert len(turn_ends) == 1
    assert turn_ends[0].origin == "agent"


async def test_turn_when_origin_none_does_report_agent():
    """Explicit origin=None → origin is 'agent'."""
    result = result_message(total_cost_usd=0.01)
    result.origin = None
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert turn_ends[0].origin == "agent"


async def test_turn_when_origin_kind_human_does_report_agent():
    """origin.kind == 'human' → origin is 'agent' (human-initiated turn)."""
    result = result_message(total_cost_usd=0.01)
    result.origin = SimpleNamespace(kind="human")
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert turn_ends[0].origin == "agent"


async def test_turn_when_origin_kind_task_notification_does_report_injected():
    """origin.kind != 'human' → origin is 'injected'."""
    result = result_message(total_cost_usd=0.01)
    result.origin = SimpleNamespace(kind="task-notification")
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert turn_ends[0].origin == "injected"


async def test_turn_when_origin_kind_via_mapping_does_detect_correctly():
    """Origin as a dict with kind='human' → 'agent'."""
    result = result_message(total_cost_usd=0.01)
    result.origin = {"kind": "human"}
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert turn_ends[0].origin == "agent"


# ---------------------------------------------------------------------------
# behavior 3: cost tracking across turns
# ---------------------------------------------------------------------------


async def test_turn_when_result_has_cost_does_emit_unsettled_usage_update():
    """A result's total_cost_usd produces a UsageUpdateEvent with settled=False."""
    result = result_message(total_cost_usd=0.10)
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    usage_updates = events_of(probe.events, UsageUpdateEvent)
    # Should have at least an unsettled one from the result, plus a settled one from end()
    unsettled = [u for u in usage_updates if not u.settled]
    assert len(unsettled) >= 1
    assert unsettled[0].cost_usd == 0.10


async def test_turn_when_result_has_cost_does_carry_cost_on_turn_end():
    """TurnEndEvent.cost_usd matches the result's total_cost_usd."""
    result = result_message(total_cost_usd=0.10)
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert turn_ends[0].cost_usd == 0.10


async def test_turn_when_two_results_with_rising_cost_does_emit_two_unsettled_updates_and_turn_ends():
    """Two turn results with rising cost produce two unsettled usage updates and two turn ends."""
    messages = [
        result_message(total_cost_usd=0.05),
        result_message(total_cost_usd=0.15),
    ]
    client = FakeClient(messages)
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain(5)
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    usage_updates = events_of(probe.events, UsageUpdateEvent)
    unsettled = [u for u in usage_updates if not u.settled]
    assert len(unsettled) == 2
    assert unsettled[0].cost_usd == 0.05
    assert unsettled[1].cost_usd == 0.15

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert len(turn_ends) == 2
    assert turn_ends[0].cost_usd == 0.05
    assert turn_ends[1].cost_usd == 0.15


# ---------------------------------------------------------------------------
# behavior 4: is_error handling and budget exhaustion
# ---------------------------------------------------------------------------


async def test_turn_when_error_result_with_regular_subtype_does_settle_error_immediately():
    """A result with is_error=True and subtype != 'error_max_budget_usd' settles error."""
    result = result_message(subtype="error", is_error=True, result="something went wrong")
    client = FakeClient([result])

    outcome = await run_outcome(client)

    assert outcome.reason == "error"
    assert outcome.message == "something went wrong"


async def test_turn_when_error_result_without_text_does_use_subtype_as_message():
    """Error result with no result text falls back to the subtype string."""
    result = result_message(subtype="error", is_error=True, result=None)
    client = FakeClient([result])

    outcome = await run_outcome(client)

    assert outcome.reason == "error"
    assert outcome.message == "error"


async def test_turn_when_budget_exhausted_does_emit_turn_end_with_flag():
    """error_max_budget_usd is a turn end with budget_exhausted=True, not an error."""
    result = result_message(
        subtype="error_max_budget_usd",
        is_error=True,
        total_cost_usd=1.50,
        result="Budget exceeded",
    )
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = events_of(probe.events, TurnEndEvent)
    assert len(turn_ends) == 1
    assert turn_ends[0].budget_exhausted is True
    assert turn_ends[0].cost_usd == 1.50


async def test_turn_when_budget_exhausted_does_not_settle_error():
    """Budget exhaustion does not settle the session as error."""
    result = result_message(
        subtype="error_max_budget_usd",
        is_error=True,
        total_cost_usd=1.50,
    )
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    # end() settles as completed, not error
    assert outcome.reason == "completed"


# ---------------------------------------------------------------------------
# behavior 5: send(text) calls client.query(text) on the same connection
# ---------------------------------------------------------------------------


async def test_send_when_called_does_forward_text_to_client_query():
    """send(text) calls client.query(text) on the same connection."""
    result = result_message(total_cost_usd=0.01)
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(kickoff="initial"), probe.observer)
    await _drain()

    await session.send("follow up message")
    await asyncio.sleep(0)

    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert len(client.query_prompts) == 2
    assert client.query_prompts[0] == "initial"
    assert client.query_prompts[1] == "follow up message"


async def test_send_when_query_raises_does_settle_error():
    """If client.query raises, the session settles as error."""
    client = QueryRaisingClient(
        [result_message(total_cost_usd=0.01)],
        raise_on_query=2,
        error=RuntimeError("connection lost"),
    )
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(kickoff="initial"), probe.observer)
    await _drain()

    await session.send("this will fail")
    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert outcome.reason == "error"
    assert outcome.message == "connection lost"


async def test_send_when_session_settled_does_noop():
    """send() on a settled session is a no-op — no exception, no query."""
    result = result_message(subtype="error", is_error=True, result="fatal")
    client = FakeClient([result])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    # The session is settled. send should be a no-op.
    # We can't easily call send on the settled session through run_session,
    # so let's use the session handle directly.
    session = driver.start(make_prompt(), noop_observer())
    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    await session.send("should be ignored")
    assert outcome.reason == "error"


# ---------------------------------------------------------------------------
# behavior 6: end() claims completed
# ---------------------------------------------------------------------------


async def test_end_when_called_does_settle_completed():
    """end() claims completed with the last known cost."""
    result = result_message(total_cost_usd=0.20)
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert outcome.reason == "completed"
    assert outcome.cost_usd == 0.20


async def test_end_when_called_does_emit_settled_usage_update():
    """end() emits a UsageUpdateEvent with settled=True."""
    result = result_message(total_cost_usd=0.20)
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    usage_updates = events_of(probe.events, UsageUpdateEvent)
    settled = [u for u in usage_updates if u.settled]
    assert len(settled) == 1
    assert settled[0].cost_usd == 0.20


async def test_end_when_called_does_disconnect_client():
    """end() disconnects the client."""
    result = result_message(total_cost_usd=0.01)
    client = FakeClient([result])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), noop_observer())
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert client.disconnect_count >= 1


async def test_end_when_called_after_interrupt_does_preserve_interrupted():
    """end() after interrupt() leaves the outcome as interrupted — no settled usage update."""
    client = FakeClient([result_message(total_cost_usd=0.10)])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    await session.interrupt()
    await session.end()
    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert outcome.reason == "interrupted"

    usage_updates = events_of(probe.events, UsageUpdateEvent)
    settled = [u for u in usage_updates if u.settled]
    assert settled == []


async def test_end_when_called_on_settled_session_does_noop():
    """end() on a settled session is a no-op."""
    result = result_message(subtype="error", is_error=True, result="fatal")
    client = FakeClient([result])
    probe = collecting_observer()
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), probe.observer)
    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    assert outcome.reason == "error"

    # end() after settlement should do nothing
    await session.end()

    # No settled usage update should have been emitted
    usage_updates = events_of(probe.events, UsageUpdateEvent)
    settled = [u for u in usage_updates if u.settled]
    assert settled == []


# ---------------------------------------------------------------------------
# behavior 7: natural stream termination
# ---------------------------------------------------------------------------


async def test_stream_end_when_after_turn_end_does_settle_completed():
    """A stream that ends naturally after at least one TurnEndEvent settles completed."""
    result = result_message(total_cost_usd=0.05)
    client = FiniteClient([result])

    outcome = await run_outcome(client)

    assert outcome.reason == "completed"
    assert outcome.cost_usd == 0.05


async def test_stream_end_when_no_turn_end_does_settle_error():
    """A stream that ends with no turn end settles error."""
    client = FiniteClient([SimpleNamespace(content=[SimpleNamespace(text="hello")])])

    outcome = await run_outcome(client)

    assert outcome.reason == "error"
    assert outcome.message is not None
    assert "result" in outcome.message.lower()


# ---------------------------------------------------------------------------
# behavior 8: interrupt and abort (confirm existing behavior preserved)
# ---------------------------------------------------------------------------


async def test_interrupt_when_called_does_settle_interrupted():
    """Interrupt settles as interrupted (same as before turn protocol)."""
    client = FakeClient([])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), noop_observer())
    await session.interrupt()
    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert outcome.reason == "interrupted"


async def test_interrupt_when_called_before_end_does_win():
    """Interrupt first-wins over a later end()."""
    client = FakeClient([result_message(total_cost_usd=0.10)])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), noop_observer())
    await session.interrupt()
    await session.end()
    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert outcome.reason == "interrupted"


# ---------------------------------------------------------------------------
# behavior 9: _build_options forwards max_budget_usd
# ---------------------------------------------------------------------------


async def test_options_when_max_budget_usd_set_does_include_max_budget_usd():
    """max_budget_usd from the prompt is forwarded in the client options."""
    client = FakeClient([result_message(total_cost_usd=0.01)])
    probe = FactoryProbe(client)
    driver = create_claude_driver(client_factory=probe)

    session = driver.start(make_prompt(max_budget_usd=5.0), noop_observer())
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert client.options is not None
    assert client.options["max_budget_usd"] == 5.0


async def test_options_when_max_budget_usd_absent_does_omit_key():
    """When max_budget_usd is not set, the key is absent from options."""
    client = FakeClient([result_message(total_cost_usd=0.01)])
    probe = FactoryProbe(client)
    driver = create_claude_driver(client_factory=probe)

    session = driver.start(make_prompt(), noop_observer())
    await _drain()
    await session.end()
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert client.options is not None
    assert "max_budget_usd" not in client.options
