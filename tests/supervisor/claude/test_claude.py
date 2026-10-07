"""Behavioral tests for the Claude Agent SDK driver.

The driver is exercised entirely through an injected fake client, so these
tests never start the real ``claude-agent-sdk`` client. The fake mimics the
streaming client surface the driver relies on (``connect``/``query``/
``receive_messages``/``interrupt``/``disconnect``) and yields the SDK's own
message and content-block dataclasses. The cost rule every driver applies to
a reported cost is pinned here too.
"""

import asyncio
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import override

import pytest
from claude_agent_sdk import (
    ConversationResetMessage,
    RateLimitEvent,
    RateLimitInfo,
    ServerToolResultBlock,
    ServerToolUseBlock,
    TaskStartedMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from gymrat.supervisor.claude import create_claude_driver, usable_cost
from gymrat.supervisor.driver import DriverSession, SessionOutcome, SessionPrompt
from gymrat.supervisor.events import (
    CompactionEvent,
    SessionEvent,
    SessionObserver,
    TextDeltaEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    summarize,
)
from tests._imports import loaded_under
from tests.supervisor._fixtures import (
    FactoryProbe,
    FakeClient,
    FiniteClient,
    assistant,
    collecting_observer,
    events_of,
    make_prompt,
    result_message,
    run_outcome,
    run_session,
    run_with_messages,
    system_message,
    tool_results,
    wait_for_event_or_task,
)

# ---------------------------------------------------------------------------
# construction and lazy loading
# ---------------------------------------------------------------------------


def test_create_claude_driver_when_given_factory_does_return_driver_without_calling_it():
    probe = FactoryProbe(FakeClient([]))

    driver = create_claude_driver(client_factory=probe)

    assert callable(driver.start)
    assert probe.calls == 0


#: Build the default driver in a fresh interpreter, then report every loaded module.
_CONSTRUCT_PROBE = (
    "import json, sys\n"
    "from gymrat.supervisor.claude import create_claude_driver\n"
    "create_claude_driver()\n"
    "json.dump(sorted(sys.modules), sys.stdout)"
)


def test_create_claude_driver_when_constructed_does_not_import_sdk():
    probe = subprocess.run(  # noqa: S603 -- fixed argv, interpreter is sys.executable
        [sys.executable, "-c", _CONSTRUCT_PROBE],
        capture_output=True,
        text=True,
        check=True,
    )

    assert loaded_under(frozenset(json.loads(probe.stdout)), "claude_agent_sdk") == []


# ---------------------------------------------------------------------------
# start — options forwarding
# ---------------------------------------------------------------------------


async def _start_with_prompt(prompt: SessionPrompt) -> FiniteClient:
    """Start a session with ``prompt`` and return the client it drove."""
    client = FiniteClient([result_message()])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    await run_session(driver, collecting_observer().observer, prompt)
    return client


#: The timeout variables every session hands the agent, at ``make_prompt``'s 60 s default.
_DEFAULT_ENV = {
    "CLAUDE_CODE_DEFAULT_TOOL_USE_TIMEOUT_MS": "60000",
    "CLAUDE_CODE_MAX_TOOL_USE_TIMEOUT_MS": "60000",
    "CLAUDE_CODE_AUTO_BACKGROUND_TIMEOUT_MS": "",
    "MCP_TOOL_TIMEOUT": "60000",
}

#: The client options ``make_prompt``'s defaults produce, with no optional field given.
_DEFAULT_OPTIONS = {
    "cwd": "/tmp/test",
    "permission_mode": "bypassPermissions",
    "include_partial_messages": True,
    "system_prompt": {"type": "preset", "preset": "claude_code", "append": "follow the runbook"},
    "env": _DEFAULT_ENV,
}

_TRACEPARENT = "00-abc123-def456-01"


async def test_start_when_launched_does_forward_options_to_client():
    client = await _start_with_prompt(
        make_prompt(
            kickoff="hello agent",
            cwd="/my/project",
            system_prompt_append="extra instructions",
            command_timeout_ms=300000,
        )
    )

    assert client.options == {
        "cwd": "/my/project",
        "permission_mode": "bypassPermissions",
        "include_partial_messages": True,
        "system_prompt": {
            "type": "preset",
            "preset": "claude_code",
            "append": "extra instructions",
        },
        "env": {
            "CLAUDE_CODE_DEFAULT_TOOL_USE_TIMEOUT_MS": "300000",
            "CLAUDE_CODE_MAX_TOOL_USE_TIMEOUT_MS": "300000",
            "CLAUDE_CODE_AUTO_BACKGROUND_TIMEOUT_MS": "",
            "MCP_TOOL_TIMEOUT": "300000",
        },
    }
    assert client.query_prompts == ["hello agent"]


@pytest.mark.parametrize(
    ("prompt", "added"),
    [
        pytest.param(
            make_prompt(model="claude-sonnet-4-20250514"),
            {"model": "claude-sonnet-4-20250514"},
            id="model",
        ),
        pytest.param(make_prompt(effort="high"), {"effort": "high"}, id="effort"),
        pytest.param(make_prompt(max_budget_usd=5.0), {"max_budget_usd": 5.0}, id="max-budget"),
        pytest.param(
            make_prompt(traceparent=_TRACEPARENT),
            {"env": {**_DEFAULT_ENV, "GYMRAT_TRACEPARENT": _TRACEPARENT}},
            id="traceparent-under-its-own-name",
        ),
    ],
)
async def test_start_when_an_optional_field_given_does_add_only_that_option(
    prompt: SessionPrompt, added: dict[str, object]
):
    client = await _start_with_prompt(prompt)

    assert client.options == _DEFAULT_OPTIONS | added


# ---------------------------------------------------------------------------
# message mapping — positive cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "parent"),
    [
        pytest.param("hello world", None, id="top-level"),
        pytest.param("subagent output", "tu_parent", id="under-a-parent-tool-call"),
    ],
)
async def test_start_when_text_block_does_emit_text_delta_carrying_its_parent(
    text: str, parent: str | None
):
    events = await run_with_messages([assistant(TextBlock(text=text), parent_tool_use_id=parent)])

    deltas = events_of(events, TextDeltaEvent)
    assert [(delta.chunk, delta.parent_tool_use_id) for delta in deltas] == [(text, parent)]


async def test_start_when_read_path_under_cwd_does_summarize_relative_to_cwd():
    tool_use = ToolUseBlock(id="tu_1", name="Read", input={"file_path": "/my/project/src/main.py"})
    driver = create_claude_driver(
        client_factory=FactoryProbe(FiniteClient([assistant(tool_use), result_message()]))
    )
    probe = collecting_observer()

    await run_session(driver, probe.observer, make_prompt(cwd="/my/project"))

    starts = events_of(probe.events, ToolStartEvent)
    assert starts[0].input_summary == "src/main.py"


async def test_start_when_tool_result_has_no_matching_start_does_use_fallback_fields():
    orphan = tool_results(ToolResultBlock(tool_use_id="tu_orphan", content="result"))

    events = await run_with_messages([orphan])

    ends = events_of(events, ToolEndEvent)
    assert len(ends) == 1
    assert ends[0].tool_name == "unknown"
    assert ends[0].tool_use_id == "tu_orphan"
    assert ends[0].duration_ms == 0


@dataclass(slots=True)
class _FakeClocks:
    """Mutable readings for the duration clock and the wall clock."""

    monotonic_ms: float = 0.0
    wall_ns: int = 0


class _ClockedClient(FiniteClient):
    """A finite client that sets both fake clocks before yielding each message."""

    def __init__(self, steps: Sequence[tuple[float, int, object]], clocks: _FakeClocks) -> None:
        super().__init__([message for _, _, message in steps])
        self._steps = steps
        self._clocks = clocks

    @override
    async def receive_messages(self):
        for monotonic_ms, wall_ns, message in self._steps:
            self._clocks.monotonic_ms = monotonic_ms
            self._clocks.wall_ns = wall_ns
            await asyncio.sleep(0)
            yield message


async def test_start_when_wall_clock_jumps_back_does_report_monotonic_tool_duration(
    monkeypatch: pytest.MonkeyPatch,
):
    clocks = _FakeClocks()
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: clocks.monotonic_ms)
    monkeypatch.setattr("time.time_ns", lambda: clocks.wall_ns)
    monkeypatch.setattr("time.time", lambda: clocks.wall_ns / 1_000_000_000)
    one_hour_ns = 3_600 * 1_000_000_000
    start_wall_ns = 1_700_000_000 * 1_000_000_000
    client = _ClockedClient(
        [
            (
                1_000.0,
                start_wall_ns,
                assistant(ToolUseBlock(id="tu_1", name="Read", input={"file_path": "/a"})),
            ),
            (
                1_250.0,
                start_wall_ns - one_hour_ns,
                tool_results(ToolResultBlock(tool_use_id="tu_1", content="done")),
            ),
        ],
        clocks,
    )
    probe = collecting_observer()

    await run_session(create_claude_driver(client_factory=FactoryProbe(client)), probe.observer)

    ends = events_of(probe.events, ToolEndEvent)
    assert [end.duration_ms for end in ends] == [250]


_JSON_CONTENT = [{"type": "text", "text": "hi"}]
_OPAQUE_CONTENT = [{"type": "opaque", "value": object()}]


@pytest.mark.parametrize(
    ("content", "expected_result"),
    [
        pytest.param(_JSON_CONTENT, json.dumps(_JSON_CONTENT), id="list-json"),
        pytest.param(_OPAQUE_CONTENT, str(_OPAQUE_CONTENT), id="not-json-encodable-str-fallback"),
        pytest.param(None, "null", id="none-null"),
    ],
)
async def test_start_when_tool_result_content_not_a_string_does_encode_it_as_the_result(
    content: list[dict[str, object]] | None, expected_result: str
):
    messages = [
        assistant(ToolUseBlock(id="tu_3", name="Bash", input={"command": "echo hi"})),
        tool_results(ToolResultBlock(tool_use_id="tu_3", content=content)),
    ]

    events = await run_with_messages(messages)

    assert [end.result for end in events_of(events, ToolEndEvent)] == [expected_result]


# ---------------------------------------------------------------------------
# message mapping — tool-use and tool-result blocks (client and server)
# ---------------------------------------------------------------------------

_WEB_SEARCH_INPUT: dict[str, object] = {"query": "benchmark noise"}
_WEB_SEARCH_RESULT: dict[str, object] = {
    "type": "web_search_tool_result",
    "content": [
        {"type": "web_search_result", "url": "https://example.com/noise", "title": "Noise"}
    ],
}


@pytest.mark.parametrize(
    ("block", "expected_input_summary"),
    [
        pytest.param(
            ToolUseBlock(id="tu_1", name="Read", input={"file_path": "/foo.ts"}),
            "/foo.ts",
            id="client-tool",
        ),
        pytest.param(
            ServerToolUseBlock(id="tu_web", name="web_search", input=_WEB_SEARCH_INPUT),
            '{"query":"benchmark noise"}',
            id="server-tool",
        ),
    ],
)
async def test_start_when_tool_use_block_does_emit_tool_start(
    block: ToolUseBlock | ServerToolUseBlock,
    expected_input_summary: str,
):
    events = await run_with_messages([assistant(block)])

    starts = events_of(events, ToolStartEvent)
    assert len(starts) == 1
    assert starts[0].tool_use_id == block.id
    assert starts[0].tool_name == block.name
    assert starts[0].input == block.input
    assert starts[0].input_summary == expected_input_summary


@pytest.mark.parametrize(
    ("messages", "tool_use_id", "tool_name", "expected_result"),
    [
        pytest.param(
            [
                assistant(ToolUseBlock(id="tu_1", name="Read", input={"file_path": "/foo.ts"})),
                tool_results(ToolResultBlock(tool_use_id="tu_1", content="file contents here")),
            ],
            "tu_1",
            "Read",
            "file contents here",
            id="client-tool",
        ),
        pytest.param(
            [
                assistant(
                    ServerToolUseBlock(id="tu_web", name="web_search", input=_WEB_SEARCH_INPUT),
                    ServerToolResultBlock(tool_use_id="tu_web", content=_WEB_SEARCH_RESULT),
                ),
            ],
            "tu_web",
            "web_search",
            json.dumps(_WEB_SEARCH_RESULT),
            id="server-tool",
        ),
    ],
)
async def test_start_when_tool_result_matches_start_does_emit_tool_end_with_tracked_name(
    messages: list[object],
    tool_use_id: str,
    tool_name: str,
    expected_result: str,
):
    events = await run_with_messages(messages)

    ends = events_of(events, ToolEndEvent)
    assert len(ends) == 1
    assert ends[0].tool_use_id == tool_use_id
    assert ends[0].tool_name == tool_name
    assert ends[0].result == expected_result
    assert ends[0].result_summary == summarize(expected_result)
    assert ends[0].duration_ms >= 0


# ---------------------------------------------------------------------------
# tool events carry parent_tool_use_id
# ---------------------------------------------------------------------------


async def test_start_when_tool_result_with_parent_does_carry_parent_tool_use_id():
    messages = [
        assistant(
            ToolUseBlock(id="tu_1", name="Read", input={"file_path": "/foo.ts"}),
            parent_tool_use_id="tu_parent",
        ),
        tool_results(
            ToolResultBlock(tool_use_id="tu_1", content="file contents"),
            parent_tool_use_id="tu_parent",
        ),
    ]

    events = await run_with_messages(messages)

    starts = events_of(events, ToolStartEvent)
    ends = events_of(events, ToolEndEvent)
    assert [start.parent_tool_use_id for start in starts] == ["tu_parent"]
    assert [end.parent_tool_use_id for end in ends] == ["tu_parent"]


# ---------------------------------------------------------------------------
# message mapping — messages without session events
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(UserMessage(content="a typed prompt"), id="user-message-string-content"),
        pytest.param(assistant(), id="assistant-without-blocks"),
        pytest.param(
            assistant(ThinkingBlock(thinking="abcd", signature="sig")),
            id="complete-thinking-block",
        ),
        pytest.param(
            TaskStartedMessage(
                subtype="task_started",
                data={},
                task_id="task-1",
                description="explore",
                uuid="task-uuid",
                session_id="sdk-session",
            ),
            id="system-message-subclass",
        ),
        pytest.param(
            RateLimitEvent(
                rate_limit_info=RateLimitInfo(status="allowed"),
                uuid="rate-uuid",
                session_id="sdk-session",
            ),
            id="rate-limit-event",
        ),
        pytest.param(
            ConversationResetMessage(
                new_conversation_id="fresh", uuid="reset-uuid", session_id="sdk-session"
            ),
            id="conversation-reset",
        ),
        pytest.param(object(), id="non-sdk-object"),
    ],
)
async def test_start_when_message_carries_no_session_content_does_emit_nothing(
    message: object,
):
    assert await run_with_messages([message]) == []


# ---------------------------------------------------------------------------
# cost ordering and interrupt
# ---------------------------------------------------------------------------


def _interrupting_observer(
    events: list[SessionEvent], holder: dict[str, DriverSession], after: int = 1
) -> SessionObserver:
    """Observer that schedules ``interrupt`` after ``after`` usage updates."""
    seen = 0
    interrupts: list[asyncio.Task[None]] = []

    def observer(event: SessionEvent) -> None:
        nonlocal seen
        events.append(event)
        if isinstance(event, UsageUpdateEvent):
            seen += 1
            if seen == after:
                interrupts.append(asyncio.ensure_future(holder["session"].interrupt()))

    return observer


async def _run_interrupting_on_first_usage_update(
    messages: Sequence[object],
) -> tuple[SessionOutcome, FakeClient]:
    """Drive a session that schedules ``interrupt`` after the first usage update.

    Usage updates come from result messages, which leave the session idle
    between turns; the soft stop lands when the stream delivers its next
    message.

    Args:
        messages: The scripted SDK messages, which must carry one more message
            after the interrupting result.

    Returns:
        The settled outcome and the fake client the session drove.
    """
    client = FakeClient(messages)
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    events: list[SessionEvent] = []
    holder: dict[str, DriverSession] = {}

    holder["session"] = driver.start(
        make_prompt(), _interrupting_observer(events, holder), asyncio.Event()
    )
    outcome = await holder["session"].outcome
    return outcome, client


async def test_interrupt_when_scheduled_on_usage_update_does_soft_stop_at_the_crossing_cost():
    outcome, client = await _run_interrupting_on_first_usage_update([
        result_message(total_cost_usd=0.15),
        assistant(TextBlock(text="late")),
    ])

    assert (outcome.reason, outcome.cost_usd) == ("interrupted", 0.15)
    assert client.interrupt_called is True
    assert client.disconnect_count == 1  # only the finally teardown, never interrupt itself


async def test_interrupt_when_first_call_wins_does_ignore_later_higher_cost():
    outcome, _client = await _run_interrupting_on_first_usage_update([
        result_message(total_cost_usd=0.1),
        result_message(total_cost_usd=0.25),
    ])

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.1


async def test_interrupt_when_called_between_messages_does_stop_before_next_message():
    gate = asyncio.Event()
    first_seen = asyncio.Event()

    class GatedClient(FakeClient):
        @override
        async def receive_messages(self):
            yield assistant(TextBlock(text="first"))
            first_seen.set()
            await gate.wait()
            yield assistant(TextBlock(text="second"))

    client = GatedClient([])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    probe = collecting_observer()

    session = driver.start(make_prompt(), probe.observer, asyncio.Event())
    await wait_for_event_or_task(first_seen, session.outcome)
    await session.interrupt()
    gate.set()
    outcome = await session.outcome

    assert outcome.reason == "interrupted"
    assert [e.chunk for e in events_of(probe.events, TextDeltaEvent)] == ["first"]


# ---------------------------------------------------------------------------
# abort
# ---------------------------------------------------------------------------


async def test_start_when_abort_fired_after_interrupt_does_preserve_interrupt_cost():
    client = FakeClient([result_message(total_cost_usd=0.1), result_message(total_cost_usd=0.3)])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    abort = asyncio.Event()
    holder: dict[str, DriverSession] = {}
    interrupts: list[asyncio.Task[None]] = []

    def observer(event: SessionEvent) -> None:
        if isinstance(event, UsageUpdateEvent):
            interrupts.append(asyncio.ensure_future(holder["session"].interrupt()))
            abort.set()

    holder["session"] = driver.start(make_prompt(), observer, abort)
    outcome = await holder["session"].outcome

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.1


async def test_start_when_abort_already_set_at_start_does_resolve_interrupted_without_client():
    probe = FactoryProbe(FakeClient([]))
    driver = create_claude_driver(client_factory=probe)
    abort = asyncio.Event()
    abort.set()

    outcome = await run_session(driver, collecting_observer().observer, abort=abort)

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.0
    assert probe.calls == 0


# ---------------------------------------------------------------------------
# result message — session settlement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result_text", "expected_message"),
    [
        pytest.param("something went wrong", "something went wrong", id="result-text-present"),
        pytest.param(None, "error", id="result-text-absent-uses-subtype"),
    ],
)
async def test_start_when_result_is_error_does_settle_error_as_the_final_result(
    result_text: str | None,
    expected_message: str,
):
    messages = [
        result_message(subtype="error", is_error=True, result=result_text, total_cost_usd=0.2)
    ]
    probe = collecting_observer()

    outcome = await run_outcome(FakeClient(messages), probe.observer)

    assert (outcome.reason, outcome.message, outcome.cost_usd) == ("error", expected_message, 0.2)
    assert [update.cost_usd for update in events_of(probe.events, UsageUpdateEvent)] == [0.2]
    assert events_of(probe.events, TurnEndEvent) == []


@pytest.mark.parametrize(
    ("cost", "expected_updates", "expected_cost"),
    [
        pytest.param(0.05, [0.05], 0.05, id="usable-cost"),
        pytest.param(None, [], 0.0, id="none-cost"),
        pytest.param(0.0, [], 0.0, id="zero-cost"),
    ],
)
async def test_start_when_stream_ends_after_a_turn_does_settle_completed_at_the_reported_cost(
    cost: float | None, expected_updates: list[float], expected_cost: float
):
    probe = collecting_observer()

    outcome = await run_outcome(FiniteClient([result_message(total_cost_usd=cost)]), probe.observer)

    assert (outcome.reason, outcome.cost_usd) == ("completed", expected_cost)
    assert [u.cost_usd for u in events_of(probe.events, UsageUpdateEvent)] == expected_updates


# ---------------------------------------------------------------------------
# system message — compact_boundary → CompactionEvent
# ---------------------------------------------------------------------------

_COMPACTED_AT_NS = 1_700_000_000_000_000_000


@pytest.mark.parametrize(
    ("subtype", "expected_compactions"),
    [
        pytest.param("init", [], id="init"),
        pytest.param("some_other_subtype", [], id="other-subtype"),
        pytest.param("compact_boundary", [_COMPACTED_AT_NS], id="compact-boundary"),
    ],
)
async def test_start_when_system_message_arrives_does_emit_compaction_only_on_compact_boundary(
    monkeypatch: pytest.MonkeyPatch, subtype: str, expected_compactions: list[int]
):
    monkeypatch.setattr("time.time_ns", lambda: _COMPACTED_AT_NS)
    messages = [system_message(subtype=subtype), result_message(total_cost_usd=0.05)]
    probe = collecting_observer()

    outcome = await run_outcome(FiniteClient(messages), probe.observer)

    assert outcome.reason == "completed"
    assert [event.at for event in events_of(probe.events, CompactionEvent)] == expected_compactions


# ---------------------------------------------------------------------------
# error — stream exception
# ---------------------------------------------------------------------------


async def test_start_when_stream_raises_does_resolve_error_without_raising():
    client = FakeClient([result_message(total_cost_usd=0.04)], throw=RuntimeError("SDK failure"))
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    outcome = await run_session(driver, collecting_observer().observer)

    assert outcome.reason == "error"
    assert outcome.message == "SDK failure"
    assert outcome.cost_usd == 0.04


# ---------------------------------------------------------------------------
# missing SDK
# ---------------------------------------------------------------------------


async def test_start_when_sdk_import_fails_does_resolve_error_naming_package(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    driver = create_claude_driver()

    outcome = await run_session(driver, collecting_observer().observer)

    assert outcome.reason == "error"
    assert outcome.message is not None
    assert "claude-agent-sdk" in outcome.message


# ---------------------------------------------------------------------------
# teardown robustness — outcome never raises
# ---------------------------------------------------------------------------


async def test_start_when_client_factory_raises_does_resolve_error_without_raising():
    error = ValueError("bad options")

    def failing_factory(options: Mapping[str, object]) -> FakeClient:
        raise error

    driver = create_claude_driver(client_factory=failing_factory)

    outcome = await run_session(driver, collecting_observer().observer)

    assert outcome.reason == "error"
    assert outcome.message == str(error)
    assert outcome.cost_usd == 0.0


# ---------------------------------------------------------------------------
# start — interrupted before client connects
# ---------------------------------------------------------------------------


async def test_start_when_interrupted_before_connect_does_resolve_once_without_sending_kickoff():
    client = FakeClient([])
    probe = FactoryProbe(client)
    driver = create_claude_driver(client_factory=probe)

    session = driver.start(
        make_prompt(kickoff="should not be sent"),
        collecting_observer().observer,
        asyncio.Event(),
    )
    await session.interrupt()  # client not built yet — the soft stop cannot reach it
    await session.interrupt()  # already stopped — a no-op that keeps the first outcome
    outcome = await session.outcome

    assert (outcome.reason, outcome.cost_usd) == ("interrupted", 0.0)
    assert client.interrupt_called is False
    assert client.query_prompts == []


# ---------------------------------------------------------------------------
# usable_cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        pytest.param(0.42, 0.42, id="positive-float"),
        pytest.param(3, 3.0, id="positive-int"),
        pytest.param(None, None, id="missing"),
        pytest.param("0.42", None, id="string"),
        pytest.param(True, None, id="true"),
        pytest.param(False, None, id="false"),
        pytest.param(math.nan, None, id="nan"),
        pytest.param(math.inf, None, id="positive-infinity"),
        pytest.param(-math.inf, None, id="negative-infinity"),
        pytest.param(0.0, None, id="zero"),
        pytest.param(-0.3, None, id="negative"),
    ],
)
def test_usable_cost_when_reported_does_accept_only_finite_positive_numbers(
    reported: object,
    expected: float | None,
):
    cost = usable_cost(reported)

    assert cost == expected
    assert type(cost) is type(expected)
