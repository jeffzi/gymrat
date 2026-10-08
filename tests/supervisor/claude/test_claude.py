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
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from operator import methodcaller
from typing import override

import pytest
from claude_agent_sdk import (
    ConversationResetMessage,
    HookMatcher,
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
from claude_agent_sdk.types import HookEvent

from gymrat.supervisor.claude import create_claude_driver, usable_cost
from gymrat.supervisor.driver import Driver, DriverSession, SessionPrompt
from gymrat.supervisor.events import (
    CompactionEvent,
    SessionEvent,
    TextDeltaEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    UsageUpdateEvent,
    summarize,
)
from gymrat.supervisor.hooks import HooksFactory
from gymrat.supervisor.tools import ToolsFactory
from tests._imports import loaded_under, modules_loaded_after
from tests.supervisor._fixtures import (
    _SENTINEL_HOOKS,
    _SENTINEL_SERVER,
    FactoryProbe,
    FakeClient,
    FiniteClient,
    HooksFactoryProbe,
    ToolsFactoryProbe,
    assistant,
    collecting_observer,
    events_of,
    make_prompt,
    result_message,
    run_interrupting_on_first_usage_update,
    run_outcome,
    run_session,
    run_with_messages,
    settled_outcome,
    start_claude_session,
    system_message,
    tool_results,
    wait_for_event_or_task,
)

# ---------------------------------------------------------------------------
# construction and lazy loading
# ---------------------------------------------------------------------------


def test_create_claude_driver_when_constructed_does_not_import_sdk():
    loaded = modules_loaded_after(
        "from gymrat.supervisor.claude import create_claude_driver\ncreate_claude_driver()"
    )

    assert loaded_under(loaded, "claude_agent_sdk") == []


# ---------------------------------------------------------------------------
# start — options forwarding
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    ("prompt", "hooks", "tools", "added"),
    [
        pytest.param(make_prompt(), None, None, {}, id="defaults"),
        pytest.param(make_prompt(cwd="/my/project"), None, None, {"cwd": "/my/project"}, id="cwd"),
        pytest.param(
            make_prompt(system_prompt_append="extra instructions"),
            None,
            None,
            {
                "system_prompt": {
                    "type": "preset",
                    "preset": "claude_code",
                    "append": "extra instructions",
                }
            },
            id="system-prompt-append",
        ),
        pytest.param(
            make_prompt(command_timeout_ms=300000),
            None,
            None,
            {
                "env": {
                    "CLAUDE_CODE_DEFAULT_TOOL_USE_TIMEOUT_MS": "300000",
                    "CLAUDE_CODE_MAX_TOOL_USE_TIMEOUT_MS": "300000",
                    "CLAUDE_CODE_AUTO_BACKGROUND_TIMEOUT_MS": "",
                    "MCP_TOOL_TIMEOUT": "300000",
                }
            },
            id="command-timeout",
        ),
        pytest.param(
            make_prompt(model="claude-sonnet-4-20250514"),
            None,
            None,
            {"model": "claude-sonnet-4-20250514"},
            id="model",
        ),
        pytest.param(make_prompt(effort="high"), None, None, {"effort": "high"}, id="effort"),
        pytest.param(
            make_prompt(max_budget_usd=5.0), None, None, {"max_budget_usd": 5.0}, id="max-budget"
        ),
        pytest.param(
            make_prompt(traceparent=_TRACEPARENT),
            None,
            None,
            {"env": {**_DEFAULT_ENV, "GYMRAT_TRACEPARENT": _TRACEPARENT}},
            id="traceparent-under-its-own-name",
        ),
        pytest.param(
            make_prompt(), HooksFactoryProbe(), None, {"hooks": _SENTINEL_HOOKS}, id="hooks-only"
        ),
        pytest.param(
            make_prompt(),
            None,
            ToolsFactoryProbe(),
            {"mcp_servers": {"gymrat": _SENTINEL_SERVER}},
            id="tools-only",
        ),
        pytest.param(
            make_prompt(),
            HooksFactoryProbe(),
            ToolsFactoryProbe(),
            {"hooks": _SENTINEL_HOOKS, "mcp_servers": {"gymrat": _SENTINEL_SERVER}},
            id="hooks-and-tools",
        ),
    ],
)
async def test_start_when_session_inputs_given_does_forward_them_as_client_options(
    prompt: SessionPrompt,
    hooks: HooksFactory | None,
    tools: ToolsFactory | None,
    added: dict[str, object],
):
    client = FiniteClient([result_message()])

    await run_outcome(client, prompt=prompt, hooks=hooks, tools=tools)

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
    client = FiniteClient([assistant(tool_use), result_message()])
    probe = collecting_observer()

    await run_outcome(client, probe.observer, prompt=make_prompt(cwd="/my/project"))

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
    async def receive_messages(self) -> AsyncIterator[object]:
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

    await run_outcome(client, probe.observer)

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
    ("block", "parent", "expected_input_summary"),
    [
        pytest.param(
            ToolUseBlock(id="tu_1", name="Read", input={"file_path": "/foo.ts"}),
            "tu_parent",
            "/foo.ts",
            id="client-tool-under-a-parent",
        ),
        pytest.param(
            ServerToolUseBlock(id="tu_web", name="web_search", input=_WEB_SEARCH_INPUT),
            None,
            '{"query":"benchmark noise"}',
            id="server-tool",
        ),
    ],
)
async def test_start_when_tool_use_block_does_emit_tool_start(
    block: ToolUseBlock | ServerToolUseBlock,
    parent: str | None,
    expected_input_summary: str,
):
    events = await run_with_messages([assistant(block, parent_tool_use_id=parent)])

    starts = events_of(events, ToolStartEvent)
    assert len(starts) == 1
    assert starts[0].parent_tool_use_id == parent
    assert starts[0].tool_use_id == block.id
    assert starts[0].tool_name == block.name
    assert starts[0].input == block.input
    assert starts[0].input_summary == expected_input_summary


@pytest.mark.parametrize(
    ("messages", "parent", "tool_use_id", "tool_name", "expected_result"),
    [
        pytest.param(
            [
                assistant(
                    ToolUseBlock(id="tu_1", name="Read", input={"file_path": "/foo.ts"}),
                    parent_tool_use_id="tu_parent",
                ),
                tool_results(
                    ToolResultBlock(tool_use_id="tu_1", content="file contents here"),
                    parent_tool_use_id="tu_parent",
                ),
            ],
            "tu_parent",
            "tu_1",
            "Read",
            "file contents here",
            id="client-tool-under-a-parent",
        ),
        pytest.param(
            [
                assistant(
                    ServerToolUseBlock(id="tu_web", name="web_search", input=_WEB_SEARCH_INPUT),
                    ServerToolResultBlock(tool_use_id="tu_web", content=_WEB_SEARCH_RESULT),
                ),
            ],
            None,
            "tu_web",
            "web_search",
            json.dumps(_WEB_SEARCH_RESULT),
            id="server-tool",
        ),
    ],
)
async def test_start_when_tool_result_matches_start_does_emit_tool_end_with_tracked_name(
    messages: list[object],
    parent: str | None,
    tool_use_id: str,
    tool_name: str,
    expected_result: str,
):
    events = await run_with_messages(messages)

    ends = events_of(events, ToolEndEvent)
    assert len(ends) == 1
    assert ends[0].parent_tool_use_id == parent
    assert ends[0].tool_use_id == tool_use_id
    assert ends[0].tool_name == tool_name
    assert ends[0].result == expected_result
    assert ends[0].result_summary == summarize(expected_result)


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
    events = await run_with_messages([message])

    assert events == []


# ---------------------------------------------------------------------------
# cost ordering and interrupt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "next_message",
    [
        pytest.param(assistant(TextBlock(text="late")), id="later-text"),
        pytest.param(result_message(total_cost_usd=0.25), id="later-result-at-a-higher-cost"),
    ],
)
async def test_interrupt_when_scheduled_on_usage_update_does_soft_stop_at_the_crossing_cost(
    next_message: object,
):
    client = FakeClient([result_message(total_cost_usd=0.15), next_message])

    outcome = await run_interrupting_on_first_usage_update(client)

    assert (outcome.reason, outcome.cost_usd) == ("interrupted", 0.15)
    assert client.interrupt_called is True


async def test_interrupt_when_called_between_messages_does_stop_before_next_message():
    gate = asyncio.Event()
    first_seen = asyncio.Event()

    class GatedClient(FakeClient):
        @override
        async def receive_messages(self) -> AsyncIterator[object]:
            yield assistant(TextBlock(text="first"))
            first_seen.set()
            await gate.wait()
            yield assistant(TextBlock(text="second"))

    client = GatedClient([])
    probe = collecting_observer()
    session = start_claude_session(client, probe.observer)
    await wait_for_event_or_task(first_seen, session.outcome)

    await session.interrupt()
    gate.set()
    outcome = await settled_outcome(session)

    assert outcome.reason == "interrupted"
    assert [e.chunk for e in events_of(probe.events, TextDeltaEvent)] == ["first"]


# ---------------------------------------------------------------------------
# abort
# ---------------------------------------------------------------------------


async def test_start_when_abort_fires_after_another_stop_does_keep_the_first_outcome():
    client = FakeClient([result_message(total_cost_usd=0.1)])
    abort = asyncio.Event()
    sessions: list[DriverSession] = []
    ends: list[asyncio.Task[None]] = []

    def end_then_abort(event: SessionEvent) -> None:
        if isinstance(event, TurnEndEvent) and not ends:
            ends.append(asyncio.ensure_future(sessions[0].end()))
            abort.set()

    sessions.append(start_claude_session(client, end_then_abort, abort=abort))
    outcome = await settled_outcome(sessions[0])
    await asyncio.gather(*ends)

    assert (outcome.reason, outcome.cost_usd) == ("completed", 0.1)


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

    await run_outcome(FiniteClient(messages), probe.observer)

    assert [event.at for event in events_of(probe.events, CompactionEvent)] == expected_compactions


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


def _failing_client_factory(_options: Mapping[str, object]) -> FakeClient:
    message = "bad options"
    raise ValueError(message)


def _failing_hooks() -> dict[HookEvent, list[HookMatcher]]:
    message = "hooks unavailable"
    raise RuntimeError(message)


def _client_factory_raises() -> Driver:
    return create_claude_driver(client_factory=_failing_client_factory)


def _hooks_factory_raises() -> Driver:
    client = FiniteClient([result_message()])
    return create_claude_driver(client_factory=FactoryProbe(client), hooks=_failing_hooks)


def _stream_raises() -> Driver:
    client = FakeClient([result_message(total_cost_usd=0.04)], throw=RuntimeError("SDK failure"))
    return create_claude_driver(client_factory=FactoryProbe(client))


@pytest.mark.parametrize(
    ("make_driver", "expected_message", "expected_cost"),
    [
        pytest.param(_client_factory_raises, "bad options", 0.0, id="client-factory-raises"),
        pytest.param(_hooks_factory_raises, "hooks unavailable", 0.0, id="hooks-factory-raises"),
        pytest.param(_stream_raises, "SDK failure", 0.04, id="stream-raises"),
    ],
)
async def test_start_when_building_or_streaming_the_client_raises_does_resolve_error_with_its_message(
    make_driver: Callable[[], Driver], expected_message: str, expected_cost: float
):
    driver = make_driver()

    outcome = await run_session(driver, collecting_observer().observer)

    assert (outcome.reason, outcome.message, outcome.cost_usd) == (
        "error",
        expected_message,
        expected_cost,
    )


# ---------------------------------------------------------------------------
# start — interrupted before client connects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stop_again",
    [
        pytest.param(methodcaller("end"), id="then-end"),
        pytest.param(methodcaller("interrupt"), id="then-interrupt"),
    ],
)
async def test_start_when_interrupted_before_connect_does_resolve_once_without_sending_kickoff(
    stop_again: Callable[[DriverSession], Awaitable[None]],
):
    client = FakeClient([result_message(total_cost_usd=0.10)])
    probe = collecting_observer()
    session = start_claude_session(
        client, probe.observer, prompt=make_prompt(kickoff="should not be sent")
    )

    await session.interrupt()  # client not built yet — the soft stop cannot reach it
    await stop_again(session)  # already stopped — a no-op that keeps the first outcome
    outcome = await settled_outcome(session)

    assert (outcome.reason, outcome.cost_usd) == ("interrupted", 0.0)
    assert client.interrupt_called is False
    assert client.query_prompts == []
    assert events_of(probe.events, UsageUpdateEvent) == []


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
