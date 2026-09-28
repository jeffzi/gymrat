"""Behavioral tests for the Claude Agent SDK driver.

The driver is exercised entirely through an injected fake client, so these
tests never start the real ``claude-agent-sdk`` client. The fake mimics the
streaming client surface the driver relies on (``connect``/``query``/
``receive_messages``/``interrupt``/``disconnect``) and yields the SDK's own
message and content-block dataclasses.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
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

from gymrat.supervisor import create_claude_driver
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
)

# ---------------------------------------------------------------------------
# test-local client variants
# ---------------------------------------------------------------------------


class Unserializable:
    """A tool-result value that JSON cannot encode, with a stable ``repr``."""

    @override
    def __repr__(self) -> str:
        return "UNSERIALIZABLE"


# ---------------------------------------------------------------------------
# construction and lazy loading
# ---------------------------------------------------------------------------


def test_create_claude_driver_when_given_factory_does_return_driver_without_calling_it():
    probe = FactoryProbe(FakeClient([]))

    driver = create_claude_driver(client_factory=probe)

    assert callable(driver.start)
    assert probe.calls == 0


def test_create_claude_driver_when_constructed_does_not_import_sdk(monkeypatch: pytest.MonkeyPatch):
    loaded = False

    def spy() -> object:
        nonlocal loaded
        loaded = True
        return object()

    monkeypatch.setattr("gymrat.supervisor.claude._load_default_factory", spy)

    create_claude_driver()

    assert loaded is False  # pyrefly: ignore[unnecessary-comparison] -- verifying spy was not called


# ---------------------------------------------------------------------------
# start — options forwarding
# ---------------------------------------------------------------------------


async def _start_with_prompt(prompt: SessionPrompt) -> FiniteClient:
    """Start a session with ``prompt`` and return the client it drove."""
    client = FiniteClient([result_message()])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    await run_session(driver, collecting_observer().observer, prompt)
    return client


async def test_start_when_launched_does_forward_options_to_client():
    client = await _start_with_prompt(make_prompt(kickoff="hello agent", cwd="/my/project"))

    assert client.options == {
        "cwd": "/my/project",
        "permission_mode": "bypassPermissions",
        "include_partial_messages": True,
    }
    assert client.query_prompts == ["hello agent"]


async def test_start_when_system_prompt_append_present_does_include_preset_append():
    client = await _start_with_prompt(make_prompt(system_prompt_append="extra instructions"))

    assert client.options is not None
    assert client.options["system_prompt"] == {
        "type": "preset",
        "preset": "claude_code",
        "append": "extra instructions",
    }


async def test_start_when_system_prompt_append_absent_does_omit_system_prompt():
    client = await _start_with_prompt(make_prompt())

    assert client.options is not None
    assert "system_prompt" not in client.options


async def test_start_when_model_given_does_include_model():
    client = await _start_with_prompt(make_prompt(model="claude-sonnet-4-20250514"))

    assert client.options is not None
    assert client.options["model"] == "claude-sonnet-4-20250514"


async def test_start_when_model_absent_does_omit_model():
    client = await _start_with_prompt(make_prompt())

    assert client.options is not None
    assert "model" not in client.options


async def test_start_when_effort_given_does_include_effort():
    client = await _start_with_prompt(make_prompt(effort="high"))

    assert client.options is not None
    assert client.options["effort"] == "high"


async def test_start_when_effort_absent_does_omit_effort():
    client = await _start_with_prompt(make_prompt())

    assert client.options is not None
    assert "effort" not in client.options


async def test_start_when_command_timeout_ms_given_does_set_timeout_env_vars():
    client = await _start_with_prompt(make_prompt(command_timeout_ms=300000))

    assert client.options is not None
    env = client.options["env"]
    assert isinstance(env, dict)
    assert env["CLAUDE_CODE_DEFAULT_TOOL_USE_TIMEOUT_MS"] == "300000"
    assert env["CLAUDE_CODE_MAX_TOOL_USE_TIMEOUT_MS"] == "300000"
    assert env["CLAUDE_CODE_AUTO_BACKGROUND_TIMEOUT_MS"] == ""
    assert env["MCP_TOOL_TIMEOUT"] == "300000"


async def test_start_when_command_timeout_ms_absent_does_omit_timeout_env_vars():
    client = await _start_with_prompt(make_prompt())

    assert client.options is not None
    env = client.options.get("env", {})
    assert isinstance(env, dict)
    assert "CLAUDE_CODE_DEFAULT_TOOL_USE_TIMEOUT_MS" not in env
    assert "CLAUDE_CODE_MAX_TOOL_USE_TIMEOUT_MS" not in env
    assert "CLAUDE_CODE_AUTO_BACKGROUND_TIMEOUT_MS" not in env
    assert "MCP_TOOL_TIMEOUT" not in env


@pytest.mark.parametrize(
    "command_timeout_ms",
    [pytest.param(300000, id="with-timeout"), pytest.param(None, id="without-timeout")],
)
async def test_start_when_traceparent_set_does_include_gymrat_traceparent_in_env(
    command_timeout_ms: int | None,
):
    tp = "00-abc123-def456-01"

    client = await _start_with_prompt(
        make_prompt(traceparent=tp, command_timeout_ms=command_timeout_ms)
    )

    assert client.options is not None
    env = client.options.get("env", {})
    assert env["GYMRAT_TRACEPARENT"] == tp  # pyrefly: ignore[bad-index]
    assert "TRACEPARENT" not in env  # pyrefly: ignore[not-iterable]


async def test_start_when_traceparent_none_does_omit_gymrat_traceparent_from_env():
    client = await _start_with_prompt(make_prompt())

    assert client.options is not None
    env = client.options.get("env", {})
    assert "GYMRAT_TRACEPARENT" not in env  # pyrefly: ignore[not-iterable]


# ---------------------------------------------------------------------------
# message mapping — positive cases
# ---------------------------------------------------------------------------


async def test_mapping_when_text_block_does_emit_text_delta():
    events = await run_with_messages([assistant(TextBlock(text="hello world"))])

    text_events = events_of(events, TextDeltaEvent)
    assert len(text_events) == 1
    assert text_events[0].chunk == "hello world"


async def test_mapping_when_read_path_under_cwd_does_summarize_relative_to_cwd():
    tool_use = ToolUseBlock(id="tu_1", name="Read", input={"file_path": "/my/project/src/main.py"})
    driver = create_claude_driver(
        client_factory=FactoryProbe(FiniteClient([assistant(tool_use), result_message()]))
    )
    probe = collecting_observer()

    await run_session(driver, probe.observer, make_prompt(cwd="/my/project"))

    starts = events_of(probe.events, ToolStartEvent)
    assert starts[0].input_summary == "src/main.py"


async def test_mapping_when_tool_result_has_no_matching_start_does_use_fallback_fields():
    orphan = tool_results(ToolResultBlock(tool_use_id="tu_orphan", content="result"))

    events = await run_with_messages([orphan])

    ends = events_of(events, ToolEndEvent)
    assert len(ends) == 1
    assert ends[0].tool_name == "unknown"
    assert ends[0].tool_use_id == "tu_orphan"
    assert ends[0].duration_ms == 0


class _FakeClocks:
    """Mutable readings for the duration clock and the wall clock."""

    def __init__(self) -> None:
        self.monotonic_ms = 0.0
        self.wall_ns = 0


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


async def test_mapping_when_wall_clock_jumps_back_does_report_monotonic_tool_duration(
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


async def test_mapping_when_tool_result_content_not_string_does_json_encode():
    payload = [{"type": "text", "text": "hi"}]
    messages = [
        assistant(ToolUseBlock(id="tu_3", name="Bash", input={"command": "echo hi"})),
        tool_results(ToolResultBlock(tool_use_id="tu_3", content=payload)),
    ]

    events = await run_with_messages(messages)

    ends = events_of(events, ToolEndEvent)
    assert len(ends) == 1
    assert ends[0].result == json.dumps(payload)


async def test_mapping_when_tool_result_content_not_json_encodable_does_fall_back_to_str():
    payload = [{"type": "opaque", "value": Unserializable()}]
    messages = [
        assistant(ToolUseBlock(id="tu_c", name="Test", input={})),
        tool_results(ToolResultBlock(tool_use_id="tu_c", content=payload)),
    ]

    events = await run_with_messages(messages)

    ends = events_of(events, ToolEndEvent)
    assert len(ends) == 1
    assert ends[0].result == str(payload)


async def test_mapping_when_tool_result_content_none_does_emit_tool_end_with_empty_result():
    messages = [
        assistant(ToolUseBlock(id="tu_n", name="Write", input={"file_path": "/a"})),
        tool_results(ToolResultBlock(tool_use_id="tu_n", content=None)),
    ]

    events = await run_with_messages(messages)

    ends = events_of(events, ToolEndEvent)
    assert [(end.tool_name, end.result) for end in ends] == [("Write", "null")]


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
async def test_mapping_when_tool_use_block_does_emit_tool_start(
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
async def test_mapping_when_tool_result_matches_start_does_emit_tool_end_with_tracked_name(
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
# message mapping — text block
# ---------------------------------------------------------------------------


async def test_mapping_when_complete_text_block_does_still_emit_text_delta():
    events = await run_with_messages([assistant(TextBlock(text="hello"))])

    text_events = events_of(events, TextDeltaEvent)
    assert len(text_events) == 1
    assert text_events[0].chunk == "hello"


async def test_mapping_when_text_block_with_parent_does_carry_parent_tool_use_id():
    msg = assistant(TextBlock(text="subagent output"), parent_tool_use_id="tu_parent")

    events = await run_with_messages([msg])

    text_events = events_of(events, TextDeltaEvent)
    assert len(text_events) == 1
    assert text_events[0].parent_tool_use_id == "tu_parent"


# ---------------------------------------------------------------------------
# tool events carry parent_tool_use_id
# ---------------------------------------------------------------------------


async def test_mapping_when_tool_use_with_parent_does_carry_parent_tool_use_id():
    tool_use = ToolUseBlock(id="tu_1", name="Read", input={"file_path": "/foo.ts"})
    msg = assistant(tool_use, parent_tool_use_id="tu_parent")

    events = await run_with_messages([msg])

    starts = events_of(events, ToolStartEvent)
    assert starts[0].parent_tool_use_id == "tu_parent"


async def test_mapping_when_tool_result_with_parent_does_carry_parent_tool_use_id():
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

    ends = events_of(events, ToolEndEvent)
    assert ends[0].parent_tool_use_id == "tu_parent"


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
async def test_mapping_when_message_carries_no_session_content_does_emit_nothing(
    message: object,
):
    assert await run_with_messages([message]) == []


# ---------------------------------------------------------------------------
# cost tracking
# ---------------------------------------------------------------------------


async def test_cost_when_no_messages_carry_cost_does_not_emit_usage_update():
    events = await run_with_messages([assistant(TextBlock(text="hello"))])

    assert events_of(events, UsageUpdateEvent) == []


# ---------------------------------------------------------------------------
# cost ordering and interrupt
# ---------------------------------------------------------------------------


def _interrupting_observer(
    events: list[SessionEvent], holder: dict[str, DriverSession], after: int = 1
) -> SessionObserver:
    """Observer that schedules ``interrupt`` after ``after`` usage updates."""
    seen = 0

    def observer(event: SessionEvent) -> None:
        nonlocal seen
        events.append(event)
        if isinstance(event, UsageUpdateEvent):
            seen += 1
            if seen == after:
                asyncio.ensure_future(holder["session"].interrupt())  # noqa: RUF006

    return observer


async def _run_interrupting_on_first_usage_update(
    messages: Sequence[object],
) -> tuple[SessionOutcome, FakeClient]:
    """Drive a session that schedules ``interrupt`` after the first usage update.

    Usage updates come from result messages, which leave the session idle
    between turns; the soft stop lands when the stream delivers its next
    message, so ``messages`` must carry one after the interrupting result.
    """
    client = FakeClient(messages)
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    events: list[SessionEvent] = []
    holder: dict[str, DriverSession] = {}

    holder["session"] = driver.start(make_prompt(), _interrupting_observer(events, holder))
    outcome = await holder["session"].outcome
    return outcome, client


async def test_interrupt_when_scheduled_on_usage_update_does_report_crossing_cost():
    outcome, _client = await _run_interrupting_on_first_usage_update([
        result_message(total_cost_usd=0.15),
        assistant(TextBlock(text="late")),
    ])

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.15


async def test_interrupt_when_first_call_wins_does_ignore_later_higher_cost():
    outcome, _client = await _run_interrupting_on_first_usage_update([
        result_message(total_cost_usd=0.1),
        result_message(total_cost_usd=0.25),
    ])

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.1


async def test_interrupt_when_called_does_soft_stop_without_disconnecting():
    _outcome, client = await _run_interrupting_on_first_usage_update([
        result_message(total_cost_usd=0.15),
        assistant(TextBlock(text="late")),
    ])

    assert client.interrupt_called is True
    assert client.disconnect_count == 1  # only the finally teardown, never interrupt itself


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

    session = driver.start(make_prompt(), probe.observer)
    await first_seen.wait()
    await session.interrupt()
    gate.set()
    outcome = await session.outcome

    assert outcome.reason == "interrupted"
    assert [e.chunk for e in events_of(probe.events, TextDeltaEvent)] == ["first"]


async def test_interrupt_when_called_repeatedly_before_client_does_resolve_interrupted_once():
    client = FakeClient([])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    session = driver.start(make_prompt(), collecting_observer().observer)
    await session.interrupt()  # client not built yet — the soft stop cannot reach it
    await session.interrupt()  # already stopped — a no-op that keeps the first outcome
    outcome = await session.outcome

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.0
    assert client.interrupt_called is False


# ---------------------------------------------------------------------------
# abort
# ---------------------------------------------------------------------------


async def test_abort_when_fired_does_resolve_interrupted():
    client = FakeClient([result_message(total_cost_usd=0.1)])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    abort = asyncio.Event()
    events: list[SessionEvent] = []

    def observer(event: SessionEvent) -> None:
        events.append(event)
        if isinstance(event, UsageUpdateEvent):
            abort.set()

    session = driver.start(make_prompt(), observer, abort)
    outcome = await session.outcome

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.1
    assert client.disconnect_count >= 1


async def test_abort_when_fired_after_interrupt_does_preserve_interrupt_cost():
    client = FakeClient([result_message(total_cost_usd=0.1), result_message(total_cost_usd=0.3)])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    abort = asyncio.Event()
    holder: dict[str, DriverSession] = {}

    def observer(event: SessionEvent) -> None:
        if isinstance(event, UsageUpdateEvent):
            asyncio.ensure_future(holder["session"].interrupt())  # noqa: RUF006
            abort.set()

    holder["session"] = driver.start(make_prompt(), observer, abort)
    outcome = await holder["session"].outcome

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.1


async def test_abort_when_already_set_at_start_does_resolve_interrupted_without_client():
    probe = FactoryProbe(FakeClient([]))
    driver = create_claude_driver(client_factory=probe)
    abort = asyncio.Event()
    abort.set()

    outcome = await run_session(driver, collecting_observer().observer, abort=abort)

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.0
    assert probe.calls == 0


async def test_abort_when_disconnect_raises_does_preserve_settled_outcome():

    class DisconnectRaisingClient(FakeClient):
        @override
        async def disconnect(self) -> None:
            self.disconnect_count += 1
            self._released.set()
            message = "abort disconnect failed"
            raise RuntimeError(message)

    client = DisconnectRaisingClient([result_message(total_cost_usd=0.10)])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    abort = asyncio.Event()

    def observer(event: SessionEvent) -> None:
        if isinstance(event, UsageUpdateEvent):
            abort.set()

    with pytest.warns(RuntimeWarning, match="disconnect failed"):
        outcome = await run_session(driver, observer, abort=abort)

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.10


# ---------------------------------------------------------------------------
# result message — session settlement
# ---------------------------------------------------------------------------


async def test_result_when_stream_yields_result_message_does_settle_completed():
    messages = [
        assistant(TextBlock(text="done")),
        result_message(total_cost_usd=0.05, num_turns=3),
    ]

    outcome = await run_outcome(FiniteClient(messages))

    assert outcome.reason == "completed"
    assert outcome.cost_usd == 0.05


@pytest.mark.parametrize(
    ("result_text", "expected_message"),
    [
        pytest.param("something went wrong", "something went wrong", id="result-text-present"),
        pytest.param(None, "error", id="result-text-absent-uses-subtype"),
    ],
)
async def test_result_when_is_error_does_settle_error(
    result_text: str | None,
    expected_message: str,
):
    messages = [result_message(subtype="error", is_error=True, result=result_text)]

    outcome = await run_outcome(FakeClient(messages))

    assert outcome.reason == "error"
    assert outcome.message == expected_message


async def test_result_when_is_error_with_cost_does_settle_as_final_result_not_turn_end():
    messages = [result_message(subtype="error", is_error=True, result="boom", total_cost_usd=0.2)]
    probe = collecting_observer()

    outcome = await run_outcome(FiniteClient(messages), probe.observer)

    updates = events_of(probe.events, UsageUpdateEvent)
    assert (outcome.reason, outcome.cost_usd) == ("error", 0.2)
    assert [(update.cost_usd, update.settled) for update in updates] == [(0.2, True)]
    assert events_of(probe.events, TurnEndEvent) == []


async def test_result_when_message_settles_and_has_cost_does_emit_usage_update():
    messages = [result_message(total_cost_usd=0.05)]
    probe = collecting_observer()

    await run_outcome(FiniteClient(messages), probe.observer)

    updates = events_of(probe.events, UsageUpdateEvent)
    assert [update.cost_usd for update in updates] == [0.05]


@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(None, id="none-cost"),
        pytest.param(0.0, id="zero-cost"),
    ],
)
async def test_result_when_message_settles_without_cost_does_not_emit_usage_update(
    cost: float | None,
):
    messages = [result_message(total_cost_usd=cost)]
    probe = collecting_observer()

    await run_outcome(FiniteClient(messages), probe.observer)

    assert events_of(probe.events, UsageUpdateEvent) == []


async def test_result_when_stream_ends_without_result_does_settle_error():
    outcome = await run_outcome(FiniteClient([assistant(TextBlock(text="hello"))]))

    assert outcome.reason == "error"
    assert outcome.message is not None
    assert "result" in outcome.message.lower()


async def test_result_when_system_message_has_subtype_does_not_end_session():
    messages = [
        system_message(subtype="init"),
        assistant(TextBlock(text="hello")),
    ]

    events = await run_with_messages(messages)

    text_events = events_of(events, TextDeltaEvent)
    assert len(text_events) == 1
    assert text_events[0].chunk == "hello"


# ---------------------------------------------------------------------------
# system message — compact_boundary → CompactionEvent
# ---------------------------------------------------------------------------


async def test_mapping_when_system_message_compact_boundary_does_emit_compaction_event():
    messages = [
        system_message(subtype="compact_boundary"),
        assistant(TextBlock(text="hello")),
    ]

    events = await run_with_messages(messages)

    compaction_events = events_of(events, CompactionEvent)
    assert len(compaction_events) == 1
    assert compaction_events[0].at > 0


async def test_mapping_when_system_message_compact_boundary_does_not_end_session():
    messages = [
        system_message(subtype="compact_boundary"),
        result_message(total_cost_usd=0.05),
    ]

    outcome = await run_outcome(FiniteClient(messages))

    assert outcome.reason == "completed"


async def test_mapping_when_system_message_other_subtype_does_not_emit_compaction_event():
    messages = [
        system_message(subtype="init"),
        system_message(subtype="some_other_subtype"),
        assistant(TextBlock(text="hello")),
    ]

    events = await run_with_messages(messages)

    compaction_events = events_of(events, CompactionEvent)
    assert compaction_events == []


# ---------------------------------------------------------------------------
# error — stream exception
# ---------------------------------------------------------------------------


async def test_outcome_when_stream_raises_does_resolve_error_without_raising():
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
    def raise_missing() -> object:
        raise ModuleNotFoundError

    monkeypatch.setattr("gymrat.supervisor.claude._load_default_factory", raise_missing)
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


async def test_start_when_disconnect_raises_after_normal_stream_does_still_resolve_completed():
    class DisconnectFailingClient(FiniteClient):
        @override
        async def disconnect(self) -> None:
            message = "teardown boom"
            raise RuntimeError(message)

    client = DisconnectFailingClient([result_message(total_cost_usd=0.05)])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    with pytest.warns(RuntimeWarning, match="disconnect failed"):
        outcome = await run_session(driver, collecting_observer().observer)

    assert outcome.reason == "completed"
    assert outcome.cost_usd == 0.05


# ---------------------------------------------------------------------------
# start — interrupted before client connects
# ---------------------------------------------------------------------------


async def test_start_when_interrupted_before_connect_does_never_send_kickoff_query():
    client = FakeClient([])
    probe = FactoryProbe(client)
    driver = create_claude_driver(client_factory=probe)

    session = driver.start(
        make_prompt(kickoff="should not be sent"), collecting_observer().observer
    )
    await session.interrupt()
    outcome = await session.outcome

    assert outcome.reason == "interrupted"
    assert client.query_prompts == []
