"""Behavioral tests for the Claude Agent SDK driver.

The driver is exercised entirely through an injected fake client, so these
tests never start the real ``claude-agent-sdk`` client. The fake mimics the
streaming client surface the driver relies on (``connect``/``query``/
``receive_messages``/``interrupt``/``disconnect``) and yields the SDK's own
message and content-block dataclasses. The cost rule every driver applies to
a reported cost is pinned here too.
"""

import asyncio
import gc
import json
import math
import sys
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import override

import pytest
from claude_agent_sdk import (
    HookMatcher,
    McpSdkServerConfig,
    ServerToolResultBlock,
    ServerToolUseBlock,
    SystemMessage,
    TaskStartedMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
)
from claude_agent_sdk.types import HookEvent

from gymrat.agent_env import TRACEPARENT_ENV
from gymrat.supervisor.claude import create_claude_driver, usable_cost
from gymrat.supervisor.driver import Driver, DriverSession, SessionPrompt
from gymrat.supervisor.events import (
    CompactionEvent,
    SessionObserver,
    TextDeltaEvent,
    ToolEndEvent,
    ToolStartEvent,
    TurnEndEvent,
    summarize,
    summarize_input,
)
from gymrat.supervisor.tool_names import MCP_SERVER
from tests._clock import install_monotonic_clock
from tests._rich import Clock
from tests._task_leaks import install_task_leak_recorder, task_leak_messages
from tests.supervisor._fixtures import (
    PROMPT_CWD,
    SDK_SESSION_ID,
    FactoryProbe,
    FakeClient,
    FiniteClient,
    abort_on_first_usage_update,
    assistant,
    collecting_observer,
    events_of,
    make_prompt,
    result_message,
    run_acting_on_first,
    run_interrupting_on_first_usage_update,
    run_outcome,
    run_session,
    run_with_messages,
    settled_outcome,
    start_claude_session,
    wait_for_event_or_task,
)

# ---------------------------------------------------------------------------
# SDK message builders
# ---------------------------------------------------------------------------


def system_message(*, subtype: str) -> SystemMessage:
    """Build an SDK ``SystemMessage`` of ``subtype`` with no data."""
    return SystemMessage(subtype=subtype, data={})


def tool_results(*blocks: ToolResultBlock, parent_tool_use_id: str | None = None) -> UserMessage:
    """Build an SDK ``UserMessage`` carrying tool results, as the CLI delivers them."""
    return UserMessage(content=list(blocks), parent_tool_use_id=parent_tool_use_id)


# ---------------------------------------------------------------------------
# start — options forwarding
# ---------------------------------------------------------------------------


#: A W3C trace context a prompt can carry for the driver to hand the agent.
TRACEPARENT = "00-abc123-def456-01"

#: The MCP server config ``ToolsFactoryProbe`` hands back, so a test can find it
#: again in the client options.
_SENTINEL_SERVER: McpSdkServerConfig = create_sdk_mcp_server("fake")

#: The hook mapping ``HooksFactoryProbe`` hands back, so a test can find it again
#: in the client options.
_SENTINEL_HOOKS: dict[HookEvent, list[HookMatcher]] = {
    "PreToolUse": [HookMatcher(matcher="Bash", hooks=[])],
}


class HooksFactoryProbe:
    """A stub hooks factory that counts its calls and returns ``_SENTINEL_HOOKS``."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> dict[HookEvent, list[HookMatcher]]:
        self.calls += 1
        return _SENTINEL_HOOKS


class ToolsFactoryProbe:
    """A stub tools factory that records each call's context and returns ``_SENTINEL_SERVER``."""

    def __init__(self) -> None:
        self.calls: list[tuple[asyncio.Event, Mapping[str, str]]] = []

    def __call__(self, abort: asyncio.Event, env: Mapping[str, str]) -> McpSdkServerConfig:
        self.calls.append((abort, env))
        return _SENTINEL_SERVER


#: The timeout variables a session sets to the prompt's command timeout.
_TIMEOUT_ENV_KEYS = (
    "CLAUDE_CODE_DEFAULT_TOOL_USE_TIMEOUT_MS",
    "CLAUDE_CODE_MAX_TOOL_USE_TIMEOUT_MS",
    "MCP_TOOL_TIMEOUT",
)


async def test_start_when_no_optional_input_given_does_pass_only_the_prompt_derived_client_options():
    client = FiniteClient([result_message()])
    prompt = make_prompt(
        cwd="/my/project", system_prompt_append="extra instructions", command_timeout_ms=300000
    )

    await run_outcome(client, prompt=prompt)

    options = client.options
    assert options is not None
    system_prompt = options["system_prompt"]
    env = options["env"]
    assert isinstance(system_prompt, dict)
    assert isinstance(env, dict)
    assert options.keys().isdisjoint({"model", "effort", "max_budget_usd", "hooks", "mcp_servers"})
    assert (options["cwd"], system_prompt["append"]) == ("/my/project", "extra instructions")
    assert [env[key] for key in _TIMEOUT_ENV_KEYS] == ["300000"] * len(_TIMEOUT_ENV_KEYS)
    assert TRACEPARENT_ENV not in env


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
    ],
)
async def test_start_when_session_inputs_given_does_forward_them_as_client_options(
    prompt: SessionPrompt, added: dict[str, object]
):
    client = FiniteClient([result_message()])

    await run_outcome(client, prompt=prompt)

    assert client.options is not None
    assert {key: client.options[key] for key in added} == added


async def test_start_when_traceparent_given_does_hand_it_to_the_agent_env():
    client = FiniteClient([result_message()])

    await run_outcome(client, prompt=make_prompt(traceparent=TRACEPARENT))

    assert client.options is not None
    env = client.options["env"]
    assert isinstance(env, dict)
    assert env[TRACEPARENT_ENV] == TRACEPARENT


async def test_start_when_factories_given_does_mount_their_results_in_client_options():
    client = FiniteClient([result_message()])

    await run_outcome(client, hooks=HooksFactoryProbe(), tools=ToolsFactoryProbe())

    assert client.options is not None
    assert {key: client.options[key] for key in ("hooks", "mcp_servers")} == {
        "hooks": _SENTINEL_HOOKS,
        "mcp_servers": {MCP_SERVER: _SENTINEL_SERVER},
    }


async def test_start_when_hooks_given_does_call_factory_once_per_session():
    probe = HooksFactoryProbe()
    driver = create_claude_driver(
        client_factory=lambda _options: FiniteClient([result_message()]),
        hooks=probe,
    )

    await run_session(driver, collecting_observer().observer)
    await run_session(driver, collecting_observer().observer)

    assert probe.calls == 2


@pytest.mark.parametrize(
    ("prompt", "env"),
    [
        pytest.param(
            make_prompt(traceparent=TRACEPARENT),
            {TRACEPARENT_ENV: TRACEPARENT},
            id="traceparent",
        ),
        pytest.param(make_prompt(), {}, id="no-traceparent"),
    ],
)
async def test_start_when_tools_given_does_call_factory_with_the_session_context(
    prompt: SessionPrompt, env: dict[str, str]
):
    probe = ToolsFactoryProbe()
    abort = asyncio.Event()

    await run_outcome(FiniteClient([result_message()]), prompt=prompt, abort=abort, tools=probe)

    assert probe.calls == [(abort, env)]


# ---------------------------------------------------------------------------
# message mapping — positive cases
# ---------------------------------------------------------------------------


async def test_start_when_text_block_does_emit_text_delta_carrying_its_parent():
    block = assistant(TextBlock(text="subagent output"), parent_tool_use_id="tu_parent")

    events = await run_with_messages([block])

    deltas = events_of(events, TextDeltaEvent)
    assert [(delta.chunk, delta.parent_tool_use_id) for delta in deltas] == [
        ("subagent output", "tu_parent")
    ]


async def test_start_when_tool_result_has_no_matching_start_does_leave_name_and_duration_unknown():
    orphan = tool_results(ToolResultBlock(tool_use_id="tu_orphan", content="result"))

    events = await run_with_messages([orphan])

    ends = events_of(events, ToolEndEvent)
    assert [(end.tool_use_id, end.tool_name, end.duration_ms) for end in ends] == [
        ("tu_orphan", None, None)
    ]


class _ClockedClient(FiniteClient):
    """A finite client that sets both fake clocks before yielding each message."""

    def __init__(
        self,
        steps: Sequence[tuple[float, int, object]],
        monotonic: Clock[float],
        wall_ns: Clock[int],
    ) -> None:
        super().__init__([message for _, _, message in steps])
        self._steps = steps
        self._monotonic = monotonic
        self._wall_ns = wall_ns

    @override
    async def receive_messages(self) -> AsyncIterator[object]:
        for monotonic_ms, wall_ns, message in self._steps:
            self._monotonic.now = monotonic_ms
            self._wall_ns.now = wall_ns
            await asyncio.sleep(0)
            yield message


async def test_start_when_wall_clock_jumps_back_does_report_monotonic_tool_duration(
    monkeypatch: pytest.MonkeyPatch,
):
    monotonic = install_monotonic_clock(monkeypatch)
    wall_ns = Clock(0)
    monkeypatch.setattr("time.time_ns", wall_ns)
    monkeypatch.setattr("time.time", lambda: wall_ns.now / 1_000_000_000)
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
        monotonic,
        wall_ns,
    )
    probe = collecting_observer()

    await run_outcome(client, probe.observer)

    ends = events_of(probe.events, ToolEndEvent)
    assert [end.duration_ms for end in ends] == [250]


_OPAQUE_CONTENT = [{"type": "opaque", "value": object()}]


async def test_start_when_tool_result_content_not_json_encodable_does_fall_back_to_its_str():
    messages = [
        assistant(ToolUseBlock(id="tu_3", name="Bash", input={"command": "echo hi"})),
        tool_results(ToolResultBlock(tool_use_id="tu_3", content=_OPAQUE_CONTENT)),
    ]

    events = await run_with_messages(messages)

    assert [end.result for end in events_of(events, ToolEndEvent)] == [str(_OPAQUE_CONTENT)]


# ---------------------------------------------------------------------------
# message mapping — tool-use and tool-result blocks (client and server)
# ---------------------------------------------------------------------------

_READ_INPUT: dict[str, object] = {"file_path": "/foo.ts"}
_READ_UNDER_CWD_INPUT: dict[str, object] = {"file_path": f"{PROMPT_CWD}/src/main.py"}
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
            ToolUseBlock(id="tu_1", name="Read", input=_READ_INPUT),
            "tu_parent",
            summarize_input(_READ_INPUT, tool_name="Read", supervised_root=PROMPT_CWD),
            id="client-tool-under-a-parent",
        ),
        pytest.param(
            ToolUseBlock(id="tu_1", name="Read", input=_READ_UNDER_CWD_INPUT),
            None,
            summarize_input(_READ_UNDER_CWD_INPUT, tool_name="Read", supervised_root=PROMPT_CWD),
            id="client-read-under-cwd-is-relative",
        ),
        pytest.param(
            ServerToolUseBlock(id="tu_web", name="web_search", input=_WEB_SEARCH_INPUT),
            None,
            summarize_input(_WEB_SEARCH_INPUT, tool_name="web_search", supervised_root=PROMPT_CWD),
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
    ("messages", "expected"),
    [
        pytest.param(
            [
                assistant(
                    ToolUseBlock(id="tu_1", name="Read", input={"file_path": "/foo.ts"}),
                    parent_tool_use_id="tu_parent",
                ),
                tool_results(
                    ToolResultBlock(tool_use_id="tu_1", content="file contents\n  here\n"),
                    parent_tool_use_id="tu_parent",
                ),
            ],
            (
                "tu_parent",
                "tu_1",
                "Read",
                "file contents\n  here\n",
                summarize("file contents\n  here\n"),
            ),
            id="client-tool-under-a-parent",
        ),
        pytest.param(
            [
                assistant(
                    ServerToolUseBlock(id="tu_web", name="web_search", input=_WEB_SEARCH_INPUT),
                    ServerToolResultBlock(tool_use_id="tu_web", content=_WEB_SEARCH_RESULT),
                ),
            ],
            (
                None,
                "tu_web",
                "web_search",
                json.dumps(_WEB_SEARCH_RESULT),
                summarize(json.dumps(_WEB_SEARCH_RESULT)),
            ),
            id="server-tool",
        ),
    ],
)
async def test_start_when_tool_result_matches_start_does_emit_tool_end_with_tracked_name(
    messages: list[object],
    expected: tuple[str | None, str, str, str, str],
):
    events = await run_with_messages(messages)

    ends = events_of(events, ToolEndEvent)
    assert [
        (end.parent_tool_use_id, end.tool_use_id, end.tool_name, end.result, end.result_summary)
        for end in ends
    ] == [expected]


# ---------------------------------------------------------------------------
# message mapping — messages without session events
# ---------------------------------------------------------------------------


async def test_start_when_system_message_subclass_arrives_does_emit_nothing():
    message = TaskStartedMessage(
        subtype="task_started",
        data={},
        task_id="task-1",
        description="explore",
        uuid="task-uuid",
        session_id=SDK_SESSION_ID,
    )

    events = await run_with_messages([message])

    assert events == []


# ---------------------------------------------------------------------------
# cost ordering and interrupt
# ---------------------------------------------------------------------------


async def test_interrupt_when_scheduled_on_usage_update_does_soft_stop_at_the_crossing_cost():
    client = FakeClient([result_message(total_cost_usd=0.15), result_message(total_cost_usd=0.25)])

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

    def end_then_abort(session: DriverSession) -> asyncio.Future[None]:
        ending = asyncio.ensure_future(session.end())
        abort.set()
        return ending

    outcome = await run_acting_on_first(client, TurnEndEvent, end_then_abort, abort=abort)

    assert (outcome.reason, outcome.cost_usd) == ("completed", 0.1)


async def test_start_when_abort_already_set_at_start_does_resolve_interrupted_without_client():
    probe = FactoryProbe(FakeClient([]))
    driver = create_claude_driver(client_factory=probe)
    abort = asyncio.Event()
    abort.set()

    outcome = await run_session(driver, collecting_observer().observer, abort=abort)

    assert outcome.reason == "interrupted"
    assert probe.calls == 0


# ---------------------------------------------------------------------------
# system message — compact_boundary → CompactionEvent
# ---------------------------------------------------------------------------


async def test_start_when_compact_boundary_arrives_does_emit_a_compaction_event():
    probe = collecting_observer()

    await run_outcome(FiniteClient([system_message(subtype="compact_boundary")]), probe.observer)

    assert len(events_of(probe.events, CompactionEvent)) == 1


# ---------------------------------------------------------------------------
# missing SDK
# ---------------------------------------------------------------------------


async def test_start_when_sdk_import_fails_does_resolve_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    driver = create_claude_driver()

    outcome = await run_session(driver, collecting_observer().observer)

    assert outcome.reason == "error"
    assert outcome.message is not None


# ---------------------------------------------------------------------------
# teardown robustness — outcome never raises
# ---------------------------------------------------------------------------


def _failing_client_factory(_options: Mapping[str, object]) -> FakeClient:
    message = "bad options"
    raise ValueError(message)


def _client_factory_raises() -> Driver:
    return create_claude_driver(client_factory=_failing_client_factory)


def _stream_raises() -> Driver:
    client = FakeClient([result_message(total_cost_usd=0.04)], throw=RuntimeError("SDK failure"))
    return create_claude_driver(client_factory=FactoryProbe(client))


@pytest.mark.parametrize(
    ("make_driver", "expected_message", "expected_cost"),
    [
        pytest.param(_client_factory_raises, "bad options", 0.0, id="client-factory-raises"),
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
# settling leaks no forgotten-task diagnostics
# ---------------------------------------------------------------------------


def _never_abort(_abort: asyncio.Event) -> SessionObserver:
    """An observer that leaves the abort unfired, so its watch task must be cleaned up."""
    return collecting_observer().observer


@pytest.mark.parametrize(
    ("make_client", "observer_for", "reason"),
    [
        pytest.param(
            lambda: FiniteClient([result_message(total_cost_usd=0.05)]),
            _never_abort,
            "completed",
            id="abort-unused-and-settles",
        ),
        pytest.param(
            # The stream hangs after the cost update, so the abort is what unblocks it:
            # the watch task starts, fires, then teardown cancels it on the settle path.
            lambda: FakeClient([result_message(total_cost_usd=0.1)]),
            abort_on_first_usage_update,
            "interrupted",
            id="abort-fires-mid-read",
        ),
    ],
)
async def test_claude_driver_when_session_settles_does_not_leak_task_diagnostics(
    make_client: Callable[[], FakeClient],
    observer_for: Callable[[asyncio.Event], SessionObserver],
    reason: str,
) -> None:
    records = install_task_leak_recorder()
    client = make_client()
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    abort = asyncio.Event()

    session = driver.start(make_prompt(), observer_for(abort), abort)
    outcome = await asyncio.wait_for(session.outcome, 10)
    del session, driver, client
    gc.collect()

    assert outcome.reason == reason
    assert task_leak_messages(records) == []


# ---------------------------------------------------------------------------
# start — interrupted before client connects
# ---------------------------------------------------------------------------


async def test_interrupt_when_called_before_connect_does_resolve_interrupted_without_sending_kickoff():
    client = FakeClient([result_message(total_cost_usd=0.10)])
    session = start_claude_session(
        client, collecting_observer().observer, prompt=make_prompt(kickoff="should not be sent")
    )

    await session.interrupt()  # client not built yet — the soft stop cannot reach it
    outcome = await settled_outcome(session)

    assert (outcome.reason, outcome.cost_usd) == ("interrupted", 0.0)
    assert client.query_prompts == []


# ---------------------------------------------------------------------------
# usable_cost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        pytest.param(0.42, 0.42, id="positive-float"),
        pytest.param(3, 3.0, id="positive-int"),
        pytest.param("0.42", None, id="string"),
        pytest.param(True, None, id="true"),
        pytest.param(math.inf, None, id="positive-infinity"),
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
