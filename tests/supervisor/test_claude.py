"""Behavioral tests for the Claude Agent SDK driver.

The driver is exercised entirely through an injected fake client, so these
tests never import the real ``claude-agent-sdk`` package. The fake mimics the
streaming client surface the driver relies on (``connect``/``query``/
``receive_messages``/``interrupt``/``disconnect``) and yields duck-typed,
attribute-only message objects shaped like the SDK's dataclasses.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
from math import ceil
from types import SimpleNamespace
from typing import override

import pytest

from gymrat.supervisor import create_claude_driver
from gymrat.supervisor.driver import Driver, DriverSession, SessionOutcome, SessionPrompt
from gymrat.supervisor.events import (
    ModelPhaseEvent,
    SessionEvent,
    SessionObserver,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    UsageUpdateEvent,
    summarize,
)
from tests.supervisor._fixtures import collecting_observer, make_prompt

# ---------------------------------------------------------------------------
# fake streaming client
# ---------------------------------------------------------------------------


class FakeClient:
    """A stand-in for the SDK streaming client that replays scripted messages.

    ``receive_messages`` yields each supplied message after an
    ``asyncio.sleep(0)`` handshake so an observer-scheduled interrupt or an
    abort lands deterministically between messages. When ``hang`` is set, the
    stream blocks after the scripted messages until ``disconnect`` releases it,
    letting a test prove that an abort tears the connection down.
    """

    def __init__(
        self,
        messages: Sequence[object],
        *,
        throw: Exception | None = None,
        hang: bool = False,
    ) -> None:
        self.messages = messages
        self.throw = throw
        self.hang = hang
        self.options: dict[str, object] | None = None
        self.query_prompt: str | None = None
        self.interrupt_called = False
        self.disconnect_count = 0
        self._released = asyncio.Event()

    async def connect(self) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self.query_prompt = prompt

    async def receive_messages(self):
        for message in self.messages:
            await asyncio.sleep(0)
            yield message
        if self.throw is not None:
            raise self.throw
        if self.hang:
            await self._released.wait()

    async def interrupt(self) -> None:
        self.interrupt_called = True

    async def disconnect(self) -> None:
        self.disconnect_count += 1
        self._released.set()


class FactoryProbe:
    """A client factory that records its call count and the options it saw."""

    def __init__(self, client: FakeClient) -> None:
        self._client = client
        self.calls = 0

    def __call__(self, options: Mapping[str, object]) -> FakeClient:
        self.calls += 1
        self._client.options = dict(options)
        return self._client


class Unserializable:
    """A tool-result payload that is neither a string nor JSON-encodable."""

    def __str__(self) -> str:
        return "UNSERIALIZABLE"


def assistant(*blocks: object, parent_tool_use_id: str | None = None) -> SimpleNamespace:
    """Build an assistant-style message carrying the given content blocks."""
    ns = SimpleNamespace(content=list(blocks))
    if parent_tool_use_id is not None:
        ns.parent_tool_use_id = parent_tool_use_id
    return ns


def stream_event(
    event: dict[str, object],
    *,
    parent_tool_use_id: str | None = None,
) -> SimpleNamespace:
    """Build a stream-event message (has ``event`` dict, no ``content`` or ``total_cost_usd``)."""
    ns = SimpleNamespace(event=event)
    if parent_tool_use_id is not None:
        ns.parent_tool_use_id = parent_tool_use_id
    return ns


async def run_session(
    driver: Driver,
    observer: SessionObserver,
    prompt: SessionPrompt | None = None,
    abort: asyncio.Event | None = None,
):
    """Start a session and await its settled outcome."""
    session = driver.start(prompt or make_prompt(), observer, abort)
    return await session.outcome


async def run_with_messages(messages: Sequence[object]) -> list[SessionEvent]:
    """Drive a session over scripted ``messages`` and return the events it emitted."""
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient(messages)))
    probe = collecting_observer()
    await run_session(driver, probe.observer)
    return probe.events


async def run_outcome(messages: Sequence[object]) -> SessionOutcome:
    """Drive a session over scripted ``messages`` and return its settled outcome."""
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient(messages)))
    return await run_session(driver, collecting_observer().observer)


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

    assert loaded is False


# ---------------------------------------------------------------------------
# start — options forwarding
# ---------------------------------------------------------------------------


async def _start_with_prompt(prompt: SessionPrompt) -> FakeClient:
    """Start a session with ``prompt`` and return the client it drove."""
    client = FakeClient([])
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
    assert client.query_prompt == "hello agent"


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


# ---------------------------------------------------------------------------
# message mapping — positive cases
# ---------------------------------------------------------------------------


async def test_mapping_when_text_block_does_emit_text_delta():
    events = await run_with_messages([assistant(SimpleNamespace(text="hello world"))])

    text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(text_events) == 1
    assert text_events[0].chunk == "hello world"


async def test_mapping_when_tool_use_block_does_emit_tool_start():
    tool_use = SimpleNamespace(id="tu_1", name="Read", input={"file_path": "/foo.ts"})

    events = await run_with_messages([assistant(tool_use)])

    starts = [e for e in events if isinstance(e, ToolStartEvent)]
    assert len(starts) == 1
    assert starts[0].tool_use_id == "tu_1"
    assert starts[0].tool_name == "Read"
    assert starts[0].input == {"file_path": "/foo.ts"}
    assert starts[0].input_summary == "/foo.ts"


async def test_mapping_when_read_path_under_cwd_does_summarize_relative_to_cwd():
    tool_use = SimpleNamespace(
        id="tu_1", name="Read", input={"file_path": "/my/project/src/main.py"}
    )
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient([assistant(tool_use)])))
    probe = collecting_observer()

    await run_session(driver, probe.observer, make_prompt(cwd="/my/project"))

    starts = [e for e in probe.events if isinstance(e, ToolStartEvent)]
    assert starts[0].input_summary == "src/main.py"


async def test_mapping_when_tool_result_matches_start_does_emit_tool_end_with_tracked_name():
    messages = [
        assistant(SimpleNamespace(id="tu_1", name="Read", input={"file_path": "/foo.ts"})),
        assistant(SimpleNamespace(tool_use_id="tu_1", content="file contents here")),
    ]

    events = await run_with_messages(messages)

    ends = [e for e in events if isinstance(e, ToolEndEvent)]
    assert len(ends) == 1
    assert ends[0].tool_use_id == "tu_1"
    assert ends[0].tool_name == "Read"
    assert ends[0].result == "file contents here"
    assert ends[0].result_summary == summarize("file contents here")
    assert ends[0].duration_ms >= 0


async def test_mapping_when_tool_result_has_no_matching_start_does_use_fallback_fields():
    orphan = assistant(SimpleNamespace(tool_use_id="tu_orphan", content="result"))

    events = await run_with_messages([orphan])

    ends = [e for e in events if isinstance(e, ToolEndEvent)]
    assert len(ends) == 1
    assert ends[0].tool_name == "unknown"
    assert ends[0].tool_use_id == "tu_orphan"
    assert ends[0].duration_ms == 0


async def test_mapping_when_tool_result_content_not_string_does_json_encode():
    payload = {"stdout": "hi", "exit_code": 0}
    messages = [
        assistant(SimpleNamespace(id="tu_3", name="Bash", input={"command": "echo hi"})),
        assistant(SimpleNamespace(tool_use_id="tu_3", content=payload)),
    ]

    events = await run_with_messages(messages)

    ends = [e for e in events if isinstance(e, ToolEndEvent)]
    assert len(ends) == 1
    assert ends[0].result == json.dumps(payload)


async def test_mapping_when_tool_result_content_not_json_encodable_does_fall_back_to_str():
    payload = Unserializable()
    messages = [
        assistant(SimpleNamespace(id="tu_c", name="Test", input={})),
        assistant(SimpleNamespace(tool_use_id="tu_c", content=payload)),
    ]

    events = await run_with_messages(messages)

    ends = [e for e in events if isinstance(e, ToolEndEvent)]
    assert len(ends) == 1
    assert ends[0].result == "UNSERIALIZABLE"


async def test_mapping_when_single_thinking_block_streamed_does_report_delta_equal_to_estimated_tokens():
    messages = [
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
        stream_event(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "abcd"},
            }
        ),
        stream_event({"type": "content_block_stop"}),
    ]

    events = await run_with_messages(messages)

    updates = [e for e in events if isinstance(e, ThinkingUpdateEvent)]
    assert len(updates) == 2
    assert updates[0].delta == 0
    assert updates[0].estimated_tokens == 0
    assert updates[-1].delta == 1
    assert updates[-1].estimated_tokens == 1


async def test_mapping_when_multiple_thinking_blocks_streamed_does_accumulate_estimated_tokens():
    messages = [
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
        stream_event(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "abcd"},
            }
        ),
        stream_event({"type": "content_block_stop"}),
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
        stream_event(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "abcdefgh"},
            }
        ),
        stream_event({"type": "content_block_stop"}),
    ]

    events = await run_with_messages(messages)

    updates = [e for e in events if isinstance(e, ThinkingUpdateEvent)]
    first_block = [u for u in updates if u.estimated_tokens <= 1]
    second_block_final = updates[-1]
    assert first_block[-1].estimated_tokens == 1
    assert second_block_final.estimated_tokens == 3


# ---------------------------------------------------------------------------
# stream events — thinking deltas with throttling
# ---------------------------------------------------------------------------


async def test_stream_when_thinking_delta_short_does_flush_only_on_block_stop():
    """A delta under 200 chars emits nothing until content_block_stop flushes."""
    messages = [
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
        stream_event(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "a" * 100},
            }
        ),
        stream_event({"type": "content_block_stop"}),
    ]

    events = await run_with_messages(messages)

    updates = [e for e in events if isinstance(e, ThinkingUpdateEvent)]
    assert updates[-1].estimated_tokens == 25  # ceil(100 / 4)


async def test_stream_when_thinking_delta_crosses_throttle_does_emit_mid_block():
    """A single delta >= 200 chars emits a ThinkingUpdateEvent immediately."""
    text = "a" * 250
    messages = [
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
        stream_event(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": text},
            }
        ),
        stream_event({"type": "content_block_stop"}),
    ]

    events = await run_with_messages(messages)

    updates = [e for e in events if isinstance(e, ThinkingUpdateEvent)]
    # block_start (delta=0), mid-block emit, block_stop flush
    assert len(updates) >= 2
    final = updates[-1]
    assert final.estimated_tokens == 63  # ceil(250 / 4)


async def test_stream_when_thinking_deltas_accumulated_does_bound_update_count():
    """Total ThinkingUpdateEvents for a block never exceed ceil(len / 200) + 2."""
    chunk = "a" * 50
    num_chunks = 20  # 1000 chars total
    messages = [
        stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
    ]
    for _ in range(num_chunks):
        messages.append(
            stream_event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": chunk},
                }
            )
        )
    messages.append(stream_event({"type": "content_block_stop"}))

    events = await run_with_messages(messages)

    updates = [e for e in events if isinstance(e, ThinkingUpdateEvent)]
    max_allowed = ceil(1000 / 200) + 2
    assert len(updates) <= max_allowed
    assert updates[-1].estimated_tokens == ceil(1000 / 4)


# ---------------------------------------------------------------------------
# stream events — phase transitions
# ---------------------------------------------------------------------------


_THINKING_BLOCK_MESSAGES = [
    stream_event({"type": "content_block_start", "content_block": {"type": "thinking"}}),
    stream_event({"type": "content_block_stop"}),
]
_TEXT_BLOCK_MESSAGES = [
    stream_event({"type": "content_block_start", "content_block": {"type": "text"}}),
    stream_event({"type": "content_block_stop"}),
]


@pytest.mark.parametrize(
    ("messages", "expected_phase"),
    [
        pytest.param(_THINKING_BLOCK_MESSAGES, "thinking", id="thinking-block-start"),
        pytest.param(_TEXT_BLOCK_MESSAGES, "responding", id="text-block-start"),
        pytest.param([stream_event({"type": "message_stop"})], "turn_end", id="message-stop"),
    ],
)
async def test_stream_when_block_start_does_emit_model_phase(
    messages: list[SimpleNamespace], expected_phase: str
):
    events = await run_with_messages(messages)

    phases = [e for e in events if isinstance(e, ModelPhaseEvent)]
    assert any(p.phase == expected_phase for p in phases)


async def test_stream_when_thinking_block_start_does_emit_initial_thinking_update():
    events = await run_with_messages(_THINKING_BLOCK_MESSAGES)

    updates = [e for e in events if isinstance(e, ThinkingUpdateEvent)]
    assert len(updates) >= 1
    assert updates[0].delta == 0
    assert updates[0].estimated_tokens == 0


async def test_stream_when_tool_use_block_start_does_emit_model_phase_tool_input():
    messages = [
        stream_event(
            {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Read"},
            }
        ),
        stream_event({"type": "content_block_stop"}),
    ]

    events = await run_with_messages(messages)

    phases = [e for e in events if isinstance(e, ModelPhaseEvent)]
    tool_phases = [p for p in phases if p.phase == "tool_input"]
    assert len(tool_phases) == 1
    assert tool_phases[0].tool_name == "Read"


# ---------------------------------------------------------------------------
# stream events — silent event types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_type",
    [
        pytest.param("text_delta", id="text-delta"),
        pytest.param("input_json_delta", id="input-json-delta"),
        pytest.param("signature_delta", id="signature-delta"),
        pytest.param("message_start", id="message-start"),
        pytest.param("message_delta", id="message-delta"),
        pytest.param("totally_unknown_type", id="unrecognized"),
    ],
)
async def test_stream_when_silent_event_type_does_emit_nothing(event_type: str):
    messages = [stream_event({"type": event_type})]

    events = await run_with_messages(messages)

    assert events == []


# ---------------------------------------------------------------------------
# stream events — per-parent thinking scoping
# ---------------------------------------------------------------------------


async def test_stream_when_subagent_thinking_does_not_inflate_top_level_total():
    """Each parent_tool_use_id keeps an independent estimated_tokens counter."""
    messages = [
        stream_event(
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            parent_tool_use_id=None,
        ),
        stream_event(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "aaaa"},
            },
            parent_tool_use_id=None,
        ),
        stream_event({"type": "content_block_stop"}, parent_tool_use_id=None),
        stream_event(
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            parent_tool_use_id="tu_sub",
        ),
        stream_event(
            {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "b" * 400},
            },
            parent_tool_use_id="tu_sub",
        ),
        stream_event({"type": "content_block_stop"}, parent_tool_use_id="tu_sub"),
    ]

    events = await run_with_messages(messages)

    updates = [e for e in events if isinstance(e, ThinkingUpdateEvent)]
    top = [u for u in updates if u.parent_tool_use_id is None]
    sub = [u for u in updates if u.parent_tool_use_id == "tu_sub"]
    assert top[-1].estimated_tokens == 1  # ceil(4 / 4)
    assert sub[-1].estimated_tokens == 100  # ceil(400 / 4)


# ---------------------------------------------------------------------------
# stream events — parent_tool_use_id propagation
# ---------------------------------------------------------------------------


async def test_stream_when_thinking_block_start_with_parent_does_carry_parent_tool_use_id():
    messages = [
        stream_event(
            {"type": "content_block_start", "content_block": {"type": "thinking"}},
            parent_tool_use_id="tu_42",
        ),
        stream_event({"type": "content_block_stop"}, parent_tool_use_id="tu_42"),
    ]

    events = await run_with_messages(messages)

    phases = [e for e in events if isinstance(e, ModelPhaseEvent)]
    assert phases[0].parent_tool_use_id == "tu_42"
    updates = [e for e in events if isinstance(e, ThinkingUpdateEvent)]
    assert updates[0].parent_tool_use_id == "tu_42"


async def test_stream_when_text_block_start_with_parent_does_carry_parent_tool_use_id():
    messages = [
        stream_event(
            {"type": "content_block_start", "content_block": {"type": "text"}},
            parent_tool_use_id="tu_99",
        ),
    ]

    events = await run_with_messages(messages)

    phases = [e for e in events if isinstance(e, ModelPhaseEvent)]
    assert phases[0].parent_tool_use_id == "tu_99"


async def test_stream_when_message_stop_with_parent_does_carry_parent_tool_use_id():
    messages = [
        stream_event({"type": "message_stop"}, parent_tool_use_id="tu_end"),
    ]

    events = await run_with_messages(messages)

    phases = [e for e in events if isinstance(e, ModelPhaseEvent)]
    assert phases[0].parent_tool_use_id == "tu_end"


# ---------------------------------------------------------------------------
# complete ThinkingBlock — no emission
# ---------------------------------------------------------------------------


async def test_mapping_when_complete_thinking_block_does_not_emit_thinking_update():
    """A complete ThinkingBlock (has ``thinking`` attr) is ignored — stream deltas counted it."""
    thinking = assistant(SimpleNamespace(thinking="abcd"))

    events = await run_with_messages([thinking])

    updates = [e for e in events if isinstance(e, ThinkingUpdateEvent)]
    assert updates == []


async def test_mapping_when_complete_text_block_does_still_emit_text_delta():
    events = await run_with_messages([assistant(SimpleNamespace(text="hello"))])

    text_events = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(text_events) == 1
    assert text_events[0].chunk == "hello"


# ---------------------------------------------------------------------------
# tool events carry parent_tool_use_id
# ---------------------------------------------------------------------------


async def test_mapping_when_tool_use_with_parent_does_carry_parent_tool_use_id():
    tool_use = SimpleNamespace(id="tu_1", name="Read", input={"file_path": "/foo.ts"})
    msg = assistant(tool_use, parent_tool_use_id="tu_parent")

    events = await run_with_messages([msg])

    starts = [e for e in events if isinstance(e, ToolStartEvent)]
    assert starts[0].parent_tool_use_id == "tu_parent"


async def test_mapping_when_tool_result_with_parent_does_carry_parent_tool_use_id():
    messages = [
        assistant(
            SimpleNamespace(id="tu_1", name="Read", input={"file_path": "/foo.ts"}),
            parent_tool_use_id="tu_parent",
        ),
        assistant(
            SimpleNamespace(tool_use_id="tu_1", content="file contents"),
            parent_tool_use_id="tu_parent",
        ),
    ]

    events = await run_with_messages(messages)

    ends = [e for e in events if isinstance(e, ToolEndEvent)]
    assert ends[0].parent_tool_use_id == "tu_parent"


# ---------------------------------------------------------------------------
# message mapping — malformed messages emit nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(SimpleNamespace(content="not a list"), id="non-list-content"),
        pytest.param(assistant(SimpleNamespace()), id="block-without-attributes"),
        pytest.param(assistant(SimpleNamespace(text=42)), id="non-string-text"),
        pytest.param(assistant(SimpleNamespace(thinking=42)), id="non-string-thinking"),
        pytest.param(assistant(SimpleNamespace(id="x", input={})), id="tool-use-missing-name"),
        pytest.param(assistant(SimpleNamespace(name="Read", input={})), id="tool-use-missing-id"),
        pytest.param(assistant("string", 42, None), id="non-object-blocks"),
        pytest.param(SimpleNamespace(foo="bar"), id="unknown-message"),
        pytest.param(
            assistant(SimpleNamespace(tool_use_id=42, content="data")),
            id="non-string-tool-use-id",
        ),
        pytest.param(SimpleNamespace(total_cost_usd=0.0), id="zero-cost"),
        pytest.param(SimpleNamespace(total_cost_usd=None), id="none-cost"),
    ],
)
async def test_mapping_when_message_malformed_does_emit_nothing(message: object):
    assert await run_with_messages([message]) == []


# ---------------------------------------------------------------------------
# cost tracking
# ---------------------------------------------------------------------------


async def test_cost_when_results_have_cost_does_emit_usage_update_per_result():
    messages = [
        assistant(SimpleNamespace(text="working...")),
        SimpleNamespace(total_cost_usd=0.03),
        assistant(SimpleNamespace(text="done")),
        SimpleNamespace(total_cost_usd=0.07),
    ]

    events = await run_with_messages(messages)

    usage = [e for e in events if isinstance(e, UsageUpdateEvent)]
    assert [e.cost_usd for e in usage] == [0.03, 0.07]


async def test_cost_when_no_result_messages_does_not_emit_usage_update():
    events = await run_with_messages([assistant(SimpleNamespace(text="hello"))])

    assert [e for e in events if isinstance(e, UsageUpdateEvent)] == []


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
    """Drive a session that schedules ``interrupt`` after the first usage update."""
    client = FakeClient(messages)
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    events: list[SessionEvent] = []
    holder: dict[str, DriverSession] = {}

    holder["session"] = driver.start(make_prompt(), _interrupting_observer(events, holder))
    outcome = await holder["session"].outcome
    return outcome, client


async def test_interrupt_when_scheduled_on_usage_update_does_report_crossing_cost():
    outcome, _client = await _run_interrupting_on_first_usage_update(
        [SimpleNamespace(total_cost_usd=0.15)]
    )

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.15


async def test_interrupt_when_first_call_wins_does_ignore_later_higher_cost():
    outcome, _client = await _run_interrupting_on_first_usage_update(
        [SimpleNamespace(total_cost_usd=0.1), SimpleNamespace(total_cost_usd=0.25)]
    )

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.1


async def test_interrupt_when_called_does_soft_stop_without_disconnecting():
    _outcome, client = await _run_interrupting_on_first_usage_update(
        [SimpleNamespace(total_cost_usd=0.15)]
    )

    assert client.interrupt_called is True
    assert client.disconnect_count == 1  # only the finally teardown, never interrupt itself


async def test_interrupt_when_called_between_messages_does_stop_before_next_message():
    gate = asyncio.Event()
    first_seen = asyncio.Event()

    class GatedClient(FakeClient):
        @override
        async def receive_messages(self):
            yield assistant(SimpleNamespace(text="first"))
            first_seen.set()
            await gate.wait()
            yield assistant(SimpleNamespace(text="second"))

    client = GatedClient([])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    probe = collecting_observer()

    session = driver.start(make_prompt(), probe.observer)
    await first_seen.wait()
    await session.interrupt()
    gate.set()
    outcome = await session.outcome

    assert outcome.reason == "interrupted"
    assert [e.chunk for e in probe.events if isinstance(e, TextDeltaEvent)] == ["first"]


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
    client = FakeClient([SimpleNamespace(total_cost_usd=0.1)], hang=True)
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
    client = FakeClient(
        [SimpleNamespace(total_cost_usd=0.1), SimpleNamespace(total_cost_usd=0.3)], hang=True
    )
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
    """A failing disconnect() must not replace the settled session outcome.

    The exception is swallowed with a warning, and the session's reason, cost,
    and ended_by remain intact.
    """

    class DisconnectRaisingClient(FakeClient):
        @override
        async def disconnect(self) -> None:
            self.disconnect_count += 1
            self._released.set()
            message = "abort disconnect failed"
            raise RuntimeError(message)

    client = DisconnectRaisingClient([SimpleNamespace(total_cost_usd=0.10)], hang=True)
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
# outcome
# ---------------------------------------------------------------------------


async def test_outcome_when_stream_ends_normally_does_report_completed_with_last_cost():
    messages = [SimpleNamespace(total_cost_usd=0.02), SimpleNamespace(total_cost_usd=0.05)]

    outcome = await run_outcome(messages)

    assert outcome.reason == "completed"
    assert outcome.cost_usd == 0.05


async def test_outcome_when_no_results_does_report_completed_with_zero_cost():
    outcome = await run_outcome([])

    assert outcome.reason == "completed"
    assert outcome.cost_usd == 0.0


async def test_outcome_when_stream_raises_does_resolve_error_without_raising():
    client = FakeClient([SimpleNamespace(total_cost_usd=0.04)], throw=RuntimeError("SDK failure"))
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
    class DisconnectFailingClient(FakeClient):
        @override
        async def disconnect(self) -> None:
            message = "teardown boom"
            raise RuntimeError(message)

    client = DisconnectFailingClient([SimpleNamespace(total_cost_usd=0.05)])
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
    assert client.query_prompt is None
