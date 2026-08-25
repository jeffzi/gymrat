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
from types import SimpleNamespace
from typing import override

import pytest

from gymrat_py.errors import message_of
from gymrat_py.supervisor import create_claude_driver
from gymrat_py.supervisor.driver import Driver, DriverSession, SessionPrompt
from gymrat_py.supervisor.events import (
    SessionEvent,
    SessionObserver,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolStartEvent,
    UsageUpdateEvent,
    summarize,
    summarize_input,
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


def assistant(*blocks: object) -> SimpleNamespace:
    """Build an assistant-style message carrying the given content blocks."""
    return SimpleNamespace(content=list(blocks))


async def run_session(
    driver: Driver,
    observer: SessionObserver,
    prompt: SessionPrompt | None = None,
    abort: asyncio.Event | None = None,
):
    """Start a session and await its settled outcome."""
    session = driver.start(prompt or make_prompt(), observer, abort)
    return await session.outcome


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

    monkeypatch.setattr("gymrat_py.supervisor.claude._load_default_factory", spy)

    create_claude_driver()

    assert loaded is False


# ---------------------------------------------------------------------------
# start — options forwarding
# ---------------------------------------------------------------------------


async def test_start_when_launched_does_forward_cwd_permission_mode_and_kickoff():
    client = FakeClient([])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    await run_session(
        driver,
        collecting_observer().observer,
        make_prompt(kickoff="hello agent", cwd="/my/project"),
    )

    assert client.options == {"cwd": "/my/project", "permission_mode": "bypassPermissions"}
    assert client.query_prompt == "hello agent"


async def test_start_when_system_prompt_append_present_does_include_preset_append():
    client = FakeClient([])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    await run_session(
        driver,
        collecting_observer().observer,
        make_prompt(system_prompt_append="extra instructions"),
    )

    assert client.options is not None
    assert client.options["system_prompt"] == {
        "type": "preset",
        "preset": "claude_code",
        "append": "extra instructions",
    }


async def test_start_when_system_prompt_append_absent_does_omit_system_prompt():
    client = FakeClient([])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    await run_session(driver, collecting_observer().observer, make_prompt())

    assert client.options is not None
    assert "system_prompt" not in client.options


async def test_start_when_model_given_does_include_model():
    client = FakeClient([])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    await run_session(
        driver,
        collecting_observer().observer,
        make_prompt(model="claude-sonnet-4-20250514"),
    )

    assert client.options is not None
    assert client.options["model"] == "claude-sonnet-4-20250514"


async def test_start_when_model_absent_does_omit_model():
    client = FakeClient([])
    driver = create_claude_driver(client_factory=FactoryProbe(client))

    await run_session(driver, collecting_observer().observer, make_prompt())

    assert client.options is not None
    assert "model" not in client.options


# ---------------------------------------------------------------------------
# message mapping — positive cases
# ---------------------------------------------------------------------------


async def test_mapping_when_text_block_does_emit_text_delta():
    driver = create_claude_driver(
        client_factory=FactoryProbe(FakeClient([assistant(SimpleNamespace(text="hello world"))]))
    )
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    text_events = [e for e in probe.events if isinstance(e, TextDeltaEvent)]
    assert len(text_events) == 1
    assert text_events[0].chunk == "hello world"


async def test_mapping_when_tool_use_block_does_emit_tool_start():
    tool_use = SimpleNamespace(id="tu_1", name="Read", input={"file_path": "/foo.ts"})
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient([assistant(tool_use)])))
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    starts = [e for e in probe.events if isinstance(e, ToolStartEvent)]
    assert len(starts) == 1
    assert starts[0].tool_use_id == "tu_1"
    assert starts[0].tool_name == "Read"
    assert starts[0].input == {"file_path": "/foo.ts"}
    assert starts[0].input_summary == summarize_input({"file_path": "/foo.ts"})


async def test_mapping_when_tool_result_matches_start_does_emit_tool_end_with_tracked_name():
    messages = [
        assistant(SimpleNamespace(id="tu_1", name="Read", input={"file_path": "/foo.ts"})),
        assistant(SimpleNamespace(tool_use_id="tu_1", content="file contents here")),
    ]
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient(messages)))
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    ends = [e for e in probe.events if isinstance(e, ToolEndEvent)]
    assert len(ends) == 1
    assert ends[0].tool_use_id == "tu_1"
    assert ends[0].tool_name == "Read"
    assert ends[0].result == "file contents here"
    assert ends[0].result_summary == summarize("file contents here")
    assert ends[0].duration_ms >= 0


async def test_mapping_when_tool_result_has_no_matching_start_does_use_unknown_and_zero_duration():
    orphan = assistant(SimpleNamespace(tool_use_id="tu_orphan", content="result"))
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient([orphan])))
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    ends = [e for e in probe.events if isinstance(e, ToolEndEvent)]
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
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient(messages)))
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    ends = [e for e in probe.events if isinstance(e, ToolEndEvent)]
    assert len(ends) == 1
    assert ends[0].result == json.dumps(payload)


async def test_mapping_when_tool_result_content_not_json_encodable_does_fall_back_to_str():
    payload = Unserializable()
    messages = [
        assistant(SimpleNamespace(id="tu_c", name="Test", input={})),
        assistant(SimpleNamespace(tool_use_id="tu_c", content=payload)),
    ]
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient(messages)))
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    ends = [e for e in probe.events if isinstance(e, ToolEndEvent)]
    assert len(ends) == 1
    assert ends[0].result == "UNSERIALIZABLE"


async def test_mapping_when_single_thinking_block_does_report_delta_equal_to_estimated_tokens():
    thinking = assistant(SimpleNamespace(thinking="abcd"))
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient([thinking])))
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    updates = [e for e in probe.events if isinstance(e, ThinkingUpdateEvent)]
    assert len(updates) == 1
    assert updates[0].delta == 1
    assert updates[0].estimated_tokens == 1


async def test_mapping_when_multiple_thinking_blocks_does_accumulate_estimated_tokens():
    messages = [
        assistant(SimpleNamespace(thinking="abcd")),
        assistant(SimpleNamespace(thinking="abcdefgh")),
    ]
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient(messages)))
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    updates = [e for e in probe.events if isinstance(e, ThinkingUpdateEvent)]
    assert len(updates) == 2
    assert (updates[0].delta, updates[0].estimated_tokens) == (1, 1)
    assert (updates[1].delta, updates[1].estimated_tokens) == (2, 3)


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
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient([message])))
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    assert probe.events == []


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
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient(messages)))
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    usage = [e for e in probe.events if isinstance(e, UsageUpdateEvent)]
    assert [e.cost_usd for e in usage] == [0.03, 0.07]


async def test_cost_when_no_result_messages_does_not_emit_usage_update():
    driver = create_claude_driver(
        client_factory=FactoryProbe(FakeClient([assistant(SimpleNamespace(text="hello"))]))
    )
    probe = collecting_observer()

    await run_session(driver, probe.observer)

    assert [e for e in probe.events if isinstance(e, UsageUpdateEvent)] == []


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


async def test_interrupt_when_scheduled_on_usage_update_does_report_crossing_cost():
    client = FakeClient([SimpleNamespace(total_cost_usd=0.15)])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    events: list[SessionEvent] = []
    holder: dict[str, DriverSession] = {}

    holder["session"] = driver.start(make_prompt(), _interrupting_observer(events, holder))
    outcome = await holder["session"].outcome

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.15


async def test_interrupt_when_first_call_wins_does_ignore_later_higher_cost():
    client = FakeClient([SimpleNamespace(total_cost_usd=0.1), SimpleNamespace(total_cost_usd=0.25)])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    events: list[SessionEvent] = []
    holder: dict[str, DriverSession] = {}

    holder["session"] = driver.start(make_prompt(), _interrupting_observer(events, holder))
    outcome = await holder["session"].outcome

    assert outcome.reason == "interrupted"
    assert outcome.cost_usd == 0.1


async def test_interrupt_when_called_does_soft_stop_without_disconnecting():
    client = FakeClient([SimpleNamespace(total_cost_usd=0.15)])
    driver = create_claude_driver(client_factory=FactoryProbe(client))
    events: list[SessionEvent] = []
    holder: dict[str, DriverSession] = {}

    holder["session"] = driver.start(make_prompt(), _interrupting_observer(events, holder))
    await holder["session"].outcome

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


async def test_abort_when_fired_does_disconnect_and_resolve_interrupted():
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


# ---------------------------------------------------------------------------
# outcome
# ---------------------------------------------------------------------------


async def test_outcome_when_stream_ends_normally_does_report_completed_with_last_cost():
    messages = [SimpleNamespace(total_cost_usd=0.02), SimpleNamespace(total_cost_usd=0.05)]
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient(messages)))

    outcome = await run_session(driver, collecting_observer().observer)

    assert outcome.reason == "completed"
    assert outcome.cost_usd == 0.05


async def test_outcome_when_no_results_does_report_completed_with_zero_cost():
    driver = create_claude_driver(client_factory=FactoryProbe(FakeClient([])))

    outcome = await run_session(driver, collecting_observer().observer)

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

    monkeypatch.setattr("gymrat_py.supervisor.claude._load_default_factory", raise_missing)
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
    assert outcome.message == message_of(error)
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
