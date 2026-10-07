"""Tests for the hooks-factory and tools-factory wiring in the Claude driver.

The driver is exercised through an injected fake client (same pattern as
``test_claude.py``). Stub hooks and tools factories replace the real ones so
most tests verify only the wiring: that each factory is called with the right
arguments once per session start and its return value lands in the options
dict under ``hooks`` or ``mcp_servers``.
"""

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING

import pytest
from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import HookEvent

from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.driver import SessionOutcome, SessionPrompt
from gymrat.supervisor.hooks import HooksFactory
from gymrat.supervisor.tools import ToolsFactory
from tests.supervisor._fixtures import (
    _SENTINEL_SERVER,
    FactoryProbe,
    FiniteClient,
    collecting_observer,
    make_prompt,
    result_message,
    run_session,
)

if TYPE_CHECKING:
    from gymrat.supervisor.claude import ClientFactory


_SENTINEL_HOOKS: dict[HookEvent, list[HookMatcher]] = {
    "PreToolUse": [HookMatcher(matcher="Bash", hooks=[])],
}


class HooksFactoryProbe:
    """A stub hooks factory that counts its calls and returns a sentinel mapping."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> dict[HookEvent, list[HookMatcher]]:
        self.calls += 1
        return _SENTINEL_HOOKS


def _sentinel_tools(abort: asyncio.Event, env: Mapping[str, str]) -> object:
    return _SENTINEL_SERVER


async def _run_session(
    client: FiniteClient,
    *,
    hooks: HooksFactory | None = None,
    tools: ToolsFactory | None = None,
) -> SessionOutcome:
    """Start one session against ``client`` with the given factories and await its outcome."""
    client_factory: ClientFactory = FactoryProbe(client)
    driver = create_claude_driver(client_factory=client_factory, hooks=hooks, tools=tools)
    return await run_session(driver, collecting_observer().observer)


async def _plain_options() -> dict[str, object]:
    """The options a driver built without hooks or tools hands its client factory."""
    client = FiniteClient([result_message()])
    await _run_session(client)
    return dict(client.options or {})


@pytest.mark.parametrize(
    ("hooks", "tools", "added"),
    [
        pytest.param(HooksFactoryProbe(), None, {"hooks": _SENTINEL_HOOKS}, id="hooks-only"),
        pytest.param(
            None, _sentinel_tools, {"mcp_servers": {"gymrat": _SENTINEL_SERVER}}, id="tools-only"
        ),
        pytest.param(
            HooksFactoryProbe(),
            _sentinel_tools,
            {"hooks": _SENTINEL_HOOKS, "mcp_servers": {"gymrat": _SENTINEL_SERVER}},
            id="hooks-and-tools",
        ),
    ],
)
async def test_start_when_factories_given_does_add_only_their_options(
    hooks: HooksFactory | None, tools: ToolsFactory | None, added: dict[str, object]
):
    plain = await _plain_options()
    client = FiniteClient([result_message()])

    await _run_session(client, hooks=hooks, tools=tools)

    assert client.options == {**plain, **added}


@pytest.mark.parametrize("session_count", [1, 2])
async def test_start_when_hooks_given_does_call_factory_once_per_session(session_count: int):
    probe = HooksFactoryProbe()
    driver = create_claude_driver(
        client_factory=lambda _options: FiniteClient([result_message()]),
        hooks=probe,
    )

    for _ in range(session_count):
        await run_session(driver, collecting_observer().observer)

    assert probe.calls == session_count


async def test_start_when_hooks_factory_raises_does_settle_error_with_message():
    error = RuntimeError("hooks unavailable")

    def failing_hooks() -> dict[HookEvent, list[HookMatcher]]:
        raise error

    outcome = await _run_session(FiniteClient([result_message()]), hooks=failing_hooks)

    assert outcome.reason == "error"
    assert outcome.message == str(error)


class ToolsFactoryProbe:
    """A stub tools factory that records each call and returns a sentinel."""

    def __init__(self, sentinel: object) -> None:
        self._sentinel = sentinel
        self.calls: list[tuple[asyncio.Event, Mapping[str, str]]] = []

    def __call__(self, abort: asyncio.Event, env: Mapping[str, str]) -> object:
        self.calls.append((abort, env))
        return self._sentinel


async def _start_with_tools(
    prompt: SessionPrompt, probe: ToolsFactoryProbe, abort: asyncio.Event
) -> None:
    """Run one session with ``probe`` as the tools factory and ``abort`` as its abort event."""
    client = FiniteClient([result_message()])
    driver = create_claude_driver(client_factory=FactoryProbe(client), tools=probe)
    await run_session(driver, collecting_observer().observer, prompt, abort)


@pytest.mark.parametrize(
    ("prompt", "env"),
    [
        pytest.param(
            make_prompt(traceparent="00-abc123-def456-01"),
            {"GYMRAT_TRACEPARENT": "00-abc123-def456-01"},
            id="traceparent",
        ),
        pytest.param(make_prompt(), {}, id="no-traceparent"),
    ],
)
async def test_start_when_tools_given_does_call_the_factory_with_the_session_context(
    prompt: SessionPrompt, env: dict[str, str]
):
    probe = ToolsFactoryProbe(_SENTINEL_SERVER)
    abort = asyncio.Event()

    await _start_with_tools(prompt, probe, abort)

    assert probe.calls == [(abort, env)]
