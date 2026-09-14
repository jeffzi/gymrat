"""Tests for the tools-factory wiring in the Claude driver.

The driver is exercised through an injected fake client (same pattern as
``test_claude.py``).  A stub tools factory replaces the real one so the tests
verify only the wiring: that the factory is called with the right arguments
and its return value lands in the options dict under ``mcp_servers``.
"""

import asyncio
from collections.abc import Mapping

from gymrat.supervisor import create_claude_driver
from gymrat.supervisor.driver import SessionPrompt
from tests.supervisor._fixtures import (
    FactoryProbe,
    FiniteClient,
    collecting_observer,
    make_prompt,
    result_message,
)


class ToolsFactoryProbe:
    """A stub tools factory that records each call and returns a sentinel."""

    def __init__(self, sentinel: object) -> None:
        self._sentinel = sentinel
        self.calls: list[tuple[asyncio.Event, Mapping[str, str]]] = []

    def __call__(self, abort: asyncio.Event, env: Mapping[str, str]) -> object:
        self.calls.append((abort, env))
        return self._sentinel


_SENTINEL_SERVER: dict[str, str] = {"type": "stdio", "command": "fake"}


async def _start_with_tools(
    prompt: SessionPrompt,
    probe: ToolsFactoryProbe,
    abort: asyncio.Event | None = None,
) -> tuple[FiniteClient, ToolsFactoryProbe]:
    """Start a session with ``probe`` as the tools factory and return (client, probe)."""
    client = FiniteClient([result_message()])
    driver = create_claude_driver(client_factory=FactoryProbe(client), tools=probe)
    await asyncio.wait_for(
        driver.start(prompt, collecting_observer().observer, abort).outcome,
        timeout=30.0,
    )
    return client, probe


async def test_start_when_tools_given_does_add_only_mcp_servers_to_options():
    prompt = make_prompt()
    plain_client = FiniteClient([result_message()])
    plain_driver = create_claude_driver(client_factory=FactoryProbe(plain_client))
    await asyncio.wait_for(
        plain_driver.start(prompt, collecting_observer().observer, None).outcome,
        timeout=30.0,
    )
    tools_probe = ToolsFactoryProbe(_SENTINEL_SERVER)

    client, _ = await _start_with_tools(prompt, tools_probe)

    assert plain_client.options is not None
    assert client.options == {**plain_client.options, "mcp_servers": {"gymrat": _SENTINEL_SERVER}}


async def test_start_when_tools_given_does_call_factory_exactly_once():
    tools_probe = ToolsFactoryProbe(_SENTINEL_SERVER)

    _, probe = await _start_with_tools(make_prompt(), tools_probe)

    assert len(probe.calls) == 1


async def test_start_when_tools_given_with_traceparent_does_pass_traceparent_env():
    tools_probe = ToolsFactoryProbe(_SENTINEL_SERVER)
    tp = "00-abc123-def456-01"

    _, probe = await _start_with_tools(make_prompt(traceparent=tp), tools_probe)

    assert len(probe.calls) == 1
    _, env = probe.calls[0]
    assert env == {"GYMRAT_TRACEPARENT": tp}


async def test_start_when_tools_given_without_traceparent_does_pass_empty_env():
    tools_probe = ToolsFactoryProbe(_SENTINEL_SERVER)

    _, probe = await _start_with_tools(make_prompt(), tools_probe)

    assert len(probe.calls) == 1
    _, env = probe.calls[0]
    assert env == {}


async def test_start_when_tools_given_with_abort_does_pass_abort_to_factory():
    tools_probe = ToolsFactoryProbe(_SENTINEL_SERVER)
    abort = asyncio.Event()

    _, probe = await _start_with_tools(make_prompt(), tools_probe, abort=abort)

    assert len(probe.calls) == 1
    received_abort, _ = probe.calls[0]
    assert received_abort is abort


async def test_start_when_tools_given_without_abort_does_pass_fresh_unset_event():
    tools_probe = ToolsFactoryProbe(_SENTINEL_SERVER)

    _, probe = await _start_with_tools(make_prompt(), tools_probe, abort=None)

    assert len(probe.calls) == 1
    received_abort, _ = probe.calls[0]
    assert isinstance(received_abort, asyncio.Event)
    assert not received_abort.is_set()
