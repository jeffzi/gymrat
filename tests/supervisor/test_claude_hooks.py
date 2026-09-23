"""Tests for the hooks-factory wiring in the Claude driver.

The driver is exercised through an injected fake client (same pattern as
``test_claude_tools.py``). A stub hooks factory replaces the real one so most
tests verify only the wiring: that the factory is called once per session start
and its return value lands in the options dict under ``hooks``.
"""

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import HookEvent

from gymrat.supervisor import HooksFactory, create_claude_driver, supervise_hooks_factory
from gymrat.supervisor.driver import SessionOutcome
from gymrat.supervisor.tools import ToolsFactory
from tests.supervisor._fixtures import (
    FactoryProbe,
    FiniteClient,
    collecting_observer,
    make_prompt,
    result_message,
)

if TYPE_CHECKING:
    from gymrat.supervisor.claude import ClientFactory

_SENTINEL_HOOKS: dict[HookEvent, list[HookMatcher]] = {
    "PreToolUse": [HookMatcher(matcher="Bash", hooks=[])],
}
_SENTINEL_SERVER: dict[str, str] = {"type": "stdio", "command": "fake"}


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
    return await asyncio.wait_for(
        driver.start(make_prompt(), collecting_observer().observer, None).outcome,
        timeout=30.0,
    )


async def _plain_options() -> dict[str, object]:
    """The options a driver built without hooks or tools hands its client factory."""
    client = FiniteClient([result_message()])
    await _run_session(client)
    assert client.options is not None
    return client.options


async def test_start_when_hooks_given_does_add_only_hooks_to_options():
    plain = await _plain_options()
    client = FiniteClient([result_message()])

    await _run_session(client, hooks=HooksFactoryProbe())

    assert client.options == {**plain, "hooks": _SENTINEL_HOOKS}


@pytest.mark.parametrize("session_count", [1, 2])
async def test_start_when_hooks_given_does_call_factory_once_per_session(session_count: int):
    probe = HooksFactoryProbe()
    driver = create_claude_driver(
        client_factory=lambda _options: FiniteClient([result_message()]),
        hooks=probe,
    )

    for _ in range(session_count):
        await asyncio.wait_for(
            driver.start(make_prompt(), collecting_observer().observer, None).outcome,
            timeout=30.0,
        )

    assert probe.calls == session_count


async def test_start_when_hooks_factory_raises_does_settle_error_with_message():
    error = RuntimeError("hooks unavailable")

    def failing_hooks() -> dict[HookEvent, list[HookMatcher]]:
        raise error

    outcome = await _run_session(FiniteClient([result_message()]), hooks=failing_hooks)

    assert outcome.reason == "error"
    assert outcome.message == str(error)


async def test_start_when_hooks_and_tools_given_does_add_both_to_options():
    plain = await _plain_options()
    client = FiniteClient([result_message()])

    await _run_session(client, hooks=HooksFactoryProbe(), tools=_sentinel_tools)

    assert client.options == {
        **plain,
        "hooks": _SENTINEL_HOOKS,
        "mcp_servers": {"gymrat": _SENTINEL_SERVER},
    }
    assert client.options is not None
    assert client.options["permission_mode"] == "bypassPermissions"


async def test_start_when_supervise_hooks_factory_given_does_register_pre_tool_use(
    tmp_path: Path,
):
    client = FiniteClient([result_message()])

    await _run_session(client, hooks=supervise_hooks_factory(tmp_path))

    assert client.options is not None
    hooks = client.options["hooks"]
    assert isinstance(hooks, dict)
    assert "PreToolUse" in hooks
