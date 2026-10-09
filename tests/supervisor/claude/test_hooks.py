"""Tests for how the Claude driver calls its hooks and tools factories.

The driver is exercised through an injected fake client (same pattern as
``test_claude.py``). Stub hooks and tools factories replace the real ones, so
these tests verify that each factory is called once per session start with the
session's context, and that what it builds lands in the client options.
"""

import asyncio

import pytest

from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.driver import SessionPrompt
from tests.supervisor._fixtures import (
    _SENTINEL_HOOKS,
    _SENTINEL_SERVER,
    FiniteClient,
    HooksFactoryProbe,
    ToolsFactoryProbe,
    collecting_observer,
    make_prompt,
    result_message,
    run_outcome,
    run_session,
)


async def test_start_when_hooks_given_does_call_factory_once_per_session():
    probe = HooksFactoryProbe()
    driver = create_claude_driver(
        client_factory=lambda _options: FiniteClient([result_message()]),
        hooks=probe,
    )

    await run_session(driver, collecting_observer().observer)
    await run_session(driver, collecting_observer().observer)

    assert probe.calls == 2


_TRACEPARENT = "00-abc123-def456-01"


def _factory_options(client: FiniteClient) -> dict[str, object]:
    """The hooks and MCP servers the driver handed the client, ``None`` when absent."""
    options = client.options or {}
    return {key: options.get(key) for key in ("hooks", "mcp_servers")}


@pytest.mark.parametrize(
    ("prompt", "env"),
    [
        pytest.param(
            make_prompt(traceparent=_TRACEPARENT),
            {"GYMRAT_TRACEPARENT": _TRACEPARENT},
            id="traceparent",
        ),
        pytest.param(make_prompt(), {}, id="no-traceparent"),
    ],
)
async def test_start_when_tools_given_does_mount_the_server_built_from_the_session_context(
    prompt: SessionPrompt, env: dict[str, str]
):
    probe = ToolsFactoryProbe()
    abort = asyncio.Event()
    client = FiniteClient([result_message()])

    await run_outcome(client, prompt=prompt, abort=abort, tools=probe)

    assert (probe.calls, _factory_options(client)) == (
        [(abort, env)],
        {"hooks": None, "mcp_servers": {"gymrat": _SENTINEL_SERVER}},
    )


@pytest.mark.parametrize(
    ("tools_cls", "expected"),
    [
        pytest.param(None, {"hooks": _SENTINEL_HOOKS, "mcp_servers": None}, id="hooks-only"),
        pytest.param(
            ToolsFactoryProbe,
            {"hooks": _SENTINEL_HOOKS, "mcp_servers": {"gymrat": _SENTINEL_SERVER}},
            id="hooks-and-tools",
        ),
    ],
)
async def test_start_when_hooks_given_does_forward_the_built_hooks_as_client_options(
    tools_cls: type[ToolsFactoryProbe] | None, expected: dict[str, object]
):
    client = FiniteClient([result_message()])
    tools = tools_cls() if tools_cls is not None else None

    await run_outcome(client, prompt=make_prompt(), hooks=HooksFactoryProbe(), tools=tools)

    assert _factory_options(client) == expected
