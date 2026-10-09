"""Tests for how the Claude driver calls its hooks and tools factories.

The driver is exercised through an injected fake client (same pattern as
``test_claude.py``). Stub hooks and tools factories replace the real ones, so
these tests verify that each factory is called once per session start with the
session's context, and that the tools it builds are mounted on the client. The
hooks landing in the client options, and both factories together, are pinned
with the other forwarded options in ``test_claude.py``.
"""

import asyncio

import pytest

from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.driver import SessionPrompt
from tests.supervisor._fixtures import (
    _SENTINEL_SERVER,
    TRACEPARENT,
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


@pytest.mark.parametrize(
    ("prompt", "env"),
    [
        pytest.param(
            make_prompt(traceparent=TRACEPARENT),
            {"GYMRAT_TRACEPARENT": TRACEPARENT},
            id="traceparent",
        ),
        pytest.param(make_prompt(), {}, id="no-traceparent"),
    ],
)
async def test_start_when_tools_given_does_mount_them_built_from_the_session_context(
    prompt: SessionPrompt, env: dict[str, str]
):
    client = FiniteClient([result_message()])
    probe = ToolsFactoryProbe()
    abort = asyncio.Event()

    await run_outcome(client, prompt=prompt, abort=abort, tools=probe)

    assert probe.calls == [(abort, env)]
    assert client.options is not None
    assert client.options["mcp_servers"] == {"gymrat": _SENTINEL_SERVER}
