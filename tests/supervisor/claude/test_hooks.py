"""Tests for how the Claude driver calls its hooks and tools factories.

The driver is exercised through an injected fake client (same pattern as
``test_claude.py``). Stub hooks and tools factories replace the real ones, so
these tests verify only that each factory is called once per session start
with the session's context. Where their return values land in the client
options is pinned with the other options in ``test_claude.py``.
"""

import asyncio

import pytest

from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.driver import SessionPrompt
from tests.supervisor._fixtures import (
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
    probe = ToolsFactoryProbe()
    abort = asyncio.Event()

    await run_outcome(FiniteClient([result_message()]), prompt=prompt, abort=abort, tools=probe)

    assert probe.calls == [(abort, env)]
