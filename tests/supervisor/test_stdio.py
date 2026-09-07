"""Behavioral tests for the subprocess stdio driver.

The driver spawns a child process and speaks a line-delimited JSON protocol over
its stdio: it writes a ``start`` command, relays the child's event lines to the
observer, and settles a :class:`SessionOutcome` from the child's terminal
``outcome`` line, its exit code, an interrupt, or an external abort.

The child is a scripted Python double (``_stdio_double.py``) invoked through
``sys.executable``; the whole module is POSIX-only because the abort path relies
on process-group tree-kill.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from gymrat.supervisor import create_stdio_driver
from gymrat.supervisor.driver import SessionOutcome
from gymrat.supervisor.events import (
    SessionEvent,
    TextDeltaEvent,
    ToolProgressEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)
from tests._cli import try_read_report
from tests._process_helpers import wait_until_dead
from tests.supervisor._fixtures import collecting_observer, make_prompt

_TEST_TIMEOUT_S = 15.0

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only process groups for tree-kill"),
    pytest.mark.filterwarnings("default::RuntimeWarning"),
]

_DOUBLE = str(Path(__file__).parent / "_stdio_double.py")


def double_argv(config: dict[str, Any]) -> list[str]:
    """Build the argv that runs the protocol double with ``config``."""
    return [sys.executable, _DOUBLE, json.dumps(config)]


async def wait_for_event(
    events: list[SessionEvent],
    event_type: type,
    timeout_s: float = _TEST_TIMEOUT_S,
) -> None:
    """Poll until an event of ``event_type`` has reached the observer's list."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not any(isinstance(event, event_type) for event in events):
        if loop.time() > deadline:
            message = f"no {event_type.__name__} arrived within {timeout_s}s"
            raise TimeoutError(message)
        await asyncio.sleep(0.02)


def resolved(path: str | Path) -> Path:
    """Resolve symlinks so a path compares equal to a spawned child's ``cwd``."""
    return Path(path).resolve()


async def read_report(report_path: Path, timeout_s: float = _TEST_TIMEOUT_S) -> dict[str, Any]:
    """Poll until ``report_path`` holds a complete JSON report, then return it."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        data = try_read_report(report_path)
        if data is not None:
            return data
        if loop.time() > deadline:
            message = f"report never appeared at {report_path}"
            raise TimeoutError(message)
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# Spawning and the start command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected_prompt"),
    [
        pytest.param(
            {},
            {"kickoff": "do the thing", "cwd": None},
            id="optionals-omitted",
        ),
        pytest.param(
            {"system_prompt_append": "extra", "model": "opus"},
            {
                "kickoff": "do the thing",
                "cwd": None,
                "systemPromptAppend": "extra",
                "model": "opus",
            },
            id="optionals-present",
        ),
        pytest.param(
            {"effort": "high", "command_timeout_ms": 300000},
            {
                "kickoff": "do the thing",
                "cwd": None,
                "effort": "high",
                "commandTimeoutMs": 300000,
            },
            id="effort-and-timeout-present",
        ),
    ],
)
async def test_stdio_driver_when_started_does_spawn_with_correct_start_line(
    tmp_path: Path,
    overrides: dict[str, Any],
    expected_prompt: dict[str, Any],
) -> None:
    report = tmp_path / "report.json"
    config = {
        "mode": "script",
        "report_path": str(report),
        "outcome": {"type": "outcome", "reason": "completed", "costUsd": 0.0},
    }
    prompt = make_prompt(cwd=str(tmp_path), **overrides)
    probe = collecting_observer()
    session = create_stdio_driver(double_argv(config)).start(prompt, probe.observer)

    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    report_data = await read_report(report)
    expected_prompt = {**expected_prompt, "cwd": str(tmp_path)}
    assert json.loads(report_data["start_line"]) == {"type": "start", "prompt": expected_prompt}
    assert resolved(report_data["cwd"]) == resolved(tmp_path)


# ---------------------------------------------------------------------------
# Relaying event lines
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_child_emits_lines_does_relay_typed_events(
    tmp_path: Path,
) -> None:
    config = {
        "mode": "script",
        "stderr": json.dumps({"type": "cap", "timestamp": 7, "cap": "spend-cap"}),
        "lines": [
            {"json": {"type": "usage_update", "timestamp": 6, "costUsd": 0.01}},
            {"text": "not json at all"},
            {"json": {"type": "text_delta", "timestamp": 5, "chunk": "hello"}},
            {"json": [1, 2, 3]},
            {"json": {"type": "mystery", "timestamp": 9}},
            {
                "json": {
                    "type": "tool_progress",
                    "timestamp": 3,
                    "toolUseId": "t1",
                    "elapsedMs": 500,
                }
            },
        ],
        "outcome": {"type": "outcome", "reason": "completed", "costUsd": 0.01},
    }
    probe = collecting_observer()
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), probe.observer
    )

    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert probe.events == [
        UsageUpdateEvent(timestamp=6, cost_usd=0.01),
        TextDeltaEvent(timestamp=5, chunk="hello"),
        ToolProgressEvent(timestamp=3, tool_use_id="t1", elapsed_ms=500),
    ]


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_outcome_line_received_does_settle_with_its_fields(
    tmp_path: Path,
) -> None:
    config = {
        "mode": "script",
        "lines": [{"json": {"type": "usage_update", "timestamp": 1, "costUsd": 0.2}}],
        "outcome": {
            "type": "outcome",
            "reason": "completed",
            "costUsd": 0.5,
            "message": "all done",
        },
    }
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert outcome == SessionOutcome(reason="completed", cost_usd=0.5, message="all done")


@pytest.mark.parametrize(
    "bool_cost", [pytest.param(True, id="true"), pytest.param(False, id="false")]
)
async def test_stdio_driver_when_outcome_cost_is_boolean_does_fall_back_to_running_cost(
    tmp_path: Path,
    bool_cost: bool,
) -> None:
    config = {
        "mode": "script",
        "lines": [{"json": {"type": "usage_update", "timestamp": 1, "costUsd": 0.42}}],
        "outcome": {
            "type": "outcome",
            "reason": "completed",
            "costUsd": bool_cost,
        },
    }
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert outcome.cost_usd == 0.42


async def test_stdio_driver_when_child_exits_without_outcome_does_error_with_exit_details(
    tmp_path: Path,
) -> None:
    config = {
        "mode": "script",
        "lines": [{"json": {"type": "usage_update", "timestamp": 1, "costUsd": 0.3}}],
        "outcome": None,
        "exit_code": 7,
    }
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert outcome.reason == "error"
    assert outcome.cost_usd == 0.3
    assert outcome.message is not None
    assert "7" in outcome.message


async def test_stdio_driver_when_child_cannot_spawn_does_settle_error_without_raising(
    tmp_path: Path,
) -> None:
    argv = [str(tmp_path / "does-not-exist")]
    session = create_stdio_driver(argv).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert outcome.reason == "error"
    assert outcome.message


# ---------------------------------------------------------------------------
# Interrupt
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_interrupt_then_child_exits_does_settle_interrupted_with_last_cost(
    tmp_path: Path,
) -> None:
    config = {
        "mode": "await_interrupt",
        "lines": [{"json": {"type": "usage_update", "timestamp": 1, "costUsd": 0.5}}],
    }
    probe = collecting_observer()
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), probe.observer
    )
    await wait_for_event(probe.events, UsageUpdateEvent)

    await session.interrupt()

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    assert outcome == SessionOutcome(reason="interrupted", cost_usd=0.5)


async def test_stdio_driver_when_interrupt_precedes_a_later_outcome_line_does_win(
    tmp_path: Path,
) -> None:
    config = {
        "mode": "await_interrupt",
        "lines": [{"json": {"type": "usage_update", "timestamp": 1, "costUsd": 0.7}}],
        "emit_outcome_on_interrupt": {"type": "outcome", "reason": "completed", "costUsd": 0.9},
    }
    probe = collecting_observer()
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), probe.observer
    )
    await wait_for_event(probe.events, UsageUpdateEvent)

    await session.interrupt()

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    assert outcome == SessionOutcome(reason="interrupted", cost_usd=0.7)


# ---------------------------------------------------------------------------
# Abort tree-kill
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_abort_fires_does_settle_interrupted(
    tmp_path: Path,
) -> None:
    report = tmp_path / "child-processes.json"
    config = {
        "mode": "sleep_forever",
        "report_path": str(report),
        "lines": [{"json": {"type": "usage_update", "timestamp": 1, "costUsd": 0.4}}],
    }
    abort = asyncio.Event()
    probe = collecting_observer()
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), probe.observer, abort
    )
    processes = await read_report(report)

    abort.set()

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    assert outcome == SessionOutcome(reason="interrupted", cost_usd=0.4)
    await wait_until_dead(int(processes["pid"]))
    await wait_until_dead(int(processes["grandchild"]))


# ---------------------------------------------------------------------------
# Turn-end events and running cost
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_child_emits_turn_end_does_relay_as_turn_end_event_and_update_cost(
    tmp_path: Path,
) -> None:
    config = {
        "mode": "script",
        "lines": [
            {"json": {"type": "usage_update", "timestamp": 1, "costUsd": 0.1}},
            {
                "json": {
                    "type": "turn_end",
                    "timestamp": 2,
                    "text": "done",
                    "costUsd": 0.25,
                    "origin": "agent",
                    "budgetExhausted": False,
                }
            },
        ],
        "outcome": {"type": "outcome", "reason": "completed", "costUsd": True},
    }
    probe = collecting_observer()
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), probe.observer
    )

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = [e for e in probe.events if isinstance(e, TurnEndEvent)]
    assert turn_ends == [
        TurnEndEvent(
            timestamp=2,
            text="done",
            cost_usd=0.25,
            origin="agent",
            budget_exhausted=False,
        )
    ]
    # costUsd=True in the outcome is boolean, so the driver falls back to the
    # running cost, which should be updated to 0.25 from the turn_end line.
    assert outcome.cost_usd == 0.25


# ---------------------------------------------------------------------------
# send / end protocol
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_send_and_end_does_write_protocol_lines_to_child_stdin(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    config = {
        "mode": "await_message",
        "report_path": str(report),
        "lines": [{"json": {"type": "usage_update", "timestamp": 1, "costUsd": 0.01}}],
        "turn_end": {
            "type": "turn_end",
            "timestamp": 2,
            "text": "hi",
            "costUsd": 0.01,
            "origin": "agent",
            "budgetExhausted": False,
        },
        "outcome": {"type": "outcome", "reason": "completed", "costUsd": 0.02},
    }
    probe = collecting_observer()
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), probe.observer
    )
    await wait_for_event(probe.events, TurnEndEvent)

    await session.send("follow-up text")
    await session.end()

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    assert outcome.reason == "completed"
    report_data = await read_report(report)
    assert report_data["message_text"] == "follow-up text"
    turn_ends = [e for e in probe.events if isinstance(e, TurnEndEvent)]
    assert len(turn_ends) == 2


# ---------------------------------------------------------------------------
# maxBudgetUsd in start command
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget", "expected_in_start"),
    [
        pytest.param(
            2.5,
            True,
            id="budget-present",
        ),
        pytest.param(
            None,
            False,
            id="budget-omitted",
        ),
    ],
)
async def test_stdio_driver_when_max_budget_usd_does_include_or_omit_in_start_command(
    tmp_path: Path,
    budget: float | None,
    expected_in_start: bool,
) -> None:
    report = tmp_path / "report.json"
    config = {
        "mode": "script",
        "report_path": str(report),
        "outcome": {"type": "outcome", "reason": "completed", "costUsd": 0.0},
    }
    prompt = make_prompt(cwd=str(tmp_path), max_budget_usd=budget)
    session = create_stdio_driver(double_argv(config)).start(prompt, collecting_observer().observer)

    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    report_data = await read_report(report)
    start_obj = json.loads(report_data["start_line"])
    if expected_in_start:
        assert start_obj["prompt"]["maxBudgetUsd"] == budget
    else:
        assert "maxBudgetUsd" not in start_obj["prompt"]
