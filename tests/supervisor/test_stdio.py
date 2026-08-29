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
    UsageUpdateEvent,
)
from tests._process_helpers import wait_until_dead
from tests.supervisor._fixtures import collecting_observer, make_prompt

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX-only process groups for tree-kill"
)

_DOUBLE = str(Path(__file__).parent / "_stdio_double.py")


def double_argv(config: dict[str, Any]) -> list[str]:
    """Build the argv that runs the protocol double with ``config``."""
    return [sys.executable, _DOUBLE, json.dumps(config)]


async def wait_for_event(
    events: list[SessionEvent],
    event_type: type,
    timeout_s: float = 5.0,
) -> None:
    """Poll until an event of ``event_type`` has reached the observer's list."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not any(isinstance(event, event_type) for event in events):
        if loop.time() > deadline:
            message = f"no {event_type.__name__} arrived within {timeout_s}s"
            raise TimeoutError(message)
        await asyncio.sleep(0.02)


def try_load_report(report_path: Path) -> dict[str, Any] | None:
    """Load the JSON report if it exists and is complete, else ``None``.

    Wrapped in a sync helper so the blocking filesystem read stays out of the
    async test body, where it would trip the async-blocking-call lint.
    """
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolved(path: str | Path) -> Path:
    """Resolve symlinks so a path compares equal to a spawned child's ``cwd``."""
    return Path(path).resolve()


async def read_report(report_path: Path, timeout_s: float = 5.0) -> dict[str, Any]:
    """Poll until ``report_path`` holds a complete JSON report, then return it."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        data = try_load_report(report_path)
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
    ],
)
async def test_stdio_driver_when_started_does_spawn_in_cwd_and_send_camelcase_start_line(
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

    await asyncio.wait_for(session.outcome, 5)

    report_data = await read_report(report)
    expected_prompt = {**expected_prompt, "cwd": str(tmp_path)}
    assert json.loads(report_data["start_line"]) == {"type": "start", "prompt": expected_prompt}
    assert resolved(report_data["cwd"]) == resolved(tmp_path)


# ---------------------------------------------------------------------------
# Relaying event lines
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_child_emits_lines_does_relay_events_and_ignore_noise(
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

    await asyncio.wait_for(session.outcome, 5)

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

    outcome = await asyncio.wait_for(session.outcome, 5)

    assert outcome == SessionOutcome(reason="completed", cost_usd=0.5, message="all done")


async def test_stdio_driver_when_child_exits_without_outcome_does_error_with_exit_code_and_last_cost(
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

    outcome = await asyncio.wait_for(session.outcome, 5)

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

    outcome = await asyncio.wait_for(session.outcome, 5)

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

    outcome = await asyncio.wait_for(session.outcome, 5)
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

    outcome = await asyncio.wait_for(session.outcome, 5)
    assert outcome == SessionOutcome(reason="interrupted", cost_usd=0.7)


# ---------------------------------------------------------------------------
# Abort tree-kill
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_abort_fires_does_kill_child_tree_and_settle_interrupted(
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

    outcome = await asyncio.wait_for(session.outcome, 5)
    assert outcome == SessionOutcome(reason="interrupted", cost_usd=0.4)
    await wait_until_dead(int(processes["pid"]))
    await wait_until_dead(int(processes["grandchild"]))
