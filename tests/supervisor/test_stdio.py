"""Behavioral tests for the subprocess stdio driver.

The driver spawns a child process and speaks a line-delimited JSON protocol over
its stdio: it writes a ``start`` command with snake_case keys, relays the child's
event lines to the observer, and settles a :class:`SessionOutcome` from the
child's terminal ``outcome`` line, its exit code, an interrupt, or an external
abort.

The child is a scripted Python double (``_stdio_double.py``) invoked through
``sys.executable``; the whole module is POSIX-only because the abort path relies
on process-group tree-kill.
"""

import asyncio
import errno
import json
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest

from gymrat import exec as gymrat_exec
from gymrat.process_group import TERMINATE_GRACE_S
from gymrat.supervisor import create_stdio_driver
from gymrat.supervisor.driver import DriverSession, SessionOutcome
from gymrat.supervisor.events import (
    SessionEvent,
    TextDeltaEvent,
    ToolProgressEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)
from tests._cli import try_read_report
from tests._process_helpers import (
    SLEEPER_ARGV,
    ZOMBIE_ONLY_GROUP_SCRIPT,
    capture_spawns,
    killpg_warnings,
    record_registry_sweep,
    refuse_resume,
    wait_for_pid_file,
    wait_until_dead,
)
from tests.supervisor._fixtures import collecting_observer, make_prompt

_TEST_TIMEOUT_S = 15.0

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only process groups for tree-kill"),
    pytest.mark.filterwarnings("default::RuntimeWarning"),
]

_DOUBLE = str(Path(__file__).parent / "_stdio_double.py")

# Sessions run back to back to hit the window where the child has written its
# outcome but is still exiting; a single run slips past it most of the time.
_RACE_SESSIONS = 20

# Upper bound on outcome-to-settle time for a child still running after its
# outcome: teardown kills it at once, so the bound only leaves margin for a
# loaded machine.
_PROMPT_TEARDOWN_S = 0.4

# Past the driver's 8 MiB line limit, with no newline, so the read overruns.
_OVERSIZED_LINE_BYTES = 12_000_000

_USAGE_LINE = {
    "json": {"type": "usage_update", "at": 1_000_000_000, "cost_usd": 0.4, "settled": False}
}

# A session child that reports, as its outcome message, the name of the signal
# its handler caught. argv: ``<pid-report-path> <signal-number>``.
_SIGNAL_REPORTER = """
import json, os, signal, sys, time
from pathlib import Path

def report(signal_number, _frame):
    name = signal.Signals(signal_number).name
    outcome = {"type": "outcome", "reason": "completed", "cost_usd": 0.0, "message": name}
    sys.stdout.write(json.dumps(outcome) + "\\n")
    sys.stdout.flush()
    os._exit(0)

signal.signal(int(sys.argv[2]), report)
sys.stdin.readline()
Path(sys.argv[1]).write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
time.sleep(30)
"""


def double_argv(config: dict[str, Any]) -> list[str]:
    """Return argv in the positional order the double's ``sys.argv`` parser expects."""
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


@pytest.fixture
def isolated_live_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start the test with an empty live-process registry, so it holds only this test's children."""
    monkeypatch.setattr(gymrat_exec, "_live_process_groups", set())


@pytest.fixture
def termination_signals_deferred() -> Iterator[None]:
    """Hold SIGINT and SIGTERM blocked in this thread, as a parent deferring them does."""
    deferred = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, deferred)
    yield
    signal.pthread_sigmask(signal.SIG_SETMASK, previous)


async def abort_session(_session: DriverSession, abort: asyncio.Event) -> None:
    """Settle the session through its abort event."""
    abort.set()


async def interrupt_session(session: DriverSession, _abort: asyncio.Event) -> None:
    """Settle the session through an interrupt command."""
    await session.interrupt()


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
# Spawning and the start command — snake_case wire
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
                "system_prompt_append": "extra",
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
                "command_timeout_ms": 300000,
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
        "outcome": {"type": "outcome", "reason": "completed", "cost_usd": 0.0},
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
# Relaying event lines — snake_case wire
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_child_emits_lines_does_relay_typed_events(
    tmp_path: Path,
) -> None:
    config = {
        "mode": "script",
        "stderr": json.dumps({"type": "cap", "at": 7_000_000_000, "cap": "spend-cap"}),
        "lines": [
            {
                "json": {
                    "type": "usage_update",
                    "at": 6_000_000_000,
                    "cost_usd": 0.01,
                    "settled": False,
                }
            },
            {"text": "not json at all"},
            {"json": {"type": "text_delta", "at": 5_000_000_000, "chunk": "hello"}},
            {"json": [1, 2, 3]},
            {"json": {"type": "mystery", "at": 9_000_000_000}},
            {
                "json": {
                    "type": "tool_progress",
                    "at": 3_000_000_000,
                    "tool_use_id": "t1",
                    "elapsed_ms": 500,
                }
            },
        ],
        "outcome": {"type": "outcome", "reason": "completed", "cost_usd": 0.01},
    }
    probe = collecting_observer()
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), probe.observer
    )

    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert probe.events == [
        UsageUpdateEvent(at=6_000_000_000, cost_usd=0.01),
        TextDeltaEvent(at=5_000_000_000, chunk="hello"),
        ToolProgressEvent(at=3_000_000_000, tool_use_id="t1", elapsed_ms=500),
    ]


# ---------------------------------------------------------------------------
# Resolution — snake_case outcome wire
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_outcome_line_received_does_settle_with_its_fields(
    tmp_path: Path,
) -> None:
    config = {
        "mode": "script",
        "lines": [
            {
                "json": {
                    "type": "usage_update",
                    "at": 1_000_000_000,
                    "cost_usd": 0.2,
                    "settled": False,
                }
            }
        ],
        "outcome": {
            "type": "outcome",
            "reason": "completed",
            "cost_usd": 0.5,
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
        "lines": [
            {
                "json": {
                    "type": "usage_update",
                    "at": 1_000_000_000,
                    "cost_usd": 0.42,
                    "settled": False,
                }
            }
        ],
        "outcome": {
            "type": "outcome",
            "reason": "completed",
            "cost_usd": bool_cost,
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
        "lines": [
            {
                "json": {
                    "type": "usage_update",
                    "at": 1_000_000_000,
                    "cost_usd": 0.3,
                    "settled": False,
                }
            }
        ],
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


@pytest.mark.parametrize(
    ("argv", "cwd", "expected_fragment"),
    [
        pytest.param(None, None, "No such file or directory", id="does-not-exist"),
        pytest.param([sys.executable, "nu\x00l"], None, "embedded null byte", id="nul-in-argv"),
        pytest.param(
            [sys.executable, "-c", "pass"], "nu\x00l", "embedded null byte", id="nul-in-cwd"
        ),
    ],
)
async def test_stdio_driver_when_spawn_rejected_before_any_process_does_settle_error_with_reason(
    tmp_path: Path,
    argv: list[str] | None,
    cwd: str | None,
    expected_fragment: str,
) -> None:
    if argv is None:
        argv = [str(tmp_path / "does-not-exist")]
    prompt = make_prompt(cwd=str(tmp_path) if cwd is None else cwd)
    session = create_stdio_driver(argv).start(prompt, collecting_observer().observer)

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert outcome.reason == "error"
    assert expected_fragment in (outcome.message or "")


# ---------------------------------------------------------------------------
# Interrupt
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_interrupt_then_child_exits_does_settle_interrupted_with_last_cost(
    tmp_path: Path,
) -> None:
    config = {
        "mode": "await_interrupt",
        "lines": [
            {
                "json": {
                    "type": "usage_update",
                    "at": 1_000_000_000,
                    "cost_usd": 0.5,
                    "settled": False,
                }
            }
        ],
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
        "lines": [
            {
                "json": {
                    "type": "usage_update",
                    "at": 1_000_000_000,
                    "cost_usd": 0.7,
                    "settled": False,
                }
            }
        ],
        "emit_outcome_on_interrupt": {"type": "outcome", "reason": "completed", "cost_usd": 0.9},
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
        "lines": [
            {
                "json": {
                    "type": "usage_update",
                    "at": 1_000_000_000,
                    "cost_usd": 0.4,
                    "settled": False,
                }
            }
        ],
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
# Teardown of the child's process group
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_child_exits_after_outcome_does_not_warn_about_killpg(
    tmp_path: Path,
    recwarn: pytest.WarningsRecorder,
) -> None:
    config = {
        "mode": "script",
        "exit_immediately": True,
        "outcome": {"type": "outcome", "reason": "completed", "cost_usd": 0.0},
    }
    driver = create_stdio_driver(double_argv(config))

    # Repeating the session is the scenario: each run is one more chance for
    # teardown to land while the child is still exiting after its outcome line.
    for _ in range(_RACE_SESSIONS):
        await asyncio.wait_for(
            driver.start(make_prompt(cwd=str(tmp_path)), collecting_observer().observer).outcome,
            _TEST_TIMEOUT_S,
        )

    assert killpg_warnings(recwarn) == []


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(
            {"type": "outcome", "reason": "completed", "cost_usd": 0.0},
            id="leader-exits-after-outcome",
        ),
        pytest.param(None, id="leader-exits-without-outcome"),
    ],
)
async def test_stdio_driver_when_grandchild_outlives_leader_does_kill_grandchild(
    tmp_path: Path,
    stray_process_ids: list[int],
    outcome: dict[str, Any] | None,
) -> None:
    report = tmp_path / "report.json"
    config = {
        "mode": "script",
        "report_path": str(report),
        "spawn_grandchild": True,
        "outcome": outcome,
    }
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )
    grandchild = int((await read_report(report))["grandchild"])
    stray_process_ids.append(grandchild)

    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    await wait_until_dead(grandchild)


async def test_stdio_driver_when_child_lingers_after_outcome_does_kill_it_promptly(
    tmp_path: Path,
    stray_process_ids: list[int],
) -> None:
    report = tmp_path / "report.json"
    config = {
        "mode": "script",
        "report_path": str(report),
        "linger_after_outcome": True,
        "lines": [
            {
                "json": {
                    "type": "usage_update",
                    "at": 1_000_000_000,
                    "cost_usd": 0.1,
                    "settled": False,
                }
            }
        ],
        "outcome": {"type": "outcome", "reason": "completed", "cost_usd": 0.1},
    }
    loop = asyncio.get_running_loop()
    # The usage line is written right before the outcome line, so its arrival
    # time stands in for the moment the driver receives the outcome.
    usage_seen_at: list[float] = []

    def observer(event: SessionEvent) -> None:
        if isinstance(event, UsageUpdateEvent):
            usage_seen_at.append(loop.time())

    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), observer
    )
    child = int((await read_report(report))["pid"])
    stray_process_ids.append(child)

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    settled_at = loop.time()

    assert outcome.reason == "completed"
    assert settled_at - usage_seen_at[0] < _PROMPT_TEARDOWN_S
    await wait_until_dead(child)


async def test_stdio_driver_when_torn_down_leaving_only_a_zombie_in_group_does_not_warn_about_killpg(
    tmp_path: Path,
    stray_process_ids: list[int],
    recwarn: pytest.WarningsRecorder,
) -> None:
    holder_pid_file = tmp_path / "holder.pid"
    abort = asyncio.Event()
    session = create_stdio_driver([
        sys.executable,
        "-c",
        ZOMBIE_ONLY_GROUP_SCRIPT,
        str(holder_pid_file),
    ]).start(make_prompt(cwd=str(tmp_path)), collecting_observer().observer, abort)
    stray_process_ids.append(await wait_for_pid_file(holder_pid_file))

    abort.set()

    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    assert killpg_warnings(recwarn) == []


# ---------------------------------------------------------------------------
# Turn-end events and running cost
# ---------------------------------------------------------------------------


async def test_stdio_driver_when_child_emits_turn_end_does_relay_as_turn_end_event_and_update_cost(
    tmp_path: Path,
) -> None:
    config = {
        "mode": "script",
        "lines": [
            {
                "json": {
                    "type": "usage_update",
                    "at": 1_000_000_000,
                    "cost_usd": 0.1,
                    "settled": False,
                }
            },
            {
                "json": {
                    "type": "turn_end",
                    "at": 2_000_000_000,
                    "text": "done",
                    "cost_usd": 0.25,
                    "origin": "agent",
                    "budget_exhausted": False,
                }
            },
        ],
        "outcome": {"type": "outcome", "reason": "completed", "cost_usd": True},
    }
    probe = collecting_observer()
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), probe.observer
    )

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    turn_ends = [e for e in probe.events if isinstance(e, TurnEndEvent)]
    assert turn_ends == [
        TurnEndEvent(
            at=2_000_000_000,
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
        "lines": [
            {
                "json": {
                    "type": "usage_update",
                    "at": 1_000_000_000,
                    "cost_usd": 0.01,
                    "settled": False,
                }
            }
        ],
        "turn_end": {
            "type": "turn_end",
            "at": 2_000_000_000,
            "text": "hi",
            "cost_usd": 0.01,
            "origin": "agent",
            "budget_exhausted": False,
        },
        "outcome": {"type": "outcome", "reason": "completed", "cost_usd": 0.02},
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
# max_budget_usd in start command — snake_case
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
        "outcome": {"type": "outcome", "reason": "completed", "cost_usd": 0.0},
    }
    prompt = make_prompt(cwd=str(tmp_path), max_budget_usd=budget)
    session = create_stdio_driver(double_argv(config)).start(prompt, collecting_observer().observer)

    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    report_data = await read_report(report)
    start_obj = json.loads(report_data["start_line"])
    if expected_in_start:
        assert start_obj["prompt"]["max_budget_usd"] == budget
    else:
        assert "max_budget_usd" not in start_obj["prompt"]


# ---------------------------------------------------------------------------
# traceparent in start command — snake_case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("traceparent", "expected_in_start"),
    [
        pytest.param("00-abc123-def456-01", True, id="traceparent-present"),
        pytest.param(None, False, id="traceparent-omitted"),
    ],
)
async def test_stdio_driver_when_traceparent_does_include_or_omit_in_start_command(
    tmp_path: Path,
    traceparent: str | None,
    expected_in_start: bool,
) -> None:
    report = tmp_path / "report.json"
    config = {
        "mode": "script",
        "report_path": str(report),
        "outcome": {"type": "outcome", "reason": "completed", "cost_usd": 0.0},
    }
    prompt = make_prompt(cwd=str(tmp_path), traceparent=traceparent)
    session = create_stdio_driver(double_argv(config)).start(prompt, collecting_observer().observer)

    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    report_data = await read_report(report)
    start_obj = json.loads(report_data["start_line"])
    if expected_in_start:
        assert start_obj["prompt"]["traceparent"] == traceparent
    else:
        assert "traceparent" not in start_obj["prompt"]


# ---------------------------------------------------------------------------
# Live-process registry — a termination signal reaches the session child
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("isolated_live_groups")
async def test_kill_live_process_groups_when_stdio_session_live_does_kill_child_tree(
    tmp_path: Path,
    stray_process_ids: list[int],
) -> None:
    report = tmp_path / "child-processes.json"
    config = {"mode": "sleep_forever", "report_path": str(report)}
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )
    processes = await read_report(report)
    child = int(processes["pid"])
    grandchild = int(processes["grandchild"])
    stray_process_ids.extend([child, grandchild])

    gymrat_exec.kill_live_process_groups()

    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    await wait_until_dead(child)
    await wait_until_dead(grandchild)


@pytest.mark.usefixtures("isolated_live_groups")
async def test_kill_live_process_groups_when_stdio_child_exits_on_terminate_does_settle_exited_leaders_silently_without_reaping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stray_process_ids: list[int],
    recwarn: pytest.WarningsRecorder,
) -> None:
    spawned = capture_spawns(monkeypatch, "create_subprocess_exec")
    sweep_times: list[float] = []
    # The zombie leader the sweep trips over only exists until the loop reaps
    # it, so the race is run several times over.
    for attempt in range(_RACE_SESSIONS):
        report = tmp_path / f"child-processes-{attempt}.json"
        config = {"mode": "sleep_forever", "report_path": str(report)}
        session = create_stdio_driver(double_argv(config)).start(
            make_prompt(cwd=str(tmp_path)), collecting_observer().observer
        )
        processes = await read_report(report)
        stray_process_ids.extend([int(processes["pid"]), int(processes["grandchild"])])
        started = time.monotonic()

        gymrat_exec.kill_live_process_groups()

        sweep_times.append(time.monotonic() - started)
        await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)

    assert killpg_warnings(recwarn) == []
    assert max(sweep_times) < TERMINATE_GRACE_S, "the sweep waited out a grace for exited leaders"
    assert {proc.returncode for proc in spawned} == {-signal.SIGTERM}, (
        "the sweep reaped a child before its event loop could collect the exit status"
    )


@pytest.mark.parametrize(
    ("seam", "error"),
    [
        pytest.param(
            "attach_process_group",
            RuntimeWarning("containment refused"),
            id="attach-warning-escalated-to-error",
        ),
        pytest.param(
            "resume_process_group",
            OSError(errno.EPERM, "containment refused"),
            id="resume-os-error",
        ),
    ],
)
@pytest.mark.usefixtures("isolated_live_groups")
async def test_stdio_driver_when_containment_raises_after_spawn_does_settle_error_after_reaping_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stray_process_ids: list[int],
    seam: str,
    error: Exception,
) -> None:
    def fail(_pid: int) -> NoReturn:
        raise error

    spawned = capture_spawns(monkeypatch, "create_subprocess_exec")
    monkeypatch.setattr(gymrat_exec, seam, fail)
    session = create_stdio_driver(list(SLEEPER_ARGV)).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    stray_process_ids.extend(proc.pid for proc in spawned)

    assert outcome == SessionOutcome(reason="error", cost_usd=0.0, message=str(error))
    assert spawned[0].returncode is not None, (
        "the child left behind by the failed spawn was never reaped"
    )
    attempted = record_registry_sweep(monkeypatch)
    gymrat_exec.kill_live_process_groups()
    assert attempted == []


@pytest.mark.parametrize(
    ("argv", "resume"),
    [
        pytest.param(
            double_argv({
                "mode": "script",
                "outcome": {"type": "outcome", "reason": "completed", "cost_usd": 0.0},
            }),
            gymrat_exec.resume_process_group,
            id="outcome-line",
        ),
        pytest.param(
            double_argv({"mode": "script", "outcome": None, "exit_code": 3}),
            gymrat_exec.resume_process_group,
            id="exit-without-outcome",
        ),
        pytest.param(
            [sys.executable, "-c", f"import sys; sys.stdout.write('x' * {_OVERSIZED_LINE_BYTES})"],
            gymrat_exec.resume_process_group,
            id="read-limit-error",
        ),
        pytest.param(
            list(SLEEPER_ARGV),
            refuse_resume,
            id="resume-refused",
        ),
    ],
)
@pytest.mark.usefixtures("isolated_live_groups")
async def test_kill_live_process_groups_when_stdio_session_settled_does_signal_nothing_for_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    resume: Callable[[int], bool],
) -> None:
    monkeypatch.setattr(gymrat_exec, "resume_process_group", resume)
    session = create_stdio_driver(argv).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    attempted = record_registry_sweep(monkeypatch)

    gymrat_exec.kill_live_process_groups()

    assert attempted == [], "a settled session left its child in the live-process registry"


@pytest.mark.parametrize(
    "settle",
    [
        pytest.param(abort_session, id="abort"),
        pytest.param(interrupt_session, id="interrupt"),
    ],
)
@pytest.mark.usefixtures("isolated_live_groups")
async def test_kill_live_process_groups_when_stdio_session_torn_down_does_signal_nothing_for_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settle: Callable[[DriverSession, asyncio.Event], Awaitable[None]],
) -> None:
    abort = asyncio.Event()
    probe = collecting_observer()
    config = {"mode": "await_interrupt", "lines": [_USAGE_LINE]}
    session = create_stdio_driver(double_argv(config)).start(
        make_prompt(cwd=str(tmp_path)), probe.observer, abort
    )
    await wait_for_event(probe.events, UsageUpdateEvent)
    await settle(session, abort)
    await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    attempted = record_registry_sweep(monkeypatch)

    gymrat_exec.kill_live_process_groups()

    assert attempted == [], "a torn-down session left its child in the live-process registry"


# ---------------------------------------------------------------------------
# Signal mask — the child is not born with termination signals blocked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "signal_number",
    [pytest.param(signal.SIGINT, id="SIGINT"), pytest.param(signal.SIGTERM, id="SIGTERM")],
)
@pytest.mark.usefixtures("termination_signals_deferred")
async def test_stdio_driver_when_spawned_while_signals_deferred_does_leave_them_unblocked_in_child(
    tmp_path: Path,
    signal_number: signal.Signals,
) -> None:
    report = tmp_path / "report.json"
    argv = [sys.executable, "-c", _SIGNAL_REPORTER, str(report), str(int(signal_number))]
    session = create_stdio_driver(argv).start(
        make_prompt(cwd=str(tmp_path)), collecting_observer().observer
    )
    child = int((await read_report(report))["pid"])

    os.kill(child, signal_number)

    outcome = await asyncio.wait_for(session.outcome, _TEST_TIMEOUT_S)
    assert outcome == SessionOutcome(reason="completed", cost_usd=0.0, message=signal_number.name)
