"""Behavioral tests for supervision ending on conditions read off the session log.

``supervise`` scans the session log on every ``ToolEndEvent`` whose log has
grown, and again at the turn-end settle, ending the run itself when a hook
fails or a stop condition becomes met. These tests drive the mock driver
through tool ends, append records to the scratch session log between steps, and
inject ``is_lock_held`` to defer an end while a command the agent left running
holds the repository lock.

A step that must never run once an end fires carries a long ``delay_ms``: the
mock's delay races the abort, so a run that ends returns at once, while a run
that misses the end sits out the delay and reports a plain session ending.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import pytest

from gymrat.config import BenchlessConfig, StopConfig
from gymrat.session.clock import now_ms, now_ns
from gymrat.session.paths import session_dir, session_jsonl_path
from gymrat.session.store import append_record
from gymrat.supervisor import supervise
from gymrat.supervisor.events import (
    CapEvent,
    FollowUpEvent,
    SessionEvent,
    TextDeltaEvent,
    ToolEndEvent,
)
from gymrat.supervisor.supervise import EndedBy
from tests.session.records._fixtures import (
    command_record,
    hook_record,
    iteration_record,
    session_record,
)
from tests.supervisor._fixtures import (
    InterruptEmitsEndDriver,
    collecting_observer,
    emit_turn_end,
    follow_ups_with_action,
    make_context,
    make_launch,
    make_prompt,
    read_log_lines,
    seed_session_log,
    sent_texts,
)
from tests.supervisor._mock_driver import (
    ActionStep,
    EmitStep,
    TurnEndStep,
    create_mock_driver,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from gymrat.session import SessionLogRecord
    from gymrat.supervisor import Driver, SupervisionResult
    from gymrat.supervisor.driver import DriverSession, SessionOutcome, SessionPrompt
    from gymrat.supervisor.events import SessionObserver
    from tests.supervisor._mock_driver import MockStep


_BLOCKED_MS = 5_000
"""A step delay long enough that only an end, never the script, cuts it short."""

_FAILED_HOOK = hook_record(stage="after", seq=2, exit_code=1, stderr_bytes=5)
_FAILED_HOOK_REASON = "after hook failed on iteration 2: exit 1 (stdout 80 B, stderr 5 B)"

_LATER_FAILED_HOOK = hook_record(stage="before", seq=3, exit_code=4, stderr_bytes=0)
_LATER_FAILED_HOOK_REASON = "before hook failed on iteration 3: exit 4 (stdout 80 B, stderr 0 B)"

_NEEDS_MODE_BITS = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="Windows lacks EACCES from chmod and root bypasses the mode bits",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path: Path) -> str:
    """A scratch repository root whose session log holds only its header."""
    path = str(tmp_path / "repo")
    seed_session_log(path)
    return path


@dataclass
class _SessionDirAccess:
    """A switch that denies and restores access to the directory holding the session log."""

    path: Path

    def deny(self) -> None:
        self.path.chmod(0o000)

    def allow(self) -> None:
        self.path.chmod(0o755)


@pytest.fixture
def session_dir_access(root: str) -> Iterator[_SessionDirAccess]:
    """An access switch on the scratch session directory, restored at teardown."""
    access = _SessionDirAccess(Path(session_dir(root)))
    yield access
    access.allow()


def _config(stop: StopConfig | None = None) -> BenchlessConfig:
    return BenchlessConfig(
        adapter="mitata",
        samples=1,
        timeout_seconds=60,
        unstable_noise_pct=5.0,
        primary="geomean",
        runbook=None,
        stop=stop,
    )


def _events_path(root: str) -> str:
    return str(Path(root).parent / "events.jsonl")


def _append_now(root: str, *records: SessionLogRecord) -> None:
    for record in records:
        append_record(session_jsonl_path(root), record)


def _append(root: str, *records: SessionLogRecord) -> ActionStep:
    async def append() -> None:
        _append_now(root, *records)

    return ActionStep(action=append)


def _log_path(root: str) -> Path:
    return Path(session_jsonl_path(root))


def _overwrite_log(root: str, content: bytes) -> None:
    _log_path(root).write_bytes(content)


def _append_log_bytes(root: str, content: bytes) -> None:
    with _log_path(root).open("ab") as handle:
        handle.write(content)


def _log_size(root: str) -> int:
    return _log_path(root).stat().st_size


def _tool_end(tool_use_id: str = "t1", tool_name: str = "Bash") -> EmitStep:
    return EmitStep(
        emit=ToolEndEvent(
            at=now_ns(),
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            duration_ms=10,
            result="ok",
            result_summary="ok",
        )
    )


def _blocked_step() -> EmitStep:
    return EmitStep(emit=TextDeltaEvent(at=now_ns(), chunk="late"), delay_ms=_BLOCKED_MS)


def _event_log_markers(root: str) -> list[str]:
    """Label each logged event as ``tool_end:<id>``, ``follow_up:<action>:<reason>``, or its type."""
    markers: list[str] = []
    for line in read_log_lines(_events_path(root)):
        match line["type"]:
            case "tool_end":
                markers.append(f"tool_end:{line['tool_use_id']}")
            case "follow_up":
                markers.append(f"follow_up:{line['action']}:{line.get('reason')}")
            case other:
                markers.append(str(other))
    return markers


def _ended_markers(markers: list[str]) -> list[str]:
    return [marker for marker in markers if marker.startswith("follow_up:ended:")]


async def _supervise(
    root: str,
    driver: Driver,
    *,
    config: BenchlessConfig | None = None,
    observer: SessionObserver | None = None,
    is_lock_held: Callable[[], bool] = lambda: False,
    grace_ms: int = 30_000,
    max_usd: float | None = None,
    deadline_ms: float | None = None,
    settle_window_ms: int = 0,
) -> SupervisionResult:
    return await supervise(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=_events_path(root),
            lock_path=str(Path(root).parent / "lockfile"),
            config=config,
            max_usd=max_usd,
            deadline_ms=deadline_ms,
        ),
        launch=make_launch(max_usd=max_usd),
        observer=observer,
        grace_ms=grace_ms,
        wall_clock_poll_ms=1,
        settle_window_ms=settle_window_ms,
        lock_poll_ms=1,
        is_lock_held=is_lock_held,
    )


@dataclass
class _LockSwitch:
    """A repository lock a test holds and releases between driver steps."""

    held: bool

    def is_held(self) -> bool:
        return self.held

    async def hold(self) -> None:
        self.held = True

    async def release(self) -> None:
        self.held = False


class _StubbornSession:
    """A session that counts ``interrupt`` calls and ignores both ``interrupt`` and ``end``."""

    def __init__(self, inner: DriverSession, driver: _StubbornDriver) -> None:
        self._inner = inner
        self._driver = driver

    @property
    def outcome(self) -> Awaitable[SessionOutcome]:
        return self._inner.outcome

    async def interrupt(self) -> None:
        self._driver.interrupts += 1

    async def send(self, text: str) -> None:
        await self._inner.send(text)

    async def end(self) -> None:
        return None


class _StubbornDriver:
    """A driver whose sessions ignore ``interrupt`` and ``end``, so only the abort event or the script's own end stops them."""

    def __init__(self, inner: Driver) -> None:
        self._inner = inner
        self.interrupts = 0
        self.abort: asyncio.Event | None = None

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        self.abort = abort
        return _StubbornSession(self._inner.start(prompt, observer, abort), self)


# ---------------------------------------------------------------------------
# EndedBy
# ---------------------------------------------------------------------------


def test_ended_by_when_inspected_does_list_every_ending():
    assert set(get_args(EndedBy)) == {
        "session",
        "wall-clock",
        "spend-cap",
        "guard",
        "stop-condition",
        "hook-failure",
    }


# ---------------------------------------------------------------------------
# detection at a tool end
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DetectionCase:
    """Records landing before a tool end, the end that follows them, and the ending they earn."""

    appended: tuple[SessionLogRecord, ...]
    stop: StopConfig | None
    tool_name: str
    ended_by: str
    reason: str


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _DetectionCase(
                appended=(_FAILED_HOOK,),
                stop=None,
                tool_name="Bash",
                ended_by="hook-failure",
                reason=_FAILED_HOOK_REASON,
            ),
            id="hook-failure-after-a-bash-end",
        ),
        pytest.param(
            _DetectionCase(
                appended=(iteration_record(seq=1),),
                stop=StopConfig(max_iterations=1),
                tool_name="mcp__gymrat__iterate",
                ended_by="stop-condition",
                reason="max iterations (1 of 1)",
            ),
            id="stop-condition-met-during-run-after-an-iterate-tool-end",
        ),
    ],
)
async def test_supervise_when_condition_lands_before_tool_end_does_end_run_after_that_tool_end(
    root: str,
    case: _DetectionCase,
):
    driver = create_mock_driver([
        _append(root, *case.appended),
        _tool_end(tool_name=case.tool_name),
        _blocked_step(),
    ])

    result = await _supervise(root, driver, config=_config(case.stop))

    markers = _event_log_markers(root)
    ended = _ended_markers(markers)
    assert result.ended_by == case.ended_by
    assert result.end_reason == case.reason
    assert result.outcome.reason == "interrupted"
    assert ended == [f"follow_up:ended:{case.reason}"]
    assert markers.index(ended[0]) > markers.index("tool_end:t1")


async def test_supervise_when_driver_ignores_condition_interrupt_does_arm_abort_after_grace(
    root: str,
):
    driver = _StubbornDriver(
        create_mock_driver([_append(root, _FAILED_HOOK), _tool_end(), _blocked_step()])
    )

    result = await _supervise(root, driver, grace_ms=50)

    assert driver.interrupts == 1
    assert driver.abort is not None
    assert driver.abort.is_set()
    assert result.ended_by == "hook-failure"
    assert result.duration_ms < _BLOCKED_MS


@pytest.mark.parametrize(
    ("seeded", "stop", "appended"),
    [
        pytest.param(
            (command_record(), _FAILED_HOOK),
            None,
            (hook_record(stage="before", seq=3),),
            id="hook-failure-before-launch-after-a-command-record",
        ),
        pytest.param(
            (iteration_record(seq=1),),
            StopConfig(max_iterations=1),
            (command_record(),),
            id="stop-condition-met-at-launch",
        ),
        pytest.param(
            (),
            None,
            (iteration_record(seq=1), hook_record(seq=1), command_record()),
            id="no-stop-configuration-and-clean-hooks",
        ),
    ],
)
async def test_supervise_when_nothing_new_ends_run_at_tool_end_does_complete_as_session(
    root: str,
    seeded: tuple[SessionLogRecord, ...],
    stop: StopConfig | None,
    appended: tuple[SessionLogRecord, ...],
):
    _append_now(root, *seeded)
    driver = create_mock_driver([_append(root, *appended), _tool_end()])

    result = await _supervise(root, driver, config=_config(stop))

    assert result.ended_by == "session"
    assert result.outcome.reason == "completed"


async def test_supervise_when_log_rewritten_at_same_size_does_not_read_it_at_tool_end(root: str):
    async def corrupt_in_place() -> None:
        _overwrite_log(root, b"x" * (_log_size(root) - 1) + b"\n")

    driver = create_mock_driver([
        _append(root, command_record()),
        _tool_end("t1"),
        ActionStep(action=corrupt_in_place),
        _tool_end("t2"),
    ])

    result = await _supervise(root, driver)

    assert result.ended_by == "session"
    assert result.outcome.reason == "completed"


async def test_supervise_when_grown_log_fails_to_read_at_tool_end_does_end_with_error_outcome(
    root: str,
):
    async def append_garbage() -> None:
        _append_log_bytes(root, b"not valid json\n")

    driver = create_mock_driver([ActionStep(action=append_garbage), _tool_end()])

    result = await _supervise(root, driver)

    assert result.outcome.reason == "error"
    assert result.outcome.message == f"Invalid JSON at {session_jsonl_path(root)}:2"


async def test_supervise_when_launch_read_fails_does_scan_hooks_only_after_first_clean_read(
    root: str,
):
    _overwrite_log(root, b"not valid json\n")

    async def repair_with_failed_hook() -> None:
        _overwrite_log(root, b"")
        _append_now(root, session_record(), _FAILED_HOOK)

    driver = create_mock_driver([
        ActionStep(action=repair_with_failed_hook),
        _tool_end("t1"),
        _append(root, _LATER_FAILED_HOOK),
        _tool_end("t2"),
        _blocked_step(),
    ])

    result = await _supervise(root, driver)

    assert result.ended_by == "hook-failure"
    assert result.end_reason == _LATER_FAILED_HOOK_REASON


@_NEEDS_MODE_BITS
@pytest.mark.filterwarnings("default::RuntimeWarning")
async def test_supervise_when_log_stat_fails_at_tool_end_does_end_with_error_outcome(
    root: str, session_dir_access: _SessionDirAccess
):
    async def deny_access() -> None:
        session_dir_access.deny()

    driver = create_mock_driver([ActionStep(action=deny_access), _tool_end()])

    result = await _supervise(root, driver)

    assert result.outcome.reason == "error"
    assert result.outcome.message is not None


def _corrupt_log_at_launch(root: str, _access: _SessionDirAccess) -> None:
    _overwrite_log(root, b"not valid json\n")


def _deny_log_at_launch(_root: str, access: _SessionDirAccess) -> None:
    access.deny()


def _repair_log(root: str, access: _SessionDirAccess, *records: SessionLogRecord) -> ActionStep:
    async def repair() -> None:
        access.allow()
        _overwrite_log(root, b"")
        _append_now(root, session_record(), *records)

    return ActionStep(action=repair)


_LAUNCH_READ_FAILURES = [
    pytest.param(_corrupt_log_at_launch, id="log-unparsable"),
    pytest.param(_deny_log_at_launch, id="log-stat-denied", marks=_NEEDS_MODE_BITS),
]


@pytest.mark.parametrize("fail_launch_read", _LAUNCH_READ_FAILURES)
async def test_supervise_when_first_clean_scan_finds_stop_condition_met_does_complete_as_session(
    root: str,
    session_dir_access: _SessionDirAccess,
    fail_launch_read: Callable[[str, _SessionDirAccess], None],
):
    fail_launch_read(root, session_dir_access)
    driver = create_mock_driver([
        _repair_log(root, session_dir_access, iteration_record(seq=1)),
        _tool_end(),
    ])

    result = await _supervise(root, driver, config=_config(StopConfig(max_iterations=1)))

    assert result.ended_by == "session"
    assert result.outcome.reason == "completed"


@pytest.mark.parametrize("fail_launch_read", _LAUNCH_READ_FAILURES)
async def test_supervise_when_stop_condition_met_after_first_clean_scan_does_end_as_stop_condition(
    root: str,
    session_dir_access: _SessionDirAccess,
    fail_launch_read: Callable[[str, _SessionDirAccess], None],
):
    fail_launch_read(root, session_dir_access)
    driver = create_mock_driver([
        _repair_log(root, session_dir_access),
        _tool_end("t1"),
        _append(root, iteration_record(seq=1)),
        _tool_end("t2"),
        _blocked_step(),
    ])

    result = await _supervise(root, driver, config=_config(StopConfig(max_iterations=1)))

    assert result.ended_by == "stop-condition"
    assert result.end_reason == "max iterations (1 of 1)"


# ---------------------------------------------------------------------------
# detection at the turn-end settle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("appended", "stop", "ended_by", "reason"),
    [
        pytest.param(_FAILED_HOOK, None, "hook-failure", _FAILED_HOOK_REASON, id="hook-failure"),
        pytest.param(
            iteration_record(seq=1),
            StopConfig(max_iterations=1),
            "stop-condition",
            "max iterations (1 of 1)",
            id="stop-condition-met-during-run",
        ),
    ],
)
async def test_supervise_when_condition_lands_after_last_tool_end_does_end_at_turn_end_unclassified(
    root: str,
    appended: SessionLogRecord,
    stop: StopConfig | None,
    ended_by: str,
    reason: str,
):
    probe = collecting_observer()
    driver = create_mock_driver([_tool_end(), _append(root, appended), TurnEndStep()])

    result = await _supervise(root, driver, config=_config(stop), observer=probe.observer)

    assert result.ended_by == ended_by
    assert result.end_reason == reason
    assert result.outcome.reason == "completed"
    assert sent_texts(driver.sessions[0]) == []
    assert [event.reason for event in follow_ups_with_action(probe.events, "ended")] == [reason]


# ---------------------------------------------------------------------------
# deferral while the repository lock is held
# ---------------------------------------------------------------------------


async def test_supervise_when_lock_held_at_detection_does_end_at_first_tool_end_finding_it_free(
    root: str,
):
    lock = _LockSwitch(held=True)
    driver = create_mock_driver([
        _append(root, _FAILED_HOOK),
        _tool_end("t1"),
        ActionStep(action=lock.release),
        _tool_end("t2"),
        _blocked_step(),
    ])

    result = await _supervise(root, driver, is_lock_held=lock.is_held)

    markers = _event_log_markers(root)
    ended = _ended_markers(markers)
    assert result.ended_by == "hook-failure"
    assert result.outcome.reason == "interrupted"
    assert ended == [f"follow_up:ended:{_FAILED_HOOK_REASON}"]
    assert markers.index(ended[0]) > markers.index("tool_end:t2")


async def test_supervise_when_end_pending_and_lock_free_at_agent_turn_end_does_end_unclassified(
    root: str,
):
    lock = _LockSwitch(held=True)
    driver = create_mock_driver([
        _append(root, _FAILED_HOOK),
        _tool_end(),
        ActionStep(action=lock.release),
        TurnEndStep(),
    ])

    result = await _supervise(root, driver, is_lock_held=lock.is_held)

    assert result.ended_by == "hook-failure"
    assert result.end_reason == _FAILED_HOOK_REASON
    assert result.outcome.reason == "completed"
    assert sent_texts(driver.sessions[0]) == []


async def test_supervise_when_end_pending_and_lock_held_at_agent_turn_end_does_wait_then_end(
    root: str,
):
    lock = _LockSwitch(held=True)
    events: list[SessionEvent] = []

    def release_on_waiting(event: SessionEvent) -> None:
        events.append(event)
        if isinstance(event, FollowUpEvent) and event.action == "waiting":
            lock.held = False

    driver = create_mock_driver([_append(root, _FAILED_HOOK), _tool_end(), TurnEndStep()])

    result = await _supervise(root, driver, observer=release_on_waiting, is_lock_held=lock.is_held)

    assert result.ended_by == "hook-failure"
    assert result.outcome.reason == "completed"
    assert follow_ups_with_action(events, "waiting") != []
    assert sent_texts(driver.sessions[0]) == []


async def test_supervise_when_end_pending_at_injected_turn_end_with_reply_outstanding_does_wait(
    root: str,
):
    lock = _LockSwitch(held=False)
    driver = create_mock_driver([
        TurnEndStep(),
        ActionStep(action=lock.hold),
        _append(root, _FAILED_HOOK),
        _tool_end("t1"),
        emit_turn_end(origin="injected"),
        ActionStep(action=lock.release),
        _tool_end("t2"),
        _blocked_step(),
    ])

    result = await _supervise(root, driver, is_lock_held=lock.is_held)

    assert result.ended_by == "hook-failure"
    assert result.outcome.reason == "interrupted"
    assert ("end", None) not in driver.sessions[0].calls


# ---------------------------------------------------------------------------
# precedence once an end or cap has fired
# ---------------------------------------------------------------------------


async def test_supervise_when_condition_end_already_fired_does_not_detect_at_later_tool_end(
    root: str,
):
    driver = _StubbornDriver(
        create_mock_driver([
            _append(root, _FAILED_HOOK),
            _tool_end("t1"),
            _append(root, _LATER_FAILED_HOOK),
            _tool_end("t2"),
            _blocked_step(),
        ])
    )

    result = await _supervise(root, driver, grace_ms=200)

    assert result.end_reason == _FAILED_HOOK_REASON
    assert _ended_markers(_event_log_markers(root)) == [f"follow_up:ended:{_FAILED_HOOK_REASON}"]


async def test_supervise_when_cap_already_fired_does_not_detect_at_later_tool_end(root: str):
    cap_seen = asyncio.Event()

    def note_cap(event: SessionEvent) -> None:
        if isinstance(event, CapEvent):
            cap_seen.set()

    async def wait_for_cap() -> None:
        await asyncio.wait_for(cap_seen.wait(), timeout=2.0)

    driver = _StubbornDriver(
        create_mock_driver([
            emit_turn_end(cost_usd=5.0),
            ActionStep(action=wait_for_cap),
            _append(root, _FAILED_HOOK),
            _tool_end(),
        ])
    )

    result = await _supervise(root, driver, observer=note_cap, max_usd=1.0)

    assert result.ended_by == "spend-cap"
    assert f"follow_up:ended:{_FAILED_HOOK_REASON}" not in _event_log_markers(root)


def _hook_failure_pending_from_tool_end(
    root: str, lock: _LockSwitch, turn: TurnEndStep
) -> list[MockStep]:
    return [_append(root, _FAILED_HOOK), _tool_end(), ActionStep(action=lock.release), turn]


def _hook_failure_detected_at_settle(
    root: str, lock: _LockSwitch, turn: TurnEndStep
) -> list[MockStep]:
    return [ActionStep(action=lock.release), _tool_end(), _append(root, _FAILED_HOOK), turn]


@pytest.mark.parametrize(
    ("turn", "max_usd"),
    [
        pytest.param(TurnEndStep(cost_usd=5.0), 1.0, id="cost-reaches-max-usd"),
        pytest.param(TurnEndStep(budget_exhausted=True), None, id="session-budget-exhausted"),
    ],
)
@pytest.mark.parametrize(
    "script",
    [
        pytest.param(_hook_failure_pending_from_tool_end, id="pending-from-tool-end"),
        pytest.param(_hook_failure_detected_at_settle, id="detected-at-settle"),
    ],
)
async def test_supervise_when_spend_cap_reached_at_turn_end_with_condition_pending_does_end_as_spend_cap(
    root: str,
    script: Callable[[str, _LockSwitch, TurnEndStep], list[MockStep]],
    turn: TurnEndStep,
    max_usd: float | None,
):
    lock = _LockSwitch(held=True)
    probe = collecting_observer()
    driver = create_mock_driver(script(root, lock, turn))

    result = await _supervise(
        root, driver, observer=probe.observer, is_lock_held=lock.is_held, max_usd=max_usd
    )

    caps = [(event.cap, event.action) for event in probe.events if isinstance(event, CapEvent)]
    ended_reasons = [event.reason for event in follow_ups_with_action(probe.events, "ended")]
    assert result.ended_by == "spend-cap"
    assert result.end_reason == "spend-cap"
    assert caps == [("spend-cap", "ending")]
    assert _FAILED_HOOK_REASON not in ended_reasons


async def test_supervise_when_wall_clock_fires_while_end_pending_does_report_the_cap(root: str):
    lock = _LockSwitch(held=True)
    driver = InterruptEmitsEndDriver(
        create_mock_driver([
            _append(root, _FAILED_HOOK),
            _tool_end(),
            ActionStep(action=lock.release),
            _blocked_step(),
        ])
    )

    result = await _supervise(
        root, driver, is_lock_held=lock.is_held, grace_ms=50, deadline_ms=now_ms() + 200
    )

    assert result.ended_by == "wall-clock"
    assert result.end_reason == "wall-clock"


class _SlowEndSession:
    """A session whose ``end`` settles only after a delay."""

    def __init__(self, inner: DriverSession, delay_ms: int) -> None:
        self._inner = inner
        self._delay_ms = delay_ms

    @property
    def outcome(self) -> Awaitable[SessionOutcome]:
        return self._inner.outcome

    async def interrupt(self) -> None:
        await self._inner.interrupt()

    async def send(self, text: str) -> None:
        await self._inner.send(text)

    async def end(self) -> None:
        await asyncio.sleep(self._delay_ms / 1000)
        await self._inner.end()


class _SlowEndDriver:
    """A driver whose sessions take ``delay_ms`` to settle after ``end``, outlasting a settle window."""

    def __init__(self, inner: Driver, delay_ms: int) -> None:
        self._inner = inner
        self._delay_ms = delay_ms

    def start(
        self,
        prompt: SessionPrompt,
        observer: SessionObserver,
        abort: asyncio.Event | None = None,
    ) -> DriverSession:
        return _SlowEndSession(self._inner.start(prompt, observer, abort), self._delay_ms)


async def test_supervise_when_cap_ends_session_during_settle_window_does_not_reply(root: str):
    probe = collecting_observer()
    driver = _SlowEndDriver(create_mock_driver([TurnEndStep(cost_usd=0.01)]), delay_ms=400)

    result = await _supervise(
        root,
        driver,
        observer=probe.observer,
        deadline_ms=now_ms() + 50,
        settle_window_ms=150,
    )

    assert result.ended_by == "wall-clock"
    assert follow_ups_with_action(probe.events, "replied") == []
