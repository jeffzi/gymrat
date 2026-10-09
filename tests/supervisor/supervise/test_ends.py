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
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import pytest

from gymrat.clock import now_ns
from gymrat.config import StopConfig
from gymrat.session.paths import session_dir, session_jsonl_path
from gymrat.supervisor.end_scan import detect_end_condition
from gymrat.supervisor.events import (
    CapEvent,
    SessionEvent,
    ToolEndEvent,
    TurnEndEvent,
    UsageUpdateEvent,
)
from tests._config import benchless_config
from tests._mode_bits import needs_mode_bits
from tests.session.records._fixtures import (
    append_records,
    command_record,
    hook_record,
    iteration_record,
    session_record,
    session_state,
)
from tests.supervisor._fixtures import (
    DelegatingSession,
    FollowUpWatch,
    InterruptEmitsEndDriver,
    LockSwitch,
    SupervisorClock,
    _supervise,
    _WrapDriver,
    append_step,
    blocked_step,
    collecting_observer,
    driver_calls,
    emit_turn_end,
    events_log_path,
    events_of,
    follow_ups_with_action,
    read_log_lines,
    sent_texts,
)
from tests.supervisor._mock_driver import (
    ActionStep,
    EmitStep,
    TurnEndStep,
    create_mock_driver,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from gymrat.session.records import SessionLogRecord
    from gymrat.supervisor.driver import Driver, DriverSession
    from gymrat.supervisor.supervise import SupervisionResult
    from tests.supervisor._mock_driver import MockStep


def _scanned_reason(
    records: list[SessionLogRecord], *, stop: StopConfig | None = None, iteration_count: int = 0
) -> str:
    """The reason the end-condition scan gives for ``records``; its wording is pinned in ``test_turns.py``.

    Args:
        records: The session log records the scan reads.
        stop: The stop configuration, or None for none.
        iteration_count: How many iterations the folded session state counts.

    Returns:
        The reason the scan reports.
    """
    condition = detect_end_condition(
        benchless_config(stop=stop),
        records,
        session_state(iteration_count=iteration_count),
        cursor=0,
        check_stop=True,
    )
    if condition is None:
        pytest.fail(f"the scan found no end condition in {records!r}")
    return condition.reason


_FAILED_HOOK = hook_record(stage="after", seq=2, exit_code=1, stderr_bytes=5)
_FAILED_HOOK_REASON = _scanned_reason([_FAILED_HOOK])

_LATER_FAILED_HOOK = hook_record(stage="before", seq=3, exit_code=4, stderr_bytes=0)
_LATER_FAILED_HOOK_REASON = _scanned_reason([_LATER_FAILED_HOOK])

_MAX_ITERATIONS_REASON = _scanned_reason([], stop=StopConfig(max_iterations=1), iteration_count=1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
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


def _log_path(root: str) -> Path:
    return Path(session_jsonl_path(root))


def _overwrite_log(root: str, content: bytes) -> None:
    _log_path(root).write_bytes(content)


def _append_log_bytes(root: str, content: bytes) -> None:
    with _log_path(root).open("ab") as handle:
        handle.write(content)


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


def _event_log_markers(root: str) -> list[str]:
    """Label each logged event as ``tool_end:<id>``, ``follow_up:<action>:<reason>``, or its type."""
    markers: list[str] = []
    for line in read_log_lines(events_log_path(root)):
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


class _StubbornSession(DelegatingSession):
    """A session that ignores ``interrupt`` and ``end``, so only the abort event or the script's own end stops it."""

    @override
    async def interrupt(self) -> None:
        return None

    @override
    async def end(self) -> None:
        return None


def _stubborn(inner: Driver) -> _WrapDriver:
    """Wrap ``inner`` so every session it starts ignores ``interrupt`` and ``end``."""
    return _WrapDriver(inner, lambda session, _abort: _StubbornSession(session))


# ---------------------------------------------------------------------------
# detection at a tool end
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
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
                reason=_MAX_ITERATIONS_REASON,
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
        append_step(root, *case.appended),
        _tool_end(tool_name=case.tool_name),
        blocked_step(),
    ])

    result = await _supervise(root, driver, config=benchless_config(stop=case.stop))

    markers = _event_log_markers(root)
    ended = _ended_markers(markers)
    assert result.ended_by == case.ended_by
    assert result.end_reason == case.reason
    assert result.outcome.reason == "interrupted"
    assert ended == [f"follow_up:ended:{case.reason}"]
    assert markers.index(ended[0]) > markers.index("tool_end:t1")


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
    append_records(root, *seeded)
    driver = create_mock_driver([append_step(root, *appended), _tool_end()])

    result = await _supervise(root, driver, config=benchless_config(stop=stop))

    assert result.ended_by == "session"
    assert result.outcome.reason == "completed"


def _replace_log_with_directory(root: str) -> None:
    _log_path(root).unlink()
    _log_path(root).mkdir()


def _directory_at_log(root: str, _access: _SessionDirAccess) -> None:
    _replace_log_with_directory(root)


def _append_invalid_json(root: str, _access: _SessionDirAccess) -> None:
    _append_log_bytes(root, b"not valid json\n")


def _deny_session_dir(_root: str, access: _SessionDirAccess) -> None:
    access.deny()


@pytest.mark.parametrize(
    ("make_unreadable", "trigger", "message"),
    [
        pytest.param(
            _directory_at_log,
            _tool_end(),
            "Cannot read session log at {log}",
            id="directory-at-tool-end-scan",
        ),
        pytest.param(
            _directory_at_log,
            TurnEndStep(),
            "Cannot read session log at {log}",
            id="directory-at-turn-end-settle",
        ),
        pytest.param(
            _append_invalid_json,
            _tool_end(),
            "Invalid JSON at {log}:2",
            id="invalid-json-at-tool-end-scan",
        ),
        pytest.param(
            _append_invalid_json,
            TurnEndStep(),
            "Invalid JSON at {log}:2",
            id="invalid-json-at-turn-end-settle",
        ),
        pytest.param(
            _deny_session_dir,
            _tool_end(),
            "Cannot inspect session log {log}",
            id="stat-denied-at-tool-end-scan",
            marks=[needs_mode_bits, pytest.mark.filterwarnings("default::RuntimeWarning")],
        ),
    ],
)
async def test_supervise_when_log_becomes_unreadable_does_end_with_error_outcome_naming_the_log(
    root: str,
    session_dir_access: _SessionDirAccess,
    make_unreadable: Callable[[str, _SessionDirAccess], None],
    trigger: MockStep,
    message: str,
):
    async def break_log() -> None:
        make_unreadable(root, session_dir_access)

    driver = create_mock_driver([
        EmitStep(emit=UsageUpdateEvent(at=now_ns(), cost_usd=0.25)),
        ActionStep(action=break_log),
        trigger,
    ])

    result = await _supervise(root, driver)

    assert (result.ended_by, result.outcome.reason, result.outcome.cost_usd) == (
        "session",
        "error",
        0.25,
    )
    assert result.end_reason == result.outcome.message
    assert message.format(log=session_jsonl_path(root)) in str(result.outcome.message)


async def test_supervise_when_two_tool_ends_find_the_log_unreadable_does_end_once(root: str):
    watch = FollowUpWatch()
    driver: _WrapDriver | None = None

    async def break_log() -> None:
        _append_log_bytes(root, b"not valid json\n")

    async def deliver_two_tool_ends() -> None:
        if driver is None:
            pytest.fail("the driver was never built, so no observer was captured")
        driver.observer(_tool_end("t1").emit)
        driver.observer(_tool_end("t2").emit)

    inner = create_mock_driver([
        ActionStep(action=break_log),
        ActionStep(action=deliver_two_tool_ends),
        TurnEndStep(),
    ])
    driver = _WrapDriver(inner)

    result = await _supervise(root, driver, observer=watch)

    assert result.outcome.reason == "error"
    assert watch.actions() == ["ended"]
    assert driver_calls(inner.sessions[0]) == ["end"]


async def test_supervise_when_launch_read_fails_does_scan_hooks_only_after_first_clean_read(
    root: str,
):
    _overwrite_log(root, b"not valid json\n")

    async def repair_with_failed_hook() -> None:
        _overwrite_log(root, b"")
        append_records(root, session_record(), _FAILED_HOOK)

    driver = create_mock_driver([
        ActionStep(action=repair_with_failed_hook),
        _tool_end("t1"),
        append_step(root, _LATER_FAILED_HOOK),
        _tool_end("t2"),
        blocked_step(),
    ])

    result = await _supervise(root, driver)

    assert result.ended_by == "hook-failure"
    assert result.end_reason == _LATER_FAILED_HOOK_REASON


def _corrupt_log_at_launch(root: str, _access: _SessionDirAccess) -> None:
    _overwrite_log(root, b"not valid json\n")


def _deny_log_at_launch(_root: str, access: _SessionDirAccess) -> None:
    access.deny()


def _repair_log(root: str, access: _SessionDirAccess, *records: SessionLogRecord) -> ActionStep:
    async def repair() -> None:
        access.allow()
        with contextlib.suppress(NotADirectoryError):
            _log_path(root).rmdir()
        _overwrite_log(root, b"")
        append_records(root, session_record(), *records)

    return ActionStep(action=repair)


_LAUNCH_READ_FAILURES = [
    pytest.param(_corrupt_log_at_launch, id="log-unparsable"),
    pytest.param(_directory_at_log, id="log-is-a-directory"),
    pytest.param(_deny_log_at_launch, id="log-stat-denied", marks=needs_mode_bits),
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

    result = await _supervise(
        root, driver, config=benchless_config(stop=StopConfig(max_iterations=1))
    )

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
        append_step(root, iteration_record(seq=1)),
        _tool_end("t2"),
        blocked_step(),
    ])

    result = await _supervise(
        root, driver, config=benchless_config(stop=StopConfig(max_iterations=1))
    )

    assert result.ended_by == "stop-condition"
    assert result.end_reason == _MAX_ITERATIONS_REASON


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
            _MAX_ITERATIONS_REASON,
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
    driver = create_mock_driver([_tool_end(), append_step(root, appended), TurnEndStep()])

    result = await _supervise(
        root, driver, config=benchless_config(stop=stop), observer=probe.observer
    )

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
    lock = LockSwitch(held=True)
    driver = create_mock_driver([
        append_step(root, _FAILED_HOOK),
        _tool_end("t1"),
        ActionStep(action=lock.release),
        _tool_end("t2"),
        blocked_step(),
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
    lock = LockSwitch(held=True)
    driver = create_mock_driver([
        append_step(root, _FAILED_HOOK),
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
    lock = LockSwitch(held=True)
    probe = collecting_observer()
    driver = create_mock_driver([append_step(root, _FAILED_HOOK), _tool_end(), TurnEndStep()])

    result = await _supervise(
        root, driver, observer=lock.release_on_waiting(probe.observer), is_lock_held=lock.is_held
    )

    assert result.ended_by == "hook-failure"
    assert result.outcome.reason == "completed"
    assert follow_ups_with_action(probe.events, "waiting") != []
    assert sent_texts(driver.sessions[0]) == []


async def test_supervise_when_end_pending_at_injected_turn_end_with_reply_outstanding_does_wait(
    root: str,
):
    lock = LockSwitch(held=False)
    driver = create_mock_driver([
        TurnEndStep(),
        ActionStep(action=lock.hold),
        append_step(root, _FAILED_HOOK),
        _tool_end("t1"),
        emit_turn_end(origin="injected"),
        ActionStep(action=lock.release),
        _tool_end("t2"),
        blocked_step(),
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
    driver = _stubborn(
        create_mock_driver([
            append_step(root, _FAILED_HOOK),
            _tool_end("t1"),
            append_step(root, _LATER_FAILED_HOOK),
            _tool_end("t2"),
            blocked_step(),
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

    driver = _stubborn(
        create_mock_driver([
            emit_turn_end(cost_usd=5.0),
            ActionStep(action=wait_for_cap),
            append_step(root, _FAILED_HOOK),
            _tool_end(),
        ])
    )

    result = await _supervise(root, driver, observer=note_cap, max_usd=1.0)

    assert result.ended_by == "spend-cap"
    assert f"follow_up:ended:{_FAILED_HOOK_REASON}" not in _event_log_markers(root)


def _hook_failure_pending_from_tool_end(
    root: str, lock: LockSwitch, turn: TurnEndStep
) -> list[MockStep]:
    return [append_step(root, _FAILED_HOOK), _tool_end(), ActionStep(action=lock.release), turn]


def _hook_failure_detected_at_settle(
    root: str, lock: LockSwitch, turn: TurnEndStep
) -> list[MockStep]:
    return [ActionStep(action=lock.release), _tool_end(), append_step(root, _FAILED_HOOK), turn]


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
    script: Callable[[str, LockSwitch, TurnEndStep], list[MockStep]],
    turn: TurnEndStep,
    max_usd: float | None,
):
    lock = LockSwitch(held=True)
    probe = collecting_observer()
    driver = create_mock_driver(script(root, lock, turn))

    result = await _supervise(
        root, driver, observer=probe.observer, is_lock_held=lock.is_held, max_usd=max_usd
    )

    caps = [(event.cap, event.action) for event in events_of(probe.events, CapEvent)]
    ended_reasons = [event.reason for event in follow_ups_with_action(probe.events, "ended")]
    assert result.ended_by == "spend-cap"
    assert result.end_reason == "spend-cap"
    assert caps == [("spend-cap", "ending")]
    assert _FAILED_HOOK_REASON not in ended_reasons


async def test_supervise_when_wall_clock_fires_while_end_pending_does_report_the_cap(
    root: str, supervisor_clock: SupervisorClock
):
    lock = LockSwitch(held=True)
    driver = InterruptEmitsEndDriver(
        create_mock_driver([
            append_step(root, _FAILED_HOOK),
            _tool_end(),
            ActionStep(action=lock.release),
            supervisor_clock.jump_step(supervisor_clock.deadline_ms),
            blocked_step(),
        ])
    )

    result = await _supervise(
        root,
        driver,
        is_lock_held=lock.is_held,
        grace_ms=50,
        deadline_ms=supervisor_clock.deadline_ms,
    )

    assert result.ended_by == "wall-clock"
    assert result.end_reason == "wall-clock"


# ---------------------------------------------------------------------------
# grace fallback when the session is ended at a turn boundary
# ---------------------------------------------------------------------------


class _AbortSettledSession(DelegatingSession):
    """A session that ignores ``end`` and settles only once the abort event is set.

    Models a backend that stays up after a polite end. ``settling`` is the task
    that interrupts the inner session on abort, for the test to await.
    """

    def __init__(self, inner: DriverSession, abort: asyncio.Event) -> None:
        super().__init__(inner)
        self.settling = asyncio.ensure_future(self._settle_on_abort(abort))

    async def _settle_on_abort(self, abort: asyncio.Event) -> None:
        await abort.wait()
        await self._inner.interrupt()

    @override
    async def end(self) -> None:
        return None


def _hook_failure_after_last_tool_end(root: str) -> list[MockStep]:
    return [_tool_end(), append_step(root, _FAILED_HOOK), TurnEndStep()]


@dataclass(frozen=True, slots=True)
class _BoundaryEndCase:
    """A script that stops at a turn end, the knobs that end the session there, and the ending."""

    script: Callable[[str], list[MockStep]]
    ended_by: str
    max_usd: float | None = None
    deadline_after_turn_end: bool = False
    settle_window_ms: int = 0


async def _supervise_boundary_case(
    root: str, driver: Driver, case: _BoundaryEndCase, clock: SupervisorClock
) -> SupervisionResult:
    """Supervise ``driver`` under ``case``'s caps, its deadline passing only after a turn end.

    Args:
        root: The repository the session runs in.
        driver: The agent driver to supervise.
        case: The script's caps and settle window.
        clock: The supervisor's wall clock, moved to its deadline on a turn end.

    Returns:
        How the supervised session ended.
    """
    return await _supervise(
        root,
        driver,
        grace_ms=50,
        max_usd=case.max_usd,
        deadline_ms=clock.deadline_ms,
        observer=clock.jump_on(TurnEndEvent) if case.deadline_after_turn_end else None,
        settle_window_ms=case.settle_window_ms,
    )


_BOUNDARY_END_CASES = [
    pytest.param(
        _BoundaryEndCase(
            script=lambda _root: [TurnEndStep(cost_usd=0.01)],
            ended_by="wall-clock",
            deadline_after_turn_end=True,
            settle_window_ms=10_000,
        ),
        id="wall-clock-while-idle",
    ),
    pytest.param(
        _BoundaryEndCase(
            script=lambda _root: [TurnEndStep(cost_usd=5.0)],
            ended_by="spend-cap",
            max_usd=1.0,
        ),
        id="spend-cap-at-settle",
    ),
    pytest.param(
        _BoundaryEndCase(script=_hook_failure_after_last_tool_end, ended_by="hook-failure"),
        id="hook-failure-at-settle",
    ),
]


@pytest.mark.parametrize("case", _BOUNDARY_END_CASES)
async def test_supervise_when_driver_ignores_end_at_turn_boundary_does_arm_abort_after_grace(
    root: str, case: _BoundaryEndCase, supervisor_clock: SupervisorClock
):
    driver = _WrapDriver(create_mock_driver(case.script(root)), _AbortSettledSession)

    result = await asyncio.wait_for(
        _supervise_boundary_case(root, driver, case, supervisor_clock), timeout=2.0
    )
    await cast("_AbortSettledSession", driver.session).settling

    assert driver.abort.is_set()
    assert result.ended_by == case.ended_by


@pytest.mark.parametrize("case", _BOUNDARY_END_CASES)
async def test_supervise_when_driver_settles_on_end_at_turn_boundary_does_not_abort(
    root: str, case: _BoundaryEndCase, supervisor_clock: SupervisorClock
):
    driver = _WrapDriver(create_mock_driver(case.script(root)))

    result = await _supervise_boundary_case(root, driver, case, supervisor_clock)
    # Outlast the grace period, so a grace timer left armed would set the abort.
    await asyncio.sleep(0.15)

    assert not driver.abort.is_set()
    assert result.ended_by == case.ended_by
    assert asyncio.all_tasks() == {asyncio.current_task()}
