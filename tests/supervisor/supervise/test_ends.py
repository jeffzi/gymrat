"""Behavioral tests for supervision ending on conditions read off the session log.

``supervise`` scans the session log on every ``ToolEndEvent`` whose log has
grown, and again at the turn-end settle, ending the run itself when a hook
fails or a stop condition becomes met. These tests drive the mock driver
through tool ends, append records to the scratch session log between steps, and
inject ``is_lock_held`` to defer an end while a command the agent left running
holds the repository lock.

A stop condition already met at launch, as on a forced run, never ends the run:
the turn ends keep getting replies until something else ends it. One that
becomes met during the run ends it at the next turn end.

A step that must never run once an end fires carries a long ``delay_ms``: the
mock's delay races the abort, so a run that ends returns at once, while a run
that misses the end sits out the delay and reports a plain session ending.

The scan itself, ``detect_end_condition``, is pinned directly in
``test_detect_end_condition.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat.config import StopConfig
from gymrat.errors import GymratError
from gymrat.session.paths import session_dir, session_jsonl_path
from gymrat.session.store import read_records
from gymrat.supervisor.events import (
    FollowUpEvent,
    SessionEvent,
    UsageUpdateEvent,
)
from gymrat.supervisor.supervise import detect_end_condition
from gymrat.supervisor.turns import CONSECUTIVE_DISCARD_LIMIT
from gymrat.utils import now_ns
from tests._config import benchless_config
from tests._mode_bits import needs_mode_bits
from tests.session.records._fixtures import (
    append_raw,
    append_records,
    command_record,
    discard_record,
    hook_record,
    iteration_record,
    session_state,
    stop_record,
)
from tests.supervisor._fixtures import (
    ActionStep,
    EmitStep,
    FollowUpWatch,
    LockSwitch,
    MockStep,
    TurnEndStep,
    WrapDriver,
    append_step,
    blocked_step,
    cap_actions,
    collecting_observer,
    create_mock_driver,
    driver_calls,
    event_log_markers,
    follow_up_reasons,
    follow_ups_with_action,
    run_supervised,
    sent_texts,
    tool_end_event,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from gymrat.session.records import SessionLogRecord


def _scanned_reason(
    records: list[SessionLogRecord], *, stop: StopConfig | None = None, iteration_count: int = 0
) -> str:
    """The reason the end-condition scan gives for ``records``.

    The reason's wording is pinned by the ``detect_end_condition`` tests in
    ``test_detect_end_condition.py``.

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


def _tool_end(tool_use_id: str = "t1", tool_name: str = "Bash") -> EmitStep:
    return EmitStep(emit=tool_end_event(tool_name, tool_use_id))


def _ended_markers(markers: list[str]) -> list[str]:
    return [marker for marker in markers if marker.startswith("follow_up:ended:")]


# ---------------------------------------------------------------------------
# detection at a tool end
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _DetectionCase:
    """Records landing before a tool end, the end that follows them, and the ending they earn."""

    appended: tuple[SessionLogRecord, ...]
    stop: StopConfig | None
    ended_by: str
    reason: str


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _DetectionCase(
                appended=(_FAILED_HOOK,),
                stop=None,
                ended_by="hook-failure",
                reason=_FAILED_HOOK_REASON,
            ),
            id="hook-failure",
        ),
        pytest.param(
            _DetectionCase(
                appended=(iteration_record(seq=1),),
                stop=StopConfig(max_iterations=1),
                ended_by="stop-condition",
                reason=_MAX_ITERATIONS_REASON,
            ),
            id="stop-condition-met-during-run",
        ),
    ],
)
async def test_supervise_when_condition_lands_before_tool_end_does_end_run_after_that_tool_end(
    root: str,
    case: _DetectionCase,
):
    driver = create_mock_driver([
        append_step(root, *case.appended),
        _tool_end(),
        blocked_step(),
    ])

    result = await run_supervised(root, driver, config=benchless_config(stop=case.stop))

    markers = event_log_markers(root)
    ended = _ended_markers(markers)
    assert result.ended_by == case.ended_by
    assert result.end_reason == case.reason
    assert result.outcome.reason == "interrupted"
    assert ended == [f"follow_up:ended:{case.reason}"]
    assert markers.index(ended[0]) > markers.index("tool_end:t1")


async def test_supervise_when_nothing_new_ends_run_at_tool_end_does_complete_as_session(
    root: str,
):
    append_records(root, command_record(), _FAILED_HOOK)
    driver = create_mock_driver([
        append_step(root, hook_record(stage="before", seq=3)),
        _tool_end(),
    ])

    result = await run_supervised(root, driver)

    assert result.ended_by == "session"
    assert result.outcome.reason == "completed"


def _breaking_steps(break_log: Callable[[], None], trigger: MockStep) -> list[MockStep]:
    """Steps that record a cost, break the session log, then let ``trigger`` read it."""

    async def run_break() -> None:
        break_log()

    return [
        EmitStep(emit=UsageUpdateEvent(at=now_ns(), cost_usd=0.25)),
        ActionStep(action=run_break),
        trigger,
    ]


@pytest.mark.parametrize(
    "trigger",
    [
        pytest.param(_tool_end(), id="at-tool-end-scan"),
        pytest.param(TurnEndStep(), id="at-turn-end-settle"),
    ],
)
async def test_supervise_when_log_holds_invalid_json_does_end_with_the_read_error_as_outcome(
    root: str,
    trigger: MockStep,
):
    driver = create_mock_driver(
        _breaking_steps(lambda: append_raw(_log_path(root), b"not valid json\n"), trigger)
    )

    result = await run_supervised(root, driver)

    with pytest.raises(GymratError) as raised:
        read_records(session_jsonl_path(root))
    assert (result.ended_by, result.outcome.reason, result.outcome.cost_usd) == (
        "session",
        "error",
        0.25,
    )
    assert result.end_reason == result.outcome.message == str(raised.value)


@needs_mode_bits
async def test_supervise_when_log_stat_denied_at_tool_end_does_end_with_error_outcome_naming_the_log(
    root: str,
    session_dir_access: _SessionDirAccess,
):
    driver = create_mock_driver(_breaking_steps(session_dir_access.deny, _tool_end()))

    result = await run_supervised(root, driver)

    assert (result.ended_by, result.outcome.reason, result.outcome.cost_usd) == (
        "session",
        "error",
        0.25,
    )
    assert result.end_reason == result.outcome.message
    assert session_jsonl_path(root) in str(result.outcome.message)


async def test_supervise_when_two_tool_ends_find_the_log_unreadable_does_end_once(root: str):
    watch = FollowUpWatch()
    driver: WrapDriver | None = None

    async def break_log() -> None:
        append_raw(_log_path(root), b"not valid json\n")

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
    driver = WrapDriver(inner)

    result = await run_supervised(root, driver, observer=watch)

    assert result.outcome.reason == "error"
    assert watch.actions() == ["ended"]
    assert driver_calls(inner.sessions[0]) == ["end"]


_DISCARD_STREAK = tuple(discard_record(seq=seq) for seq in range(1, CONSECUTIVE_DISCARD_LIMIT + 1))


async def test_supervise_when_launch_read_fails_does_end_with_error_outcome(root: str):
    append_raw(_log_path(root), b"not valid json\n")
    driver = create_mock_driver([])

    result = await run_supervised(root, driver)

    assert driver.sessions == []
    assert (result.ended_by, result.outcome.reason) == ("session", "error")
    assert result.end_reason == result.outcome.message


@pytest.mark.parametrize(
    ("appended", "ended_by", "end_reason"),
    [
        pytest.param(
            (discard_record(seq=1),), "session", None, id="one-discard-after-a-launch-streak"
        ),
        pytest.param(
            _DISCARD_STREAK, "guard", "consecutive-discards", id="a-full-streak-after-launch"
        ),
    ],
)
async def test_supervise_when_discards_precede_launch_does_count_only_discards_after_it(
    root: str, appended: tuple[SessionLogRecord, ...], ended_by: str, end_reason: str | None
):
    append_records(root, *_DISCARD_STREAK)
    driver = create_mock_driver([append_step(root, *appended), TurnEndStep(cost_usd=0.01)])

    result = await run_supervised(root, driver)

    assert (result.ended_by, result.end_reason) == (ended_by, end_reason)


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

    result = await run_supervised(
        root, driver, config=benchless_config(stop=stop), observer=probe.observer
    )

    assert result.ended_by == ended_by
    assert result.end_reason == reason
    assert result.outcome.reason == "completed"
    assert sent_texts(driver.sessions[0]) == []
    assert [event.reason for event in follow_ups_with_action(probe.events, "ended")] == [reason]


# ---------------------------------------------------------------------------
# a stop condition met at launch versus during the run
# ---------------------------------------------------------------------------


async def test_supervise_when_stop_condition_met_at_launch_does_reply_until_a_later_end(
    root: str,
):
    append_records(root, iteration_record(seq=1))
    probe = collecting_observer()
    driver = create_mock_driver([TurnEndStep(), append_step(root, stop_record()), TurnEndStep()])

    result = await run_supervised(
        root,
        driver,
        config=benchless_config(stop=StopConfig(max_iterations=1)),
        observer=probe.observer,
    )

    assert result.ended_by == "session"
    assert follow_up_reasons(probe.events) == [("replied", None), ("ended", "finished")]


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

    result = await run_supervised(root, driver, is_lock_held=lock.is_held)

    markers = event_log_markers(root)
    ended = _ended_markers(markers)
    assert result.ended_by == "hook-failure"
    assert result.outcome.reason == "interrupted"
    assert ended == [f"follow_up:ended:{_FAILED_HOOK_REASON}"]
    assert markers.index(ended[0]) > markers.index("tool_end:t2")


async def test_supervise_when_end_pending_and_lock_held_at_agent_turn_end_does_wait_then_end(
    root: str,
):
    lock = LockSwitch(held=True)
    probe = collecting_observer()
    driver = create_mock_driver([append_step(root, _FAILED_HOOK), _tool_end(), TurnEndStep()])

    result = await run_supervised(
        root, driver, observer=lock.release_on_waiting(probe.observer), is_lock_held=lock.is_held
    )

    assert result.ended_by == "hook-failure"
    assert result.outcome.reason == "completed"
    assert follow_ups_with_action(probe.events, "waiting") != []
    assert sent_texts(driver.sessions[0]) == []


# ---------------------------------------------------------------------------
# a failing lock probe
# ---------------------------------------------------------------------------

_PROBE_FAILURE = "lock probe broke"


@dataclass(slots=True)
class _BreakableLockProbe:
    """A repository lock probe that reports the lock held until broken, then raises.

    A probe that fails once raises on its first probe only and then reports the
    lock held for good, so only the first probe site to reach it can end the
    session with the failure.
    """

    broken: bool
    fails_once: bool = False

    def is_held(self) -> bool:
        if self.broken:
            self.broken = not self.fails_once
            raise RuntimeError(_PROBE_FAILURE)
        return True

    def break_on_waiting(self, event: SessionEvent) -> None:
        if self.fails_once:
            return
        if isinstance(event, FollowUpEvent) and event.action == "waiting":
            self.broken = True


@dataclass(frozen=True, slots=True)
class _FailingProbeCase:
    """A script that runs one probe site into a broken lock probe, and when the probe breaks."""

    script: Callable[[str], list[MockStep]]
    broken_at_launch: bool
    fails_once: bool = False


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _FailingProbeCase(
                script=lambda root: [append_step(root, _FAILED_HOOK), _tool_end()],
                broken_at_launch=True,
            ),
            id="tool-end-scan",
        ),
        pytest.param(
            _FailingProbeCase(
                script=lambda _root: [TurnEndStep()], broken_at_launch=True, fails_once=True
            ),
            id="settle",
        ),
        pytest.param(
            _FailingProbeCase(
                script=lambda root: [append_step(root, _FAILED_HOOK), _tool_end(), TurnEndStep()],
                broken_at_launch=False,
            ),
            id="lock-poll",
        ),
    ],
)
async def test_supervise_when_lock_probe_raises_does_end_promptly_with_error_outcome(
    root: str, case: _FailingProbeCase
):
    probe = _BreakableLockProbe(broken=case.broken_at_launch, fails_once=case.fails_once)
    driver = create_mock_driver(case.script(root))

    result = await asyncio.wait_for(
        run_supervised(root, driver, observer=probe.break_on_waiting, is_lock_held=probe.is_held),
        timeout=2.0,
    )

    assert (result.ended_by, result.outcome.reason) == ("session", "error")
    assert result.end_reason == result.outcome.message
    assert _PROBE_FAILURE in str(result.outcome.message)


# ---------------------------------------------------------------------------
# precedence once an end or cap has fired
# ---------------------------------------------------------------------------


async def test_supervise_when_spend_cap_reached_at_turn_end_with_condition_pending_does_end_as_spend_cap(
    root: str,
):
    lock = LockSwitch(held=True)
    probe = collecting_observer()
    driver = create_mock_driver([
        append_step(root, _FAILED_HOOK),
        _tool_end(),
        ActionStep(action=lock.release),
        TurnEndStep(cost_usd=5.0),
    ])

    result = await run_supervised(
        root, driver, observer=probe.observer, is_lock_held=lock.is_held, max_usd=1.0
    )

    assert result.ended_by == "spend-cap"
    assert result.end_reason == "spend-cap"
    assert cap_actions(probe.events) == [("spend-cap", "ending")]
