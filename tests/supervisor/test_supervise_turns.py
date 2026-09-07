"""Behavioral tests for the supervisor turn-loop integration.

``supervise`` processes ``TurnEndEvent`` by entering an idle state, running a
settle window, reading and folding the session log, probing the repository lock,
and delegating to ``classify`` for the decision. These tests exercise the full
supervisor-classifier wiring: scheduling, caps, in-flight detection, settle
windows, lock polling, guard propagation, and the follow-up events that each
decision emits through the combined observer and into the JSONL log.

Unless otherwise noted, tests pass ``settle_window_ms=0`` and ``lock_poll_ms=1``
to keep runs instantaneous, and inject ``is_lock_held`` as a callable to avoid
filesystem contention.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from gymrat.session.clock import now_ms
from gymrat.session.paths import session_jsonl_path
from gymrat.session.store import append_record
from gymrat.supervisor import (
    SessionOutcome,
    TextDeltaEvent,
    supervise,
)
from gymrat.supervisor.supervise import EndedBy, SupervisionResult
from tests.conftest import hold_lock
from tests.session.records._fixtures import (
    stop_record,
)
from tests.supervisor._fixtures import (
    _cap_events,
    add_stop_async,
    collecting_observer,
    emit_turn_end,
    follow_up_events,
    follow_ups_with_action,
    make_context,
    make_launch,
    make_prompt,
    seed_session_log,
    seed_with_stop,
    sent_texts,
    supervise_fast,
)
from tests.supervisor._mock_driver import (
    ActionStep,
    CostStep,
    EmitStep,
    TurnEndStep,
    create_mock_driver,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# behavior 1: module constants and new parameters
# ---------------------------------------------------------------------------


def test_settle_window_ms_when_imported_does_equal_800():
    from gymrat.supervisor.supervise import SETTLE_WINDOW_MS

    assert SETTLE_WINDOW_MS == 800


def test_lock_poll_ms_when_imported_does_equal_5000():
    from gymrat.supervisor.supervise import LOCK_POLL_MS

    assert LOCK_POLL_MS == 5000


# ---------------------------------------------------------------------------
# behavior 2: EndedBy gains "guard" and SupervisionResult gains end_reason
# ---------------------------------------------------------------------------


def test_ended_by_when_guard_value_used_does_be_valid():
    assert "guard" in EndedBy.__args__  # type: ignore[attr-defined]


def test_supervision_result_when_constructed_with_end_reason_does_carry_it():
    result = SupervisionResult(
        outcome=SessionOutcome(reason="completed", cost_usd=0.0),
        ended_by="session",
        end_reason="test-reason",
        duration_ms=100,
        cost_usd=0.0,
    )

    assert result.end_reason == "test-reason"


def test_supervision_result_when_no_end_reason_does_default_to_none():
    result = SupervisionResult(
        outcome=SessionOutcome(reason="completed", cost_usd=0.0),
        ended_by="session",
        duration_ms=100,
        cost_usd=0.0,
    )

    assert result.end_reason is None


# ---------------------------------------------------------------------------
# behavior 3: turn end triggers settle → fold → classify
# ---------------------------------------------------------------------------


async def test_supervise_when_turn_end_with_stop_record_does_end_as_session(
    tmp_path: Path,
):
    """The stop record makes the classifier return ``End("finished")``."""
    root = str(tmp_path / "repo")
    seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01),
        ]
    )

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
    )

    assert result.ended_by == "session"
    assert result.end_reason is None
    assert result.outcome.reason in {"completed", "interrupted"}


# ---------------------------------------------------------------------------
# behavior 4: End("spend-cap") → CapEvent, session.end(), ended_by="spend-cap"
# ---------------------------------------------------------------------------


async def test_supervise_when_cost_exceeds_max_usd_at_turn_end_does_end_as_spend_cap(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=5.0),
        ]
    )

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
            max_usd=1.0,
        ),
        launch=make_launch(max_usd=1.0),
        observer=probe.observer,
    )

    assert result.ended_by == "spend-cap"
    assert result.end_reason == "spend-cap"

    caps = _cap_events(probe.events)
    assert len(caps) == 1
    assert caps[0].cap == "spend-cap"


# ---------------------------------------------------------------------------
# behavior 4: guard End(reason) → session.end(), ended_by="guard", end_reason
# ---------------------------------------------------------------------------


async def test_supervise_when_no_progress_guard_trips_does_end_as_guard_from_supervisor_state(
    tmp_path: Path,
):
    """Guard state is seeded from the record count at launch.

    It is the supervisor — not the classifier — that owns the fruitless-reply
    tally.
    """
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    # Four turn ends with no new session records → no-progress guard fires
    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
            TurnEndStep(cost_usd=0.01, origin="agent"),
            TurnEndStep(cost_usd=0.01, origin="agent"),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    assert result.ended_by == "guard"
    assert result.end_reason == "no-progress"


# ---------------------------------------------------------------------------
# behavior 4: Reply(text) → session.send(text)
# ---------------------------------------------------------------------------


async def test_supervise_when_classifier_replies_does_send_text_to_session(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")

    # First turn end → Reply → send → second turn end → add stop → End
    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
            ActionStep(action=lambda: add_stop_async(root)),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
    )

    # Session ended normally after the stop record was written
    assert result.ended_by == "session"


# ---------------------------------------------------------------------------
# behavior 5: turn end while reply outstanding
# ---------------------------------------------------------------------------


async def test_supervise_when_turn_end_while_reply_outstanding_does_not_schedule(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    # First agent turn → Reply (outstanding). The injected turn end is emitted
    # via EmitStep (not TurnEndStep) so the mock does not block waiting for
    # send/end — the supervisor ignores it because reply is outstanding.
    # Then the agent turn end arrives, clears outstanding, classifies, stop → end.
    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
            emit_turn_end(origin="injected"),
            ActionStep(action=lambda: add_stop_async(root)),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    assert result.ended_by == "session"


# ---------------------------------------------------------------------------
# behavior 6: WaitForLock → poll → settle → classify with after_wait=True
# ---------------------------------------------------------------------------


async def test_supervise_when_lock_held_does_wait_then_classify_with_after_wait(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()
    poll_count = 0

    def lock_held() -> bool:
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            return True
        # On release, add the stop record so the re-classification ends the session
        append_record(session_jsonl_path(root), stop_record())
        return False

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
        is_lock_held=lock_held,
    )

    assert result.ended_by == "session"

    waiting_events = follow_ups_with_action(probe.events, "waiting")
    assert len(waiting_events) >= 1


async def test_supervise_when_lock_held_after_two_fruitless_replies_does_wait_instead_of_ending(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()
    lock_held_calls = 0

    def lock_held() -> bool:
        nonlocal lock_held_calls
        lock_held_calls += 1
        return lock_held_calls <= 1

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),  # Reply #1
            TurnEndStep(cost_usd=0.01, origin="agent"),  # Reply #2
            ActionStep(action=lambda: add_stop_async(root)),
            TurnEndStep(
                cost_usd=0.01, origin="agent"
            ),  # Would be no-progress #3 but lock held → wait
        ]
    )

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
        is_lock_held=lock_held,
    )

    waiting_events = follow_ups_with_action(probe.events, "waiting")
    assert len(waiting_events) >= 1
    assert result.ended_by == "session"


async def test_supervise_when_real_lock_held_does_wait_then_reply_with_after_wait_line(
    tmp_path: Path,
):
    """Uses ``hold_lock`` on the real lock path instead of injecting ``is_lock_held``.

    The OS-level file lock drives the poll loop.  The lock is released from a
    background task after the ``waiting`` follow-up appears, proving the poll
    iterates at least once.
    """
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    lock = hold_lock(lock_path)

    released = False

    async def release_after_waiting() -> None:
        nonlocal released
        # probe.events is a plain list mutated by another coroutine — no
        # callback hook to wire an asyncio.Event to; polling is the only option.
        while not any(  # noqa: ASYNC110
            e.action == "waiting" for e in follow_up_events(probe.events)
        ):
            await asyncio.sleep(0.005)
        lock.release()
        released = True

    _background_task: asyncio.Task[None] | None = None

    async def schedule_release() -> None:
        nonlocal _background_task
        _background_task = asyncio.create_task(release_after_waiting())

    driver = create_mock_driver(
        [
            ActionStep(action=schedule_release),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await supervise(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
        settle_window_ms=0,
        lock_poll_ms=1,
    )

    assert released
    assert result.ended_by == "session"

    follow_ups = follow_up_events(probe.events)
    assert any(e.action == "waiting" for e in follow_ups)
    assert any(e.action == "replied" for e in follow_ups)

    session = driver.sessions[0]
    sent = sent_texts(session)
    assert any(
        "The command you left running has finished" in text for text in sent if text is not None
    )


async def test_supervise_when_text_delta_during_lock_poll_does_cancel_poll(
    tmp_path: Path,
):
    """No ``replied`` follow-up appears for that turn.

    The next turn end is classified afresh.
    """
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver(
        [
            emit_turn_end(),
            EmitStep(
                emit=TextDeltaEvent(timestamp=now_ms(), chunk="typing"),
                delay_ms=30,
            ),
            ActionStep(action=lambda: add_stop_async(root)),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
        is_lock_held=lambda: True,
    )

    assert result.ended_by == "session"

    waiting_events = follow_ups_with_action(probe.events, "waiting")
    replied_events = follow_ups_with_action(probe.events, "replied")
    assert len(waiting_events) >= 1
    assert len(replied_events) == 0

    session = driver.sessions[0]
    assert len(sent_texts(session)) == 0


# ---------------------------------------------------------------------------
# behavior 7: every decision emitted as FollowUpEvent
# ---------------------------------------------------------------------------


async def test_supervise_when_session_ends_normally_does_emit_ended_follow_up(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01),
        ]
    )

    await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    ended = follow_ups_with_action(probe.events, "ended")
    assert len(ended) >= 1
    assert ended[0].reason == "finished"


async def test_supervise_when_classifier_replies_does_emit_replied_follow_up(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")
    probe = collecting_observer()

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=0.01, origin="agent"),
            ActionStep(action=lambda: add_stop_async(root)),
            TurnEndStep(cost_usd=0.01, origin="agent"),
        ]
    )

    await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
        ),
        launch=make_launch(),
        observer=probe.observer,
    )

    replied = follow_ups_with_action(probe.events, "replied")
    assert len(replied) >= 1
    assert replied[0].text is not None


# ---------------------------------------------------------------------------
# behavior 9: spend cap at turn end, not via _cost_observer
# ---------------------------------------------------------------------------


async def test_supervise_when_usage_update_alone_does_not_end_session(
    tmp_path: Path,
):
    """A UsageUpdateEvent alone never ends the session."""
    root = str(tmp_path / "repo")
    seed_with_stop(root)
    lock_path = str(tmp_path / "lockfile")

    driver = create_mock_driver(
        [
            CostStep(cost_usd=5.0),
            TurnEndStep(cost_usd=5.0),
        ]
    )

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
            max_usd=1.0,
        ),
        launch=make_launch(max_usd=1.0),
    )

    # A stop record makes classify() return End("finished") ahead of the budget check.
    assert result.ended_by == "session"


async def test_supervise_when_turn_end_cost_exceeds_max_usd_without_stop_does_end_spend_cap(
    tmp_path: Path,
):
    root = str(tmp_path / "repo")
    seed_session_log(root)
    lock_path = str(tmp_path / "lockfile")

    driver = create_mock_driver(
        [
            TurnEndStep(cost_usd=5.0),
        ]
    )

    result = await supervise_fast(
        driver,
        make_prompt(cwd=root),
        context=make_context(
            root=root,
            log_path=str(tmp_path / "events.jsonl"),
            lock_path=lock_path,
            max_usd=1.0,
        ),
        launch=make_launch(max_usd=1.0),
    )

    assert result.ended_by == "spend-cap"
    assert result.end_reason == "spend-cap"
