"""Behavioral tests for the turn classifier, guard state, and reply text.

The classifier is a pure function over its arguments plus the ``GuardState``
it mutates. It never reads the filesystem or the driver. Tests build states
with ``SessionState`` fixtures and record lists from the session record models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

from gymrat.config import BenchlessConfig, StopConfig
from gymrat.eta import format_duration
from gymrat.supervisor.events import TurnEndEvent

if TYPE_CHECKING:
    from gymrat.session import SessionLogRecord
    from gymrat.session.store import SessionState
from gymrat.supervisor.turns import (
    CONSECUTIVE_DISCARD_LIMIT,
    FOLLOW_UP_CEILING,
    NO_PROGRESS_LIMIT,
    Decision,
    End,
    GuardState,
    Reply,
    WaitForLock,
    classify,
)
from tests.cli.supervise._fixtures import session_state
from tests.session.records._fixtures import (
    blocked_keep,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    iteration_record,
    stop_record,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _benchless_config(**overrides: object) -> BenchlessConfig:
    """A minimal ``BenchlessConfig`` for classifier tests."""
    defaults: dict[str, object] = {
        "adapter": "mitata",
        "samples": 1,
        "timeout_seconds": 60,
        "unstable_noise_pct": 5.0,
        "primary": "geomean",
        "runbook": None,
        "stop": None,
    }
    defaults.update(overrides)
    return BenchlessConfig(**defaults)  # type: ignore[arg-type]


def _turn_end(
    *,
    cost_usd: float = 0.01,
    budget_exhausted: bool = False,
    text: str = "done",
    origin: Literal["agent", "injected"] = "agent",
    timestamp: int = 5000,
) -> TurnEndEvent:
    return TurnEndEvent(
        timestamp=timestamp,
        text=text,
        cost_usd=cost_usd,
        origin=origin,
        budget_exhausted=budget_exhausted,
    )


def _guards(
    *,
    initial_record_count: int = 0,
    replies_sent: int = 0,
    no_progress_count: int = 0,
    last_record_count: int | None = None,
) -> GuardState:
    gs = GuardState(initial_record_count=initial_record_count)
    gs.replies_sent = replies_sent
    gs.no_progress_count = no_progress_count
    if last_record_count is not None:
        gs.last_record_count = last_record_count
    return gs


def _classify(
    *,
    config: BenchlessConfig,
    state: SessionState,
    records: list[SessionLogRecord],
    guards: GuardState,
    turn: TurnEndEvent,
    **overrides: object,
) -> Decision:
    """Delegates to ``classify`` with defaults for the boilerplate keyword args."""
    defaults: dict[str, object] = {
        "lock_held": False,
        "max_usd": None,
        "deadline_ms": 999_999_999.0,
        "max_minutes": 60,
        "now_ms": 0.0,
    }
    defaults.update(overrides)
    return classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=turn,
        **defaults,  # type: ignore[arg-type]
    )


def _classify_discards(records: list[SessionLogRecord]) -> Decision:
    """Runs classify with the discard-streak defaults, varying only ``records``."""
    return _classify(
        config=_benchless_config(),
        state=session_state(),
        records=records,
        guards=_guards(),
        turn=_turn_end(),
    )


# ---------------------------------------------------------------------------
# behavior 2: constants
# ---------------------------------------------------------------------------


def test_follow_up_ceiling_when_imported_does_equal_100():
    assert FOLLOW_UP_CEILING == 100


def test_no_progress_limit_when_imported_does_equal_3():
    assert NO_PROGRESS_LIMIT == 3


def test_consecutive_discard_limit_when_imported_does_equal_5():
    assert CONSECUTIVE_DISCARD_LIMIT == 5


# ---------------------------------------------------------------------------
# behavior 3: GuardState and Decision types
# ---------------------------------------------------------------------------


def test_guard_state_when_constructed_does_start_counting_from_initial_record_count():
    gs = GuardState(initial_record_count=7)

    assert gs.last_record_count == 7
    assert gs.replies_sent == 0
    assert gs.no_progress_count == 0


def test_end_when_constructed_does_carry_reason():
    d = End(reason="finished")

    assert isinstance(d, Decision)
    assert d.reason == "finished"


def test_reply_when_constructed_does_carry_text():
    d = Reply(text="hello")

    assert isinstance(d, Decision)
    assert d.text == "hello"


def test_wait_for_lock_when_constructed_does_be_a_decision():
    d = WaitForLock()

    assert isinstance(d, Decision)


# ---------------------------------------------------------------------------
# behavior 1: stop_condition accepts BenchlessConfig
# ---------------------------------------------------------------------------


def test_classify_when_stop_condition_met_via_benchless_config_does_end_finished():
    """stop_condition reads only config.stop; a BenchlessConfig suffices."""
    config = _benchless_config(stop=StopConfig(max_iterations=2))
    state = session_state(iteration_count=2)
    guards = _guards()

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
    )

    assert isinstance(result, End)
    assert result.reason == "finished"


# ---------------------------------------------------------------------------
# behavior 4: classify evaluation order
# ---------------------------------------------------------------------------

# 4.i: finalized / ends_on_stop / stop_condition -> End("finished")


def test_classify_when_state_finalized_does_end_finished():
    config = _benchless_config()
    state = session_state(finalized=finalize_record())
    guards = _guards()

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
    )

    assert isinstance(result, End)
    assert result.reason == "finished"


def test_classify_when_ends_on_stop_does_end_finished():
    config = _benchless_config()
    state = session_state(ends_on_stop=True)
    guards = _guards()

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
    )

    assert isinstance(result, End)
    assert result.reason == "finished"


# 4.ii: budget_exhausted / max_usd -> End("spend-cap")


def test_classify_when_budget_exhausted_does_end_spend_cap():
    config = _benchless_config()
    state = session_state()
    guards = _guards()

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(budget_exhausted=True),
    )

    assert isinstance(result, End)
    assert result.reason == "spend-cap"


def test_classify_when_cost_exceeds_max_usd_does_end_spend_cap():
    config = _benchless_config()
    state = session_state()
    guards = _guards()

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(cost_usd=5.0),
        max_usd=4.0,
    )

    assert isinstance(result, End)
    assert result.reason == "spend-cap"


def test_classify_when_budget_exhausted_but_ends_on_stop_does_end_finished():
    """Rule 1 (finished) fires before rule 2 (budget).

    A budget-exhausted turn whose log ends on a stop record reads End("finished").
    """
    config = _benchless_config()
    state = session_state(ends_on_stop=True)
    guards = _guards()

    result = _classify(
        config=config,
        state=state,
        records=[stop_record()],
        guards=guards,
        turn=_turn_end(budget_exhausted=True),
    )

    assert isinstance(result, End)
    assert result.reason == "finished"


# 4.iii: lock_held -> WaitForLock


def test_classify_when_lock_held_does_return_wait_for_lock():
    config = _benchless_config()
    state = session_state()
    guards = _guards()

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
        lock_held=True,
    )

    assert isinstance(result, WaitForLock)


def test_classify_when_lock_held_does_not_mutate_guard_counters():
    """Every guard counter and the record-count baseline stay exactly as they were."""
    config = _benchless_config()
    state = session_state()
    guards = _guards(replies_sent=5, no_progress_count=1, last_record_count=3)
    original_replies = guards.replies_sent
    original_no_progress = guards.no_progress_count
    original_last_count = guards.last_record_count

    _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
        lock_held=True,
    )

    assert guards.replies_sent == original_replies
    assert guards.no_progress_count == original_no_progress
    assert guards.last_record_count == original_last_count


# 4.iv: guards -> End(reason)


def test_classify_when_replies_equal_follow_up_ceiling_does_end_follow_up_ceiling():
    config = _benchless_config()
    state = session_state()
    guards = _guards(replies_sent=FOLLOW_UP_CEILING)

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
    )

    assert isinstance(result, End)
    assert result.reason == "follow-up-ceiling"


def test_classify_when_no_progress_reaches_limit_does_end_no_progress():
    config = _benchless_config()
    state = session_state()
    # One below the limit; classify will increment to reach it.
    guards = _guards(
        replies_sent=1,
        no_progress_count=NO_PROGRESS_LIMIT - 1,
        last_record_count=0,
    )

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
    )

    assert isinstance(result, End)
    assert result.reason == "no-progress"


def test_classify_when_consecutive_discards_reach_limit_does_end_consecutive_discards():
    records: list[SessionLogRecord] = [
        discard_record(seq=i) for i in range(1, CONSECUTIVE_DISCARD_LIMIT + 1)
    ]

    result = _classify_discards(records)

    assert isinstance(result, End)
    assert result.reason == "consecutive-discards"


def test_classify_when_committed_keep_between_discards_does_reset_streak():
    """A committed keep resets the consecutive-discard counter."""
    records: list[SessionLogRecord] = [
        discard_record(seq=1),
        discard_record(seq=2),
        discard_record(seq=3),
        discard_record(seq=4),
        committed_keep(seq=5),
        discard_record(seq=6),
        discard_record(seq=7),
    ]

    result = _classify_discards(records)

    assert isinstance(result, Reply)


def test_classify_when_iteration_and_hook_records_between_discards_does_not_break_streak():
    """Iteration and hook records between discards do not break the run."""
    records: list[SessionLogRecord] = [
        discard_record(seq=1),
        iteration_record(seq=2),
        hook_record(seq=2),
        discard_record(seq=2),
        discard_record(seq=3),
        iteration_record(seq=4),
        discard_record(seq=4),
        discard_record(seq=5),
    ]

    result = _classify_discards(records)

    assert isinstance(result, End)
    assert result.reason == "consecutive-discards"


def test_classify_when_blocked_keep_between_discards_does_not_reset_streak():
    """Five discards with a blocked keep interleaved still trips the guard.

    Only a committed keep resets the counter.
    """
    records: list[SessionLogRecord] = [
        discard_record(seq=1),
        discard_record(seq=2),
        blocked_keep(seq=3),
        discard_record(seq=4),
        discard_record(seq=5),
        discard_record(seq=6),
    ]

    result = _classify_discards(records)

    assert isinstance(result, End)
    assert result.reason == "consecutive-discards"


# 4.v: otherwise -> Reply


def test_classify_when_no_condition_triggered_does_reply():
    config = _benchless_config()
    state = session_state()
    guards = _guards()

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
    )

    assert isinstance(result, Reply)


# ---------------------------------------------------------------------------
# behavior 5: no-progress accounting
# ---------------------------------------------------------------------------


def test_classify_when_no_new_records_since_last_reply_does_increment_no_progress():
    config = _benchless_config()
    state = session_state()
    guards = _guards(replies_sent=1, no_progress_count=0, last_record_count=0)

    _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
    )

    assert guards.no_progress_count == 1


def test_classify_when_new_records_appended_does_reset_no_progress():
    config = _benchless_config()
    state = session_state()
    guards = _guards(replies_sent=1, no_progress_count=2, last_record_count=0)
    records: list[SessionLogRecord] = [iteration_record()]

    _classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=_turn_end(),
    )

    assert guards.no_progress_count == 0


def test_classify_when_record_count_grows_does_move_baseline():
    """The record-count baseline moves to the current count on every non-WaitForLock."""
    config = _benchless_config()
    state = session_state()
    guards = _guards(last_record_count=0)
    records: list[SessionLogRecord] = [iteration_record(), iteration_record(seq=2)]

    _classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=_turn_end(),
    )

    assert guards.last_record_count == 2


def test_classify_when_first_turn_plus_three_stale_replies_does_end_no_progress_on_fourth():
    """Four turns with no new records: first is free, next three trip the guard.

    Turn 1: first classification (replies_sent=0, no-progress not counted)
    Turn 2: no new records, no_progress=1
    Turn 3: no new records, no_progress=2
    Turn 4: no new records, no_progress=3 -> End("no-progress")
    """
    config = _benchless_config()
    state = session_state()
    guards = _guards()
    records: list[SessionLogRecord] = []

    result1 = _classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=_turn_end(),
    )
    assert isinstance(result1, Reply)

    result2 = _classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=_turn_end(),
        now_ms=1000.0,
    )
    assert isinstance(result2, Reply)

    result3 = _classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=_turn_end(),
        now_ms=2000.0,
    )
    assert isinstance(result3, Reply)

    result4 = _classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=_turn_end(),
        now_ms=3000.0,
    )
    assert isinstance(result4, End)
    assert result4.reason == "no-progress"


def test_classify_when_record_appended_mid_sequence_does_reset_no_progress_counter():
    """A turn end that appended one record of any type resets the counter."""
    config = _benchless_config()
    state = session_state()
    guards = _guards()
    records: list[SessionLogRecord] = []

    _classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=_turn_end(),
    )

    _classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=_turn_end(),
        now_ms=1000.0,
    )
    assert guards.no_progress_count == 1

    records.append(iteration_record())
    _classify(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=_turn_end(),
        now_ms=2000.0,
    )
    assert guards.no_progress_count == 0


# ---------------------------------------------------------------------------
# behavior 6: empty records -> classifier replies
# ---------------------------------------------------------------------------


def test_classify_when_no_session_record_in_log_does_reply():
    config = _benchless_config()
    state = session_state()
    guards = _guards()

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
    )

    assert isinstance(result, Reply)


# ---------------------------------------------------------------------------
# behavior 7: Reply.text exact format
# ---------------------------------------------------------------------------


def test_classify_when_replying_does_include_runbook_instruction_and_time_left():
    config = _benchless_config()
    state = session_state()
    guards = _guards()
    deadline_ms = 600_000.0
    now_ms = 0.0
    max_minutes = 10.0

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
        deadline_ms=deadline_ms,
        max_minutes=max_minutes,
    )

    assert isinstance(result, Reply)
    remaining = deadline_ms - now_ms
    expected = (
        "No human is present. Run gymrat status to re-read the session, "
        "then decide from the runbook and continue. When the work is done, "
        "record your report with gymrat stop -m and end the turn.\n"
        f"{format_duration(remaining)} left of {max_minutes:g}m"
    )
    assert result.text == expected


def test_classify_when_replying_with_after_wait_does_append_wait_finished_line():
    config = _benchless_config()
    state = session_state()
    guards = _guards()
    deadline_ms = 600_000.0
    now_ms = 0.0
    max_minutes = 10.0

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
        deadline_ms=deadline_ms,
        max_minutes=max_minutes,
        after_wait=True,
    )

    assert isinstance(result, Reply)
    remaining = deadline_ms - now_ms
    expected = (
        "No human is present. Run gymrat status to re-read the session, "
        "then decide from the runbook and continue. When the work is done, "
        "record your report with gymrat stop -m and end the turn.\n"
        f"{format_duration(remaining)} left of {max_minutes:g}m\n"
        "The command you left running has finished; its record, if any, is in the session log."
    )
    assert result.text == expected


def test_classify_when_time_past_deadline_does_clamp_remaining_at_zero():
    config = _benchless_config()
    state = session_state()
    guards = _guards()

    result = _classify(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=_turn_end(),
        deadline_ms=1000.0,
        max_minutes=10.0,
        now_ms=5000.0,
    )

    assert isinstance(result, Reply)
    assert f"{format_duration(0)} left of 10m" in result.text


# ---------------------------------------------------------------------------
# behavior 8: turn text and origin are not inputs to any rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "origin"),
    [
        pytest.param("What should I do next?", "agent", id="question"),
        pytest.param("I completed the benchmark run.", "agent", id="report"),
        pytest.param("", "agent", id="empty-text"),
        pytest.param("done", "injected", id="injected-origin"),
    ],
)
def test_classify_when_turn_text_and_origin_vary_does_produce_same_decision_type(
    text: str, origin: Literal["agent", "injected"]
):
    config = _benchless_config()
    state = session_state()

    results = []
    pairs: list[tuple[str, Literal["agent", "injected"]]] = [
        ("baseline text", "agent"),
        (text, origin),
    ]
    for t, o in pairs:
        guards = _guards()
        result = _classify(
            config=config,
            state=state,
            records=[],
            guards=guards,
            turn=_turn_end(text=t, origin=o),
        )
        results.append(result)

    assert type(results[0]) is type(results[1])
