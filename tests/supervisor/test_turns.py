"""Behavioral tests for the turn classifier, guard state, and reply text.

The classifier is a pure function over its arguments plus the ``GuardState``
it mutates. It never reads the filesystem or the driver. Tests build states
with ``SessionState`` fixtures and record lists from the session record models.

The guards' record accounting is covered too: which records count as progress
for the no-progress guard (outcome records only), and the consecutive-discard
streak, which skips the records already present when the session launched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from gymrat.session.budget import Budget
from gymrat.supervisor.turns import (
    FOLLOW_UP_CEILING,
    NO_PROGRESS_LIMIT,
    Decision,
    End,
    GuardState,
    Reply,
    WaitForLock,
    classify,
    outcome_record_count,
)
from gymrat.utils import format_time_left
from tests.session.records._fixtures import (
    blocked_keep,
    command_record,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    iteration_record,
    session_state,
    stop_record,
)
from tests.supervisor._fixtures import make_turn_end

if TYPE_CHECKING:
    from gymrat.session.records import SessionLogRecord
    from gymrat.session.store import SessionState
    from gymrat.supervisor.events import TurnEndEvent


def guard_state(
    *,
    initial_record_count: int = 0,
    replies_sent: int = 0,
    no_progress_count: int = 0,
    last_record_count: int | None = None,
) -> GuardState:
    """A guard state preset to the given counters."""
    gs = GuardState(initial_record_count=initial_record_count)
    gs.replies_sent = replies_sent
    gs.no_progress_count = no_progress_count
    if last_record_count is not None:
        gs.last_record_count = last_record_count
    return gs


def classify_with_defaults(
    *,
    state: SessionState | None = None,
    records: list[SessionLogRecord] | None = None,
    guards: GuardState | None = None,
    turn: TurnEndEvent | None = None,
    lock_held: bool = False,
    max_usd: float | None = None,
    budget: Budget | None = None,
    now_ms: float = 0.0,
    after_wait: bool = False,
) -> Decision:
    """Delegates to ``classify``, filling every argument a test leaves out with a neutral default.

    Args:
        state: The folded session state; an empty open session when ``None``.
        records: The session log records; none when ``None``.
        guards: The guard counters ``classify`` mutates; fresh ones when ``None``.
        turn: The turn that just ended; an agent turn with budget left when ``None``.
        lock_held: Whether another command holds the repository lock.
        max_usd: The spend cap, or ``None`` for none.
        budget: The session's wall-clock budget; an hour that never runs out when ``None``.
        now_ms: The current monotonic time, in milliseconds.
        after_wait: Whether the turn follows a wait for the lock.

    Returns:
        The decision ``classify`` reaches.
    """
    return classify(
        state=state if state is not None else session_state(),
        records=records if records is not None else [],
        guards=guards if guards is not None else guard_state(),
        turn=turn if turn is not None else make_turn_end(),
        lock_held=lock_held,
        max_usd=max_usd,
        budget=budget if budget is not None else Budget(max_minutes=60, deadline_ms=999_999_999.0),
        now_ms=now_ms,
        after_wait=after_wait,
    )


def _shape(decision: Decision) -> Decision | type[Reply]:
    """``decision`` itself, or the ``Reply`` class for a reply, whose text the reply tests pin."""
    return Reply if isinstance(decision, Reply) else decision


# ---------------------------------------------------------------------------
# classify evaluation order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param(
            {
                "state": session_state(ends_on_stop=True),
                "records": [stop_record()],
                "turn": make_turn_end(budget_exhausted=True),
            },
            End(reason="finished"),
            id="budget-exhausted-but-ends-on-stop",
        ),
        pytest.param(
            {"turn": make_turn_end(cost_usd=4.0), "max_usd": 4.0},
            End(reason="spend-cap"),
            id="cost-equal-to-max-usd",
        ),
        pytest.param(
            {"state": session_state(finalized=finalize_record()), "lock_held": True},
            End(reason="finished"),
            id="finished-before-lock",
        ),
        pytest.param(
            {"turn": make_turn_end(budget_exhausted=True), "lock_held": True},
            End(reason="spend-cap"),
            id="spend-cap-before-lock",
        ),
        pytest.param(
            {"lock_held": True, "guards": guard_state(replies_sent=FOLLOW_UP_CEILING)},
            WaitForLock(),
            id="lock-before-follow-up-ceiling",
        ),
        pytest.param(
            {
                "guards": guard_state(
                    replies_sent=FOLLOW_UP_CEILING,
                    no_progress_count=NO_PROGRESS_LIMIT - 1,
                    last_record_count=0,
                )
            },
            End(reason="follow-up-ceiling"),
            id="follow-up-ceiling-before-no-progress",
        ),
    ],
)
def test_classify_when_conditions_vary_does_decide_in_evaluation_order(
    arguments: dict[str, Any], expected: Decision | type[Reply]
):
    result = classify_with_defaults(**arguments)

    assert _shape(result) == expected


def test_classify_when_lock_held_does_wait_without_touching_guard_counters():
    guards = guard_state(replies_sent=5, no_progress_count=1, last_record_count=3)

    result = classify_with_defaults(guards=guards, lock_held=True)

    assert result == WaitForLock()
    assert (guards.replies_sent, guards.no_progress_count, guards.last_record_count) == (5, 1, 3)


# ---------------------------------------------------------------------------
# Reply.text time left and trailing line
# ---------------------------------------------------------------------------


def test_classify_when_replying_does_state_the_time_left():
    budget = Budget(max_minutes=10.0, deadline_ms=600_000.0)

    result = classify_with_defaults(budget=budget, now_ms=0.0)

    assert isinstance(result, Reply)
    assert result.text.splitlines()[1] == format_time_left(600_000, 10.0)


@pytest.mark.parametrize(
    ("after_wait", "line_count"),
    [
        pytest.param(True, 3, id="after-wait-appends-the-finished-line"),
        pytest.param(False, 2, id="no-wait-omits-the-finished-line"),
    ],
)
def test_classify_when_replying_does_append_the_finished_line_only_after_a_wait(
    *, after_wait: bool, line_count: int
):
    budget = Budget(max_minutes=10.0, deadline_ms=600_000.0)

    result = classify_with_defaults(budget=budget, now_ms=0.0, after_wait=after_wait)

    assert isinstance(result, Reply)
    assert len(result.text.splitlines()) == line_count


# ---------------------------------------------------------------------------
# no-progress accounting
# ---------------------------------------------------------------------------


def test_classify_when_new_outcome_records_appended_does_reset_the_no_progress_streak():
    guards = guard_state(replies_sent=1, no_progress_count=2, last_record_count=0)

    classify_with_defaults(records=[iteration_record(), iteration_record(seq=2)], guards=guards)

    assert (guards.no_progress_count, guards.last_record_count) == (0, 2)


def test_classify_when_four_stale_turns_does_end_no_progress_on_fourth():
    guards = guard_state()

    results = [
        classify_with_defaults(records=[], guards=guards, now_ms=now_ms)
        for now_ms in (0.0, 1000.0, 2000.0, 3000.0)
    ]

    assert [type(result) for result in results[:3]] == [Reply] * 3
    assert results[3] == End(reason="no-progress")


def test_classify_when_turns_add_only_command_records_does_end_no_progress():
    guards = guard_state(
        replies_sent=1, no_progress_count=NO_PROGRESS_LIMIT - 1, last_record_count=0
    )

    result = classify_with_defaults(
        records=[command_record(), command_record(seq=2)], guards=guards
    )

    assert result == End(reason="no-progress")


# ---------------------------------------------------------------------------
# no-progress guard counts outcome records only
# ---------------------------------------------------------------------------


def test_outcome_record_count_when_records_mix_commands_does_exclude_command_records():
    records: list[SessionLogRecord] = [
        iteration_record(),
        command_record(),
        discard_record(seq=1),
        command_record(seq=4),
        stop_record(),
    ]

    count = outcome_record_count(records)

    assert count == 3


# ---------------------------------------------------------------------------
# consecutive-discard streak, interleaved records and launch-time records
# ---------------------------------------------------------------------------


_SEVEN_DISCARDS_AFTER_COMMANDS: list[SessionLogRecord] = [
    command_record(),
    discard_record(seq=1),
    command_record(seq=2),
    *(discard_record(seq=i) for i in range(2, 8)),
]


@pytest.mark.parametrize(
    ("records", "initial_record_count", "ends"),
    [
        pytest.param(
            [
                discard_record(seq=1),
                iteration_record(seq=2),
                hook_record(seq=2),
                discard_record(seq=2),
                discard_record(seq=3),
                iteration_record(seq=4),
                discard_record(seq=4),
                discard_record(seq=5),
            ],
            0,
            True,
            id="iteration-and-hook-interleaved",
        ),
        pytest.param(
            [
                discard_record(seq=1),
                discard_record(seq=2),
                blocked_keep(seq=3),
                discard_record(seq=4),
                discard_record(seq=5),
                discard_record(seq=6),
            ],
            0,
            True,
            id="blocked-keep-interleaved",
        ),
        pytest.param(
            [
                discard_record(seq=1),
                discard_record(seq=2),
                discard_record(seq=3),
                discard_record(seq=4),
                committed_keep(seq=5),
                discard_record(seq=6),
                discard_record(seq=7),
            ],
            0,
            False,
            id="committed-keep-resets",
        ),
        pytest.param(
            _SEVEN_DISCARDS_AFTER_COMMANDS,
            3,
            False,
            id="commands-at-launch-not-counted-as-skipped",
        ),
    ],
)
def test_classify_when_discards_follow_launch_does_end_on_the_consecutive_limit(
    records: list[SessionLogRecord], initial_record_count: int, *, ends: bool
):
    result = classify_with_defaults(
        records=records, guards=guard_state(initial_record_count=initial_record_count)
    )

    assert _shape(result) == (End(reason="consecutive-discards") if ends else Reply)
