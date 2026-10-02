"""Behavioral tests for the turn classifier, guard state, and reply text.

The classifier is a pure function over its arguments plus the ``GuardState``
it mutates. It never reads the filesystem or the driver. Tests build states
with ``SessionState`` fixtures and record lists from the session record models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

from gymrat.config.types import StopConfig
from gymrat.eta import format_duration

if TYPE_CHECKING:
    from gymrat.session.records import SessionLogRecord
from gymrat.supervisor.turns import (
    CONSECUTIVE_DISCARD_LIMIT,
    FOLLOW_UP_CEILING,
    NO_PROGRESS_LIMIT,
    Decision,
    End,
    EndCondition,
    GuardState,
    Reply,
    WaitForLock,
    detect_end_condition,
)
from tests.session.records._fixtures import (
    command_record,
    finalize_record,
    hook_record,
    iteration_record,
    session_state,
    stop_record,
)
from tests.supervisor._fixtures import default_benchless_config
from tests.supervisor._turn_inputs import (
    classify_with_defaults,
    guard_state,
    turn_end,
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
    config = default_benchless_config(stop=StopConfig(max_iterations=2))
    state = session_state(iteration_count=2)
    guards = guard_state()

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
    )

    assert isinstance(result, End)
    assert result.reason == "finished"


# ---------------------------------------------------------------------------
# behavior 4: classify evaluation order
# ---------------------------------------------------------------------------

# 4.i: finalized / ends_on_stop / stop_condition -> End("finished")


def test_classify_when_state_finalized_does_end_finished():
    config = default_benchless_config()
    state = session_state(finalized=finalize_record())
    guards = guard_state()

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
    )

    assert isinstance(result, End)
    assert result.reason == "finished"


def test_classify_when_ends_on_stop_does_end_finished():
    config = default_benchless_config()
    state = session_state(ends_on_stop=True)
    guards = guard_state()

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
    )

    assert isinstance(result, End)
    assert result.reason == "finished"


# 4.ii: budget_exhausted / max_usd -> End("spend-cap")


def test_classify_when_budget_exhausted_does_end_spend_cap():
    config = default_benchless_config()
    state = session_state()
    guards = guard_state()

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(budget_exhausted=True),
    )

    assert isinstance(result, End)
    assert result.reason == "spend-cap"


def test_classify_when_cost_exceeds_max_usd_does_end_spend_cap():
    config = default_benchless_config()
    state = session_state()
    guards = guard_state()

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(cost_usd=5.0),
        max_usd=4.0,
    )

    assert isinstance(result, End)
    assert result.reason == "spend-cap"


def test_classify_when_budget_exhausted_but_ends_on_stop_does_end_finished():
    config = default_benchless_config()
    state = session_state(ends_on_stop=True)
    guards = guard_state()

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[stop_record()],
        guards=guards,
        turn=turn_end(budget_exhausted=True),
    )

    assert isinstance(result, End)
    assert result.reason == "finished"


# 4.iii: lock_held -> WaitForLock


def test_classify_when_lock_held_does_return_wait_for_lock():
    config = default_benchless_config()
    state = session_state()
    guards = guard_state()

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
        lock_held=True,
    )

    assert isinstance(result, WaitForLock)


def test_classify_when_lock_held_does_not_mutate_guard_counters():
    config = default_benchless_config()
    state = session_state()
    guards = guard_state(replies_sent=5, no_progress_count=1, last_record_count=3)
    original_replies = guards.replies_sent
    original_no_progress = guards.no_progress_count
    original_last_count = guards.last_record_count

    classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
        lock_held=True,
    )

    assert guards.replies_sent == original_replies
    assert guards.no_progress_count == original_no_progress
    assert guards.last_record_count == original_last_count


# 4.iv: guards -> End(reason)


def test_classify_when_replies_equal_follow_up_ceiling_does_end_follow_up_ceiling():
    config = default_benchless_config()
    state = session_state()
    guards = guard_state(replies_sent=FOLLOW_UP_CEILING)

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
    )

    assert isinstance(result, End)
    assert result.reason == "follow-up-ceiling"


# 4.v: otherwise -> Reply, including the empty-records case (behavior 6)


def test_classify_when_no_condition_triggered_does_reply():
    config = default_benchless_config()
    state = session_state()
    guards = guard_state()

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
    )

    assert isinstance(result, Reply)


# ---------------------------------------------------------------------------
# behavior 7: Reply.text exact format
# ---------------------------------------------------------------------------


def test_classify_when_replying_does_include_runbook_instruction_and_time_left():
    config = default_benchless_config()
    state = session_state()
    guards = guard_state()
    deadline_ms = 600_000.0
    now_ms = 0.0
    max_minutes = 10.0

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
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
    config = default_benchless_config()
    state = session_state()
    guards = guard_state()
    deadline_ms = 600_000.0
    now_ms = 0.0
    max_minutes = 10.0

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
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
    config = default_benchless_config()
    state = session_state()
    guards = guard_state()

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
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
    config = default_benchless_config()
    state = session_state()

    results = []
    pairs: list[tuple[str, Literal["agent", "injected"]]] = [
        ("baseline text", "agent"),
        (text, origin),
    ]
    for t, o in pairs:
        guards = guard_state()
        result = classify_with_defaults(
            config=config,
            state=state,
            records=[],
            guards=guards,
            turn=turn_end(text=t, origin=o),
        )
        results.append(result)

    assert type(results[0]) is type(results[1])


# ---------------------------------------------------------------------------
# behavior 10: detect_end_condition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cursor", [pytest.param(None, id="no-cursor"), pytest.param(0, id="cursor")]
)
def test_detect_end_condition_when_stop_condition_met_does_report_stop_condition(
    cursor: int | None,
):
    config = default_benchless_config(stop=StopConfig(max_iterations=2))
    state = session_state(iteration_count=2)
    records: list[SessionLogRecord] = [iteration_record(seq=1), iteration_record(seq=2)]

    result = detect_end_condition(config, records, state, cursor=cursor, check_stop=True)

    assert result == (
        EndCondition(ended_by="stop-condition", reason="max iterations (2 of 2)"),
        2,
    )


def test_detect_end_condition_when_state_is_met_but_records_are_not_does_use_state():
    config = default_benchless_config(stop=StopConfig(max_iterations=2))
    state = session_state(iteration_count=2)

    result = detect_end_condition(config, [], state, cursor=0, check_stop=True)

    assert result == (
        EndCondition(ended_by="stop-condition", reason="max iterations (2 of 2)"),
        0,
    )


def test_detect_end_condition_when_stop_met_but_check_stop_false_does_report_nothing():
    config = default_benchless_config(stop=StopConfig(max_iterations=2))
    state = session_state(iteration_count=2)
    records: list[SessionLogRecord] = [iteration_record(seq=1), iteration_record(seq=2)]

    result = detect_end_condition(config, records, state, cursor=0, check_stop=False)

    assert result == (None, 2)


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        pytest.param(
            [
                hook_record(stage="before", seq=2, exit_code=3, stdout_bytes=80, stderr_bytes=12),
                iteration_record(seq=2),
                hook_record(stage="after", seq=2),
            ],
            "before hook failed on iteration 2: exit 3 (stdout 80 B, stderr 12 B)",
            id="non-zero-exit-two-back-from-tail",
        ),
        pytest.param(
            [
                command_record(),
                hook_record(stage="after", seq=4, exit_code=1, timed_out=True, stderr_bytes=None),
            ],
            "after hook failed on iteration 4: timed out (stdout 80 B, stderr ? B)",
            id="timed-out-without-stderr-count",
        ),
        pytest.param(
            [
                hook_record(stage="after", seq=2, exit_code=5, stderr_bytes=1),
                iteration_record(seq=3),
                hook_record(stage="after", seq=3, exit_code=6, stderr_bytes=2),
            ],
            "after hook failed on iteration 2: exit 5 (stdout 80 B, stderr 1 B)",
            id="earliest-of-several-failures",
        ),
    ],
)
def test_detect_end_condition_when_hook_failed_after_cursor_does_report_hook_failure(
    records: list[SessionLogRecord], reason: str
):
    result = detect_end_condition(
        default_benchless_config(), records, session_state(), cursor=0, check_stop=True
    )

    assert result == (EndCondition(ended_by="hook-failure", reason=reason), len(records))


@pytest.mark.parametrize(
    ("records", "cursor"),
    [
        pytest.param(
            [hook_record(seq=1), command_record(), iteration_record(seq=1)],
            0,
            id="clean-hooks",
        ),
        pytest.param(
            [hook_record(seq=1, exit_code=1), iteration_record(seq=1)],
            1,
            id="failure-before-cursor",
        ),
        pytest.param(
            [hook_record(seq=1, exit_code=1), iteration_record(seq=1)],
            None,
            id="no-cursor",
        ),
    ],
)
def test_detect_end_condition_when_no_failure_in_scan_and_no_stop_does_report_nothing(
    records: list[SessionLogRecord], cursor: int | None
):
    result = detect_end_condition(
        default_benchless_config(), records, session_state(), cursor=cursor, check_stop=True
    )

    assert result == (None, len(records))


def test_detect_end_condition_when_hook_failed_and_stop_met_does_report_hook_failure():
    config = default_benchless_config(stop=StopConfig(max_iterations=1))
    state = session_state(iteration_count=1)
    records: list[SessionLogRecord] = [
        iteration_record(seq=1),
        hook_record(stage="after", seq=1, exit_code=2, stderr_bytes=0),
    ]

    result = detect_end_condition(config, records, state, cursor=0, check_stop=True)

    assert result == (
        EndCondition(
            ended_by="hook-failure",
            reason="after hook failed on iteration 1: exit 2 (stdout 80 B, stderr 0 B)",
        ),
        2,
    )
