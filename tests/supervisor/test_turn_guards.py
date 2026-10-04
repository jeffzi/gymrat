"""Record accounting in the turn classifier's guards.

These cover the no-progress guard — which records count as progress, and only
outcome records at that — and the consecutive-discard streak, which skips the
records already present when the session launched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from gymrat.session.records import SessionLogRecord
from gymrat.supervisor.turns import (
    CONSECUTIVE_DISCARD_LIMIT,
    NO_PROGRESS_LIMIT,
    Decision,
    End,
    Reply,
    outcome_record_count,
)
from tests.cli.supervise._fixtures import session_state
from tests.session.records._fixtures import (
    blocked_keep,
    command_record,
    committed_keep,
    discard_record,
    hook_record,
    iteration_record,
    stop_record,
)
from tests.supervisor._fixtures import default_benchless_config
from tests.supervisor._turn_inputs import (
    classify_discards,
    classify_with_defaults,
    guard_state,
    turn_end,
)

# ---------------------------------------------------------------------------
# behavior 5: no-progress accounting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "records",
    [
        pytest.param([], id="no-records"),
        pytest.param([command_record()], id="commands-only"),
    ],
)
def test_classify_when_no_new_outcome_records_since_last_reply_does_increment_no_progress(
    records: list[SessionLogRecord],
):
    config = default_benchless_config()
    state = session_state()
    guards = guard_state(replies_sent=1, no_progress_count=0, last_record_count=0)

    classify_with_defaults(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=turn_end(),
    )

    assert guards.no_progress_count == 1


@pytest.mark.parametrize(
    "records",
    [
        pytest.param([iteration_record()], id="no-commands"),
        pytest.param(
            [command_record(), iteration_record(), command_record(seq=4)],
            id="commands-interleaved",
        ),
    ],
)
def test_classify_when_new_outcome_records_appended_does_reset_no_progress(
    records: list[SessionLogRecord],
):
    config = default_benchless_config()
    state = session_state()
    guards = guard_state(replies_sent=1, no_progress_count=2, last_record_count=0)

    classify_with_defaults(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=turn_end(),
    )

    assert guards.no_progress_count == 0


@pytest.mark.parametrize(
    "records",
    [
        pytest.param([iteration_record(), iteration_record(seq=2)], id="no-commands"),
        pytest.param(
            [iteration_record(), command_record(), iteration_record(seq=2)],
            id="commands-interleaved",
        ),
    ],
)
def test_classify_when_outcome_record_count_grows_does_move_baseline(
    records: list[SessionLogRecord],
):
    config = default_benchless_config()
    state = session_state()
    guards = guard_state(last_record_count=0)

    classify_with_defaults(
        config=config,
        state=state,
        records=records,
        guards=guards,
        turn=turn_end(),
    )

    assert guards.last_record_count == 2


@pytest.mark.parametrize(
    "records_per_turn",
    [
        pytest.param([[], [], [], []], id="no-records"),
        pytest.param(
            [
                [],
                [command_record()],
                [command_record(), command_record(seq=4)],
                [command_record(), command_record(seq=4), command_record(seq=5)],
            ],
            id="commands-only",
        ),
    ],
)
def test_classify_when_four_stale_turns_does_end_no_progress_on_fourth(
    records_per_turn: list[list[SessionLogRecord]],
):
    config = default_benchless_config()
    state = session_state()
    guards = guard_state()
    now_ms_steps = (0.0, 1000.0, 2000.0, 3000.0)

    results = [
        classify_with_defaults(
            config=config,
            state=state,
            records=records,
            guards=guards,
            turn=turn_end(),
            now_ms=now_ms,
        )
        for records, now_ms in zip(records_per_turn, now_ms_steps, strict=True)
    ]

    assert [type(result) for result in results] == [Reply, Reply, Reply, End]
    last_result = results[-1]
    assert isinstance(last_result, End)
    assert last_result.reason == "no-progress"


def test_classify_when_record_appended_mid_sequence_does_reset_no_progress_counter():
    config = default_benchless_config()
    state = session_state()
    guards = guard_state()
    records: list[SessionLogRecord] = []
    now_ms_steps = (0.0, 1000.0, 2000.0)

    no_progress_counts = []
    for index, now_ms in enumerate(now_ms_steps):
        if index == 2:
            records.append(iteration_record())
        classify_with_defaults(
            config=config,
            state=state,
            records=records,
            guards=guards,
            turn=turn_end(),
            now_ms=now_ms,
        )
        no_progress_counts.append(guards.no_progress_count)

    assert no_progress_counts == [0, 1, 0]


def test_classify_when_no_progress_reaches_limit_does_end_no_progress():
    config = default_benchless_config()
    state = session_state()
    # One below the limit; classify will increment to reach it.
    guards = guard_state(
        replies_sent=1,
        no_progress_count=NO_PROGRESS_LIMIT - 1,
        last_record_count=0,
    )

    result = classify_with_defaults(
        config=config,
        state=state,
        records=[],
        guards=guards,
        turn=turn_end(),
    )

    assert isinstance(result, End)
    assert result.reason == "no-progress"


# ---------------------------------------------------------------------------
# behavior 9: no-progress guard counts outcome records only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        pytest.param(
            [iteration_record(), discard_record(seq=1), hook_record()], 3, id="no-commands"
        ),
        pytest.param([command_record(), command_record(seq=4)], 0, id="only-commands"),
        pytest.param(
            [
                iteration_record(),
                command_record(),
                discard_record(seq=1),
                command_record(seq=4),
                stop_record(),
            ],
            3,
            id="mixed",
        ),
    ],
)
def test_outcome_record_count_when_records_vary_does_exclude_command_records(
    records: list[SessionLogRecord], expected: int
):
    assert outcome_record_count(records) == expected


# ---------------------------------------------------------------------------
# behavior 9a: consecutive-discard streak with interleaved settlement records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        pytest.param(
            [discard_record(seq=i) for i in range(1, CONSECUTIVE_DISCARD_LIMIT + 1)],
            "consecutive-discards",
            id="no-interleaving",
        ),
        pytest.param(
            [
                discard_record(seq=1),
                command_record(),
                discard_record(seq=2),
                discard_record(seq=3),
                command_record(seq=4),
                discard_record(seq=4),
                discard_record(seq=5),
            ],
            "consecutive-discards",
            id="commands-interleaved",
        ),
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
            "consecutive-discards",
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
            "consecutive-discards",
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
            None,
            id="committed-keep-resets",
        ),
    ],
)
def test_classify_when_settlement_records_interleaved_does_evaluate_discard_streak(
    records: list[SessionLogRecord], reason: str | None
):
    result = classify_discards(records)

    if reason is None:
        assert isinstance(result, Reply)
    else:
        assert isinstance(result, End)
        assert result.reason == reason


# ---------------------------------------------------------------------------
# behavior 9b: the discard streak skips records present at launch
# ---------------------------------------------------------------------------

_SEVEN_DISCARDS_AFTER_COMMANDS: list[SessionLogRecord] = [
    command_record(),
    discard_record(seq=1),
    command_record(seq=2),
    *(discard_record(seq=i) for i in range(2, 8)),
]


def _classify_discards_since_launch(
    records: list[SessionLogRecord], initial_record_count: int
) -> Decision:
    """Runs classify on the discard-streak defaults with a launch-time record count."""
    return classify_with_defaults(
        config=default_benchless_config(),
        state=session_state(),
        records=records,
        guards=guard_state(initial_record_count=initial_record_count),
        turn=turn_end(),
    )


@pytest.mark.parametrize(
    ("records", "initial_record_count"),
    [
        pytest.param(
            [
                discard_record(seq=1),
                command_record(),
                discard_record(seq=2),
                command_record(seq=3),
                *(discard_record(seq=i) for i in range(3, 6)),
            ],
            0,
            id="nothing-at-launch-commands-interleaved",
        ),
        pytest.param(
            _SEVEN_DISCARDS_AFTER_COMMANDS, 2, id="five-discards-after-launch-commands-skipped"
        ),
    ],
)
def test_classify_when_launch_records_skipped_leave_discard_limit_does_end_consecutive_discards(
    records: list[SessionLogRecord], initial_record_count: int
):
    result = _classify_discards_since_launch(records, initial_record_count)

    assert isinstance(result, End)
    assert result.reason == "consecutive-discards"


@pytest.mark.parametrize(
    ("records", "initial_record_count"),
    [
        pytest.param(
            _SEVEN_DISCARDS_AFTER_COMMANDS, 3, id="commands-at-launch-not-counted-as-skipped"
        ),
        pytest.param(
            [*(discard_record(seq=i) for i in range(1, 6)), command_record(seq=6)],
            5,
            id="launch-count-equals-outcome-records",
        ),
        pytest.param(
            [command_record(), *(discard_record(seq=i) for i in range(1, 6))],
            9,
            id="launch-count-exceeds-outcome-records",
        ),
    ],
)
def test_classify_when_launch_records_skipped_leave_fewer_than_limit_does_reply(
    records: list[SessionLogRecord], initial_record_count: int
):
    result = _classify_discards_since_launch(records, initial_record_count)

    assert isinstance(result, Reply)
