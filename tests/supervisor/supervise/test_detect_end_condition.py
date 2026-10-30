"""Unit tests for ``detect_end_condition``, the scan that ends a supervised run.

They pin which stop conditions end the session (and when ``check_stop`` silences
them), which hook failures end it and with what reason (a hook failure wins over
a met stop condition), and how the scan cursor bounds the hook-failure scan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.config import StopConfig
from gymrat.loop.stop_condition import stop_reason
from gymrat.supervisor.supervise import EndCondition, detect_end_condition
from tests._config import benchless_config
from tests.session.records._fixtures import (
    command_record,
    hook_record,
    iteration_record,
    session_state,
)

if TYPE_CHECKING:
    from gymrat.session.records import SessionLogRecord


def test_detect_end_condition_when_stop_condition_met_does_report_it_from_the_state():
    config = benchless_config(stop=StopConfig(max_iterations=2))
    state = session_state(iteration_count=2)

    result = detect_end_condition(config, [], state, cursor=0, check_stop=True)

    assert result == EndCondition(ended_by="stop-condition", reason=str(stop_reason(config, state)))


def test_detect_end_condition_when_stop_met_but_check_stop_false_does_report_nothing():
    config = benchless_config(stop=StopConfig(max_iterations=2))
    state = session_state(iteration_count=2)
    records: list[SessionLogRecord] = [iteration_record(seq=1), iteration_record(seq=2)]

    result = detect_end_condition(config, records, state, cursor=0, check_stop=False)

    assert result is None


@pytest.mark.parametrize(
    ("records", "fragments"),
    [
        pytest.param(
            [
                hook_record(stage="before", seq=7, exit_code=3, stdout_bytes=80, stderr_bytes=12),
                iteration_record(seq=7),
                hook_record(stage="after", seq=7),
            ],
            ("before", "7", "exit 3", "80", "12"),
            id="non-zero-exit-two-back-from-tail",
        ),
        pytest.param(
            [
                command_record(),
                hook_record(stage="after", seq=4, exit_code=1, timed_out=True, stderr_bytes=None),
            ],
            ("after", "4", "timed out", "?"),
            id="timed-out-without-stderr-count",
        ),
        pytest.param(
            [
                hook_record(stage="after", seq=2, exit_code=5, stderr_bytes=1),
                iteration_record(seq=3),
                hook_record(stage="after", seq=3, exit_code=6, stderr_bytes=4),
            ],
            ("2", "exit 5"),
            id="earliest-of-several-failures",
        ),
    ],
)
def test_detect_end_condition_when_hook_failed_after_cursor_does_report_hook_failure(
    records: list[SessionLogRecord], fragments: tuple[str, ...]
):
    result = detect_end_condition(
        benchless_config(), records, session_state(), cursor=0, check_stop=True
    )

    assert result is not None
    assert result.ended_by == "hook-failure"
    assert [fragment for fragment in fragments if fragment not in result.reason] == []


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
    ],
)
def test_detect_end_condition_when_no_failure_in_scan_and_no_stop_does_report_nothing(
    records: list[SessionLogRecord], cursor: int
):
    result = detect_end_condition(
        benchless_config(), records, session_state(), cursor=cursor, check_stop=True
    )

    assert result is None


def test_detect_end_condition_when_hook_failed_and_stop_met_does_report_hook_failure():
    config = benchless_config(stop=StopConfig(max_iterations=1))
    state = session_state(iteration_count=1)
    records: list[SessionLogRecord] = [
        iteration_record(seq=1),
        hook_record(stage="after", seq=1, exit_code=2, stderr_bytes=0),
    ]

    result = detect_end_condition(config, records, state, cursor=0, check_stop=True)

    assert result is not None
    assert result.ended_by == "hook-failure"
