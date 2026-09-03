"""Behavioral tests for the supervise reporter in plain (non-Live) mode.

Tests assert on recorded milestone lines.
"""

from __future__ import annotations

import pytest

from tests.cli.supervise._fixtures import (
    _throwing_read,
    fire_cap,
    fire_launch,
    fire_tool_end,
    fire_tool_start,
    fire_usage_update,
    make_iteration,
    make_plain_reporter,
    make_read_session,
    session_state,
)


def test_plain_when_launched_with_spend_cap_does_print_caps_with_dollars():
    plain = make_plain_reporter(max_usd=5.0, max_minutes=60)

    fire_launch(plain.observer, 1000, max_usd=5.0)

    caps_line = plain.writes[-1]
    assert caps_line == "caps 60m, $5.00"


@pytest.mark.parametrize(
    ("max_minutes", "expected"),
    [
        pytest.param(30, "caps 30m", id="whole-int"),
        pytest.param(5.5, "caps 5.5m", id="fractional-keeps-decimal"),
        pytest.param(10.0, "caps 10m", id="whole-float-drops-decimal"),
    ],
)
def test_plain_caps_when_max_minutes_given_does_render_the_actual_cap_value(
    max_minutes: float, expected: str
):
    plain = make_plain_reporter(max_minutes=max_minutes)

    fire_launch(plain.observer, 1000, max_minutes=max_minutes)

    assert plain.writes[-1] == expected


def test_plain_when_usage_update_does_print_cost():
    plain = make_plain_reporter(max_usd=5.0)

    fire_launch(plain.observer, 1000, max_usd=5.0)
    fire_usage_update(plain.observer, 1.42, 2000)

    assert plain.writes[-1] == "cost $1.42"


def test_plain_when_loop_changes_does_print_loop_segment():
    state = session_state(
        iteration_count=2,
        keep_count=1,
        discard_count=1,
        last_iteration=make_iteration(3.2, "regressed"),
    )
    plain = make_plain_reporter(
        max_iterations=20,
        read_session=make_read_session(state, has_baseline=True),
    )

    fire_launch(plain.observer, 1000)
    fire_tool_start(plain.observer, "Bash", "bash-1", 2000)
    fire_tool_end(plain.observer, "Bash", "bash-1", 3000)

    assert plain.writes[-1] == "2/20 iterations · 1 kept · 1 discarded · last +3.2% regressed"


def test_plain_when_no_session_yet_does_not_print_loop_segment():
    plain = make_plain_reporter(read_session=_throwing_read)

    fire_launch(plain.observer, 1000)

    assert plain.writes[-1] == "caps 60m"
    assert all("no session yet" not in w for w in plain.writes)


def test_plain_when_capped_does_print_cap_interrupting():
    plain = make_plain_reporter()

    fire_launch(plain.observer, 1000)
    fire_cap(plain.observer, "wall-clock")

    cap_line = plain.writes[-1]
    assert cap_line == "cap wall-clock — interrupting"


def test_plain_when_warn_called_does_record_warning():
    plain = make_plain_reporter()
    fire_launch(plain.observer, 1000)

    plain.reporter.warn("heads up")

    assert plain.writes[-1] == "heads up"


def test_plain_stop_when_called_does_not_raise():
    plain = make_plain_reporter()
    fire_launch(plain.observer, 1000)

    plain.reporter.stop()
