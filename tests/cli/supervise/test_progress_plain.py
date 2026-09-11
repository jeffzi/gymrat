"""Behavioral tests for the supervise reporter in plain (non-Live) mode.

Tests assert on recorded milestone lines.
"""

from __future__ import annotations

import pytest

from tests.cli.supervise._fixtures import (
    _throwing_read,
    fire_cap,
    fire_compaction,
    fire_follow_up,
    fire_launch,
    fire_tool_end,
    fire_tool_start,
    fire_turn_end,
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


# ---------------------------------------------------------------------------
# turn end / follow-up in plain mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "expected_suffix"),
    [
        pytest.param("replied", "replied", id="replied"),
        pytest.param("waiting", "waiting for gymrat", id="waiting"),
        pytest.param("ended", "ended budget exhausted", id="ended"),
    ],
)
def test_plain_when_follow_up_does_print_turn_count_and_action(action: str, expected_suffix: str):
    plain = make_plain_reporter()
    fire_launch(plain.observer, 1000)
    fire_turn_end(plain.observer, 2000, text="done")

    reason = "budget exhausted" if action == "ended" else None
    fire_follow_up(plain.observer, 3000, action=action, reason=reason)  # type: ignore[arg-type]

    assert any(f"turn 1 ended · {expected_suffix}" in w for w in plain.writes)


# ---------------------------------------------------------------------------
# cap event — interrupting vs ending
# ---------------------------------------------------------------------------


def test_plain_when_capped_while_idle_after_turn_end_does_print_cap_ending():
    plain = make_plain_reporter()
    fire_launch(plain.observer, 1000)
    fire_turn_end(plain.observer, 2000, text="done")
    fire_cap(plain.observer, "wall-clock", action="ending")

    assert plain.writes[-1] == "cap wall-clock — ending"


# ---------------------------------------------------------------------------
# compaction event — plain mode
# ---------------------------------------------------------------------------


def test_plain_when_compaction_does_print_context_compacted():
    plain = make_plain_reporter()
    fire_launch(plain.observer, 1000)
    fire_compaction(plain.observer, 3000)

    assert any("context compacted" in w for w in plain.writes)
