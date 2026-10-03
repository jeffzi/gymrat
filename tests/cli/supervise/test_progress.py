"""Behavioral tests for the supervise Rich dashboard reporter.

The reporter turns a stream of :class:`~gymrat.supervisor.events.SessionEvent`
values into a bordered ``Live`` dashboard with time/cost/loop summary rows and a
liveness section showing tool activity.  Every test injects the clock (``now``)
and the session reader (``read_session``) so nothing depends on real time or
disk.

**Live mode** tests render ``reporter.frame()`` through ``frame_text()`` from
``tests._rich`` at a fixed width, pinning frame content with syrupy snapshots.
The liveness line's own behavior lives in ``test_progress_liveness.py``; plain
mode tests live in ``test_progress_plain.py``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from gymrat.cli.supervise.progress import make_default_read
from gymrat.cli.supervise.types import ReadSessionResult
from gymrat.eta import NS_PER_MS
from gymrat.session.records import IterationPrimary
from gymrat.supervisor.events import CompactionEvent, TextDeltaEvent
from tests.cli.supervise._fixtures import (
    _throwing_read,
    cap_event,
    fire_launch_and_bash_cycle,
    follow_up_event,
    launch_event,
    make_iteration,
    make_read_session,
    make_reporter,
    render_frame,
    session_state_three_iterations,
    tool_end_event,
    tool_start_event,
    turn_end_event,
    usage_event,
)
from tests.session.records._fixtures import (
    baseline_record,
    blocked_keep,
    committed_keep,
    empty_session_state,
    finalize_record,
    iteration_record,
    session_record,
    session_state,
    stop_record,
    write_session_log,
)

if TYPE_CHECKING:
    from pathlib import Path

    from syrupy.assertion import SnapshotAssertion

    from gymrat.session.records import SessionLogRecord
    from gymrat.session.schema import PrimaryKind
    from gymrat.supervisor.events import CapAction, CapType


# ---------------------------------------------------------------------------
# factory / contract
# ---------------------------------------------------------------------------


def test_create_reporter_when_built_does_expose_frame():
    kit = make_reporter()

    frame = kit.reporter.frame()

    assert frame is not None


def test_create_reporter_when_session_read_does_expose_the_latest_session_result():
    state = session_state_three_iterations(-4.2, "improved", seq=3)
    kit = make_reporter(read_session=make_read_session(state, has_baseline=True))

    fire_launch_and_bash_cycle(kit.reporter.observer)

    session_result = kit.reporter.session_result()
    assert session_result is not None
    assert session_result.state == state


def _read_back(tmp_path: Path, history: tuple[SessionLogRecord, ...]) -> ReadSessionResult:
    """Write a session log under ``tmp_path`` and read it back the dashboard's way."""
    write_session_log(str(tmp_path), session_record(), history)
    return make_default_read(str(tmp_path))()


@pytest.mark.parametrize(
    ("kind", "name", "primary_label"),
    [
        pytest.param("geomean", None, "geomean", id="geomean-primary"),
        pytest.param("metric", "decode/time", "decode/time", id="metric-primary"),
    ],
)
def test_make_default_read_when_keeps_committed_does_report_the_best_committed_iteration(
    tmp_path: Path, kind: PrimaryKind, name: str | None, primary_label: str
):
    history = (
        iteration_record(seq=1, primary=IterationPrimary(kind=kind, name=name, delta_pct=-7.2)),
        committed_keep(1),
        iteration_record(seq=2, primary=IterationPrimary(kind=kind, name=name, delta_pct=-9.0)),
        committed_keep(2),
        iteration_record(seq=3, primary=IterationPrimary(kind=kind, name=name, delta_pct=-20.0)),
        blocked_keep(3),
    )

    result = _read_back(tmp_path, history)

    assert (
        result.best_delta_pct,
        result.best_seq,
        result.primary_label,
        result.baseline_sha,
        result.has_baseline,
        result.stop_message,
    ) == (-9.0, 2, primary_label, "a" * 40, False, None)


@pytest.mark.parametrize(
    ("history", "stop_message"),
    [
        pytest.param((stop_record(message="done for today"),), "done for today", id="ends-on-stop"),
        pytest.param(
            (stop_record(message="done for today"), iteration_record(seq=1)),
            None,
            id="stop-then-iteration",
        ),
    ],
)
def test_make_default_read_when_stop_recorded_does_report_it_only_while_last(
    tmp_path: Path, history: tuple[SessionLogRecord, ...], stop_message: str | None
):
    result = _read_back(tmp_path, history)

    assert result.stop_message == stop_message


@pytest.mark.parametrize(
    ("history", "has_baseline"),
    [
        pytest.param((), False, id="no-baseline"),
        pytest.param(
            (baseline_record(),),
            True,
            id="baseline-recorded",
        ),
    ],
)
def test_make_default_read_when_baseline_presence_varies_does_report_it(
    tmp_path: Path, history: tuple[SessionLogRecord, ...], has_baseline: bool
):
    result = _read_back(tmp_path, history)

    assert result.has_baseline is has_baseline


# ---------------------------------------------------------------------------
# panel structure — title
# ---------------------------------------------------------------------------


def test_panel_title_when_launched_does_contain_session_and_branch(
    snapshot: SnapshotAssertion,
):
    kit = make_reporter(
        session_id="20260813-125044-34ec",
        branch="gymrat/20260813-125044-34ec",
    )
    kit.reporter.observer(launch_event(1000))

    frame = render_frame(kit.reporter)

    assert frame == snapshot


def test_panel_title_when_all_identity_empty_does_show_bare_supervise():
    kit = make_reporter(session_id="", branch="")
    kit.reporter.observer(launch_event(1000))

    frame = render_frame(kit.reporter)
    title_line = frame.splitlines()[0]

    assert "supervise" in title_line
    assert "session" not in title_line
    assert "branch" not in title_line


# ---------------------------------------------------------------------------
# time bar
# ---------------------------------------------------------------------------


def test_time_bar_when_launched_does_show_elapsed_and_cap_in_remaining(
    snapshot: SnapshotAssertion,
):
    kit = make_reporter(max_minutes=480, clock_start=1000)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 1000 + (2 * 3600 + 41 * 60) * 1000

    frame = render_frame(kit.reporter)

    assert "2h 41m" in frame
    assert "cap in 5h 19m" in frame
    assert "eta" not in frame
    assert "/ 8h" not in frame
    assert frame == snapshot


def test_time_bar_when_elapsed_exceeds_max_does_clamp_remaining_to_zero(
    snapshot: SnapshotAssertion,
):
    kit = make_reporter(max_minutes=60, clock_start=1000)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 1000 + (2 * 3600) * 1000

    frame = render_frame(kit.reporter)

    assert "2h 00m" in frame
    assert "cap in 0s" in frame
    assert frame == snapshot


# ---------------------------------------------------------------------------
# cost row
# ---------------------------------------------------------------------------


def test_cost_when_no_cap_and_no_usage_does_show_zero_cost(snapshot: SnapshotAssertion):
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))

    frame = render_frame(kit.reporter)

    assert "cost" in frame
    assert "$0.00" in frame
    assert frame == snapshot


def test_cost_when_cap_set_and_usage_received_does_show_cost_against_cap(
    snapshot: SnapshotAssertion,
):
    kit = make_reporter(max_usd=10.0)
    kit.reporter.observer(launch_event(1000, max_usd=10.0))
    kit.clock.now = 2000
    kit.reporter.observer(usage_event(4.12, 2000))

    frame = render_frame(kit.reporter)

    assert "$4.12" in frame
    assert "$10.00" in frame
    assert frame == snapshot


def test_cost_when_no_cap_and_usage_received_does_show_bare_cost():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(usage_event(4.12, 2000))

    frame = render_frame(kit.reporter)

    assert "$4.12" in frame
    assert "/ $" not in frame


# ---------------------------------------------------------------------------
# loop row
# ---------------------------------------------------------------------------


def test_loop_when_read_session_throws_does_show_no_session_yet():
    kit = make_reporter(read_session=_throwing_read)
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "no session yet" in frame


def test_loop_when_baseline_recorded_does_name_the_missing_iterations():
    kit = make_reporter(
        max_iterations=20,
        read_session=make_read_session(empty_session_state(), has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "baseline recorded · no iterations yet" in frame


def test_loop_when_iterations_present_does_show_counts_and_last(snapshot: SnapshotAssertion):
    state = session_state_three_iterations(-3.2, "improved")
    kit = make_reporter(
        max_iterations=20,
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "3/20 iterations" in frame
    assert "2 kept" in frame
    assert "1 discarded" in frame
    assert "-3.2%" in frame
    assert "improved" in frame
    assert frame == snapshot


def test_loop_when_max_iterations_absent_does_omit_the_denominator():
    state = session_state(
        iteration_count=2,
        keep_count=1,
        discard_count=1,
        last_iteration=make_iteration(1.5, "regressed"),
    )
    kit = make_reporter(
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "2 iterations" in frame
    assert re.search(r"\d+/\d+ iterations", frame) is None


@pytest.mark.parametrize(
    "delta_pct",
    [
        pytest.param(None, id="missing"),
        pytest.param(float("nan"), id="undefined-arithmetic"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_loop_when_last_delta_is_missing_or_non_finite_does_render_em_dash(
    delta_pct: float | None,
):
    state = session_state(
        iteration_count=1,
        last_iteration=make_iteration(delta_pct, "no-signal"),
    )
    kit = make_reporter(
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "last — no-signal" in frame


@pytest.mark.parametrize(
    ("delta_pct", "outcome", "expected_segment"),
    [
        pytest.param(2.2, "regressed", "last +2.2% regressed", id="signed-regression"),
        pytest.param(-0.04, "neutral", "last 0.0% neutral", id="rounds-to-zero-unsigned"),
    ],
)
def test_loop_when_last_delta_is_finite_does_render_formatted_delta(
    delta_pct: float, outcome: str, expected_segment: str
):
    state = session_state(
        iteration_count=1,
        last_iteration=make_iteration(delta_pct, outcome),
    )
    kit = make_reporter(
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert expected_segment in frame


def test_loop_when_unsettled_does_append_unsettled():
    state = session_state(
        iteration_count=1,
        unsettled=True,
        last_iteration=make_iteration(-2.0, "improved"),
    )
    kit = make_reporter(
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "unsettled" in frame


def test_loop_when_finalized_does_report_finalized():
    state = session_state(
        iteration_count=3,
        keep_count=2,
        discard_count=1,
        last_iteration=make_iteration(-5.0, "improved"),
        finalized=finalize_record(),
    )
    kit = make_reporter(
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "finalized" in frame


# ---------------------------------------------------------------------------
# best row
# ---------------------------------------------------------------------------


def test_best_when_no_kept_iteration_does_omit_best_row():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))

    frame = render_frame(kit.reporter)

    assert "best" not in frame


def test_best_when_kept_iteration_exists_does_show_the_best_row():
    state = session_state(
        iteration_count=3,
        keep_count=1,
        discard_count=2,
        last_iteration=make_iteration(-6.8, "improved", seq=3),
    )
    kit = make_reporter(
        read_session=make_read_session(
            state,
            has_baseline=True,
            best_delta_pct=-6.8,
            best_seq=3,
        ),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "best" in frame


# ---------------------------------------------------------------------------
# true best tracking (#22) — new ReadSessionResult fields
# ---------------------------------------------------------------------------


def test_best_when_session_has_best_fields_does_show_delta_label_sha_and_iteration():
    state = session_state(
        iteration_count=5,
        keep_count=2,
        discard_count=3,
        last_iteration=make_iteration(-2.0, "improved", seq=5),
    )
    kit = make_reporter(
        read_session=make_read_session(
            state,
            has_baseline=True,
            best_delta_pct=-6.8,
            best_seq=3,
            primary_label="geomean",
            baseline_sha="2ec6e05abcdef1234567890abcdef1234567890a",
        ),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "best" in frame
    assert "-6.8%" in frame
    assert "geomean" in frame
    assert "2ec6e05" in frame
    assert "(iteration 3)" in frame


def test_best_when_last_kept_differs_from_best_does_show_best_not_last():
    state = session_state(
        iteration_count=5,
        keep_count=3,
        discard_count=2,
        last_iteration=make_iteration(-2.0, "improved", seq=5),
    )
    kit = make_reporter(
        read_session=make_read_session(
            state,
            has_baseline=True,
            best_delta_pct=-6.8,
            best_seq=3,
            primary_label="geomean",
            baseline_sha="abcdef1234567890abcdef1234567890abcdef12",
        ),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "(iteration 3)" in frame
    assert "(iteration 5)" not in frame


# ---------------------------------------------------------------------------
# session re-read
# ---------------------------------------------------------------------------


class CountingRead:
    """A ``read_session`` that tallies how many times it was called."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> ReadSessionResult:
        self.count += 1
        return ReadSessionResult(state=empty_session_state(), has_baseline=False)


def test_reread_when_any_tool_ends_does_reread():
    counting = CountingRead()
    kit = make_reporter(read_session=counting)
    observer = kit.reporter.observer

    observer(launch_event(1000))
    after_launch = counting.count
    observer(tool_start_event("Read", "read-1", 2000))
    observer(tool_end_event("Read", "read-1", 3000))
    after_read = counting.count
    observer(tool_start_event("Bash", "bash-1", 4000))
    observer(tool_end_event("Bash", "bash-1", 5000))
    after_bash = counting.count

    assert after_read > after_launch
    assert after_bash > after_read


def test_reread_when_tool_end_has_unknown_id_does_reread():
    counting = CountingRead()
    kit = make_reporter(read_session=counting)
    observer = kit.reporter.observer

    observer(launch_event(1000))
    after_launch = counting.count
    observer(tool_end_event("Bash", "unknown-id", 3000))

    assert counting.count > after_launch


# ---------------------------------------------------------------------------
# full dashboard golden snapshot
# ---------------------------------------------------------------------------


def test_dashboard_when_mid_session_does_render_full_layout(snapshot: SnapshotAssertion):
    state = session_state(
        iteration_count=3,
        keep_count=1,
        discard_count=1,
        last_iteration=make_iteration(-6.8, "improved", seq=3),
    )
    kit = make_reporter(
        max_minutes=480,
        max_usd=10.0,
        max_iterations=5,
        read_session=make_read_session(
            state,
            has_baseline=True,
            best_delta_pct=-6.8,
            best_seq=3,
            primary_label="geomean",
            baseline_sha="abc1234567890abcdef1234567890abcdef123456",
        ),
    )
    kit.reporter.observer(launch_event(1000, max_minutes=480, max_usd=10.0))

    kit.clock.now = 1000 + (2 * 3600 + 41 * 60) * 1000
    kit.reporter.observer(usage_event(4.12, kit.clock.now))

    kit.clock.now += 1000
    kit.reporter.observer(
        tool_start_event("Read", "read-1", kit.clock.now, input_summary="src/archetype.ts")
    )
    kit.clock.now += 500
    kit.reporter.observer(tool_end_event("Read", "read-1", kit.clock.now))

    kit.clock.now += 200
    kit.reporter.observer(
        tool_start_event("Edit", "edit-1", kit.clock.now, input_summary="src/archetype.ts")
    )
    kit.clock.now += 800
    kit.reporter.observer(tool_end_event("Edit", "edit-1", kit.clock.now))

    kit.clock.now += 100
    kit.reporter.observer(
        tool_start_event("Bash", "bash-1", kit.clock.now, input_summary="gymrat iterate")
    )
    kit.clock.now += 60_000

    frame = render_frame(kit.reporter)

    assert frame == snapshot


# ---------------------------------------------------------------------------
# cap event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", ["wall-clock", "spend-cap"])
@pytest.mark.parametrize("action", ["ending", "interrupting"])
def test_cap_when_fired_does_show_action_with_cap_type(cap: CapType, action: CapAction):
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.reporter.observer(cap_event(cap, action=action))

    frame = render_frame(kit.reporter)

    assert f"{action} ({cap})" in frame


def test_cap_when_fired_does_freeze_liveness_against_later_tool_events():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.reporter.observer(cap_event("wall-clock"))
    kit.clock.now = 6000
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 6000))
    kit.reporter.observer(tool_end_event("Bash", "bash-1", 7000))

    frame = render_frame(kit.reporter)

    assert "interrupting" in frame


# ---------------------------------------------------------------------------
# warn
# ---------------------------------------------------------------------------


def test_warn_when_called_in_live_mode_does_not_crash():
    kit = make_reporter(mode="live")
    kit.reporter.observer(launch_event(1000))

    kit.reporter.warn("something is wrong")

    frame = render_frame(kit.reporter)
    assert frame


# ---------------------------------------------------------------------------
# final_text
# ---------------------------------------------------------------------------


def test_final_text_when_no_text_received_does_return_none():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))

    assert kit.reporter.final_text() is None


def test_final_text_when_agent_turn_ends_does_return_its_text():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(turn_end_event(2000, text="first turn summary"))
    observer(turn_end_event(3000, text="second turn summary"))

    assert kit.reporter.final_text() == "second turn summary"


def test_final_text_when_injected_turn_ends_does_not_replace_agent_text():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(turn_end_event(2000, text="agent said this", origin="agent"))
    observer(turn_end_event(3000, text="injected turn text", origin="injected"))

    assert kit.reporter.final_text() == "agent said this"


def test_final_text_when_text_delta_received_does_not_set_final_text():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(TextDeltaEvent(at=2_000_000_000, chunk="streamed text"))

    assert kit.reporter.final_text() is None


# ---------------------------------------------------------------------------
# turn counting and follow-up rendering (live mode)
# ---------------------------------------------------------------------------


def test_follow_up_when_replied_does_show_turn_count_and_replied():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(turn_end_event(2000, text="done"))
    observer(follow_up_event(3000, action="replied"))

    frame = render_frame(kit.reporter)

    assert "turn 1 ended" in frame
    assert "replied" in frame


def test_follow_up_when_waiting_does_show_turn_count_and_waiting_for_gymrat():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(turn_end_event(2000, text="done"))
    observer(follow_up_event(3000, action="waiting"))

    frame = render_frame(kit.reporter)

    assert "turn 1 ended" in frame
    assert "waiting for gymrat" in frame


def test_follow_up_when_ended_does_show_turn_count_and_ended_with_reason():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(turn_end_event(2000, text="done"))
    observer(follow_up_event(3000, action="ended", reason="budget exhausted"))

    frame = render_frame(kit.reporter)

    assert "turn 1 ended" in frame
    assert "ended budget exhausted" in frame


def test_follow_up_when_multiple_turns_does_increment_turn_count():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(turn_end_event(2000, text="first"))
    observer(follow_up_event(3000, action="replied"))
    observer(turn_end_event(4000, text="second"))
    observer(follow_up_event(5000, action="replied"))

    frame = render_frame(kit.reporter)

    assert "turn 2 ended" in frame


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def test_stop_when_called_does_not_raise():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))

    kit.reporter.stop()


# ---------------------------------------------------------------------------
# compaction event — context compacted line
# ---------------------------------------------------------------------------


def test_compaction_when_fired_does_show_context_compacted_in_frame():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 3000
    kit.reporter.observer(CompactionEvent(at=3000 * NS_PER_MS))

    frame = render_frame(kit.reporter)

    assert "context compacted" in frame
