"""Behavioral tests for the supervise Rich dashboard reporter.

The reporter turns a stream of :class:`~gymrat.supervisor.events.SessionEvent`
values into a bordered ``Live`` dashboard with time/cost/loop summary rows and a
liveness section showing tool activity.  Every test injects the clock (``now``)
and the session reader (``read_session``) so nothing depends on real time or
disk.

**Live mode** tests render ``reporter.frame()`` through ``frame_text()`` from
``tests._rich`` at a fixed width, pinning frame content with syrupy snapshots.
Liveness, waiting, sidecar, cap, and non-rendering event tests also live here.
Plain mode tests live in ``test_progress_plain.py``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from gymrat.cli.supervise.progress import IDLE_WARN_MS, CapType, ReadSessionResult
from gymrat.session.progress_file import ProgressSnapshot
from gymrat.supervisor.events import (
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolProgressEvent,
)
from tests.cli.supervise._fixtures import (
    _epoch_ms_to_local_hms,
    _throwing_read,
    empty_session_state,
    finalize_record,
    fire_cap,
    fire_launch,
    fire_launch_and_bash_cycle,
    fire_launch_and_bash_start,
    fire_model_phase,
    fire_thinking_update,
    fire_tool_end,
    fire_tool_start,
    fire_usage_update,
    make_iteration,
    make_read_session,
    make_reporter,
    render_frame,
    session_state,
    session_state_three_iterations,
)

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


# ---------------------------------------------------------------------------
# factory / contract
# ---------------------------------------------------------------------------


def test_create_reporter_when_built_does_expose_frame():
    kit = make_reporter()

    frame = kit.reporter.frame()

    assert frame is not None


def test_create_reporter_when_session_read_does_expose_the_latest_session_result():
    """The closing summary reads the final session state off the reporter."""
    state = session_state_three_iterations(-4.2, "improved", seq=3)
    kit = make_reporter(read_session=make_read_session(state, has_baseline=True))

    fire_launch_and_bash_cycle(kit.reporter.observer)

    session_result = kit.reporter.session_result()
    assert session_result is not None
    assert session_result.state == state


def test_create_reporter_when_color_false_does_build_colorless_console():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        make_reporter(mode="live", color=False)

        call_kwargs = mock_live_cls.call_args.kwargs
        console = call_kwargs.get("console")
        assert console is not None
        assert console.color_system is None


# ---------------------------------------------------------------------------
# panel structure — title
# ---------------------------------------------------------------------------


def test_panel_title_when_launched_does_contain_label_session_and_branch(
    snapshot: SnapshotAssertion,
):
    kit = make_reporter(
        label="ecstatic-ts",
        session_id="20260813-125044-34ec",
        branch="gymrat/20260813-125044-34ec",
    )
    fire_launch(kit.reporter.observer, 1000)

    frame = render_frame(kit.reporter)

    assert frame == snapshot


def test_panel_title_when_all_identity_empty_does_show_bare_supervise():
    kit = make_reporter(label="", session_id="", branch="")
    fire_launch(kit.reporter.observer, 1000)

    frame = render_frame(kit.reporter)
    title_line = frame.splitlines()[0]

    assert "supervise" in title_line
    assert "session" not in title_line
    assert "branch" not in title_line


def test_panel_title_when_only_label_present_does_omit_session_and_branch():
    kit = make_reporter(label="ecstatic-ts", session_id="", branch="")
    fire_launch(kit.reporter.observer, 1000)

    frame = render_frame(kit.reporter)
    title_line = frame.splitlines()[0]

    assert "supervise ecstatic-ts" in title_line
    assert "session" not in title_line
    assert "branch" not in title_line


# ---------------------------------------------------------------------------
# time bar
# ---------------------------------------------------------------------------


def test_time_bar_when_launched_does_show_elapsed_and_cap_in_remaining(
    snapshot: SnapshotAssertion,
):
    kit = make_reporter(max_minutes=480, clock_start=1000)
    fire_launch(kit.reporter.observer, 1000)
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
    fire_launch(kit.reporter.observer, 1000)
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
    fire_launch(kit.reporter.observer, 1000)

    frame = render_frame(kit.reporter)

    assert "cost" in frame
    assert "$0.00" in frame
    assert frame == snapshot


def test_cost_when_cap_set_and_usage_received_does_show_cost_against_cap(
    snapshot: SnapshotAssertion,
):
    kit = make_reporter(max_usd=10.0)
    fire_launch(kit.reporter.observer, 1000, max_usd=10.0)
    kit.clock.now = 2000
    fire_usage_update(kit.reporter.observer, 4.12, 2000)

    frame = render_frame(kit.reporter)

    assert "$4.12" in frame
    assert "$10.00" in frame
    assert frame == snapshot


def test_cost_when_no_cap_and_usage_received_does_show_bare_cost():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_usage_update(kit.reporter.observer, 4.12, 2000)

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


def test_loop_when_last_delta_is_none_does_render_em_dash():
    state = session_state(
        iteration_count=1,
        last_iteration=make_iteration(None, "no-signal"),
    )
    kit = make_reporter(
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "—" in frame
    assert "no-signal" in frame


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
    fire_launch(kit.reporter.observer, 1000)

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


def test_reread_when_bash_tool_ends_does_reread_unlike_non_bash():
    counting = CountingRead()
    kit = make_reporter(read_session=counting)
    observer = kit.reporter.observer

    fire_launch(observer, 1000)
    after_launch = counting.count
    fire_tool_start(observer, "Read", "read-1", 2000)
    fire_tool_end(observer, "Read", "read-1", 3000)
    after_read = counting.count
    fire_tool_start(observer, "Bash", "bash-1", 4000)
    fire_tool_end(observer, "Bash", "bash-1", 5000)
    after_bash = counting.count

    assert after_read == after_launch
    assert after_bash > after_launch


def test_reread_when_tool_end_has_unknown_id_does_reread():
    counting = CountingRead()
    kit = make_reporter(read_session=counting)
    observer = kit.reporter.observer

    fire_launch(observer, 1000)
    after_launch = counting.count
    fire_tool_end(observer, "Bash", "unknown-id", 3000)

    assert counting.count > after_launch


# ---------------------------------------------------------------------------
# full dashboard golden snapshot
# ---------------------------------------------------------------------------


def test_dashboard_when_mid_session_does_render_full_layout(snapshot: SnapshotAssertion):
    """A mid-session snapshot with time, cost, loop, best, and liveness."""
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
    fire_launch(kit.reporter.observer, 1000, max_minutes=480, max_usd=10.0)

    # Simulate 2h41m elapsed
    kit.clock.now = 1000 + (2 * 3600 + 41 * 60) * 1000
    fire_usage_update(kit.reporter.observer, 4.12, kit.clock.now)

    kit.clock.now += 1000
    fire_tool_start(
        kit.reporter.observer, "Read", "read-1", kit.clock.now, input_summary="src/archetype.ts"
    )
    kit.clock.now += 500
    fire_tool_end(kit.reporter.observer, "Read", "read-1", kit.clock.now)

    kit.clock.now += 200
    fire_tool_start(
        kit.reporter.observer, "Edit", "edit-1", kit.clock.now, input_summary="src/archetype.ts"
    )
    kit.clock.now += 800
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", kit.clock.now)

    kit.clock.now += 100
    fire_tool_start(
        kit.reporter.observer,
        "Bash",
        "bash-1",
        kit.clock.now,
        input_summary="gymrat iterate",
    )
    kit.clock.now += 60_000

    frame = render_frame(kit.reporter)

    assert frame == snapshot


# ---------------------------------------------------------------------------
# ticking display (#28) — Live uses get_renderable
# ---------------------------------------------------------------------------

LIVE_CLASS_PATH = "gymrat.cli.supervise.progress.Live"


def test_create_reporter_when_live_mode_does_use_get_renderable_for_ticking():
    """Live is constructed with ``get_renderable`` so it rebuilds on its 1 Hz refresh."""
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        make_reporter(mode="live")

        call_kwargs = mock_live_cls.call_args.kwargs
        assert "get_renderable" in call_kwargs
        assert callable(call_kwargs["get_renderable"])


# ---------------------------------------------------------------------------
# mounted display (#30) — Live.start() called during creation
# ---------------------------------------------------------------------------


def test_create_reporter_when_live_mode_does_mount_the_live_display():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        make_reporter(mode="live")

        mock_live.start.assert_called_once()


# ---------------------------------------------------------------------------
# liveness — starting state
# ---------------------------------------------------------------------------


def test_liveness_before_any_tool_does_show_starting():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    frame = render_frame(kit.reporter)

    assert "starting" in frame


# ---------------------------------------------------------------------------
# liveness — in-flight tool
# ---------------------------------------------------------------------------


def test_liveness_when_tool_in_flight_does_show_tool_name_and_elapsed():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000, input_summary="gymrat iterate")
    kit.clock.now = 7000

    frame = render_frame(kit.reporter)

    assert "Bash" in frame
    assert "5s" in frame


def test_liveness_when_tool_starts_does_show_tool_name():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Read", "read-1", 2000, input_summary="src/archetype.ts")

    frame = render_frame(kit.reporter)

    assert "Read" in frame


# ---------------------------------------------------------------------------
# liveness — ended tools
# ---------------------------------------------------------------------------


def test_liveness_when_tool_ends_does_show_finished_tool_in_log():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 3000)

    frame = render_frame(kit.reporter)

    assert "Edit" in frame


def test_liveness_when_three_tools_finish_does_show_last_three(snapshot: SnapshotAssertion):
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    for i, (name, summary) in enumerate(
        [("Read", "src/a.ts"), ("Edit", "src/b.ts"), ("Bash", "npm test"), ("Read", "src/c.ts")],
        start=1,
    ):
        ts = 1000 + i * 1000
        kit.clock.now = ts
        fire_tool_start(kit.reporter.observer, name, f"t-{i}", ts, input_summary=summary)
        kit.clock.now = ts + 500
        fire_tool_end(kit.reporter.observer, name, f"t-{i}", ts + 500)

    frame = render_frame(kit.reporter)

    assert frame == snapshot


def test_liveness_when_one_of_several_tools_ends_does_fall_back_to_remaining():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Read", "read-1", 2000)
    kit.clock.now = 2500
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2500)
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Read", "read-1", 3000)

    frame = render_frame(kit.reporter)

    assert "Bash" in frame


def test_liveness_when_untracked_tool_ends_does_not_change():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_tool_end(kit.reporter.observer, "Bash", "never-started", 3000)

    frame = render_frame(kit.reporter)

    assert "starting" in frame


# ---------------------------------------------------------------------------
# finished tool marks
# ---------------------------------------------------------------------------


def test_finished_tool_when_succeeded_does_not_show_success_mark():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 3000)

    frame = render_frame(kit.reporter)

    assert "Edit" in frame
    assert "✔" not in frame


def test_finished_tool_when_failed_does_show_error_mark():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 3000, result="error")

    frame = render_frame(kit.reporter)

    assert "Edit" in frame
    assert "✗" in frame


# ---------------------------------------------------------------------------
# sub-second finished tool
# ---------------------------------------------------------------------------


def test_finished_tool_when_under_one_second_does_show_less_than_one_second():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/a.ts")
    kit.clock.now = 2500
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 2500)

    frame = render_frame(kit.reporter)

    assert "<1s" in frame


# ---------------------------------------------------------------------------
# in-flight tool truncation
# ---------------------------------------------------------------------------


def test_liveness_when_in_flight_summary_exceeds_width_does_truncate_to_one_line():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    long_summary = "src/" + "/".join(f"level{i}" for i in range(20)) + "/file.ts"
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary=long_summary)
    kit.clock.now = 7000

    frame = render_frame(kit.reporter)
    liveness_lines = [line for line in frame.splitlines() if "Edit" in line]

    assert len(liveness_lines) == 1, f"expected one liveness line, got {liveness_lines}"
    assert "…" in liveness_lines[0]


# ---------------------------------------------------------------------------
# wall-clock finished-tool lines
# ---------------------------------------------------------------------------


def test_finished_tool_when_ended_does_show_wall_clock_timestamp():
    """Wall-clock uses the reporter's timezone.

    The reporter is built with ``tz=UTC`` (fixture default), so the wall-clock
    string is a fixed UTC value regardless of the host timezone.
    """
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 3000)

    frame = render_frame(kit.reporter)

    assert "00:00:03" in frame


# ---------------------------------------------------------------------------
# wall-clock — local-timezone default
# ---------------------------------------------------------------------------


def test_finished_tool_when_no_explicit_tz_does_use_system_local_time():
    """Verify local-time fallback without an explicit timezone.

    The expected timestamp is computed from the same epoch-to-local conversion
    the implementation should use, so this assertion is the only remaining user
    of ``_epoch_ms_to_local_hms``.
    """
    ended_at_ms = 3000
    expected_time = _epoch_ms_to_local_hms(ended_at_ms)

    kit = make_reporter(tz=None)
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = ended_at_ms
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", ended_at_ms)

    frame = render_frame(kit.reporter)

    assert expected_time in frame


# ---------------------------------------------------------------------------
# liveness — waiting state
# ---------------------------------------------------------------------------


def test_liveness_when_tool_ends_below_threshold_does_show_waiting():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Read", "read-1", 2000)
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Read", "read-1", 3000)

    frame = render_frame(kit.reporter)

    assert "waiting" in frame
    assert "no output" not in frame


# ---------------------------------------------------------------------------
# above-threshold waiting — last tool context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected_fragment"),
    [
        pytest.param("ok", "(last tool: Bash at 00:00:03)", id="ok-tool"),
        pytest.param("error", "(last tool: Bash ✗ at 00:00:03)", id="errored-tool"),
    ],
)
def test_liveness_when_waiting_past_threshold_does_show_last_tool_context(
    result: str, expected_fragment: str
):
    """Reporter is built with ``tz=UTC`` so wall-clock is fixed."""
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000)
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Bash", "bash-1", 3000, result=result)
    kit.clock.now = 3000 + IDLE_WARN_MS + 1

    frame = render_frame(kit.reporter)

    assert "no output" in frame
    assert "idle" not in frame
    assert expected_fragment in frame


def test_liveness_when_waiting_past_threshold_no_tool_does_omit_parenthetical():
    """When no tool has finished, above-threshold waiting shows bare ``no output for Xs``."""
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_model_phase(kit.reporter.observer, 2000, "turn_end")
    kit.clock.now = 2000 + IDLE_WARN_MS + 1

    frame = render_frame(kit.reporter)

    assert "no output" in frame
    assert "(last tool" not in frame


# ---------------------------------------------------------------------------
# liveness — model phase transitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        pytest.param("thinking", "thinking", id="thinking"),
        pytest.param("responding", "responding", id="responding"),
        pytest.param("turn_end", "waiting", id="turn_end"),
    ],
)
def test_liveness_when_model_phase_does_show_expected_state(phase: str, expected: str):
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    fire_model_phase(observer, 2000, phase)

    assert expected in render_frame(kit.reporter)


def test_liveness_when_model_phase_thinking_after_thinking_update_does_preserve_token_count():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    fire_thinking_update(observer, 1500, estimated_tokens=200)
    fire_model_phase(observer, 2000, "thinking")

    frame = render_frame(kit.reporter)
    assert "thinking" in frame
    assert "200" in frame


@pytest.mark.parametrize(
    ("tool_name", "expected_in_frame"),
    [
        pytest.param("Edit", "Edit", id="with-tool-name"),
        pytest.param(None, "unknown", id="without-tool-name"),
    ],
)
def test_liveness_when_model_phase_tool_input_does_show_preparing(
    tool_name: str | None, expected_in_frame: str
):
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    fire_model_phase(observer, 2000, "tool_input", tool_name=tool_name)

    frame = render_frame(kit.reporter)
    assert "preparing" in frame
    assert expected_in_frame in frame


# ---------------------------------------------------------------------------
# liveness — model phase ignored when capped or in-flight
# ---------------------------------------------------------------------------


def test_liveness_when_model_phase_while_capped_does_stay_capped():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    fire_cap(observer, "wall-clock")

    fire_model_phase(observer, 6000, "responding")

    frame = render_frame(kit.reporter)
    assert "interrupting" in frame
    assert "responding" not in frame


def test_liveness_when_model_phase_while_in_flight_does_stay_in_flight():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    fire_model_phase(observer, 2000, "responding")

    assert "Bash" in render_frame(kit.reporter)


def test_liveness_when_thinking_update_while_in_flight_does_stay_in_flight():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    fire_thinking_update(observer, 2000, estimated_tokens=100)

    assert "Bash" in render_frame(kit.reporter)


def test_liveness_when_thinking_update_while_capped_does_stay_capped():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    fire_cap(observer, "wall-clock")
    fire_thinking_update(observer, 6000, estimated_tokens=100)

    assert "interrupting" in render_frame(kit.reporter)


# ---------------------------------------------------------------------------
# liveness — nested subagent activity
# ---------------------------------------------------------------------------


def _liveness_lines(frame: str, needle: str) -> list[str]:
    """Top-level liveness lines containing *needle*, excluding nested (``↳``) lines."""
    return [line for line in frame.splitlines() if needle in line and "↳" not in line]


def test_liveness_when_nested_tool_starts_does_not_change_top_level_liveness():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    fire_tool_start(observer, "Read", "nested-read-1", 2000, parent_tool_use_id="bash-1")

    frame = render_frame(kit.reporter)
    assert len(_liveness_lines(frame, "Bash")) == 1


def test_liveness_when_nested_tool_ends_does_not_appear_in_finished_tools():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    fire_tool_start(observer, "Read", "nested-read-1", 2000, parent_tool_use_id="bash-1")
    fire_tool_end(observer, "Read", "nested-read-1", 2500, parent_tool_use_id="bash-1")

    frame = render_frame(kit.reporter)
    assert "Bash" in frame
    assert "Read" not in frame


def test_liveness_when_nested_thinking_update_does_not_change_top_level():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    fire_thinking_update(observer, 2000, estimated_tokens=999, parent_tool_use_id="bash-1")

    assert "Bash" in render_frame(kit.reporter)
    assert "999" not in render_frame(kit.reporter)


def test_liveness_when_nested_model_phase_does_not_change_top_level():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    fire_model_phase(observer, 2000, "responding", parent_tool_use_id="bash-1")

    frame = render_frame(kit.reporter)
    assert len(_liveness_lines(frame, "Bash")) == 1


def test_liveness_when_nested_event_has_no_matching_parent_does_ignore():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    fire_tool_start(observer, "Read", "nested-read-1", 2000, parent_tool_use_id="nonexistent")

    assert "starting" in render_frame(kit.reporter)
    assert "Read" not in render_frame(kit.reporter)


# ---------------------------------------------------------------------------
# liveness — nested iterate sidecar
# ---------------------------------------------------------------------------


def test_liveness_when_iterate_tool_has_sidecar_does_show_passes_bar(
    snapshot: SnapshotAssertion,
):
    sidecar = ProgressSnapshot(
        passes_completed=7,
        passes_total=10,
        last_pass_duration_ms=225_000.0,
    )

    def fake_read_progress(_root: str) -> ProgressSnapshot | None:
        return sidecar

    kit = make_reporter(read_progress=fake_read_progress)
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000, input_summary="gymrat iterate")
    kit.clock.now = 2000 + 31 * 60 * 1000

    frame = render_frame(kit.reporter)

    assert "7/10" in frame
    assert frame == snapshot


def test_liveness_when_iterate_tool_has_no_sidecar_does_show_plain_elapsed():
    def fake_read_progress(_root: str) -> ProgressSnapshot | None:
        return None

    kit = make_reporter(read_progress=fake_read_progress)
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000, input_summary="gymrat iterate")
    kit.clock.now = 7000

    frame = render_frame(kit.reporter)

    assert "Bash" in frame
    assert "7/10" not in frame


# ---------------------------------------------------------------------------
# cap event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", ["wall-clock", "spend-cap"])
def test_cap_when_fired_does_show_interrupting_with_cap_type(cap: CapType):
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_cap(kit.reporter.observer, cap)

    frame = render_frame(kit.reporter)

    assert f"interrupting ({cap})" in frame


def test_cap_when_fired_does_freeze_liveness_against_later_tool_events():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    fire_cap(kit.reporter.observer, "wall-clock")
    kit.clock.now = 6000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 6000)
    fire_tool_end(kit.reporter.observer, "Bash", "bash-1", 7000)

    frame = render_frame(kit.reporter)

    assert "interrupting" in frame


# ---------------------------------------------------------------------------
# events that do not render
# ---------------------------------------------------------------------------


def test_liveness_when_text_delta_after_launch_does_stay_starting():
    kit = make_reporter()
    observer = kit.reporter.observer

    fire_launch(observer, 1000)
    observer(TextDeltaEvent(timestamp=2500, chunk="hello"))

    frame = render_frame(kit.reporter)

    assert "starting" in frame


def test_liveness_when_thinking_after_launch_does_show_thinking():
    kit = make_reporter()
    observer = kit.reporter.observer

    fire_launch(observer, 1000)
    observer(ThinkingUpdateEvent(timestamp=2500, estimated_tokens=100, delta=10))

    frame = render_frame(kit.reporter)

    assert "thinking" in frame
    assert "100" in frame


def test_liveness_when_tool_progress_after_launch_does_not_crash():
    kit = make_reporter()
    observer = kit.reporter.observer

    fire_launch(observer, 1000)
    observer(ToolProgressEvent(timestamp=2000, tool_use_id="tp-1", elapsed_ms=500))

    frame = render_frame(kit.reporter)

    assert frame


# ---------------------------------------------------------------------------
# warn
# ---------------------------------------------------------------------------


def test_warn_when_called_in_live_mode_does_not_crash():
    kit = make_reporter(mode="live")
    fire_launch(kit.reporter.observer, 1000)

    kit.reporter.warn("something is wrong")

    frame = render_frame(kit.reporter)
    assert frame


# ---------------------------------------------------------------------------
# final_text
# ---------------------------------------------------------------------------


def test_final_text_when_no_text_received_does_return_none():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    assert kit.reporter.final_text() is None


def test_final_text_when_top_level_text_deltas_received_does_return_last_chunk():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    observer(TextDeltaEvent(timestamp=2000, chunk="first message"))
    observer(TextDeltaEvent(timestamp=3000, chunk="final message"))

    assert kit.reporter.final_text() == "final message"


def test_final_text_when_nested_text_delta_received_does_not_replace_top_level():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    observer(TextDeltaEvent(timestamp=2000, chunk="top-level text"))
    observer(TextDeltaEvent(timestamp=3000, chunk="nested text", parent_tool_use_id="tu_sub"))

    assert kit.reporter.final_text() == "top-level text"


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def test_stop_when_called_does_not_raise():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    kit.reporter.stop()
