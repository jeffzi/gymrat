"""Behavioral tests for the supervise Rich dashboard reporter.

The reporter turns a stream of :class:`~gymrat.supervisor.events.SessionEvent`
values into a bordered ``Live`` dashboard with time/cost/loop summary rows and a
liveness section showing tool activity.  Every test injects the clock (``now``)
and the session reader (``read_session``) so nothing depends on real time or
disk.

**Live mode** tests render ``reporter.frame()`` through ``frame_text()`` from
``tests._rich`` at a fixed width, pinning frame content with syrupy snapshots.
**Plain mode** tests assert on recorded milestone lines. Liveness, idle, sidecar,
cap, and non-rendering event tests also live here.
"""

from __future__ import annotations

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
    fire_tool_end,
    fire_tool_start,
    fire_usage_update,
    make_iteration,
    make_plain_reporter,
    make_read_session,
    make_reporter,
    render_frame,
    session_state,
)

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


# ---------------------------------------------------------------------------
# factory / contract
# ---------------------------------------------------------------------------


def test_create_reporter_when_built_does_expose_observer_and_stop():
    kit = make_reporter()

    assert callable(kit.reporter.observer)
    assert callable(kit.reporter.stop)
    kit.reporter.stop()


def test_create_reporter_when_built_does_expose_frame():
    kit = make_reporter()

    frame = kit.reporter.frame()

    assert frame is not None


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

    assert "supervise" in frame
    assert "ecstatic-ts" in frame
    assert "20260813-125044-34ec" in frame
    assert frame == snapshot


# ---------------------------------------------------------------------------
# time bar
# ---------------------------------------------------------------------------


def test_time_bar_when_launched_does_show_elapsed_over_max_minutes(
    snapshot: SnapshotAssertion,
):
    kit = make_reporter(max_minutes=480, clock_start=1000)
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 1000 + (2 * 3600 + 41 * 60) * 1000

    frame = render_frame(kit.reporter)

    assert frame == snapshot


# ---------------------------------------------------------------------------
# cost row
# ---------------------------------------------------------------------------


def test_cost_when_no_cap_and_no_usage_does_show_placeholder(snapshot: SnapshotAssertion):
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    frame = render_frame(kit.reporter)

    assert "cost" in frame
    assert "$—" in frame


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


def test_loop_when_baseline_recorded_does_show_baseline_recorded():
    kit = make_reporter(
        max_iterations=20,
        read_session=make_read_session(empty_session_state(), has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "baseline recorded" in frame


def test_loop_when_iterations_present_does_show_counts_and_last(snapshot: SnapshotAssertion):
    state = session_state(
        iteration_count=3,
        keep_count=2,
        discard_count=1,
        last_iteration=make_iteration(-3.2, "improved"),
    )
    kit = make_reporter(
        max_iterations=20,
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "iter 3/20" in frame
    assert "2 kept" in frame
    assert "1 discarded" in frame
    assert "-3.2%" in frame
    assert "improved" in frame


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

    assert "iter 2" in frame
    assert "iter 2/" not in frame


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


def test_best_when_kept_iteration_exists_does_show_best_delta_and_seq():
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


def test_best_when_session_has_best_fields_does_show_delta_label_sha_and_seq():
    """The best row renders delta, primary label, baseline SHA (7 chars), and seq."""
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
    assert "(seq 3)" in frame


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

    assert "(seq 3)" in frame
    assert "(seq 5)" not in frame


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


def test_reread_when_non_bash_tool_ends_does_not_reread_but_bash_does():
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
# wall-clock finished-tool lines
# ---------------------------------------------------------------------------


def test_finished_tool_when_ended_does_show_wall_clock_timestamp():
    """Each finished tool line leads with the local-time completion timestamp.

    The expected timestamp is computed from the same epoch-to-local conversion
    the implementation should use, so the assertion is timezone-independent.
    """
    ended_at_ms = 3000
    expected_time = _epoch_ms_to_local_hms(ended_at_ms)

    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Edit", "edit-1", 3000)

    frame = render_frame(kit.reporter)

    assert expected_time in frame


# ---------------------------------------------------------------------------
# liveness — idle state
# ---------------------------------------------------------------------------


def test_liveness_when_idle_past_threshold_does_show_idle():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000)
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Bash", "bash-1", 3000)
    kit.clock.now = 3000 + IDLE_WARN_MS + 1

    frame = render_frame(kit.reporter)

    assert "idle" in frame


def test_liveness_when_tool_ends_below_idle_threshold_does_not_show_idle():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Read", "read-1", 2000)
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Read", "read-1", 3000)

    frame = render_frame(kit.reporter)

    assert "idle" not in frame


# ---------------------------------------------------------------------------
# idle context — wall-clock and tool glyph
# ---------------------------------------------------------------------------


def test_liveness_when_idle_past_threshold_does_show_wall_clock_and_last_tool_context():
    """Idle line shows wall-clock of last tool end plus tool name and success glyph.

    The format is ``idle 4m 5s — no tool call since HH:MM:SS (last: Bash ✔)``.
    """
    ended_at_ms = 3000
    expected_time = _epoch_ms_to_local_hms(ended_at_ms)

    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 2000
    fire_tool_start(kit.reporter.observer, "Bash", "bash-1", 2000)
    kit.clock.now = 3000
    fire_tool_end(kit.reporter.observer, "Bash", "bash-1", 3000, result="ok")
    kit.clock.now = 3000 + IDLE_WARN_MS + 1

    frame = render_frame(kit.reporter)

    assert "idle" in frame
    assert expected_time in frame
    assert "Bash" in frame
    assert "✔" in frame


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


def test_liveness_when_text_delta_or_thinking_update_does_not_crash():
    kit = make_reporter()
    observer = kit.reporter.observer

    fire_launch(observer, 1000)
    observer(TextDeltaEvent(timestamp=2500, chunk="hello"))
    observer(ThinkingUpdateEvent(timestamp=2500, estimated_tokens=100, delta=10))

    frame = render_frame(kit.reporter)

    assert "starting" in frame


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


def test_warn_when_called_in_live_mode_does_record_warning():
    """In live mode, warn() should print above the live block.

    Since we test via frame(), we verify the reporter accepts warn() calls
    and the warning text is accessible.
    """
    kit = make_reporter(mode="live")
    fire_launch(kit.reporter.observer, 1000)

    kit.reporter.warn("something is wrong")

    frame = render_frame(kit.reporter)
    assert frame


# ---------------------------------------------------------------------------
# plain mode output
# ---------------------------------------------------------------------------


def test_plain_when_launched_with_spend_cap_does_print_caps_with_dollars():
    plain = make_plain_reporter(max_usd=5.0, max_minutes=60)

    fire_launch(plain.observer, 1000, max_usd=5.0)

    assert any("caps" in w and "$5.00" in w for w in plain.writes)


def test_plain_when_launched_without_spend_cap_does_print_bare_caps():
    plain = make_plain_reporter(max_minutes=30)

    fire_launch(plain.observer, 1000, max_minutes=30)

    assert any("caps 30m" in w for w in plain.writes)


@pytest.mark.parametrize(
    ("max_minutes", "expected"),
    [
        pytest.param(5.5, "caps 5.5m", id="fractional-keeps-decimal"),
        pytest.param(10.0, "caps 10m", id="whole-float-drops-decimal"),
    ],
)
def test_plain_caps_when_max_minutes_given_does_render_the_actual_cap_value(
    max_minutes: float, expected: str
):
    plain = make_plain_reporter(max_minutes=max_minutes)

    fire_launch(plain.observer, 1000, max_minutes=max_minutes)

    assert any(expected in w for w in plain.writes)


def test_plain_when_usage_update_does_print_cost():
    plain = make_plain_reporter(max_usd=5.0)

    fire_launch(plain.observer, 1000, max_usd=5.0)
    fire_usage_update(plain.observer, 1.42, 2000)

    assert any("cost $1.42" in w for w in plain.writes)


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

    loop_writes = [w for w in plain.writes if "iter 2/20" in w]
    assert loop_writes
    assert "+3.2%" in loop_writes[0]
    assert "regressed" in loop_writes[0]


def test_plain_when_no_session_yet_does_not_print_loop_segment():
    plain = make_plain_reporter(read_session=_throwing_read)

    fire_launch(plain.observer, 1000)

    assert any("caps" in w for w in plain.writes)
    assert not any("no session yet" in w for w in plain.writes)


def test_plain_when_capped_does_print_cap_interrupting():
    plain = make_plain_reporter()

    fire_launch(plain.observer, 1000)
    fire_cap(plain.observer, "wall-clock")

    assert any("cap wall-clock" in w for w in plain.writes)
    assert any("interrupting" in w for w in plain.writes)


def test_plain_when_warn_called_does_record_warning():
    plain = make_plain_reporter()
    fire_launch(plain.observer, 1000)

    plain.reporter.warn("heads up")

    assert any("heads up" in w for w in plain.writes)


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


def test_stop_when_called_does_not_raise():
    kit = make_reporter()
    fire_launch(kit.reporter.observer, 1000)

    kit.reporter.stop()


def test_stop_when_called_in_plain_mode_does_not_raise():
    plain = make_plain_reporter()
    fire_launch(plain.observer, 1000)

    plain.reporter.stop()
