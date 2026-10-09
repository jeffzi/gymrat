"""Behavioral tests for the supervise Rich dashboard reporter.

The reporter turns a stream of :class:`~gymrat.supervisor.events.SessionEvent`
values into a bordered ``Live`` dashboard with time/cost/loop summary rows and a
liveness section showing tool activity.  Reporter tests inject the clock
(``now``) and the session reader (``read_session``) so they depend on neither
real time nor disk.  The ``read_live_session`` tests write a session log under
``tmp_path`` and read it back the way the dashboard does.

**Live mode** tests render ``reporter.frame()`` through ``frame_text()`` from
``tests._rich`` at a fixed width, pinning frame content with syrupy snapshots.
The liveness line covers in-flight and finished tools (with their marks,
truncation and wall-clock times), the waiting state and its idle threshold,
model phase transitions, nested subagent lines and the tool-name column, the
iterate sidecar and MCP tool detection, and the follow-up transition. The panel
title's model and effort text is pinned here too; the frame's styling is pinned
in ``test_frame.py``. During the run-end exit sequence the reporter shows each
exit phase as the liveness line, and a session refresh re-reads the session,
keeping the last good read when one fails. The reducer transitions behind them
are pinned in ``test_reducer.py``.

**Plain mode** tests assert on the recorded milestone lines.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, timedelta, timezone
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from gymrat.cli.supervise.progress import (
    IDLE_WARN_MS,
    create_supervise_reporter,
    read_live_session,
)
from gymrat.cli.supervise.types import BestIteration
from gymrat.session.progress_file import ProgressSnapshot
from gymrat.session.records import IterationPrimary
from gymrat.supervisor.exit_sequence import ExitPhase
from tests._rich import track
from tests.cli.supervise._fixtures import (
    BASH_CYCLE_END_MS,
    EMPTY_READ,
    FRAME_WIDTH,
    KEPT_READ,
    ReporterKit,
    _throwing_read,
    cap_event,
    content_line,
    fire_launch_and_bash_cycle,
    fire_launch_and_bash_start,
    fire_launch_and_edit_cycle,
    fire_launch_and_iterate_start,
    follow_up_event,
    launch_event,
    line_after,
    make_read_session,
    make_reporter,
    model_phase_event,
    read_result,
    render_frame,
    reporter_with_nested_read,
    row_content,
    session_state_three_iterations,
    tool_end_event,
    tool_start_event,
    turn_end_event,
    usage_event,
)
from tests.session.records._fixtures import (
    BASELINE_SHA,
    COMMIT,
    baseline_record,
    blocked_keep,
    committed_keep,
    discard_record,
    iteration_record,
    make_iteration,
    session_record,
    session_state,
    stop_record,
    write_session_log,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import tzinfo
    from pathlib import Path

    from syrupy.assertion import SnapshotAssertion

    from gymrat.cli.supervise.types import ReadSessionResult
    from gymrat.config import Effort
    from gymrat.session.records import SessionLogRecord
    from gymrat.session.schema import PrimaryKind
    from gymrat.supervisor.events import ModelPhase, SessionEvent


# ---------------------------------------------------------------------------
# factory / contract
# ---------------------------------------------------------------------------


def test_session_result_when_a_tool_ends_does_return_the_latest_read():
    at_launch = read_result()
    at_tool_end = read_result(
        session_state_three_iterations(-4.2, "improved", seq=3), has_baseline=True
    )
    kit = make_reporter(read_session=Mock(side_effect=[at_launch, at_tool_end]))

    fire_launch_and_bash_cycle(kit.reporter.observer)

    assert kit.reporter.session_result() == at_tool_end


def _read_back(tmp_path: Path, history: tuple[SessionLogRecord, ...]) -> ReadSessionResult:
    """Write a session log under ``tmp_path`` and read it back the dashboard's way."""
    write_session_log(str(tmp_path), session_record(), history)
    return read_live_session(str(tmp_path), "lower")


def _committed_throughput_history(*deltas: float) -> tuple[SessionLogRecord, ...]:
    """One committed ``throughput`` iteration per delta, numbered from 1."""
    return tuple(
        record
        for seq, delta_pct in enumerate(deltas, start=1)
        for record in (
            iteration_record(
                seq=seq,
                primary=IterationPrimary(kind="metric", name="throughput", delta_pct=delta_pct),
            ),
            committed_keep(seq),
        )
    )


@pytest.mark.parametrize(
    ("kind", "name", "primary_label"),
    [
        pytest.param("geomean", None, "geomean", id="geomean-primary"),
        pytest.param("metric", "decode/time", "decode/time", id="metric-primary"),
    ],
)
def test_read_live_session_when_keeps_committed_does_report_the_best_committed_iteration(
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

    assert result.best == BestIteration(
        delta_pct=-9.0, seq=2, label=primary_label, baseline_sha=COMMIT
    )


_FIRST_KEEP_SHA = "1" * 40


@pytest.mark.parametrize(
    ("deltas", "discarded", "expected"),
    [
        pytest.param((-9.0, -7.2), (), (1, BASELINE_SHA), id="no-keep-before-it"),
        pytest.param((-7.2, -9.0), (), (2, _FIRST_KEEP_SHA), id="previous-keep"),
        pytest.param(
            (-7.2, -20.0, -9.0), (2,), (3, _FIRST_KEEP_SHA), id="previous-keep-across-a-discard"
        ),
    ],
)
def test_read_live_session_when_keeps_precede_the_best_does_name_its_baseline(
    tmp_path: Path,
    deltas: tuple[float, ...],
    discarded: tuple[int, ...],
    expected: tuple[int, str],
):
    history = tuple(
        record
        for seq, delta_pct in enumerate(deltas, start=1)
        for record in (
            iteration_record(
                seq=seq, primary=IterationPrimary(kind="geomean", delta_pct=delta_pct)
            ),
            discard_record(seq) if seq in discarded else committed_keep(seq, commit=str(seq) * 40),
        )
    )

    result = _read_back(tmp_path, history)

    assert result.best is not None
    assert (result.best.seq, result.best.baseline_sha) == expected


def test_create_reporter_when_no_session_reader_given_does_read_with_the_primary_direction(
    tmp_path: Path,
):
    write_session_log(
        str(tmp_path), session_record(), _committed_throughput_history(2.0, 9.0, -3.0)
    )
    reporter = track(
        create_supervise_reporter(
            root=str(tmp_path),
            max_minutes=60,
            mode="plain",
            plain_write=lambda _line: None,
            primary_direction="higher",
        )
    )

    reporter.observer(launch_event(1000))

    session_result = reporter.session_result()
    assert session_result is not None
    assert session_result.best == BestIteration(
        delta_pct=9.0, seq=2, label="throughput", baseline_sha=COMMIT, direction="higher"
    )


@pytest.mark.parametrize(
    "history",
    [
        pytest.param(
            (
                iteration_record(seq=1, primary=IterationPrimary(kind="geomean", delta_pct=-7.2)),
                blocked_keep(1),
            ),
            id="no-committed-keep",
        ),
        pytest.param(
            (
                iteration_record(seq=1, primary=IterationPrimary(kind="geomean", delta_pct=None)),
                committed_keep(1),
            ),
            id="committed-keep-without-a-delta",
        ),
    ],
)
def test_read_live_session_when_no_committed_keep_has_a_delta_does_report_no_best(
    tmp_path: Path, history: tuple[SessionLogRecord, ...]
):
    result = _read_back(tmp_path, history)

    assert result.best is None


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
def test_read_live_session_when_stop_recorded_does_report_it_only_while_last(
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
def test_read_live_session_when_baseline_presence_varies_does_report_it(
    tmp_path: Path, history: tuple[SessionLogRecord, ...], has_baseline: bool
):
    result = _read_back(tmp_path, history)

    assert result.has_baseline is has_baseline


# ---------------------------------------------------------------------------
# dashboard title — model and effort text
# ---------------------------------------------------------------------------


def _panel_title_text(frame: str) -> str:
    """The panel's title text, with the top border's box-drawing chars stripped."""
    return frame.splitlines()[0].strip("╭╮─ ")


@pytest.mark.parametrize(
    ("model", "effort", "expected_title"),
    [
        pytest.param("opus", None, "supervise · model opus", id="model-only"),
        pytest.param(None, "high", "supervise · effort high", id="effort-only"),
        pytest.param("opus", "max", "supervise · model opus · effort max", id="model-and-effort"),
        pytest.param(None, None, "supervise", id="neither"),
    ],
)
def test_panel_title_when_model_or_effort_in_force_does_show_labelled_value(
    model: str | None, effort: Effort | None, expected_title: str
) -> None:
    kit = make_reporter(session_id="", branch="", model=model, effort=effort)
    kit.reporter.observer(launch_event(1000))

    frame = render_frame(kit.reporter)

    assert _panel_title_text(frame) == expected_title


# ---------------------------------------------------------------------------
# time bar
# ---------------------------------------------------------------------------


def test_time_bar_when_elapsed_exceeds_max_does_clamp_remaining_to_zero():
    kit = make_reporter(max_minutes=60, clock_start=1000)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 1000 + (2 * 3600) * 1000

    frame = render_frame(kit.reporter)

    assert content_line(frame, "cap in") == "time " + "━" * 73 + " 2h 00m  cap in 0s"


# ---------------------------------------------------------------------------
# starting layout
# ---------------------------------------------------------------------------


def test_frame_when_just_launched_does_render_the_starting_layout(
    snapshot: SnapshotAssertion,
):
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))

    frame = render_frame(kit.reporter)

    assert frame == snapshot


# ---------------------------------------------------------------------------
# best row
# ---------------------------------------------------------------------------


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
            best=BestIteration(delta_pct=-6.8, seq=3, label="geomean"),
        ),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert content_line(frame, "best") == "best -6.8% geomean (iteration 3)"


# ---------------------------------------------------------------------------
# session re-read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("event", "expected_reads"),
    [
        pytest.param(tool_start_event("Read", "read-1", 2000), 0, id="tool-start"),
        pytest.param(tool_end_event("Read", "read-1", 3000), 1, id="tool-end"),
    ],
)
def test_observer_when_tool_event_arrives_does_reread_only_on_the_end(
    event: SessionEvent, expected_reads: int
):
    read = Mock(return_value=read_result())
    kit = make_reporter(read_session=read)
    kit.reporter.observer(launch_event(1000))
    reads_before = read.call_count

    kit.reporter.observer(event)

    assert read.call_count - reads_before == expected_reads


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
            best=BestIteration(
                delta_pct=-6.8,
                seq=3,
                label="geomean",
                baseline_sha="abc1234567890abcdef1234567890abcdef123456",
            ),
        ),
    )
    kit.reporter.observer(launch_event(1000, max_minutes=480, max_usd=10.0))

    kit.clock.now = 1000 + (2 * 3600 + 41 * 60) * 1000
    kit.reporter.observer(usage_event(4.12, kit.clock.now))

    kit.clock.now += 1000
    read_started_at = kit.clock.now
    kit.reporter.observer(
        tool_start_event("Read", "read-1", read_started_at, input_summary="src/archetype.ts")
    )
    kit.clock.now += 500
    kit.reporter.observer(
        tool_end_event("Read", "read-1", kit.clock.now, started_at_ms=read_started_at)
    )

    kit.clock.now += 200
    edit_started_at = kit.clock.now
    kit.reporter.observer(
        tool_start_event("Edit", "edit-1", edit_started_at, input_summary="src/archetype.ts")
    )
    kit.clock.now += 800
    kit.reporter.observer(
        tool_end_event("Edit", "edit-1", kit.clock.now, started_at_ms=edit_started_at)
    )

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


def test_cap_when_fired_does_show_action_with_cap_type():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.reporter.observer(cap_event("spend-cap", action="ending"))

    frame = render_frame(kit.reporter)

    assert content_line(frame, "(spend-cap)") == "ending (spend-cap)"


# ---------------------------------------------------------------------------
# final_text
# ---------------------------------------------------------------------------


def test_final_text_when_agent_turn_ends_does_return_its_text():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(turn_end_event(2000, text="first turn summary"))
    observer(turn_end_event(3000, text="second turn summary"))

    assert kit.reporter.final_text() == "second turn summary"


def _liveness_rows(frame: str) -> list[str]:
    """The frame rows below the loop row, with panel border and padding stripped."""
    rows = [row_content(line) for line in frame.splitlines() if line.startswith("│")]
    loop_index = next(i for i, row in enumerate(rows) if row.startswith("loop "))
    return rows[loop_index + 1 :]


# ---------------------------------------------------------------------------
# finished tool marks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected_row"),
    [
        pytest.param("ok", "00:00:03  Edit   src/archetype.ts  1s", id="ok"),
        pytest.param("error", "00:00:03  Edit   src/archetype.ts  ✗ 1s", id="error"),
    ],
)
def test_finished_tool_when_ended_does_mark_only_an_error(result: str, expected_row: str):
    kit = make_reporter()
    fire_launch_and_edit_cycle(kit, result=result)

    frame = render_frame(kit.reporter)

    assert content_line(frame, "Edit") == expected_row


# ---------------------------------------------------------------------------
# in-flight tool truncation
# ---------------------------------------------------------------------------


def test_liveness_when_in_flight_summary_exceeds_width_does_truncate_to_one_line():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    long_summary = "src/" + "/".join(f"level{i}" for i in range(20)) + "/file.ts"
    kit.reporter.observer(tool_start_event("Edit", "edit-1", 2000, input_summary=long_summary))
    kit.clock.now = 7000

    frame = render_frame(kit.reporter)

    assert content_line(frame, "Edit") == (
        "00:00:02  Edit   src/level0/level1/level2/level3/level4/level5/level6/level7/level8/level9/le…"
    )


# ---------------------------------------------------------------------------
# liveness — nested subagent lines
# ---------------------------------------------------------------------------


def test_nested_tool_when_in_flight_does_render_an_arrow_line_under_parent():
    kit = reporter_with_nested_read()

    nested_line = line_after(render_frame(kit.reporter), "Bash")

    assert row_content(nested_line) == "↳ Read src/config.ts  3s"


@pytest.mark.parametrize(
    ("phase", "tool_name", "now", "expected"),
    [
        pytest.param("thinking", None, 4000, "↳ thinking  2s", id="thinking"),
        pytest.param("responding", None, 3000, "↳ responding  1s", id="responding"),
        pytest.param("tool_input", "Edit", 3000, "↳ preparing Edit  1s", id="composing"),
    ],
)
def test_nested_phase_when_reported_does_render_an_arrow_line_naming_it(
    phase: ModelPhase, tool_name: str | None, now: int, expected: str
):
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch_and_bash_start(observer)
    kit.clock.now = 2000
    observer(model_phase_event(2000, phase, tool_name=tool_name, parent_tool_use_id="bash-1"))
    kit.clock.now = now

    nested_line = line_after(render_frame(kit.reporter), "Bash")

    assert row_content(nested_line) == expected


def test_nested_when_no_activity_does_end_the_panel_on_the_bash_row():
    kit = make_reporter()
    fire_launch_and_iterate_start(kit)
    kit.clock.now = 5000

    row_after_bash = line_after(render_frame(kit.reporter), "Bash")

    assert row_after_bash == "╰" + "─" * (FRAME_WIDTH - 2) + "╯"


# ---------------------------------------------------------------------------
# tool-name column width — nested tools excluded
# ---------------------------------------------------------------------------


def test_tool_name_column_width_when_nested_tool_present_does_ignore_nested_width():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    kit.clock.now = 2000
    observer(tool_start_event("Bash", "bash-1", 2000, input_summary="run tests"))
    kit.clock.now = 2500
    observer(
        tool_start_event(
            "LongNestedToolName",
            "nested-1",
            2500,
            parent_tool_use_id="bash-1",
            input_summary="something",
        )
    )
    kit.clock.now = 3000

    frame = render_frame(kit.reporter)

    # "Bash" padded to the 5-column floor; counting the nested name would widen it.
    assert content_line(frame, "Bash") == "00:00:02  Bash   run tests  1s"


# ---------------------------------------------------------------------------
# wall-clock finished-tool lines
# ---------------------------------------------------------------------------


# 2023-11-14 22:13:20 UTC.
_WALL_CLOCK_EPOCH_MS = 1_700_000_000_000


@pytest.mark.parametrize(
    ("tz", "expected_clock"),
    [
        pytest.param(timezone(timedelta(hours=5, minutes=30)), "03:43:20", id="half-hour-ahead"),
        pytest.param(timezone(timedelta(hours=-5)), "17:13:20", id="five-hours-behind"),
    ],
)
def test_finished_tool_when_ended_does_show_wall_clock_in_the_given_tz(
    tz: tzinfo, expected_clock: str
):
    started_at = _WALL_CLOCK_EPOCH_MS - 1000
    kit = make_reporter(tz=tz, clock_start=started_at)
    kit.reporter.observer(launch_event(started_at))
    kit.reporter.observer(
        tool_start_event("Edit", "edit-1", started_at, input_summary="src/archetype.ts")
    )
    kit.clock.now = _WALL_CLOCK_EPOCH_MS
    kit.reporter.observer(
        tool_end_event("Edit", "edit-1", _WALL_CLOCK_EPOCH_MS, started_at_ms=started_at)
    )

    frame = render_frame(kit.reporter)

    assert content_line(frame, "Edit") == f"{expected_clock}  Edit   src/archetype.ts  1s"


# ---------------------------------------------------------------------------
# wall-clock — local-timezone default
# ---------------------------------------------------------------------------


@pytest.fixture
def india_local_time() -> Iterator[None]:
    """Pin the process's local time zone to UTC+05:30 for one test."""
    original = os.environ.get("TZ")
    os.environ["TZ"] = "IST-5:30"
    time.tzset()
    yield
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


@pytest.mark.skipif(sys.platform == "win32", reason="time.tzset is POSIX-only")
@pytest.mark.usefixtures("india_local_time")
def test_finished_tool_when_no_explicit_tz_does_use_system_local_time():
    kit = make_reporter(tz=None)
    fire_launch_and_edit_cycle(kit)

    frame = render_frame(kit.reporter)

    assert content_line(frame, "Edit") == "05:30:03  Edit   src/archetype.ts  1s"


# ---------------------------------------------------------------------------
# above-threshold waiting — last tool context
# ---------------------------------------------------------------------------


def _make_reporter_past_bash_end(
    *, idle_warn_ms: int = IDLE_WARN_MS, result: str = "ok"
) -> ReporterKit:
    """A reporter with the given idle-warn threshold, clock frozen right after a Bash end."""
    kit = make_reporter(idle_warn_ms=idle_warn_ms)
    fire_launch_and_bash_cycle(kit.reporter.observer, clock=kit.clock, result=result)
    return kit


def test_liveness_when_waiting_past_threshold_after_errored_tool_does_mark_it_in_the_context():
    kit = _make_reporter_past_bash_end(result="error")
    kit.clock.now = BASH_CYCLE_END_MS + IDLE_WARN_MS + 1

    frame = render_frame(kit.reporter)

    assert content_line(frame, "no output") == "no output for 30s (last tool: Bash ✗ at 00:00:03)"


def test_liveness_when_waiting_past_threshold_no_tool_does_omit_parenthetical():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.reporter.observer(model_phase_event(2000, "turn_end"))
    kit.clock.now = 2000 + IDLE_WARN_MS + 1

    frame = render_frame(kit.reporter)

    assert _liveness_rows(frame) == ["no output for 30s"]


# ---------------------------------------------------------------------------
# custom idle_warn_ms — configurable threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "expected_state"),
    [
        pytest.param(
            1, "no output for 0s (last tool: Bash at 00:00:03)", id="past-custom-idle-warn"
        ),
        pytest.param(-1, "waiting  0s", id="below-custom-idle-warn"),
    ],
)
def test_liveness_when_waiting_around_custom_idle_warn_does_escalate_only_past_it(
    offset: int, expected_state: str
):
    custom_ms = 100
    kit = _make_reporter_past_bash_end(idle_warn_ms=custom_ms)
    kit.clock.now = BASH_CYCLE_END_MS + custom_ms + offset

    frame = render_frame(kit.reporter)

    assert _liveness_rows(frame) == [expected_state, "00:00:03  Bash   ...  1s"]


# ---------------------------------------------------------------------------
# liveness — model phase transitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "tool_name", "expected"),
    [
        pytest.param("thinking", None, "thinking  ~0 tokens  0s", id="thinking"),
        pytest.param("responding", None, "responding  0s", id="responding"),
        pytest.param("turn_end", None, "waiting  0s", id="turn_end"),
        pytest.param("tool_input", "Edit", "preparing Edit  0s", id="with-tool-name"),
        pytest.param("tool_input", None, "preparing unknown  0s", id="without-tool-name"),
    ],
)
def test_liveness_when_model_phase_reported_does_show_the_phase_with_its_elapsed(
    phase: ModelPhase, tool_name: str | None, expected: str
):
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(model_phase_event(2000, phase, tool_name=tool_name))

    frame = render_frame(kit.reporter)

    assert _liveness_rows(frame) == [expected]


# ---------------------------------------------------------------------------
# liveness — iterate sidecar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("passes_completed", "expected_row"),
    [
        pytest.param(7, "passes 7/10 · 31m 0s · ~11m 15s left", id="passes-left"),
        pytest.param(10, "passes 10/10 · 31m 0s", id="no-pass-left"),
    ],
)
def test_liveness_when_iterate_tool_has_sidecar_does_show_its_passes(
    passes_completed: int, expected_row: str
):
    sidecar = ProgressSnapshot(
        passes_completed=passes_completed,
        passes_total=10,
        last_pass_duration_ms=225_000.0,
    )
    kit = make_reporter(read_progress=lambda _root: sidecar)
    fire_launch_and_iterate_start(kit)
    kit.clock.now = 2000 + 31 * 60 * 1000

    frame = render_frame(kit.reporter)

    assert content_line(frame, "passes") == expected_row


def test_liveness_when_iterate_tool_has_no_sidecar_does_show_plain_elapsed():
    kit = make_reporter(read_progress=lambda _root: None)
    fire_launch_and_iterate_start(kit)
    kit.clock.now = 7000

    frame = render_frame(kit.reporter)

    assert content_line(frame, "Bash") == "00:00:02  Bash   gymrat iterate  5s"


# ---------------------------------------------------------------------------
# MCP iterate tool detection
# ---------------------------------------------------------------------------

#: An iterate sidecar halfway through its passes.
_HALFWAY = ProgressSnapshot(passes_completed=4, passes_total=8, last_pass_duration_ms=120_000.0)


def test_liveness_when_mcp_iterate_tool_in_flight_does_show_the_sidecar_passes():
    kit = make_reporter(read_progress=lambda _root: _HALFWAY)
    fire_launch_and_iterate_start(kit, tool_name="mcp__gymrat__iterate")
    kit.clock.now = 2000 + 10 * 60 * 1000

    frame = render_frame(kit.reporter)

    assert _liveness_rows(frame) == [
        "00:00:02  mcp__gymrat__iterate  gymrat iterate  10m 0s",
        "passes 4/8 · 10m 0s · ~8m left",
    ]


@pytest.mark.parametrize(
    ("tool_name", "input_summary", "expected_row"),
    [
        pytest.param(
            "mcp__gymrat__probe",
            "gymrat probe a b",
            "00:00:02  mcp__gymrat__probe  gymrat probe a b  3s",
            id="other-mcp-tool",
        ),
        pytest.param(
            "Bash", "npm test", "00:00:02  Bash   npm test  3s", id="bash-running-something-else"
        ),
        pytest.param(
            "Read",
            "notes on gymrat iterate",
            "00:00:02  Read   notes on gymrat iterate  3s",
            id="other-tool-naming-iterate",
        ),
    ],
)
def test_liveness_when_non_iterate_tool_in_flight_does_show_its_summary_without_the_sidecar(
    tool_name: str, input_summary: str, expected_row: str
):
    kit = make_reporter(read_progress=lambda _root: _HALFWAY)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event(tool_name, "tool-2", 2000, input_summary=input_summary))
    kit.clock.now = 5000

    frame = render_frame(kit.reporter)

    assert _liveness_rows(frame) == [expected_row]


# ---------------------------------------------------------------------------
# liveness — turn end and follow-up transitions
# ---------------------------------------------------------------------------


def test_liveness_when_follow_up_replied_does_show_it_on_the_turns_row():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(model_phase_event(1500, "responding"))
    observer(turn_end_event(2000, text="done"))
    observer(follow_up_event(3000, action="replied"))

    frame = render_frame(kit.reporter)

    assert _liveness_rows(frame) == ["turns  turn 1 ended · replied", "waiting  0s"]


# ---------------------------------------------------------------------------
# plain mode
# ---------------------------------------------------------------------------


def _plain(**options: Any) -> tuple[ReporterKit, list[str]]:
    """A plain-mode reporter in UTC whose milestone lines land in the returned list."""
    writes: list[str] = []
    return make_reporter(mode="plain", plain_write=writes.append, tz=UTC, **options), writes


def test_plain_when_no_writer_given_does_print_each_line_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    kit = make_reporter(mode="plain", max_minutes=60)

    kit.reporter.observer(launch_event(1000))

    assert capsys.readouterr().err == "caps 60m\n"


def test_plain_when_session_read_fails_again_after_recovering_does_not_warn_a_second_time():
    recovered = make_read_session(session_state(), has_baseline=True)
    reads = iter([_throwing_read, recovered, _throwing_read])

    def read_in_turn() -> ReadSessionResult:
        return next(reads)()

    kit, writes = _plain(read_session=read_in_turn)

    kit.reporter.refresh_session()
    kit.reporter.refresh_session()
    kit.reporter.refresh_session()

    assert writes == ["session read failed: no session file"]


def test_plain_when_warn_called_does_record_warning():
    kit, writes = _plain()
    kit.reporter.observer(launch_event(1000))

    kit.reporter.warn("heads up")

    assert writes[-1] == "heads up"


# ---------------------------------------------------------------------------
# exit phase — plain mode
# ---------------------------------------------------------------------------


_WAITING_LOCK = ExitPhase(kind="waiting-lock", pid=4242)


@pytest.mark.parametrize(
    ("second", "expected_lines"),
    [
        pytest.param(
            ExitPhase(kind="settling", pid=None),
            ["waiting for gymrat (PID 4242)", "settling…"],
            id="phase-changes",
        ),
        pytest.param(_WAITING_LOCK, ["waiting for gymrat (PID 4242)"], id="same-phase-repeats"),
    ],
)
def test_exit_phase_when_plain_does_write_a_line_only_when_the_phase_changes(
    second: ExitPhase, expected_lines: list[str]
):
    kit, writes = _plain()
    kit.reporter.observer(launch_event(1000))
    launched = len(writes)

    kit.reporter.exit_phase(_WAITING_LOCK)
    kit.clock.now = 5000
    kit.reporter.exit_phase(second)

    assert writes[launched:] == expected_lines


# ---------------------------------------------------------------------------
# run-end exit sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "expected_line"),
    [
        pytest.param(
            ExitPhase(kind="waiting-lock", pid=4242),
            "waiting for gymrat (PID 4242)  3s",
            id="waiting-lock",
        ),
        pytest.param(
            ExitPhase(kind="waiting-lock", pid=None),
            "waiting for gymrat (PID unknown)  3s",
            id="waiting-lock-unknown-pid",
        ),
        pytest.param(ExitPhase(kind="settling", pid=None), "settling…  3s", id="settling"),
    ],
)
def test_liveness_when_exit_phase_reported_does_show_the_phase_with_advancing_elapsed(
    phase: ExitPhase, expected_line: str
):
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 5000
    kit.reporter.exit_phase(phase)
    kit.clock.now = 8000

    frame = render_frame(kit.reporter)

    assert _liveness_rows(frame) == [expected_line]


_UNREADABLE = RuntimeError("session file unreadable")


@pytest.mark.parametrize(
    ("reread", "expected"),
    [
        pytest.param(KEPT_READ, KEPT_READ, id="rereads"),
        pytest.param(_UNREADABLE, EMPTY_READ, id="reread-fails-keeps-previous"),
    ],
)
def test_refresh_session_when_called_does_reread_the_session_keeping_the_last_good_one(
    reread: ReadSessionResult | Exception, expected: ReadSessionResult
):
    kit = make_reporter(
        mode="plain",
        read_session=Mock(side_effect=[EMPTY_READ, reread]),
        plain_write=lambda _line: None,
    )
    kit.reporter.observer(launch_event(1000))

    kit.reporter.refresh_session()

    assert kit.reporter.session_result() == expected
