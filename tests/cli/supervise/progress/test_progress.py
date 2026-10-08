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
model phase transitions, the iterate sidecar and MCP tool detection, and the
follow-up transition. During the run-end exit sequence the reporter shows each
exit phase as the liveness line, and a session refresh re-reads the session,
keeping the last good read when one fails. The reducer transitions behind them
are pinned in ``test_reducer.py``.

**Plain mode** tests assert on the recorded milestone lines.
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from datetime import UTC, timedelta, timezone
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest

from gymrat.cli.supervise.progress import (
    IDLE_WARN_MS,
    create_supervise_reporter,
    read_live_session,
)
from gymrat.cli.supervise.types import BestIteration, ReadSessionResult
from gymrat.session.progress_file import ProgressSnapshot
from gymrat.session.records import IterationPrimary
from gymrat.supervisor.exit_sequence import ExitPhase
from tests._rich import track
from tests.cli.supervise._fixtures import (
    BASH_CYCLE_END_MS,
    LIVE_CLASS_PATH,
    ReporterKit,
    _throwing_read,
    cap_event,
    fire_launch_and_bash_cycle,
    follow_up_event,
    launch_event,
    lines_containing,
    make_read_session,
    make_reporter,
    model_phase_event,
    render_frame,
    session_state_three_iterations,
    thinking_event,
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
    empty_session_state,
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

    from gymrat.session.records import SessionLogRecord
    from gymrat.session.schema import PrimaryKind
    from gymrat.supervisor.events import CapAction, CapType, ModelPhase


# ---------------------------------------------------------------------------
# factory / contract
# ---------------------------------------------------------------------------


def test_session_result_when_a_tool_ends_does_return_the_latest_read():
    state = session_state_three_iterations(-4.2, "improved", seq=3)
    kit = make_reporter(read_session=make_read_session(state, has_baseline=True))

    fire_launch_and_bash_cycle(kit.reporter.observer)

    session_result = kit.reporter.session_result()
    assert session_result is not None
    assert session_result.state == state


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
    reporter.stop()

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
# time bar
# ---------------------------------------------------------------------------


def test_time_bar_when_elapsed_exceeds_max_does_clamp_remaining_to_zero():
    kit = make_reporter(max_minutes=60, clock_start=1000)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 1000 + (2 * 3600) * 1000

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "cap in") == "time " + "━" * 73 + " 2h 00m  cap in 0s"


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
# loop row
# ---------------------------------------------------------------------------


def test_loop_when_iterations_present_does_show_counts_and_last():
    state = session_state_three_iterations(-3.2, "improved")
    kit = make_reporter(
        max_iterations=20,
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "loop") == (
        "loop   3/20 iterations · 2 kept · 1 discarded · last -3.2% improved"
    )


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

    assert _content_line(frame, "best") == "best -6.8% geomean (iteration 3)"


# ---------------------------------------------------------------------------
# session re-read
# ---------------------------------------------------------------------------


def test_reread_when_any_tool_ends_does_reread_on_the_end_not_the_start():
    read = Mock(return_value=ReadSessionResult(state=empty_session_state(), has_baseline=False))
    kit = make_reporter(read_session=read)
    observer = kit.reporter.observer
    events = [
        tool_start_event("Read", "read-1", 2000),
        tool_end_event("Read", "read-1", 3000),
        tool_start_event("Bash", "bash-1", 4000),
        tool_start_event("Read", "nested-1", 4100, parent_tool_use_id="bash-1"),
        tool_end_event("Read", "nested-1", 4200, parent_tool_use_id="bash-1"),
        tool_end_event("Bash", "bash-1", 5000),
    ]
    observer(launch_event(1000))
    reads = [read.call_count]

    for event in events:
        observer(event)
        reads.append(read.call_count)

    assert [after - before for before, after in itertools.pairwise(reads)] == [0, 1, 0, 0, 1, 1]


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


@pytest.mark.parametrize("cap", ["wall-clock", "spend-cap"])
@pytest.mark.parametrize("action", ["ending", "interrupting"])
def test_cap_when_fired_does_show_action_with_cap_type(cap: CapType, action: CapAction):
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.reporter.observer(cap_event(cap, action=action))

    frame = render_frame(kit.reporter)

    assert _content_line(frame, f"({cap})") == f"{action} ({cap})"


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


def _content_line(frame: str, needle: str) -> str:
    """The sole frame row containing *needle*, with panel border and padding stripped."""
    lines = lines_containing(frame, needle)
    assert len(lines) == 1, f"expected exactly one line containing {needle!r}, got {lines}"
    return lines[0].split("│")[1].strip()


def _liveness_rows(frame: str) -> list[str]:
    """The frame rows below the loop row, with panel border and padding stripped."""
    rows = [line.split("│")[1].strip() for line in frame.splitlines() if line.startswith("│")]
    loop_index = next(i for i, row in enumerate(rows) if row.startswith("loop "))
    return rows[loop_index + 1 :]


# ---------------------------------------------------------------------------
# liveness — ended tools
# ---------------------------------------------------------------------------


def test_liveness_when_four_tools_finish_does_show_only_last_three():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))

    for i, (name, summary) in enumerate(
        [("Read", "src/a.ts"), ("Edit", "src/b.ts"), ("Bash", "npm test"), ("Read", "src/c.ts")],
        start=1,
    ):
        ts = 1000 + i * 1000
        kit.clock.now = ts
        kit.reporter.observer(tool_start_event(name, f"t-{i}", ts, input_summary=summary))
        kit.clock.now = ts + 500
        kit.reporter.observer(tool_end_event(name, f"t-{i}", ts + 500, started_at_ms=ts))

    frame = render_frame(kit.reporter)

    assert _liveness_rows(frame) == [
        "waiting  0s",
        "00:00:05  Read   src/c.ts  <1s",
        "00:00:04  Bash   npm test  <1s",
        "00:00:03  Edit   src/b.ts  <1s",
    ]


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
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event("Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    )
    kit.clock.now = 3000
    kit.reporter.observer(tool_end_event("Edit", "edit-1", 3000, result=result))

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "Edit") == expected_row


# ---------------------------------------------------------------------------
# sub-second finished tool
# ---------------------------------------------------------------------------


def test_finished_tool_when_under_one_second_does_show_less_than_one_second():
    kit = make_reporter()
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Edit", "edit-1", 2000, input_summary="src/a.ts"))
    kit.clock.now = 2500
    kit.reporter.observer(tool_end_event("Edit", "edit-1", 2500))

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "Edit") == "00:00:02  Edit   src/a.ts  <1s"


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

    assert _content_line(frame, "Edit") == (
        "00:00:02  Edit   src/level0/level1/level2/level3/level4/level5/level6/level7/level8/level9/le…"
    )


# ---------------------------------------------------------------------------
# wall-clock finished-tool lines
# ---------------------------------------------------------------------------


# 2023-11-14 22:13:20 UTC.
_WALL_CLOCK_EPOCH_MS = 1_700_000_000_000


@pytest.mark.parametrize(
    ("tz", "expected_clock"),
    [
        pytest.param(UTC, "22:13:20", id="utc"),
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

    assert _content_line(frame, "Edit") == f"{expected_clock}  Edit   src/archetype.ts  1s"


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
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event("Edit", "edit-1", 2000, input_summary="src/archetype.ts")
    )
    kit.clock.now = 3000
    kit.reporter.observer(tool_end_event("Edit", "edit-1", 3000))

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "Edit") == "05:30:03  Edit   src/archetype.ts  1s"


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

    assert _content_line(frame, "no output") == "no output for 30s (last tool: Bash ✗ at 00:00:03)"


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
    ("phase", "expected"),
    [
        pytest.param("thinking", "thinking  ~0 tokens  0s", id="thinking"),
        pytest.param("responding", "responding  0s", id="responding"),
        pytest.param("turn_end", "waiting  0s", id="turn_end"),
    ],
)
def test_liveness_when_model_phase_reported_does_show_the_phase_with_its_elapsed(
    phase: ModelPhase, expected: str
):
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(model_phase_event(2000, phase))

    frame = render_frame(kit.reporter)

    assert _liveness_rows(frame) == [expected]


def test_liveness_when_model_phase_thinking_after_thinking_update_does_preserve_token_count():
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(thinking_event(1500, estimated_tokens=200))
    observer(model_phase_event(2000, "thinking"))

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "thinking") == "thinking  ~200 tokens  0s"


@pytest.mark.parametrize(
    ("tool_name", "expected_state"),
    [
        pytest.param("Edit", "preparing Edit  0s", id="with-tool-name"),
        pytest.param(None, "preparing unknown  0s", id="without-tool-name"),
    ],
)
def test_liveness_when_model_phase_tool_input_does_show_preparing(
    tool_name: str | None, expected_state: str
):
    kit = make_reporter()
    observer = kit.reporter.observer
    observer(launch_event(1000))
    observer(model_phase_event(2000, "tool_input", tool_name=tool_name))

    frame = render_frame(kit.reporter)

    assert _liveness_rows(frame) == [expected_state]


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
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate"))
    kit.clock.now = 2000 + 31 * 60 * 1000

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "passes") == expected_row


def test_liveness_when_iterate_tool_has_no_sidecar_does_show_plain_elapsed():
    kit = make_reporter(read_progress=lambda _root: None)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(tool_start_event("Bash", "bash-1", 2000, input_summary="gymrat iterate"))
    kit.clock.now = 7000

    frame = render_frame(kit.reporter)

    assert _content_line(frame, "Bash") == "00:00:02  Bash   gymrat iterate  5s"


# ---------------------------------------------------------------------------
# MCP iterate tool detection
# ---------------------------------------------------------------------------

#: An iterate sidecar halfway through its passes.
_HALFWAY = ProgressSnapshot(passes_completed=4, passes_total=8, last_pass_duration_ms=120_000.0)


def test_liveness_when_mcp_iterate_tool_in_flight_does_show_the_sidecar_passes():
    kit = make_reporter(read_progress=lambda _root: _HALFWAY)
    kit.reporter.observer(launch_event(1000))
    kit.clock.now = 2000
    kit.reporter.observer(
        tool_start_event("mcp__gymrat__iterate", "mcp-1", 2000, input_summary="gymrat iterate")
    )
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

    reports = [line for line in writes if "session read failed" in line]
    assert reports == ["session read failed: no session file"]


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

_EMPTY = ReadSessionResult(state=empty_session_state(), has_baseline=True)
_KEPT = ReadSessionResult(
    state=session_state(
        iteration_count=1, keep_count=1, last_iteration=make_iteration(-2.0, "improved")
    ),
    has_baseline=True,
)


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
        pytest.param(_KEPT, _KEPT, id="rereads"),
        pytest.param(_UNREADABLE, _EMPTY, id="reread-fails-keeps-previous"),
    ],
)
def test_refresh_session_when_called_does_reread_the_session_keeping_the_last_good_one(
    reread: ReadSessionResult | Exception, expected: ReadSessionResult
):
    kit = make_reporter(
        mode="plain",
        read_session=Mock(side_effect=[_EMPTY, reread]),
        plain_write=lambda _line: None,
    )
    kit.reporter.observer(launch_event(1000))

    kit.reporter.refresh_session()

    assert kit.reporter.session_result() == expected


@pytest.mark.parametrize(
    ("reread", "expected_repaints"),
    [
        pytest.param(_KEPT, 1, id="reread"),
        pytest.param(_UNREADABLE, 0, id="reread-fails"),
    ],
)
def test_refresh_session_when_live_does_repaint_only_after_a_successful_reread(
    reread: ReadSessionResult | Exception, expected_repaints: int
):
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        live = mock_live_cls.return_value
        kit = make_reporter(mode="live", read_session=Mock(side_effect=[_EMPTY, reread]))
        kit.reporter.observer(launch_event(1000))
        painted = live.refresh.call_count

        kit.reporter.refresh_session()

        assert live.refresh.call_count - painted == expected_repaints
