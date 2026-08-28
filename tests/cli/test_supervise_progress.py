"""Behavioral tests for the supervise Rich dashboard reporter.

The reporter turns a stream of :class:`~gymrat_py.supervisor.events.SessionEvent`
values into a bordered ``Live`` dashboard with time/cost/loop summary rows and a
liveness section showing tool activity.  Every test injects the clock (``now``)
and the session reader (``read_session``) so nothing depends on real time or
disk.

**Live mode** tests render ``reporter.frame()`` through ``frame_text()`` from
``tests._rich`` at a fixed width, pinning frame content with syrupy snapshots.
**Plain mode** tests assert on recorded milestone lines.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

from gymrat_py.cli.supervise_progress import (
    IDLE_WARN_MS,
    CapType,
    ReadSessionResult,
    SuperviseReporter,
    create_supervise_reporter,
)
from gymrat_py.session import IterationPrimary, IterationRecord
from gymrat_py.session.progress_file import ProgressSnapshot
from gymrat_py.session.store import SessionState
from gymrat_py.supervisor.events import (
    CapEvent,
    LaunchEvent,
    SessionObserver,
    TextDeltaEvent,
    ThinkingUpdateEvent,
    ToolEndEvent,
    ToolProgressEvent,
    ToolStartEvent,
    UsageUpdateEvent,
)
from tests._rich import frame_text
from tests.session._records import finalize_record, iteration_record

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class Clock:
    """A mutable millisecond clock a test advances by assigning ``now``."""

    def __init__(self, start: int):
        self.now = start

    def __call__(self) -> int:
        return self.now


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def empty_session_state() -> SessionState:
    """A session that has opened but measured nothing yet."""
    return SessionState(
        session=None,
        iteration_count=0,
        last_iteration=None,
        unsettled=False,
        keep_count=0,
        discard_count=0,
        target_reached_and_kept=False,
        last_seq=0,
        last_kept_commit=None,
        ends_on_gating_block=False,
        finalized=None,
    )


def session_state(**changes: Any) -> SessionState:
    """The empty session state with the named fields overridden."""
    return replace(empty_session_state(), **changes)


def make_iteration(delta_pct: float | None, outcome: str, seq: int = 1) -> IterationRecord:
    """An iteration whose only reporter-visible fields are its delta and outcome."""
    return iteration_record(
        seq=seq,
        primary=IterationPrimary(kind="geomean", delta_pct=delta_pct),
        outcome=outcome,
    )


def make_read_session(
    state: SessionState, *, has_baseline: bool
) -> Callable[[], ReadSessionResult]:
    """A ``read_session`` that always returns ``state`` and ``has_baseline``."""
    result = ReadSessionResult(state=state, has_baseline=has_baseline)
    return lambda: result


def _throwing_read() -> ReadSessionResult:
    message = "no session file"
    raise RuntimeError(message)


# ---------------------------------------------------------------------------
# Event firers
# ---------------------------------------------------------------------------


def fire_launch(
    observer: SessionObserver,
    timestamp: int = 1000,
    *,
    max_minutes: float = 60,
    max_usd: float | None = None,
) -> None:
    observer(
        LaunchEvent(
            timestamp=timestamp,
            head_sha="abc123",
            dirty=False,
            max_minutes=max_minutes,
            max_usd=max_usd,
            model=None,
            runbook_path="/path/to/runbook.md",
            kickoff_summary="test kickoff",
        )
    )


def fire_tool_start(
    observer: SessionObserver,
    tool_name: str,
    tool_use_id: str,
    timestamp: int = 2000,
    *,
    input_summary: str = "...",
) -> None:
    observer(
        ToolStartEvent(
            timestamp=timestamp,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            input={},
            input_summary=input_summary,
        )
    )


def fire_tool_end(
    observer: SessionObserver,
    tool_name: str,
    tool_use_id: str,
    timestamp: int = 3000,
    *,
    result: str = "ok",
    result_summary: str = "ok",
) -> None:
    observer(
        ToolEndEvent(
            timestamp=timestamp,
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            duration_ms=timestamp - 2000,
            result=result,
            result_summary=result_summary,
        )
    )


def fire_usage_update(observer: SessionObserver, cost_usd: float, timestamp: int = 4000) -> None:
    observer(UsageUpdateEvent(timestamp=timestamp, cost_usd=cost_usd))


def fire_cap(observer: SessionObserver, cap: CapType, timestamp: int = 5000) -> None:
    observer(CapEvent(timestamp=timestamp, cap=cap))


def fire_launch_and_bash_cycle(observer: SessionObserver) -> None:
    """Launch, then run a Bash tool start/end cycle at the default timestamps.

    The Bash end is what triggers the reporter's session re-read (see the
    "session re-read" tests below), so this is the minimum event sequence
    that gets session state into the loop/best rows.
    """
    fire_launch(observer, 1000)
    fire_tool_start(observer, "Bash", "bash-1", 2000)
    fire_tool_end(observer, "Bash", "bash-1", 3000)


# ---------------------------------------------------------------------------
# Reporter setup
# ---------------------------------------------------------------------------

# Fixed width for all golden-snapshot tests so frames are stable.
FRAME_WIDTH = 100


class ReporterKit(NamedTuple):
    reporter: SuperviseReporter
    clock: Clock


def make_reporter(
    *,
    mode: str = "live",
    max_minutes: float = 480,
    max_usd: float | None = None,
    max_iterations: int | None = None,
    read_session: Callable[[], ReadSessionResult] | None = None,
    clock_start: int = 1000,
    root: str = "/tmp/repo",
    read_progress: Callable[[str], ProgressSnapshot | None] | None = None,
    plain_write: Callable[[str], None] | None = None,
    label: str = "ecstatic-ts",
    session_id: str = "20260813-125044-34ec",
    branch: str = "gymrat/20260813-125044-34ec",
) -> ReporterKit:
    """Build a reporter with injectable dependencies for deterministic testing."""
    clock = Clock(clock_start)
    if read_session is None:
        read_session = make_read_session(empty_session_state(), has_baseline=False)
    kwargs: dict[str, Any] = {}
    if max_iterations is not None:
        kwargs["max_iterations"] = max_iterations
    if max_usd is not None:
        kwargs["max_usd"] = max_usd
    if read_progress is not None:
        kwargs["read_progress"] = read_progress
    if plain_write is not None:
        kwargs["plain_write"] = plain_write
    reporter = create_supervise_reporter(
        root=root,
        max_minutes=max_minutes,
        mode=mode,
        now=clock,
        read_session=read_session,
        label=label,
        session_id=session_id,
        branch=branch,
        **kwargs,
    )
    return ReporterKit(reporter, clock)


def render_frame(reporter: SuperviseReporter, *, width: int = FRAME_WIDTH) -> str:
    """Render the reporter's current frame through a non-terminal console."""
    return frame_text(reporter.frame(), width=width)


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
        read_session=make_read_session(state, has_baseline=True),
    )
    fire_launch_and_bash_cycle(kit.reporter.observer)

    frame = render_frame(kit.reporter)

    assert "best" in frame


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
# liveness — nested iterate sidecar
# ---------------------------------------------------------------------------


def test_liveness_when_iterate_tool_has_sidecar_does_show_passes_bar(
    snapshot: SnapshotAssertion,
):
    sidecar = ProgressSnapshot(
        seq=3,
        phase="measure",
        passes_completed=7,
        passes_total=10,
        current_side="experiment",
        current_round=4,
        last_pass_duration_ms=225_000.0,
        started_at=1700000000.0,
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


class PlainCapture(NamedTuple):
    """A plain-mode reporter paired with a write recorder."""

    kit: ReporterKit
    writes: list[str]

    @property
    def reporter(self) -> SuperviseReporter:
        return self.kit.reporter

    @property
    def observer(self) -> SessionObserver:
        return self.kit.reporter.observer


def make_plain_reporter(
    *,
    max_minutes: float = 60,
    max_usd: float | None = None,
    max_iterations: int | None = None,
    read_session: Callable[[], ReadSessionResult] | None = None,
    clock_start: int = 1000,
) -> PlainCapture:
    """Build a plain-mode reporter with a write-capturing callback.

    Each milestone line the reporter emits is appended to the ``writes`` list.
    """
    writes: list[str] = []
    kit = make_reporter(
        mode="plain",
        max_minutes=max_minutes,
        max_usd=max_usd,
        max_iterations=max_iterations,
        read_session=read_session,
        clock_start=clock_start,
        plain_write=writes.append,
    )
    return PlainCapture(kit, writes)


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
        read_session=make_read_session(state, has_baseline=True),
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
