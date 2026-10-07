"""Behavioral tests for the supervise reporter in plain (non-Live) mode.

Tests assert on recorded milestone lines.
"""

from __future__ import annotations

from datetime import UTC, tzinfo
from typing import TYPE_CHECKING, NamedTuple

import pytest

from gymrat.supervisor.events import CompactionEvent
from gymrat.supervisor.exit_sequence import ExitPhase
from gymrat.utils import NS_PER_MS
from tests._logging import unhandled_logging
from tests.cli.supervise._fixtures import (
    ReporterKit,
    _throwing_read,
    cap_event,
    fire_launch_and_bash_cycle,
    follow_up_event,
    launch_event,
    make_read_session,
    make_reporter,
    tool_end_event,
    tool_start_event,
    turn_end_event,
    usage_event,
)
from tests.session.records._fixtures import make_iteration, session_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from gymrat.cli.supervise.progress import SuperviseReporter
    from gymrat.cli.supervise.types import ReadSessionResult
    from gymrat.supervisor.events import SessionObserver


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
    tz: tzinfo | None = UTC,
) -> PlainCapture:
    """Build a plain-mode reporter with a write-capturing callback.

    Each milestone line the reporter emits is appended to the ``writes`` list.
    The ``tz`` parameter defaults to ``UTC`` for stable assertions.
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
        tz=tz,
    )
    return PlainCapture(kit, writes)


def test_plain_when_no_writer_given_does_print_each_line_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    kit = make_reporter(mode="plain", max_minutes=60)

    kit.reporter.observer(launch_event(1000))

    assert capsys.readouterr().err == "caps 60m\n"


def test_plain_when_launched_with_spend_cap_does_print_caps_with_dollars():
    plain = make_plain_reporter(max_usd=5.0, max_minutes=60)

    plain.observer(launch_event(1000, max_usd=5.0))

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

    plain.observer(launch_event(1000, max_minutes=max_minutes))

    assert plain.writes[-1] == expected


def test_plain_when_usage_update_does_print_cost():
    plain = make_plain_reporter(max_usd=5.0)

    plain.observer(launch_event(1000, max_usd=5.0))
    plain.observer(usage_event(1.42, 2000))

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

    plain.observer(launch_event(1000))
    plain.observer(tool_start_event("Bash", "bash-1", 2000))
    plain.observer(tool_end_event("Bash", "bash-1", 3000))

    assert plain.writes[-1] == "2/20 iterations · 1 kept · 1 discarded · last +3.2% regressed"


def test_plain_when_no_session_yet_does_not_print_loop_segment():
    plain = make_plain_reporter(read_session=_throwing_read)

    plain.observer(launch_event(1000))

    assert plain.writes[-1] == "caps 60m"
    assert all("no session yet" not in w for w in plain.writes)


def test_plain_when_session_read_keeps_failing_does_warn_once_and_leave_stderr_empty(
    capsys: pytest.CaptureFixture[str],
):
    plain = make_plain_reporter(read_session=_throwing_read)

    with unhandled_logging():
        fire_launch_and_bash_cycle(plain.observer)
        plain.reporter.refresh_session()

    reports = [line for line in plain.writes if "session read failed" in line]
    assert (reports, capsys.readouterr().err) == (["session read failed: no session file"], "")


def test_plain_when_session_read_fails_again_after_recovering_does_not_warn_a_second_time():
    recovered = make_read_session(session_state(), has_baseline=True)
    reads = iter([_throwing_read, recovered, _throwing_read])

    def read_in_turn() -> ReadSessionResult:
        return next(reads)()

    plain = make_plain_reporter(read_session=read_in_turn)

    plain.reporter.refresh_session()
    plain.reporter.refresh_session()
    plain.reporter.refresh_session()

    reports = [line for line in plain.writes if "session read failed" in line]
    assert reports == ["session read failed: no session file"]


def test_plain_when_capped_does_print_cap_interrupting():
    plain = make_plain_reporter()

    plain.observer(launch_event(1000))
    plain.observer(cap_event("wall-clock"))

    cap_line = plain.writes[-1]
    assert cap_line == "cap wall-clock — interrupting"


def test_plain_when_warn_called_does_record_warning():
    plain = make_plain_reporter()
    plain.observer(launch_event(1000))

    plain.reporter.warn("heads up")

    assert plain.writes[-1] == "heads up"


def test_plain_stop_when_called_does_not_raise():
    plain = make_plain_reporter()
    plain.observer(launch_event(1000))

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
    plain.observer(launch_event(1000))
    plain.observer(turn_end_event(2000, text="done"))

    reason = "budget exhausted" if action == "ended" else None
    plain.observer(follow_up_event(3000, action=action, reason=reason))  # type: ignore[arg-type]

    assert any(f"turn 1 ended · {expected_suffix}" in w for w in plain.writes)


# ---------------------------------------------------------------------------
# cap event — interrupting vs ending
# ---------------------------------------------------------------------------


def test_plain_when_capped_while_idle_after_turn_end_does_print_cap_ending():
    plain = make_plain_reporter()
    plain.observer(launch_event(1000))
    plain.observer(turn_end_event(2000, text="done"))
    plain.observer(cap_event("wall-clock", action="ending"))

    assert plain.writes[-1] == "cap wall-clock — ending"


# ---------------------------------------------------------------------------
# compaction event — plain mode
# ---------------------------------------------------------------------------


def test_plain_when_compaction_does_print_context_compacted():
    plain = make_plain_reporter()
    plain.observer(launch_event(1000))
    plain.observer(CompactionEvent(at=3000 * NS_PER_MS))

    assert any("context compacted" in w for w in plain.writes)


# ---------------------------------------------------------------------------
# exit phase — plain mode
# ---------------------------------------------------------------------------


def test_plain_exit_phase_when_phase_changes_does_write_each_phase_line():
    plain = make_plain_reporter()
    plain.observer(launch_event(1000))
    launched = len(plain.writes)

    plain.reporter.exit_phase(ExitPhase(kind="waiting-lock", pid=4242))
    plain.reporter.exit_phase(ExitPhase(kind="settling", pid=None))

    assert plain.writes[launched:] == ["waiting for gymrat (PID 4242)", "settling…"]


def test_plain_exit_phase_when_same_phase_repeats_does_write_nothing():
    plain = make_plain_reporter()
    plain.observer(launch_event(1000))
    plain.reporter.exit_phase(ExitPhase(kind="waiting-lock", pid=4242))
    written = list(plain.writes)
    plain.kit.clock.now = 5000

    plain.reporter.exit_phase(ExitPhase(kind="waiting-lock", pid=4242))

    assert plain.writes == written
