"""Behavioral tests for the supervise dashboard during the run-end exit sequence.

The reporter shows each exit phase as the liveness line and, when a follow-up
ends while an exit phase is showing, records the exit step and re-reads the
session so the closing summary describes the session as the sequence left it.
Plain-mode phase lines live in ``test_progress_plain.py``.
"""

from __future__ import annotations

import pytest

from gymrat.cli.supervise.state import ReadSessionResult
from gymrat.supervisor.exit_sequence import ExitPhase
from tests.cli.supervise._fixtures import (
    empty_session_state,
    fire_follow_up,
    fire_launch,
    make_iteration,
    make_reporter,
    render_frame,
    session_state,
)


class SwitchableRead:
    """A ``read_session`` returning whichever result the test last assigned, raising on ``None``."""

    def __init__(self, result: ReadSessionResult | None) -> None:
        self.result = result

    def __call__(self) -> ReadSessionResult:
        if self.result is None:
            message = "session file unreadable"
            raise RuntimeError(message)
        return self.result


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
    fire_launch(kit.reporter.observer, 1000)
    kit.clock.now = 5000
    kit.reporter.exit_phase(phase)
    kit.clock.now = 8000

    frame = render_frame(kit.reporter)

    assert expected_line in frame


def test_follow_up_when_ended_while_exiting_does_show_the_exit_step():
    kit = make_reporter()
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    kit.reporter.exit_phase(ExitPhase(kind="settling", pid=None))
    fire_follow_up(observer, 8000, action="ended", reason="finalize")

    frame = render_frame(kit.reporter)

    assert "exit · finalize" in frame


@pytest.mark.parametrize(
    ("exiting", "reread", "expected"),
    [
        pytest.param(True, _KEPT, _KEPT, id="exiting-rereads"),
        pytest.param(True, None, _EMPTY, id="exiting-reread-fails-keeps-previous"),
        pytest.param(False, _KEPT, _EMPTY, id="not-exiting-skips-reread"),
    ],
)
def test_session_result_when_follow_up_ends_does_reread_only_while_exiting(
    exiting: bool, reread: ReadSessionResult | None, expected: ReadSessionResult
):
    reader = SwitchableRead(_EMPTY)
    kit = make_reporter(read_session=reader)
    observer = kit.reporter.observer
    fire_launch(observer, 1000)
    if exiting:
        kit.reporter.exit_phase(ExitPhase(kind="settling", pid=None))
    reader.result = reread

    fire_follow_up(observer, 8000, action="ended", reason="keep")

    assert kit.reporter.session_result() == expected
