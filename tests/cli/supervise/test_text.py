"""Tests for the Rich-free loop-summary and exit-phase text of the supervise dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gymrat.cli.supervise.text import exit_phase_text, loop_segments
from gymrat.supervisor.exit_sequence import ExitPhase
from tests.cli.supervise._fixtures import read_result
from tests.session.records._fixtures import make_iteration, session_state

if TYPE_CHECKING:
    from gymrat.cli.supervise.text import LoopStyle
    from gymrat.session.schema import Outcome


# ---------------------------------------------------------------------------
# loop_segments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "expected_role"),
    [
        pytest.param("improved", "done", id="improved"),
        pytest.param("regressed", "regressed", id="regressed"),
        pytest.param("no-signal", "meta", id="no-signal"),
    ],
)
def test_loop_segments_when_last_iteration_has_an_outcome_does_style_its_delta_and_outcome_by_it(
    outcome: Outcome, expected_role: LoopStyle
):
    state = session_state(iteration_count=1, last_iteration=make_iteration(-3.2, outcome))

    segments = loop_segments(read_result(state), None)

    delta, _space, word = segments[-3:]
    assert (delta.text, delta.role, word.text, word.role) == (
        "-3.2%",
        expected_role,
        outcome,
        expected_role,
    )


# ---------------------------------------------------------------------------
# exit_phase_text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        pytest.param(ExitPhase(kind="settling", pid=None), "settling…", id="settling"),
        pytest.param(
            ExitPhase(kind="waiting-lock", pid=4242),
            "waiting for gymrat (PID 4242)",
            id="waiting-lock",
        ),
        pytest.param(
            ExitPhase(kind="waiting-lock", pid=None),
            "waiting for gymrat (PID unknown)",
            id="waiting-lock-unknown-pid",
        ),
    ],
)
def test_exit_phase_text_when_phase_varies_does_name_the_phase_and_lock_holder(
    phase: ExitPhase, expected: str
):
    assert exit_phase_text(phase) == expected
