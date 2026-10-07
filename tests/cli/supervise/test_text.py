"""Tests for the plain-text loop label the supervise display builds from a session."""

from __future__ import annotations

import pytest

from gymrat.cli.supervise.reducer import loop_plain_text
from tests.cli.supervise._fixtures import make_read_session
from tests.session.records._fixtures import session_state


@pytest.mark.parametrize(
    ("iteration_count", "max_iterations", "expected_label"),
    [
        pytest.param(1, None, "1 iteration", id="one-uncapped"),
        pytest.param(2, None, "2 iterations", id="two-uncapped"),
        pytest.param(1, 20, "1/20 iterations", id="one-capped"),
        pytest.param(2, 20, "2/20 iterations", id="two-capped"),
    ],
)
def test_loop_plain_text_when_iteration_count_varies_does_lead_with_the_iteration_label(
    iteration_count: int, max_iterations: int | None, expected_label: str
) -> None:
    session_result = make_read_session(
        session_state(iteration_count=iteration_count), has_baseline=True
    )()

    text = loop_plain_text(session_result, max_iterations)

    assert text.split(" · ")[0] == expected_label
