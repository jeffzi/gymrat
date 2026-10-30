"""Behavioral tests for ``stop_reason``: which configured stop condition a session has met."""

import pytest

from gymrat.config import ResolvedConfig, StopConfig
from gymrat.loop.stop_condition import stop_reason
from gymrat.session.store import SessionState
from tests._config import resolved_config
from tests.session.records._fixtures import session_state

# ---------------------------------------------------------------------------
# stop_reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "state", "inserted"),
    [
        pytest.param(
            resolved_config(primary="total_ms", stop=StopConfig(target_value=95)),
            session_state(iteration_count=1, target_reached_and_kept=True),
            (),
            id="target-reached-and-kept",
        ),
        pytest.param(
            resolved_config(primary="total_ms", stop=StopConfig(target_value=95)),
            session_state(iteration_count=1, target_reached_and_kept=False),
            None,
            id="target-not-kept",
        ),
        pytest.param(
            resolved_config(stop=StopConfig(max_iterations=2)),
            session_state(iteration_count=2),
            ("2",),
            id="max-iterations-met",
        ),
        pytest.param(
            resolved_config(),
            session_state(iteration_count=5, target_reached_and_kept=True),
            None,
            id="no-stop-configured",
        ),
    ],
)
def test_stop_reason_when_a_configured_condition_is_met_does_name_it(
    config: ResolvedConfig, state: SessionState, inserted: tuple[str, ...] | None
):
    result = stop_reason(config, state)

    assert (result is None) == (inserted is None)
    assert all(value in (result or "") for value in inserted or ())
