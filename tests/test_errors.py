from collections.abc import Callable
from functools import partial

import pytest

from gymrat.errors import GATE_EXIT_CODE, TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.loop.iterate.run import BudgetExceededError, LoopStopError

# ---------------------------------------------------------------------------
# exit codes
# ---------------------------------------------------------------------------


def test_exit_codes_when_imported_does_match_the_documented_cli_contract():
    assert (GATE_EXIT_CODE, TOOL_FAILURE_EXIT_CODE) == (1, 2)


# ---------------------------------------------------------------------------
# GymratError
# ---------------------------------------------------------------------------


def test_gymrat_error_when_given_only_a_message_does_default_hint_and_reason_to_none():
    error = GymratError("something broke")

    assert (str(error), error.hint, error.reason) == ("something broke", None, None)


# ---------------------------------------------------------------------------
# LoopStopError / BudgetExceededError — reason defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "reason"),
    [
        pytest.param(LoopStopError, "stop-condition", id="loop-stop-default"),
        pytest.param(partial(LoopStopError, reason="error"), "error", id="loop-stop-overridden"),
        pytest.param(BudgetExceededError, "budget-exceeded", id="budget-default"),
        pytest.param(partial(BudgetExceededError, reason="error"), "error", id="budget-overridden"),
    ],
)
def test_stop_error_when_constructed_does_default_its_reason_unless_given(
    build: Callable[..., GymratError], reason: str
):
    err = build("stopped", hint="check it")

    assert (str(err), err.hint, err.reason) == ("stopped", "check it", reason)
