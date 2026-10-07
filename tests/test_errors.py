from collections.abc import Callable
from functools import partial

import pytest

from gymrat.errors import GymratError
from gymrat.loop.iterate.run import BudgetExceededError, LoopStopError
from gymrat.session.schema import CommandReason

# ---------------------------------------------------------------------------
# GymratError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hint", "reason"),
    [
        pytest.param(None, None, id="bare"),
        pytest.param("try this", "error", id="hint-and-reason"),
    ],
)
def test_gymrat_error_when_constructed_does_expose_its_message_hint_and_reason(
    hint: str | None, reason: CommandReason | None
):
    error = GymratError("something broke", hint=hint, reason=reason)

    assert (str(error), error.hint, error.reason) == ("something broke", hint, reason)


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
