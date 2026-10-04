import pytest

from gymrat.errors import (
    CommandError,
    GymratError,
)
from gymrat.loop.iterate.run import BudgetExceededError, LoopStopError

# ---------------------------------------------------------------------------
# GymratError
# ---------------------------------------------------------------------------


def test_gymrat_error_when_constructed_does_expose_message_and_optional_hint():
    without_hint = GymratError("something broke")
    with_hint = GymratError("something broke", hint="try restarting")

    assert isinstance(without_hint, Exception)
    assert str(without_hint) == "something broke"
    assert without_hint.hint is None
    assert str(with_hint) == "something broke"
    assert with_hint.hint == "try restarting"


def test_gymrat_error_subclass_when_declared_inline_does_inherit_signature_and_handling():
    class CustomError(GymratError):
        pass

    err = CustomError("boom", hint="reset it")

    assert isinstance(err, GymratError)
    assert str(err) == "boom"
    assert err.hint == "reset it"
    assert CustomError("boom").hint is None


def test_gymrat_error_when_constructed_does_expose_reason():
    without_reason = GymratError("something broke")
    with_reason = GymratError("something broke", reason="error")
    with_both = GymratError("something broke", hint="try this", reason="error")

    assert without_reason.reason is None
    assert with_reason.reason == "error"
    assert with_both.reason == "error"
    assert with_both.hint == "try this"


@pytest.mark.parametrize(
    "cls",
    [GymratError, CommandError],
    ids=["GymratError", "CommandError"],
)
def test_error_when_inspected_does_annotate_reason_as_command_reason(
    cls: type[GymratError],
):
    from typing import get_type_hints

    from gymrat.session.schema import CommandReason

    hints = get_type_hints(cls.__init__, None, {"CommandReason": CommandReason})

    assert hints["reason"] == CommandReason | None


# ---------------------------------------------------------------------------
# LoopStopError / BudgetExceededError — reason defaults
# ---------------------------------------------------------------------------


def test_loop_stop_error_when_constructed_does_default_reason_to_stop_condition():
    err = LoopStopError("max iterations reached", hint="increase limit")

    assert err.reason == "stop-condition"
    assert str(err) == "max iterations reached"
    assert err.hint == "increase limit"


def test_loop_stop_error_when_reason_overridden_does_use_given_value():
    err = LoopStopError("stopped", reason="error")

    assert err.reason == "error"


def test_budget_exceeded_error_when_constructed_does_default_reason_to_budget_exceeded():
    err = BudgetExceededError("over budget", hint="check costs")

    assert err.reason == "budget-exceeded"
    assert str(err) == "over budget"
    assert err.hint == "check costs"


def test_budget_exceeded_error_when_reason_overridden_does_use_given_value():
    err = BudgetExceededError("over budget", reason="error")

    assert err.reason == "error"


# ---------------------------------------------------------------------------
# CommandError
# ---------------------------------------------------------------------------


def test_command_error_when_raised_does_subclass_gymrat_error_and_share_signature():
    err = CommandError("command failed", hint="check the target")

    assert isinstance(err, GymratError)
    assert str(err) == "command failed"
    assert err.hint == "check the target"
