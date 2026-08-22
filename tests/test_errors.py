import pytest

from gymrat_py.errors import CommandError, GymratError

# ---------------------------------------------------------------------------
# GymratError
# ---------------------------------------------------------------------------


def test_gymrat_error_when_only_message_given_does_set_message_and_null_hint():
    err = GymratError("something broke")

    assert isinstance(err, Exception)
    assert str(err) == "something broke"
    assert err.hint is None


def test_gymrat_error_when_hint_given_does_expose_hint_without_changing_message():
    err = GymratError("something broke", hint="try restarting")

    assert err.hint == "try restarting"
    assert str(err) == "something broke"


def test_gymrat_error_subclass_when_declared_inline_does_inherit_signature():
    class CustomError(GymratError):
        pass

    with_hint = CustomError("boom", hint="reset it")
    without_hint = CustomError("boom")

    assert without_hint.hint is None
    assert with_hint.hint == "reset it"
    assert str(with_hint) == "boom"


def test_gymrat_error_subclass_when_declared_inline_does_catch_as_gymrat_error():
    class CustomError(GymratError):
        pass

    err = CustomError("boom")

    with pytest.raises(GymratError) as caught:
        raise err
    assert str(caught.value) == "boom"


# ---------------------------------------------------------------------------
# CommandError
# ---------------------------------------------------------------------------


def test_command_error_when_raised_does_subclass_gymrat_error_and_share_signature():
    err = CommandError("command failed", hint="check the target")

    assert isinstance(err, GymratError)
    assert str(err) == "command failed"
    assert err.hint == "check the target"
