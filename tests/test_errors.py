import subprocess

import pytest

from gymrat_py.errors import (
    CommandError,
    GymratError,
    hint_of,
    message_of,
    stderr_text_of,
)


def _error_with_stderr(message: str, stderr: str | bytes) -> Exception:
    """Build a plain exception carrying a ``stderr`` attribute for the helpers."""
    error = Exception(message)
    error.stderr = stderr  # type: ignore[attr-defined]
    return error


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


# ---------------------------------------------------------------------------
# message_of
# ---------------------------------------------------------------------------


def test_message_of_when_given_exception_does_return_its_message():
    error = ValueError("boom")

    assert message_of(error) == "boom"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("boom", "boom"), (42, "42")],
)
def test_message_of_when_given_non_exception_does_stringify_value(value: object, expected: str):
    assert message_of(value) == expected


# ---------------------------------------------------------------------------
# hint_of
# ---------------------------------------------------------------------------


def test_hint_of_when_gymrat_error_has_hint_does_return_hint():
    assert hint_of(GymratError("boom", hint="try that")) == "try that"


def test_hint_of_when_gymrat_error_has_no_hint_does_return_none():
    assert hint_of(GymratError("boom")) is None


def test_hint_of_when_plain_exception_does_return_none():
    assert hint_of(ValueError("boom")) is None


# ---------------------------------------------------------------------------
# stderr_text_of
# ---------------------------------------------------------------------------


def test_stderr_text_of_when_stderr_non_blank_does_prefer_it_over_message():
    error = subprocess.CalledProcessError(1, ["git"], stderr="fatal: bad thing\n")

    assert stderr_text_of(error) == "fatal: bad thing"


@pytest.mark.parametrize("stderr", ["", "   \n\t"])
def test_stderr_text_of_when_stderr_blank_does_fall_back_to_message(stderr: str):
    error = _error_with_stderr("real message", stderr)

    assert stderr_text_of(error) == "real message"


def test_stderr_text_of_when_stderr_absent_does_fall_back_to_message():
    assert stderr_text_of(ValueError("plain message")) == "plain message"


def test_stderr_text_of_when_stderr_is_bytes_does_decode_and_trim():
    error = subprocess.CalledProcessError(1, ["git"], stderr=b"fatal: bad thing\n")

    assert stderr_text_of(error) == "fatal: bad thing"
