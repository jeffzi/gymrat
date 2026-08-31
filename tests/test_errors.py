import subprocess

import pytest

from gymrat.errors import (
    CommandError,
    GymratError,
    hint_of,
    stderr_text_of,
)


def _error_with_stderr(message: str, stderr: str | bytes) -> Exception:
    """Build a plain exception carrying a ``stderr`` attribute for the helpers."""
    error = Exception(message)
    error.stderr = stderr  # type: ignore[attr-defined]
    return error


def _error_with_streams(message: str, *, stdout: str | bytes, stderr: str | bytes) -> Exception:
    """Build a plain exception carrying both stream attributes for the helpers."""
    error = Exception(message)
    error.stdout = stdout  # type: ignore[attr-defined]
    error.stderr = stderr  # type: ignore[attr-defined]
    return error


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


# ---------------------------------------------------------------------------
# CommandError
# ---------------------------------------------------------------------------


def test_command_error_when_raised_does_subclass_gymrat_error_and_share_signature():
    err = CommandError("command failed", hint="check the target")

    assert isinstance(err, GymratError)
    assert str(err) == "command failed"
    assert err.hint == "check the target"


# ---------------------------------------------------------------------------
# hint_of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        pytest.param(GymratError("boom", hint="try that"), "try that", id="gymrat-with-hint"),
        pytest.param(GymratError("boom"), None, id="gymrat-no-hint"),
        pytest.param(ValueError("boom"), None, id="plain-exception"),
    ],
)
def test_hint_of_when_called_does_return_hint_or_none(error: Exception, expected: str | None):
    assert hint_of(error) == expected


# ---------------------------------------------------------------------------
# stderr_text_of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stderr",
    [
        pytest.param("fatal: bad thing\n", id="str"),
        pytest.param(b"fatal: bad thing\n", id="bytes"),
    ],
)
def test_stderr_text_of_when_stderr_non_blank_does_prefer_it_over_message(stderr: str | bytes):
    error = subprocess.CalledProcessError(1, ["git"], stderr=stderr)

    assert stderr_text_of(error) == "fatal: bad thing"


@pytest.mark.parametrize("stderr", ["", "   \n\t"])
def test_stderr_text_of_when_stderr_blank_does_fall_back_to_message(stderr: str):
    error = _error_with_stderr("real message", stderr)

    assert stderr_text_of(error) == "real message"


def test_stderr_text_of_when_stderr_absent_does_fall_back_to_message():
    assert stderr_text_of(ValueError("plain message")) == "plain message"


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("hook rejected the commit\n", id="str"),
        pytest.param(b"hook rejected the commit\n", id="bytes"),
    ],
)
def test_stderr_text_of_when_stderr_blank_does_fall_back_to_stdout(stdout: str | bytes):
    error = subprocess.CalledProcessError(1, ["git", "commit"], output=stdout, stderr="")

    assert stderr_text_of(error) == "hook rejected the commit"


def test_stderr_text_of_when_both_streams_non_blank_does_prefer_stderr():
    error = subprocess.CalledProcessError(
        1, ["git", "commit"], output="on stdout", stderr="on stderr"
    )

    assert stderr_text_of(error) == "on stderr"


def test_stderr_text_of_when_both_streams_blank_does_fall_back_to_message():
    error = _error_with_streams("real message", stdout="  \n\t", stderr="")

    assert stderr_text_of(error) == "real message"
