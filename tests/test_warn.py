import pytest

from gymrat.warn import warn_to_stderr

# ---------------------------------------------------------------------------
# warn_to_stderr
# ---------------------------------------------------------------------------


def test_warn_to_stderr_when_called_does_write_message_with_newline_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    warn_to_stderr("hello")

    captured = capsys.readouterr()
    assert captured.err == "hello\n"
    assert captured.out == ""
