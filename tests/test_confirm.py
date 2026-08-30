import io
import sys

import pytest

from gymrat.confirm import confirm_action

# ---------------------------------------------------------------------------
# confirm_action
# ---------------------------------------------------------------------------


def test_confirm_action_when_called_does_write_prompt_to_stderr(
    capsys: pytest.CaptureFixture[str],
):
    confirm_action("Proceed?", io.StringIO("y\n"))

    captured = capsys.readouterr()
    assert captured.err == "Proceed? [y/N] "
    assert captured.out == ""


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        pytest.param("y\n", True, id="lowercase-y"),
        pytest.param("Y\n", True, id="uppercase-y"),
        pytest.param("n\n", False, id="lowercase-n"),
        pytest.param("N\n", False, id="uppercase-n"),
        pytest.param("\n", False, id="empty-line"),
        pytest.param("", False, id="eof"),
        pytest.param("yes\n", False, id="full-word-yes"),
        pytest.param("nope\n", False, id="arbitrary-text"),
    ],
)
def test_confirm_action_when_answer_given_does_return_true_only_for_exact_y(
    line: str,
    expected: bool,
):
    result = confirm_action("Proceed?", io.StringIO(line))

    assert result is expected


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(BrokenPipeError, id="broken-pipe"),
        pytest.param(OSError, id="closed-stream"),
    ],
)
def test_confirm_action_when_prompt_write_fails_does_return_false(
    error: type[OSError],
    monkeypatch: pytest.MonkeyPatch,
):
    class BrokenStream:
        def write(self, data: str) -> None:
            raise error

        def flush(self) -> None:
            raise error

    monkeypatch.setattr(sys, "stderr", BrokenStream())

    result = confirm_action("Proceed?", io.StringIO("y\n"))

    assert result is False
