import io
import sys
from unittest.mock import MagicMock

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


def test_confirm_action_when_prompt_written_does_flush_stderr(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []

    def record_write(_data: str) -> None:
        calls.append("write")

    def record_flush() -> None:
        calls.append("flush")

    mock_stderr = MagicMock(spec=sys.stderr)
    mock_stderr.write = MagicMock(side_effect=record_write)
    mock_stderr.flush = MagicMock(side_effect=record_flush)
    monkeypatch.setattr("sys.stderr", mock_stderr)

    confirm_action("Proceed?", io.StringIO("y\n"))

    assert "write" in calls
    assert "flush" in calls
    write_idx = calls.index("write")
    flush_idx = calls.index("flush")
    assert flush_idx > write_idx, "flush must be called after write"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("y\n", True),
        ("Y\n", True),
        ("n\n", False),
        ("N\n", False),
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
