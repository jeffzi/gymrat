"""Tests for CLI error rendering and the error-exit path.

These cover the ``format_cli_error`` structure (red label, adapter class-name
prefix, debug-only stack, hint, bug footer) and the ``exit_with_error``
guarantee that a failing stderr write never changes the exit code.
"""

import io

import pytest
import typer

from gymrat.adapters.types import AdapterError
from gymrat.cli.shared import (
    BUGS_URL,
    TOOL_FAILURE_EXIT_CODE,
    exit_with_error,
    format_cli_error,
    set_debug_mode,
)
from gymrat.errors import GymratError


@pytest.fixture(autouse=True)
def _reset_debug_mode():
    yield
    set_debug_mode(False)


# ---------------------------------------------------------------------------
# format_cli_error
# ---------------------------------------------------------------------------


def test_format_cli_error_when_colored_paints_the_error_label_red(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    output = format_cli_error(ValueError("boom"))

    assert "\x1b[31m" in output
    assert "Error" in output


def test_format_cli_error_when_no_color_renders_plain_label_and_message(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error(ValueError("boom"))

    assert "Error: boom" in output
    assert "\x1b[" not in output


def test_format_cli_error_when_adapter_error_keeps_its_class_name_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    output = format_cli_error(AdapterError("parse failed"))

    assert "AdapterError: parse failed" in output


def test_format_cli_error_includes_stack_only_under_debug():
    message = "boom"
    try:
        raise ValueError(message)
    except ValueError as error:
        with_stack = format_cli_error(error, debug=True)
        without_stack = format_cli_error(error, debug=False)

    assert "Traceback" in with_stack
    assert "Traceback" not in without_stack


def test_format_cli_error_when_gymrat_error_carries_hint_appends_unlabeled_line_without_footer(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    output = format_cli_error(GymratError("boom", hint="run gymrat doctor"))

    assert output.splitlines()[-1] == "run gymrat doctor"
    assert "Hint" not in output
    assert BUGS_URL not in output


def test_format_cli_error_when_hint_colored_does_dim_the_line_and_paint_inline_code_blue(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    output = format_cli_error(GymratError("boom", hint="run `gymrat doctor` first"))

    hint_line = output.splitlines()[-1]
    assert hint_line.startswith("\x1b[2mrun ")  # cspell:disable-line
    assert "\x1b[2;34mgymrat doctor" in hint_line  # cspell:disable-line


def test_format_cli_error_when_not_gymrat_error_appends_bug_footer():
    output = format_cli_error(ValueError("boom"))

    assert BUGS_URL in output


def test_format_cli_error_when_value_is_not_an_exception_still_renders(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)

    output = format_cli_error("plain failure")

    assert "Error: plain failure" in output
    assert BUGS_URL in output


# ---------------------------------------------------------------------------
# exit_with_error
# ---------------------------------------------------------------------------


def test_exit_with_error_writes_to_stderr_and_exits_on_the_given_code(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = io.StringIO()
    monkeypatch.setattr("sys.stderr", captured)

    with pytest.raises(typer.Exit) as exc:
        exit_with_error(ValueError("boom"), code=TOOL_FAILURE_EXIT_CODE)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE
    assert "Error: boom" in captured.getvalue()


def test_exit_with_error_when_stderr_write_fails_keeps_the_exit_code(
    monkeypatch: pytest.MonkeyPatch,
):
    class BrokenStderr:
        def write(self, _data: str) -> int:
            raise OSError

        def flush(self) -> None:
            raise OSError

        def isatty(self) -> bool:
            return False

    monkeypatch.setattr("sys.stderr", BrokenStderr())

    with pytest.raises(typer.Exit) as exc:
        exit_with_error(ValueError("boom"), code=TOOL_FAILURE_EXIT_CODE)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE


def test_exit_with_error_honors_debug_mode_for_the_stack(monkeypatch: pytest.MonkeyPatch):
    captured = io.StringIO()
    monkeypatch.setattr("sys.stderr", captured)
    set_debug_mode(True)
    message = "boom"

    try:
        raise ValueError(message)
    except ValueError as error:
        with pytest.raises(typer.Exit):
            exit_with_error(error, code=TOOL_FAILURE_EXIT_CODE)

    assert "Traceback" in captured.getvalue()
