"""Tests for the CLI error rendering and exit routing.

These cover stdout writing, the error formatter and the exit path, plus the
import-latency guard. Run-setup tests live in ``test_run_setup.py``; flag parser
tests live in ``test_options.py``; stream-helper and stderr-console tests live in
``test_console.py``; budget report tests live in ``test_budget_report.py``; lock
and trace tests live in ``tests/test_command_run.py``.
"""

import errno
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call

import pytest
import typer

from gymrat.adapters import AdapterError
from gymrat.cli.app import app
from gymrat.cli.console import set_color_override, set_debug_mode
from gymrat.cli.exit import (
    BUGS_URL,
    exit_with_error,
    format_cli_error,
    write_and_flush,
    write_stdout,
)
from gymrat.errors import GATE_EXIT_CODE, TOOL_FAILURE_EXIT_CODE, GymratError
from tests._process_helpers import run_with_closed_reader
from tests._rich import unwrap_panel
from tests._streams import FakeStream, RaisingStream
from tests.cli._session import (
    runner,
    stub_resolve,
)

# Execs ``argv[1:]`` with the file-size limit at zero, so every write the new
# program makes to a regular file fails with EFBIG. Bytecode caching is off so
# the child never trips the limit on its own ``.pyc`` files.
_ZERO_FILE_SIZE_TRAMPOLINE = """
import os, resource, sys
resource.setrlimit(resource.RLIMIT_FSIZE, (0, resource.RLIM_INFINITY))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.execv(sys.argv[1], sys.argv[1:])
"""


def _run_with_failing_stdout(
    argv: list[str], **run_kwargs: Any
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` with every stdout write failing for a reason other than a closed pipe.

    Stdout is a regular file and the child's file-size limit is zero, so each
    write fails with EFBIG, the portable stand-in for a full disk (macOS has no
    ``/dev/full``). Stderr is captured through a pipe, which the limit does not
    cover. POSIX only: Windows has no file-size limit.

    Args:
        argv: The command to run; ``argv[0]`` must be an executable path.
        **run_kwargs: Extra ``subprocess.run`` arguments such as ``cwd`` and
            ``timeout``.

    Returns:
        The finished child, run with ``check=False`` and text-decoded stderr.
    """
    with tempfile.TemporaryFile() as stdout:
        return subprocess.run(  # noqa: S603 -- caller passes a fixed argv
            [sys.executable, "-c", _ZERO_FILE_SIZE_TRAMPOLINE, *argv],
            stdout=stdout,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            **run_kwargs,
        )


#: How each platform reports a stdout reader that has gone: ``(error, sys.platform)``.
CLOSED_STDOUT_ERRORS = [
    pytest.param(BrokenPipeError(errno.EPIPE, "Broken pipe"), "linux", id="posix-broken-pipe"),
    pytest.param(OSError(errno.EINVAL, "Invalid argument"), "win32", id="windows-einval"),
]


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------


def test_constants_when_checked_does_match_the_shipped_contract():
    assert GATE_EXIT_CODE == 1
    assert TOOL_FAILURE_EXIT_CODE == 2
    assert BUGS_URL == "https://github.com/jeffzi/gymrat/issues"


# ---------------------------------------------------------------------------
# stdout writing
# ---------------------------------------------------------------------------


def test_write_and_flush_when_called_does_write_then_flush():
    stream = Mock()

    write_and_flush(stream, "hello")

    assert stream.method_calls == [call.write("hello"), call.flush()]


#: Lines each flood probe prints: 16384 lines of 64 bytes overflow any pipe buffer.
_FLOOD_LINES = 1 << 14

#: Characters per flood line; with the trailing newline, each line is 64 bytes.
_FLOOD_LINE_CHARS = 63

_FLOOD_THROUGH_RUN_CLI = f"""
import typer
from gymrat.cli.exit import run_cli, write_stdout

app = typer.Typer()

@app.command()
def flood() -> None:
    async def run() -> None:
        for _ in range({_FLOOD_LINES}):
            write_stdout("x" * {_FLOOD_LINE_CHARS} + "\\n")

    run_cli(run)

app()
"""

_FLOOD_OUTSIDE_RUN_CLI = f"""
import typer
from gymrat.cli.exit import write_stdout

app = typer.Typer()

@app.command()
def flood() -> None:
    for _ in range({_FLOOD_LINES}):
        write_stdout("x" * {_FLOOD_LINE_CHARS} + "\\n")

app()
"""


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(_FLOOD_THROUGH_RUN_CLI, id="run_cli"),
        pytest.param(_FLOOD_OUTSIDE_RUN_CLI, id="bare-command"),
    ],
)
def test_write_stdout_when_reader_closed_does_exit_zero_without_stderr(probe: str):
    result = run_with_closed_reader(
        [sys.executable, "-c", probe], stream="stdout", text=True, timeout=60
    )

    assert (result.returncode, result.stderr) == (0, "")


@pytest.mark.parametrize(("error", "platform"), CLOSED_STDOUT_ERRORS)
def test_write_stdout_when_pipe_closed_does_return_without_raising(
    monkeypatch: pytest.MonkeyPatch, error: OSError, platform: str
):
    monkeypatch.setattr("sys.platform", platform)
    monkeypatch.setattr("sys.stdout", RaisingStream(error))

    write_stdout("first line\n")


@pytest.mark.usefixtures("repo")
def test_run_cli_when_body_raises_broken_pipe_does_exit_two_with_error(
    monkeypatch: pytest.MonkeyPatch,
):
    stub_resolve(monkeypatch)

    async def _explode(*_args: object, **_kwargs: object) -> None:
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    monkeypatch.setattr("gymrat.measure.measure", _explode)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert "Broken pipe" in unwrap_panel(result.stderr)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no file-size limit")
@pytest.mark.parametrize("fmt", ["text", "json"])
def test_gymrat_when_stdout_write_fails_otherwise_does_exit_two_without_shutdown_traceback(
    tmp_path: Path, fmt: str
):
    result = _run_with_failing_stdout(
        [sys.executable, "-m", "gymrat", "doctor", "--format", fmt], cwd=tmp_path, timeout=60
    )

    assert result.returncode == TOOL_FAILURE_EXIT_CODE
    assert os.strerror(errno.EFBIG) in unwrap_panel(result.stderr)
    assert "Exception ignored" not in result.stderr


# ---------------------------------------------------------------------------
# color control
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "env_var", "tty", "colored"),
    [
        pytest.param(True, "NO_COLOR", False, True, id="color-beats-no-color-off-a-tty"),
        pytest.param(False, "FORCE_COLOR", True, False, id="no-color-beats-force-color-on-a-tty"),
    ],
)
def test_format_cli_error_when_color_override_set_does_beat_the_color_env_vars(
    override: bool,
    env_var: str,
    tty: bool,
    colored: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=tty))
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv(env_var, "1")
    set_color_override(override)

    result = format_cli_error(ValueError("boom"))

    assert ("\x1b[" in result) is colored


# ---------------------------------------------------------------------------
# format_cli_error
# ---------------------------------------------------------------------------


def test_format_cli_error_when_colored_does_paint_the_error_label_red(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    output = format_cli_error(ValueError("boom"))

    assert "\x1b[31m" in output
    assert "Error" in output


def test_format_cli_error_when_adapter_error_does_keep_its_class_name_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error(AdapterError("parse failed"))

    assert "AdapterError: parse failed" in output


@pytest.mark.parametrize(
    ("debug", "has_stack"),
    [
        pytest.param(True, True, id="debug"),
        pytest.param(False, False, id="no-debug"),
    ],
)
def test_format_cli_error_when_debug_toggled_does_include_the_stack_only_under_debug(
    debug: bool, has_stack: bool
):
    message = "boom"
    try:
        raise ValueError(message)
    except ValueError as caught:
        error = caught

    output = format_cli_error(error, debug=debug)

    assert ("Traceback" in output) is has_stack


def test_format_cli_error_when_gymrat_error_carries_hint_does_append_an_unlabeled_line_without_footer(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error(GymratError("boom", hint="run gymrat doctor"))

    assert output.splitlines()[-1] == "run gymrat doctor"
    assert "Hint" not in output
    assert BUGS_URL not in output


def test_format_cli_error_when_hint_colored_does_render_inline_code_blue_on_a_dim_line(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    output = format_cli_error(GymratError("boom", hint="run `gymrat doctor` first"))

    hint_line = output.splitlines()[-1]
    assert hint_line.startswith("\x1b[2mrun ")  # cspell:disable-line
    assert "\x1b[2;34mgymrat doctor" in hint_line  # cspell:disable-line


def test_format_cli_error_when_not_gymrat_error_does_render_plain_label_message_and_bug_footer(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error(ValueError("boom"))

    assert output.splitlines() == [
        "Error: boom",
        "Run with gymrat --debug for details. If this is a bug, please report it at",
        BUGS_URL,
    ]


def test_format_cli_error_when_value_is_not_an_exception_does_still_render(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error("plain failure")

    assert "Error: plain failure" in output
    assert BUGS_URL in output


# ---------------------------------------------------------------------------
# exit_with_error
# ---------------------------------------------------------------------------


def test_exit_with_error_when_stderr_writable_does_exit_on_the_code_with_the_error_reported(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = io.StringIO()
    monkeypatch.setattr("sys.stderr", captured)

    with pytest.raises(typer.Exit) as exc:
        exit_with_error(ValueError("boom"), code=TOOL_FAILURE_EXIT_CODE)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE
    assert "Error: boom" in captured.getvalue()


def test_exit_with_error_when_stderr_write_fails_does_keep_the_exit_code(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", RaisingStream(OSError()))

    with pytest.raises(typer.Exit) as exc:
        exit_with_error(ValueError("boom"), code=TOOL_FAILURE_EXIT_CODE)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE


def test_exit_with_error_when_debug_mode_on_does_print_the_stack(monkeypatch: pytest.MonkeyPatch):
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
