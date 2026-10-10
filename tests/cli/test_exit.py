"""Tests for the CLI error rendering and exit routing.

These cover stdout writing, the error formatter and the exit path. Run-setup
tests live in ``test_run_setup.py``; flag parser tests live in ``test_options.py``;
stream-helper and stderr-console tests live in ``test_console.py``; budget report
tests live in ``test_budget_report.py``; lock and trace tests live in
``tests/test_command_run.py``; import-latency tests live in
``tests/test_import_latency.py``.
"""

import errno
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, override

import pytest
import typer

from gymrat.adapters import AdapterError
from gymrat.cli.console import set_color_override, set_debug_mode
from gymrat.cli.exit import (
    BUGS_URL,
    exit_with_error,
    format_cli_error,
    run_cli,
    write_and_flush,
    write_stdout,
)
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from tests._ansi import TRAILING_SGR_RUN, sgr_codes
from tests._process_helpers import run_with_closed_reader
from tests._rich import screen_cells, unwrap_panel
from tests._streams import FakeStream, RaisingStream

# The SGR code for dim, which a pyte screen does not track.
_SGR_DIM = "2"


def _style_opened_before(line: str, text: str) -> set[str]:
    """The SGR codes of the escape run directly before ``text``'s first occurrence in ``line``."""
    before, _, _ = line.partition(text)
    run = TRAILING_SGR_RUN.search(before)
    assert run is not None
    return sgr_codes(run.group())


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


# ---------------------------------------------------------------------------
# bug-report link
# ---------------------------------------------------------------------------


def test_bugs_url_when_imported_does_point_at_the_issue_tracker():
    assert BUGS_URL == "https://github.com/jeffzi/gymrat/issues"


# ---------------------------------------------------------------------------
# stdout writing
# ---------------------------------------------------------------------------


class _FlushRecordingStream(io.StringIO):
    """A text stream that records what it held at each flush."""

    def __init__(self) -> None:
        super().__init__()
        self.flushed_with: list[str] = []

    @override
    def flush(self) -> None:
        self.flushed_with.append(self.getvalue())
        super().flush()


def test_write_and_flush_when_called_does_write_then_flush():
    stream = _FlushRecordingStream()

    write_and_flush(stream, "hello")

    assert stream.flushed_with == ["hello"]


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


def test_write_stdout_when_pipe_closed_on_stream_without_descriptor_does_return_without_raising(
    monkeypatch: pytest.MonkeyPatch,
):
    stream = RaisingStream(BrokenPipeError(errno.EPIPE, "Broken pipe"))
    monkeypatch.setattr("sys.stdout", stream)

    result = write_stdout("first line\n")

    assert result is None


def test_run_cli_when_body_raises_broken_pipe_does_exit_two_with_error(
    monkeypatch: pytest.MonkeyPatch,
):
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr)

    async def _explode() -> None:
        raise BrokenPipeError(errno.EPIPE, "Broken pipe")

    with pytest.raises(typer.Exit) as exc:
        run_cli(_explode)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE
    assert "Broken pipe" in unwrap_panel(stderr.getvalue())


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


def test_format_cli_error_when_color_override_set_does_color_despite_no_color_off_a_tty(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=False))
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("NO_COLOR", "1")
    set_color_override(override=True)

    result = format_cli_error(ValueError("boom"))

    assert "\x1b[" in result


# ---------------------------------------------------------------------------
# format_cli_error
# ---------------------------------------------------------------------------


def test_format_cli_error_when_colored_does_paint_the_error_label_red(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    output = format_cli_error(ValueError("boom"))

    label = screen_cells(output)[0][: len("Error")]
    assert [(cell.data, cell.fg) for cell in label] == [(char, "red") for char in "Error"]


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
    hint_row = screen_cells(output)[len(output.splitlines()) - 1]
    assert [(cell.data, cell.fg) for cell in hint_row[: len("run gymrat doctor first")]] == [
        *((char, "default") for char in "run "),
        *((char, "blue") for char in "gymrat doctor"),
        *((char, "default") for char in " first"),
    ]
    assert _SGR_DIM in _style_opened_before(hint_line, "run ")
    assert _SGR_DIM in _style_opened_before(hint_line, "gymrat doctor")


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ValueError("boom"), id="exception"),
        pytest.param("boom", id="not-an-exception"),
    ],
)
def test_format_cli_error_when_not_gymrat_error_does_render_plain_label_message_and_bug_footer(
    monkeypatch: pytest.MonkeyPatch, error: object
):
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error(error)

    assert output.splitlines() == [
        "Error: boom",
        "Run with gymrat --debug for details. If this is a bug, please report it at",
        BUGS_URL,
    ]


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
    set_debug_mode(enabled=True)
    message = "boom"
    try:
        raise ValueError(message)
    except ValueError as caught:
        error = caught

    with pytest.raises(typer.Exit):
        exit_with_error(error, code=TOOL_FAILURE_EXIT_CODE)

    assert "Traceback" in captured.getvalue()
