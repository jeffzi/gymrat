"""Tests for the CLI error rendering and exit routing.

These cover stdout writing, the error formatter and the exit path. Run-setup
tests live in ``test_run_setup.py``; flag parser tests live in ``test_options.py``;
stream-helper tests live in ``tests/utils/test_utils.py``; stderr-console tests
live in ``test_console.py``; budget report
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
from typing import Any

import pytest
import typer

from gymrat.adapters.contract import AdapterError
from gymrat.cli.exit import (
    BUGS_URL,
    exit_with_error,
    format_cli_error,
    run_cli,
)
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from tests._process_helpers import run_with_closed_reader
from tests._rich import unwrap_panel
from tests._streams import RaisingStream

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


#: Lines each flood probe prints: 16384 lines of 64 bytes overflow any pipe buffer.
_FLOOD_LINES = 1 << 14

#: Characters per flood line; with the trailing newline, each line is 64 bytes.
_FLOOD_LINE_CHARS = 63

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


def test_write_stdout_when_reader_closed_does_exit_zero_without_stderr():
    result = run_with_closed_reader(
        [sys.executable, "-c", _FLOOD_OUTSIDE_RUN_CLI], stream="stdout", text=True, timeout=60
    )

    assert (result.returncode, result.stderr) == (0, "")


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
def test_gymrat_when_stdout_write_fails_otherwise_does_exit_two_without_shutdown_traceback(
    tmp_path: Path,
):
    result = _run_with_failing_stdout(
        [sys.executable, "-m", "gymrat", "doctor", "--format", "text"], cwd=tmp_path, timeout=60
    )

    assert result.returncode == TOOL_FAILURE_EXIT_CODE
    assert os.strerror(errno.EFBIG) in unwrap_panel(result.stderr)


# ---------------------------------------------------------------------------
# format_cli_error
# ---------------------------------------------------------------------------


def test_format_cli_error_when_adapter_error_does_keep_its_class_name_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error(AdapterError("parse failed"))

    assert "AdapterError: parse failed" in output


def _raised(message: str) -> ValueError:
    """A ``ValueError`` that was raised and caught, so it carries a stack."""
    try:
        raise ValueError(message)
    except ValueError as caught:
        return caught


@pytest.mark.parametrize(
    ("error", "debug", "has_stack"),
    [
        pytest.param(_raised("boom"), True, True, id="debug"),
        pytest.param(_raised("boom"), False, False, id="no-debug"),
        pytest.param("boom", True, False, id="debug-and-not-an-exception"),
    ],
)
def test_format_cli_error_when_debug_toggled_does_include_the_stack_only_under_debug(
    error: object, debug: bool, has_stack: bool
):
    output = format_cli_error(error, debug=debug)

    assert ("Traceback" in output) is has_stack


def test_format_cli_error_when_gymrat_error_carries_hint_does_append_an_unlabeled_line_without_footer(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error(GymratError("boom", hint="run gymrat doctor"))

    assert output.splitlines()[-1] == "run gymrat doctor"
    assert BUGS_URL not in output


def test_format_cli_error_when_not_gymrat_error_does_append_the_bug_footer(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error(ValueError("boom"))

    assert BUGS_URL in output


def test_format_cli_error_when_plain_error_does_render_the_bare_message(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")

    output = format_cli_error(ValueError("boom"))

    assert output.splitlines()[0].endswith("boom")


# ---------------------------------------------------------------------------
# exit_with_error
# ---------------------------------------------------------------------------


def test_exit_with_error_when_stderr_write_fails_does_keep_the_exit_code(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", RaisingStream(OSError()))

    with pytest.raises(typer.Exit) as exc:
        exit_with_error(ValueError("boom"), code=TOOL_FAILURE_EXIT_CODE)

    assert exc.value.exit_code == TOOL_FAILURE_EXIT_CODE
