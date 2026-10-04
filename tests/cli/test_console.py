"""Tests for the CLI console state: stream helpers and the stderr console factory.

The console module owns the TTY, colour and debug state every command reads,
and sits below the shared CLI infrastructure: importing it must never pull the
shared module back in.
"""

import errno
import io
import os
import sys
from pathlib import Path
from typing import override

import pytest

from gymrat.cli.console import (
    apply_command_flags,
    is_broken_pipe,
    is_debug_mode,
    point_stream_at_devnull,
    resolve_stream_color,
    set_color_override,
    set_debug_mode,
    stderr_console,
)
from tests._imports import modules_imported_by
from tests._process_helpers import run_with_closed_reader
from tests._streams import FakeStream, RaisingStream


class _BadDescriptorStream(io.StringIO):
    """A stream whose descriptor is invalid, so redirecting it fails."""

    @override
    def fileno(self) -> int:
        return -1


def _open_descriptors() -> set[str]:
    """List the descriptors this process holds open."""
    return {entry.name for entry in Path("/dev/fd").iterdir()}


# ---------------------------------------------------------------------------
# module dependencies
# ---------------------------------------------------------------------------


def test_importing_console_does_not_import_the_error_module():
    loaded = modules_imported_by("gymrat.cli.console")

    assert "gymrat.cli.exit" not in loaded


# ---------------------------------------------------------------------------
# stream helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "platform", "expected"),
    [
        pytest.param(BrokenPipeError(), "linux", True, id="posix-broken-pipe"),
        pytest.param(OSError(errno.EINVAL, "Invalid argument"), "win32", True, id="windows-einval"),
        pytest.param(OSError(errno.EINVAL, "Invalid argument"), "linux", False, id="posix-einval"),
        pytest.param(OSError(errno.EACCES, "Access denied"), "win32", False, id="windows-eacces"),
        pytest.param(ValueError("banana"), "win32", False, id="not-an-os-error"),
    ],
)
def test_is_broken_pipe_when_called_does_recognize_each_platforms_closed_pipe_error(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, platform: str, expected: bool
):
    monkeypatch.setattr("sys.platform", platform)

    assert is_broken_pipe(error) is expected


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor semantics")
def test_point_stream_at_devnull_when_redirected_does_point_descriptor_at_devnull(tmp_path: Path):
    with (tmp_path / "out.txt").open("w") as stream:
        point_stream_at_devnull(stream)

        assert os.path.samestat(os.fstat(stream.fileno()), Path(os.devnull).stat())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor semantics")
def test_point_stream_at_devnull_when_redirected_does_close_devnull(tmp_path: Path):
    with (tmp_path / "out.txt").open("w") as stream:
        before = _open_descriptors()

        point_stream_at_devnull(stream)

        assert _open_descriptors() == before


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX descriptor semantics")
def test_point_stream_at_devnull_when_redirect_fails_does_close_devnull_and_raise():
    before = _open_descriptors()

    with pytest.raises(OSError, match="Bad file descriptor"):
        point_stream_at_devnull(_BadDescriptorStream())

    assert _open_descriptors() == before


# ---------------------------------------------------------------------------
# stderr target
# ---------------------------------------------------------------------------


def test_stderr_console_does_write_to_stderr(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=False))

    console = stderr_console()

    assert console.file is sys.stderr


# ---------------------------------------------------------------------------
# color resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("color_flag", "tty", "env", "expected_no_color"),
    [
        pytest.param(False, True, {}, True, id="no-color-flag-vetoes"),
        pytest.param(True, True, {}, False, id="color-flag-forces-color"),
        pytest.param(True, True, {"FORCE_COLOR": "0"}, False, id="color-flag-overrides-force-zero"),
        pytest.param(True, True, {"NO_COLOR": "1"}, False, id="color-flag-overrides-no-color-env"),
        pytest.param(True, False, {}, False, id="color-flag-forces-color-even-without-tty"),
        pytest.param(None, True, {}, False, id="none-flag-tty-detects-color"),
        pytest.param(None, False, {}, True, id="none-flag-no-tty-detects-no-color"),
        pytest.param(None, True, {"NO_COLOR": "1"}, True, id="no-color-env-overrides-tty"),
        pytest.param(
            None, False, {"FORCE_COLOR": "1"}, False, id="force-color-env-overrides-no-tty"
        ),
    ],
)
def test_stderr_console_resolves_color_from_flag_env_and_tty(
    color_flag: bool | None,
    tty: bool,
    env: dict[str, str],
    expected_no_color: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("sys.stderr", FakeStream(tty=tty))

    console = stderr_console(color_flag=color_flag)

    assert console.no_color is expected_no_color


@pytest.mark.parametrize(
    ("color", "env", "expected_no_color"),
    [
        pytest.param(True, "NO_COLOR", False, id="color-flag-outranks-no-color-env"),
        pytest.param(False, "FORCE_COLOR", True, id="no-color-flag-outranks-force-color-env"),
    ],
)
def test_stderr_console_when_command_color_flag_installed_does_follow_it(
    color: bool, env: str, expected_no_color: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(env, "1")
    monkeypatch.setattr("sys.stderr", FakeStream(tty=False))
    apply_command_flags(debug=False, color=color)

    console = stderr_console()

    assert console.no_color is expected_no_color


def test_apply_command_flags_when_command_gives_no_flags_does_keep_the_root_flags(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NO_COLOR", "1")
    set_color_override(True)
    set_debug_mode(True)

    apply_command_flags(debug=False, color=None)

    assert (is_debug_mode(), resolve_stream_color(None, FakeStream(tty=False))) == (True, True)


# ---------------------------------------------------------------------------
# width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("columns", "expected_width"),
    [
        pytest.param("120", 120, id="explicit-columns"),
        pytest.param("0", 0, id="zero-is-valid"),
    ],
)
def test_stderr_console_when_columns_set_does_use_env_width(
    columns: str,
    expected_width: int,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("COLUMNS", columns)
    monkeypatch.setattr("sys.stderr", FakeStream(tty=False))

    console = stderr_console(color_flag=False)

    assert console.width == expected_width


def test_stderr_console_when_columns_unset_does_use_terminal_width(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=False))

    console = stderr_console(color_flag=False)

    assert console.width > 0


@pytest.mark.parametrize(
    "columns",
    [
        pytest.param("", id="empty-string"),
        pytest.param("abc", id="non-numeric"),
        pytest.param("  ", id="whitespace-only"),
    ],
)
def test_stderr_console_when_columns_is_not_a_valid_integer_does_not_crash(
    columns: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("COLUMNS", columns)
    monkeypatch.setattr("sys.stderr", FakeStream(tty=False))

    console = stderr_console(color_flag=False)

    assert console.width > 0


# ---------------------------------------------------------------------------
# colorless SGR suppression (color_system=None, not just no_color=True)
# ---------------------------------------------------------------------------


def test_stderr_console_when_colorless_does_strip_all_sgr_including_bold(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=True))

    console = stderr_console(color_flag=False)
    with console.capture() as capture:
        console.print("probe", style="bold", end="")

    assert "\x1b[" not in capture.get()


# ---------------------------------------------------------------------------
# broken pipe on the console's own stream
# ---------------------------------------------------------------------------


def test_stderr_console_when_stderr_pipe_breaks_does_exit_one_without_output():
    probe = (
        "from gymrat.cli.console import stderr_console\n"
        "stderr_console(color_flag=False).print('probe')\n"
        "print('after the broken pipe')\n"
    )

    result = run_with_closed_reader([sys.executable, "-c", probe], stream="stderr", timeout=60)

    assert (result.returncode, result.stdout) == (1, b"")


@pytest.mark.parametrize(
    ("platform", "error"),
    [
        pytest.param("linux", BrokenPipeError(), id="posix-broken-pipe"),
        pytest.param("win32", OSError(errno.EINVAL, "Invalid argument"), id="windows-einval"),
    ],
)
def test_stderr_console_when_stream_has_no_file_descriptor_does_exit_one_on_broken_pipe(
    monkeypatch: pytest.MonkeyPatch, platform: str, error: OSError
):
    monkeypatch.setattr("sys.platform", platform)
    monkeypatch.setattr("sys.stdout", io.StringIO())
    monkeypatch.setattr("sys.stderr", RaisingStream(error))
    console = stderr_console(color_flag=False)

    with pytest.raises(SystemExit) as exc:
        console.print("probe")

    assert exc.value.code == 1
