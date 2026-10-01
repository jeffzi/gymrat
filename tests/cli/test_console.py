"""Tests for the stderr console factory."""

import errno
import io
import sys
from typing import override

import pytest

from gymrat.cli.console import stderr_console
from tests._process_helpers import run_with_closed_reader
from tests._streams import RaisingStream


class _FakeStderr(io.StringIO):
    """A stderr stand-in whose TTY status the test controls."""

    def __init__(self, *, tty: bool):
        super().__init__()
        self._tty = tty

    @override
    def isatty(self) -> bool:
        return self._tty


# ---------------------------------------------------------------------------
# stderr target
# ---------------------------------------------------------------------------


def test_stderr_console_does_write_to_stderr(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.stderr", _FakeStderr(tty=False))

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
    monkeypatch.setattr("sys.stderr", _FakeStderr(tty=tty))

    console = stderr_console(color_flag=color_flag)

    assert console.no_color is expected_no_color


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
    monkeypatch.setattr("sys.stderr", _FakeStderr(tty=False))

    console = stderr_console(color_flag=False)

    assert console.width == expected_width


def test_stderr_console_when_columns_unset_does_use_terminal_width(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStderr(tty=False))

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
    monkeypatch.setattr("sys.stderr", _FakeStderr(tty=False))

    console = stderr_console(color_flag=False)

    assert console.width > 0


# ---------------------------------------------------------------------------
# colorless SGR suppression (color_system=None, not just no_color=True)
# ---------------------------------------------------------------------------


def test_stderr_console_when_colorless_does_strip_all_sgr_including_bold(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", _FakeStderr(tty=True))

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
