"""Tests for the CLI console state and the stderr console factory.

The console module owns the TTY, colour and debug state every command reads,
and sits below the shared CLI infrastructure: importing it must never pull
``gymrat.cli.exit`` back in.
"""

import errno
import io
import sys

import pytest

from gymrat.cli.console import (
    apply_command_flags,
    is_debug_mode,
    resolve_stream_color,
    set_color_override,
    set_debug_mode,
    stderr_console,
)
from tests._imports import modules_imported_by
from tests._process_helpers import run_with_closed_reader
from tests._streams import FakeStream, RaisingStream

# ---------------------------------------------------------------------------
# module dependencies
# ---------------------------------------------------------------------------


def test_console_module_when_imported_does_not_import_the_exit_module():
    loaded = modules_imported_by("gymrat.cli.console")

    assert "gymrat.cli.exit" not in loaded


# ---------------------------------------------------------------------------
# color resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("color_flag", "tty", "env", "expected_no_color"),
    [
        pytest.param(False, True, {}, True, id="no-color-flag-vetoes"),
        pytest.param(True, True, {"FORCE_COLOR": "0"}, False, id="color-flag-overrides-force-zero"),
        pytest.param(True, True, {"NO_COLOR": "1"}, False, id="color-flag-overrides-no-color-env"),
        pytest.param(True, False, {}, False, id="color-flag-forces-color-even-without-tty"),
    ],
)
def test_stderr_console_when_flag_env_and_tty_vary_does_resolve_color(
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
    set_color_override(override=True)
    set_debug_mode(enabled=True)

    apply_command_flags(debug=False, color=None)

    assert (is_debug_mode(), resolve_stream_color(None, FakeStream(tty=False))) == (True, True)


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


# ---------------------------------------------------------------------------
# emoji codes stay literal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("markup", "highlight", "emoji"),
    [
        pytest.param(None, None, None, id="default-flags"),
        pytest.param(False, False, None, id="no-markup-no-highlight"),
        pytest.param(None, None, True, id="emoji-requested"),
    ],
)
def test_stderr_console_when_text_has_an_emoji_code_does_print_it_literally(
    markup: bool | None,
    highlight: bool | None,
    emoji: bool | None,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=False))
    console = stderr_console(color_flag=False)

    with console.capture() as capture:
        console.print("lat:100:p99", end="", markup=markup, highlight=highlight, emoji=emoji)

    assert capture.get() == "lat:100:p99"
