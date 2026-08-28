"""Tests for the stderr console factory."""

import io
import sys
from typing import override

import pytest

from gymrat_py.cli.console import stderr_console


class _FakeStderr(io.StringIO):
    """A stderr stand-in whose TTY status the test controls."""

    def __init__(self, *, tty: bool):
        super().__init__()
        self._tty = tty

    @override
    def isatty(self) -> bool:
        return self._tty


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize env vars that influence color and width decisions."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("COLUMNS", raising=False)


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
        pytest.param(True, True, {}, False, id="color-flag-defers-tty-detects-color"),
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
