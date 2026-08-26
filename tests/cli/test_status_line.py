"""Tests for the content-agnostic status-line primitive.

These cover ``create_status_line``: plain newline lines, in-place overwrite with
clear/redraw on warn, the minimal spinner's first-write start, the periodic
tick callback, and thread-safety around ``stop``.
"""

import io
import threading
import time
from typing import override

import pytest

from gymrat_py.cli import status_line as status_line_module
from gymrat_py.cli.status_line import create_status_line
from gymrat_py.report.style import shorten_label

CLEAR_LINE = "\r\x1b[K"


class _FakeStderr(io.StringIO):
    """A stderr stand-in that reports a fixed TTY status."""

    def __init__(self, *, tty: bool = True):
        super().__init__()
        self._tty = tty

    @override
    def isatty(self) -> bool:
        return self._tty


class _SignalingStderr(_FakeStderr):
    """A TTY stderr that sets an event once a marker string is written."""

    def __init__(self, marker: str):
        super().__init__(tty=True)
        self._marker = marker
        self.wrote = threading.Event()

    @override
    def write(self, data: str) -> int:
        written = super().write(data)
        if self._marker in data:
            self.wrote.set()
        return written


# ---------------------------------------------------------------------------
# plain mode
# ---------------------------------------------------------------------------


def test_plain_status_line_writes_newline_lines_and_stop_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeStderr(tty=False)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("plain")

    line.write("one")
    line.warn("careful")
    line.stop()

    assert fake.getvalue() == "one\ncareful\n"


# ---------------------------------------------------------------------------
# overwrite mode
# ---------------------------------------------------------------------------


def test_overwrite_status_line_writes_clear_line_then_text_fitted_to_width(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("COLUMNS", "10")
    fake = _FakeStderr(tty=True)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("overwrite")

    line.write("abcdefghijklmnop")

    assert fake.getvalue() == CLEAR_LINE + shorten_label("abcdefghijklmnop", 9)


def test_overwrite_status_line_off_tty_passes_text_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeStderr(tty=False)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("overwrite")

    line.write("abcdefghijklmnop")

    assert fake.getvalue() == CLEAR_LINE + "abcdefghijklmnop"


def test_overwrite_status_line_warn_clears_prints_warning_then_redraws(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("COLUMNS", "80")
    fake = _FakeStderr(tty=True)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("overwrite")
    line.write("progress")
    fake.seek(0)
    fake.truncate(0)

    line.warn("careful")

    assert fake.getvalue() == f"{CLEAR_LINE}careful\n{CLEAR_LINE}progress"


def test_overwrite_status_line_stop_clears_the_row(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeStderr(tty=True)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("overwrite")
    line.write("progress")
    fake.seek(0)
    fake.truncate(0)

    line.stop()

    assert fake.getvalue() == CLEAR_LINE


# ---------------------------------------------------------------------------
# spinner mode
# ---------------------------------------------------------------------------


def test_spinner_status_line_starts_on_the_first_write(monkeypatch: pytest.MonkeyPatch):
    fake = _FakeStderr(tty=True)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("spinner")

    line.write("running")

    output = fake.getvalue()
    assert "running" in output
    assert output.startswith(CLEAR_LINE)


def test_spinner_status_line_warn_before_any_progress_prints_the_message_bare(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeStderr(tty=True)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("spinner")

    line.warn("heads up")

    assert fake.getvalue() == "heads up\n"


# ---------------------------------------------------------------------------
# periodic tick
# ---------------------------------------------------------------------------


def test_status_line_on_tick_fires_periodically_until_stopped(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(status_line_module, "TICK_INTERVAL_MS", 10)
    fake = _SignalingStderr("tick")
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("overwrite", lambda: "tick")

    try:
        assert fake.wrote.wait(2.0)
    finally:
        line.stop()

    assert "tick" in fake.getvalue()


# ---------------------------------------------------------------------------
# stop / tick thread safety
# ---------------------------------------------------------------------------


def test_status_line_when_tick_fires_during_stop_does_not_write_stale_tick(
    monkeypatch: pytest.MonkeyPatch,
):
    """A tick whose callback returns *after* ``stop()`` must not write.

    The tick callback blocks, giving ``stop()`` time to run while the timer
    thread is still inside the callback. When the callback returns, the timer
    thread holds stale text — a lock around ``_write_text`` and a ``_stopped``
    re-check inside it must prevent this text from reaching stderr.
    """
    tick_entered = threading.Event()
    tick_proceed = threading.Event()

    def slow_tick() -> str:
        tick_entered.set()
        tick_proceed.wait(timeout=2.0)
        return "stale-tick"

    monkeypatch.setattr(status_line_module, "TICK_INTERVAL_MS", 1)
    fake = _FakeStderr(tty=True)
    monkeypatch.setattr("sys.stderr", fake)
    line = create_status_line("overwrite", slow_tick)

    assert tick_entered.wait(2.0), "tick callback never fired"

    line.stop()

    fake.seek(0)
    fake.truncate(0)
    tick_proceed.set()
    time.sleep(0.05)

    assert "stale-tick" not in fake.getvalue(), "tick wrote stale text after stop() returned"
