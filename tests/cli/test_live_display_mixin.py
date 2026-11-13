"""Tests for the bookkeeping every CLI progress renderer inherits from ``LiveDisplayMixin``.

The zero-width fallback, the ``stop()`` guards and their closed-stream
suppression, and a termination signal after ``stop()`` are shared by every
renderer, so they are driven through the measure/compare ``ProgressReporter``,
which keeps the mixin's ``warn()`` and ``stop()`` unchanged.
"""

from __future__ import annotations

import errno
import sys
from functools import partial
from io import StringIO
from typing import TYPE_CHECKING, Literal, override

import pytest

from gymrat.progress_events import PrepareStarted
from gymrat.signals import install_termination_cleanup
from tests._rich import TERMINATION_SIGNAL, console_output, sealed_console
from tests._streams import RaisingStream
from tests.cli._progress_helpers import progress_reporter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rich.console import Console

    from gymrat.cli.live_display import ErasableLive
    from gymrat.cli.progress import ProgressReporter

    ReporterFactory = Callable[[Literal["live", "plain"], Console], ProgressReporter]


@pytest.fixture
def build_reporter() -> Iterator[ReporterFactory]:
    """Build progress reporters showing their first event; stops each one, even after a failure."""
    built: list[ProgressReporter] = []

    def build(mode: Literal["live", "plain"], console: Console) -> ProgressReporter:
        _console, _clock, reporter = progress_reporter(mode, console=console, command="measure")
        reporter.report(PrepareStarted(label="bench", at_ms=0))
        built.append(reporter)
        return reporter

    yield build
    for reporter in built:
        # A termination signal marks the reporter stopped (as os._exit would
        # follow in production), so stop() is then a no-op; stop the Live
        # display directly so no refresh thread or console registration leaks.
        if reporter.live is not None:
            reporter.live.stop()
        reporter.stop()


def _live(reporter: ProgressReporter) -> ErasableLive:
    """The live display ``reporter`` mounted, failing the test when it mounted none."""
    live = reporter.live
    if live is None:
        pytest.fail("no live display mounted")
    return live


# ---------------------------------------------------------------------------
# mounting
# ---------------------------------------------------------------------------


def test_live_renderer_when_console_width_zero_does_mount_no_live_display(
    build_reporter: ReporterFactory,
):
    reporter = build_reporter("live", sealed_console(width=0))

    assert reporter.live is None


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class _FailsFirstWriteStream(StringIO):
    """A stream whose first write raises ``error``; later writes succeed."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error: Exception | None = error

    @override
    def write(self, s: str, /) -> int:
        error, self._error = self._error, None
        if error is not None:
            raise error
        return super().write(s)


def test_stop_when_signal_already_erased_the_display_does_write_nothing(
    build_reporter: ReporterFactory,
):
    console = sealed_console()
    reporter = build_reporter("live", console)
    _live(reporter).erase_for_exit()
    before = console_output(console)

    reporter.stop()

    assert console_output(console) == before


@pytest.mark.usefixtures("manual_refresh")
def test_stop_when_called_again_after_a_failed_stop_does_write_nothing_more(
    build_reporter: ReporterFactory,
):
    console = sealed_console()
    reporter = build_reporter("live", console)
    console.file = _FailsFirstWriteStream(ValueError("unexpected"))
    with pytest.raises(ValueError, match="unexpected"):
        reporter.stop()
    after_first_stop = console_output(console)

    reporter.stop()

    assert console_output(console) == after_first_stop


def _closed_stream() -> StringIO:
    """A stream already closed, so every write raises ``ValueError``."""
    stream = StringIO()
    stream.close()
    return stream


@pytest.mark.usefixtures("manual_refresh")
@pytest.mark.parametrize(
    "make_stream",
    [
        pytest.param(
            partial(RaisingStream, OSError(errno.EIO, "Input/output error")), id="os-error"
        ),
        pytest.param(_closed_stream, id="closed-stream"),
    ],
)
def test_stop_when_stderr_fails_every_write_does_not_raise(
    build_reporter: ReporterFactory, make_stream: Callable[[], StringIO]
):
    console = sealed_console()
    reporter = build_reporter("live", console)
    console.file = make_stream()

    reporter.stop()

    assert reporter.live is None


# ---------------------------------------------------------------------------
# termination signal
# ---------------------------------------------------------------------------


def test_progress_reporter_when_signal_arrives_after_stop_does_write_nothing(
    build_reporter: ReporterFactory,
    monkeypatch: pytest.MonkeyPatch,
    raise_signal: Callable[[int], int],
):
    reporter = build_reporter("live", sealed_console())
    reporter.stop()
    install_termination_cleanup(lambda: None)
    buf = StringIO()
    monkeypatch.setattr(sys, "stderr", buf)

    raise_signal(TERMINATION_SIGNAL)

    assert buf.getvalue() == ""
