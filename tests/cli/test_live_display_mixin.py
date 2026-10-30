"""Tests for the bookkeeping every CLI progress renderer inherits from ``LiveDisplayMixin``.

The zero-width fallback, the ``stop()`` guards and their closed-stream
suppression, and a termination signal after ``stop()`` are shared by every
renderer, so they are driven through the measure/compare ``ProgressReporter``,
which keeps the mixin's ``warn()`` and ``stop()`` unchanged.
"""

from __future__ import annotations

import errno
import sys
from io import StringIO
from typing import TYPE_CHECKING, Literal, override

import pytest

from gymrat.cli.progress import ProgressReporter, RunShape
from gymrat.progress_events import PrepareStarted
from gymrat.signals import install_termination_cleanup
from tests._rich import TERMINATION_SIGNAL, Clock, console_output, sealed_console, track

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rich.console import Console

    from gymrat.cli.live_display import ErasableLive

    ReporterFactory = Callable[[Literal["live", "plain"], Console], ProgressReporter]


@pytest.fixture
def build_reporter() -> Iterator[ReporterFactory]:
    """Build progress reporters showing their first event; stops each one, even after a failure."""
    built: list[ProgressReporter] = []

    def build(mode: Literal["live", "plain"], console: Console) -> ProgressReporter:
        reporter = track(
            ProgressReporter(
                mode=mode,
                console=console,
                shape=RunShape(target_count=1, sample_count=3, command="measure"),
                clock=Clock(0.0),
            )
        )
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


def _fail_live_stop_once(
    monkeypatch: pytest.MonkeyPatch, live: ErasableLive, error: Exception
) -> None:
    """Make ``live.stop`` raise ``error`` once, then stop the display for real."""
    real_stop = live.stop
    failures = iter([error])

    def stop_once() -> None:
        failure = next(failures, None)
        if failure is not None:
            raise failure
        real_stop()

    monkeypatch.setattr(live, "stop", stop_once)


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


@pytest.mark.usefixtures("manual_refresh")
def test_stop_when_live_does_clear_live(build_reporter: ReporterFactory):
    reporter = build_reporter("live", sealed_console())

    reporter.stop()

    assert reporter.live is None


def test_stop_when_signal_already_erased_the_display_does_write_nothing(
    build_reporter: ReporterFactory,
):
    console = sealed_console()
    reporter = build_reporter("live", console)
    _live(reporter).erase_for_exit()
    before = console_output(console)

    reporter.stop()

    assert console_output(console) == before


def test_stop_when_called_again_does_write_nothing_more(build_reporter: ReporterFactory):
    console = sealed_console()
    reporter = build_reporter("live", console)
    reporter.stop()
    after_first_stop = console_output(console)

    reporter.stop()

    assert console_output(console) == after_first_stop


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OSError("stderr closed"), id="os-error"),
        pytest.param(ValueError("I/O operation on closed file"), id="closed-stream-value-error"),
    ],
)
def test_stop_when_live_stop_raises_a_closed_stream_error_does_suppress(
    build_reporter: ReporterFactory, monkeypatch: pytest.MonkeyPatch, error: Exception
):
    reporter = build_reporter("live", sealed_console())
    _fail_live_stop_once(monkeypatch, _live(reporter), error)

    reporter.stop()


class _FailingStream(StringIO):
    """A stream whose every write fails with an I/O error, as a broken stderr does."""

    @override
    def write(self, s: str, /) -> int:
        raise OSError(errno.EIO, "Input/output error")


def _closed_stream() -> StringIO:
    """A stream already closed, so every write raises ``ValueError``."""
    stream = StringIO()
    stream.close()
    return stream


@pytest.mark.usefixtures("manual_refresh")
@pytest.mark.parametrize(
    "make_stream",
    [
        pytest.param(_FailingStream, id="os-error"),
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


def test_stop_when_live_stop_raises_an_unrelated_value_error_does_propagate(
    build_reporter: ReporterFactory, monkeypatch: pytest.MonkeyPatch
):
    reporter = build_reporter("live", sealed_console())
    _fail_live_stop_once(monkeypatch, _live(reporter), ValueError("unexpected"))

    with pytest.raises(ValueError, match="unexpected"):
        reporter.stop()


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
