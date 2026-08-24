"""Tests for progress-line rendering and the live ETA countdown.

These cover ``render_progress_line`` (prepare vs sample, ETA suffix vs pending
label) and the reporter's tick countdown that resets on each emit and
disappears when an emit yields no ETA.
"""

from collections.abc import Callable

import pytest

from gymrat_py.cli.progress import (
    PLAIN_STYLE,
    STYLED,
    create_progress_reporter,
    render_progress_line,
)
from gymrat_py.sampling import PrepareProgressStep, SampleProgressStep


class _Clock:
    """A hand-advanced millisecond clock shared by the tracker and countdown."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _SpyStatusLine:
    """Records what the reporter writes, standing in for the real status line."""

    def __init__(self):
        self.writes: list[str] = []
        self.warns: list[str] = []
        self.stopped = False

    def write(self, text: str) -> None:
        self.writes.append(text)

    def warn(self, message: str) -> None:
        self.warns.append(message)

    def stop(self) -> None:
        self.stopped = True


# ---------------------------------------------------------------------------
# render_progress_line
# ---------------------------------------------------------------------------


def test_render_progress_line_prepare_never_shows_an_eta():
    line = render_progress_line(PrepareProgressStep(label="A"), None, PLAIN_STYLE)

    assert line == "prepare · A"


@pytest.mark.parametrize(
    ("eta_ms", "expected"),
    [
        pytest.param(1000, "sample 1/2 · A · ~1s left", id="with-eta"),
        pytest.param(None, "sample 1/2 · A · estimating time left…", id="pending-label"),
    ],
)
def test_render_progress_line_sample_appends_eta_or_pending_label(
    eta_ms: int | None, expected: str
):
    line = render_progress_line(
        SampleProgressStep(index=1, total=2, label="A"), eta_ms, PLAIN_STYLE
    )

    assert line == expected


def test_render_progress_line_styled_wraps_fields_with_ansi():
    line = render_progress_line(SampleProgressStep(index=1, total=2, label="A"), 1000, STYLED)

    assert "\x1b[" in line
    assert "A" in line
    assert "1/2" in line


# ---------------------------------------------------------------------------
# ProgressReporter countdown
# ---------------------------------------------------------------------------


def test_progress_reporter_countdown_resets_per_emit_and_disappears_without_eta(
    monkeypatch: pytest.MonkeyPatch,
):
    spy = _SpyStatusLine()
    captured_on_tick: list[Callable[[], str]] = []

    def fake_create_status_line(
        mode: str, on_tick: Callable[[], str] | None = None
    ) -> _SpyStatusLine:
        if on_tick is not None:
            captured_on_tick.append(on_tick)
        return spy

    monkeypatch.setattr("gymrat_py.cli.progress.create_status_line", fake_create_status_line)
    clock = _Clock()
    reporter = create_progress_reporter("overwrite", 3, clock=clock)
    on_tick = captured_on_tick[0]

    reporter.report(SampleProgressStep(index=1, total=3, label="A"))
    assert spy.writes[-1] == "sample 1/3 · A · estimating time left…"

    clock.now = 6000
    reporter.report(SampleProgressStep(index=2, total=3, label="A"))
    assert spy.writes[-1] == "sample 2/3 · A · ~48s left"
    assert on_tick() == "sample 2/3 · A · ~48s left"

    clock.now = 16000
    assert on_tick() == "sample 2/3 · A · ~38s left"

    reporter.report(PrepareProgressStep(label="B"))
    assert spy.writes[-1] == "prepare · B"
    assert on_tick() == "prepare · B"


def test_progress_reporter_stop_stops_the_status_line(monkeypatch: pytest.MonkeyPatch):
    spy = _SpyStatusLine()

    def fake_create_status_line(
        mode: str, on_tick: Callable[[], str] | None = None
    ) -> _SpyStatusLine:
        return spy

    monkeypatch.setattr("gymrat_py.cli.progress.create_status_line", fake_create_status_line)
    reporter = create_progress_reporter("plain", 2)

    reporter.stop()

    assert spy.stopped is True
