"""Live-wiring, render-refresh, and tick-task tests.

Tests for the ``Live`` construction contract (``auto_refresh=False``,
``transient=True``, mounted via ``start()``), the explicit ``refresh=True`` on
every ``update`` call, ``_stop_live`` suppression scope, and the tick task
lifecycle including its error-handling path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from tests.cli.supervise._fixtures import (
    fire_launch,
    make_reporter,
)

LIVE_CLASS_PATH = "gymrat.cli.supervise.progress.Live"


async def _wait_for_call(mock: MagicMock, *, timeout_s: float = 1) -> None:
    """Await the next call to *mock*, bounded by *timeout_s* seconds."""
    called = asyncio.Event()
    prior = mock.side_effect

    def _signal(*a: object, **kw: object) -> None:
        if callable(prior):
            prior(*a, **kw)
        called.set()

    mock.side_effect = _signal
    try:
        async with asyncio.timeout(timeout_s):
            await called.wait()
    finally:
        mock.side_effect = prior


# ---------------------------------------------------------------------------
# Live color override
# ---------------------------------------------------------------------------


def test_create_reporter_when_color_false_does_build_colorless_console():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        make_reporter(mode="live", color=False)

        call_kwargs = mock_live_cls.call_args.kwargs
        console = call_kwargs.get("console")
        assert console is not None
        assert console.color_system is None


# ---------------------------------------------------------------------------
# Live construction — auto_refresh=False, transient, mounted, initial paint
# ---------------------------------------------------------------------------


def test_create_reporter_when_live_mode_does_configure_and_mount_live():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        make_reporter(mode="live")

        call_kwargs = mock_live_cls.call_args.kwargs
        assert call_kwargs.get("auto_refresh") is False
        assert call_kwargs.get("transient") is True
        mock_live.start.assert_called_once()
        mock_live.update.assert_called_once()
        _, update_kwargs = mock_live.update.call_args
        assert update_kwargs.get("refresh") is True


# ---------------------------------------------------------------------------
# render calls — explicit refresh=True on every update
# ---------------------------------------------------------------------------


def test_render_when_event_fires_in_live_mode_does_call_update_with_refresh():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        kit = make_reporter(mode="live")

        mock_live.update.reset_mock()
        fire_launch(kit.reporter.observer, 1000)

        mock_live.update.assert_called()
        _, update_kwargs = mock_live.update.call_args
        assert update_kwargs.get("refresh") is True


def test_render_when_plain_mode_does_not_create_live():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        make_reporter(mode="plain", plain_write=lambda _: None)

        mock_live_cls.assert_not_called()


# ---------------------------------------------------------------------------
# _stop_live — suppresses only OSError
# ---------------------------------------------------------------------------


def test_stop_when_live_stop_raises_os_error_does_suppress():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        mock_live.stop.side_effect = OSError("stderr closed")
        kit = make_reporter(mode="live")

        kit.reporter.stop()


def test_stop_when_live_stop_raises_non_os_error_does_propagate():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        mock_live.stop.side_effect = ValueError("unexpected")
        kit = make_reporter(mode="live")

        with pytest.raises(ValueError, match="unexpected"):
            kit.reporter.stop()


# ---------------------------------------------------------------------------
# tick task — start / stop (Behavior 4)
# ---------------------------------------------------------------------------


async def test_start_when_live_mode_does_render_tick_after_interval():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        kit = make_reporter(mode="live", refresh_ms=10)
        fire_launch(kit.reporter.observer, 1000)
        kit.clock.now = 2000

        mock_live.update.reset_mock()
        kit.reporter.start()
        await _wait_for_call(mock_live.update)

        assert mock_live.update.call_count == 1
        _, update_kwargs = mock_live.update.call_args
        assert update_kwargs.get("refresh") is True

        kit.reporter.stop()


async def test_start_when_plain_mode_does_nothing():
    writes: list[str] = []
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        kit = make_reporter(mode="plain", plain_write=writes.append, refresh_ms=10)
        fire_launch(kit.reporter.observer, 1000)
        writes_before_start = list(writes)

        kit.reporter.start()
        await asyncio.sleep(0.05)
        kit.reporter.stop()

        mock_live_cls.assert_not_called()
        assert writes == writes_before_start


async def test_stop_when_tick_running_does_cancel_tick_task():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        kit = make_reporter(mode="live", refresh_ms=10)
        fire_launch(kit.reporter.observer, 1000)

        mock_live.update.reset_mock()
        kit.reporter.start()
        await _wait_for_call(mock_live.update)
        kit.reporter.stop()
        call_count_after_stop = mock_live.update.call_count
        await asyncio.sleep(0.05)

        assert mock_live.update.call_count == call_count_after_stop


# ---------------------------------------------------------------------------
# tick error handling (Behavior 5)
# ---------------------------------------------------------------------------

RENDER_LIVE_PATH = "gymrat.cli.supervise.progress.render_live"


async def test_tick_when_render_raises_does_warn_and_end_tick():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        mock_live.console = MagicMock()
        kit = make_reporter(mode="live", refresh_ms=10)
        fire_launch(kit.reporter.observer, 1000)

        with patch(RENDER_LIVE_PATH, autospec=True, side_effect=RuntimeError("render boom")):
            kit.reporter.start()
            await _wait_for_call(mock_live.console.print)

        update_count_after_error = mock_live.update.call_count
        await asyncio.sleep(0.05)

        assert mock_live.update.call_count == update_count_after_error
        mock_live.console.print.assert_called_once()
        (warn_message,), _ = mock_live.console.print.call_args
        assert "render boom" in warn_message

        kit.reporter.stop()


async def test_tick_when_render_raises_does_allow_subsequent_event_render():
    with patch(LIVE_CLASS_PATH, autospec=True) as mock_live_cls:
        mock_live = mock_live_cls.return_value
        mock_live.console = MagicMock()
        kit = make_reporter(mode="live", refresh_ms=10)
        fire_launch(kit.reporter.observer, 1000)

        with patch(RENDER_LIVE_PATH, autospec=True, side_effect=RuntimeError("render boom")):
            kit.reporter.start()
            await _wait_for_call(mock_live.console.print)

        mock_live.update.reset_mock()
        fire_launch(kit.reporter.observer, 2000)

        assert mock_live.update.call_count >= 1

        kit.reporter.stop()
