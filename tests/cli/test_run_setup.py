"""Tests for the run setup the benchmarking commands share.

These cover the render-mode resolution, the progress reporter a run starts
with, and the abort event a termination signal trips.
"""

import asyncio
from collections.abc import Callable

import pytest

from gymrat.cli import run_setup
from gymrat.cli.run_setup import (
    SharedFlags,
    begin_run,
    resolve_render_mode,
    run_with_signal_abort,
)
from tests._process_helpers import fake_install
from tests._rich import track
from tests._streams import FakeStream

# ---------------------------------------------------------------------------
# resolve_render_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tty", "force_color", "no_color", "expected"),
    [
        pytest.param(False, None, None, "plain", id="non-tty-plain"),
        pytest.param(True, None, None, "live", id="tty-live"),
        pytest.param(True, None, "1", "live", id="no-color-on-a-tty-still-live"),
        pytest.param(False, "1", None, "plain", id="force-color-off-a-tty-still-plain"),
    ],
)
def test_resolve_render_mode_when_called_does_follow_the_tty_alone(
    *,
    monkeypatch: pytest.MonkeyPatch,
    color_env: Callable[[str | None, str | None], None],
    tty: bool,
    force_color: str | None,
    no_color: str | None,
    expected: str,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=tty))
    color_env(force_color, no_color)

    assert resolve_render_mode() == expected


# ---------------------------------------------------------------------------
# begin_run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tty", "live"),
    [
        pytest.param(True, True, id="tty-mounts-live-display"),
        pytest.param(False, False, id="non-tty-prints-plain"),
    ],
)
def test_begin_run_when_stderr_tty_varies_does_return_a_reporter_live_only_on_a_tty(
    monkeypatch: pytest.MonkeyPatch, tty: bool, live: bool
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=tty))

    reporter = track(begin_run(SharedFlags(bench="b", samples=7), target_count=3))

    assert (reporter.live is not None) is live


# ---------------------------------------------------------------------------
# run_with_signal_abort
# ---------------------------------------------------------------------------


async def test_run_with_signal_abort_when_cleanup_invoked_does_kill_groups_before_setting_abort(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_cleanup: list[Callable[[], None]] = []
    captured_abort: list[asyncio.Event] = []
    monkeypatch.setattr(run_setup, "install_termination_cleanup", fake_install(captured_cleanup))

    observed: dict[str, bool] = {}

    def _kill() -> None:
        observed["kill_ran"] = True
        observed["abort_set_at_kill"] = captured_abort[0].is_set()

    monkeypatch.setattr(run_setup, "kill_live_process_groups", _kill)

    async def execute(abort: asyncio.Event) -> str:
        captured_abort.append(abort)
        captured_cleanup[0]()
        observed["abort_after_cleanup"] = abort.is_set()
        return "done"

    result = await run_with_signal_abort(execute)

    assert result == "done"
    assert observed["kill_ran"] is True
    assert observed["abort_set_at_kill"] is False
    assert observed["abort_after_cleanup"] is True
