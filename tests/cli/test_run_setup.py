"""Tests for the run setup the benchmarking commands share.

These cover the shared flags, the render-mode resolution, the progress reporter
a run starts with, and the abort event a termination signal trips.
"""

import asyncio
from collections.abc import Callable

import pytest

from gymrat.cli import run_setup
from gymrat.cli.progress import ProgressReporter
from gymrat.cli.run_setup import (
    SharedFlags,
    begin_run,
    resolve_render_mode,
    run_with_signal_abort,
)
from tests._process_helpers import fake_install
from tests._streams import FakeStream


class _StubReporter:
    """A minimal double satisfying the ProgressReporter report/stop contract."""

    def report(self, event: object) -> None: ...
    def stop(self) -> None: ...


def _capturing_progress_reporter(
    captured: dict[str, object],
) -> Callable[..., _StubReporter]:
    """A ``ProgressReporter`` stub that records the constructor's mode and counts."""

    def fake_init(
        mode: str,
        console: object,
        target_count: int,
        sample_count: int | None = None,
        *,
        clock: object = None,
        command: str | None = None,
        target_labels: list[str] | None = None,
    ) -> _StubReporter:
        captured["mode"] = mode
        captured["target_count"] = target_count
        captured["sample_count"] = sample_count
        return _StubReporter()

    return fake_init


# ---------------------------------------------------------------------------
# resolve_render_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tty", "expected"),
    [
        pytest.param(False, "plain", id="non-tty-plain"),
        pytest.param(True, "live", id="tty-live"),
    ],
)
def test_resolve_render_mode_when_called_does_map_tty_to_strategy(
    tty: bool,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=tty))

    assert resolve_render_mode() == expected


def test_resolve_render_mode_when_no_color_set_does_still_use_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stderr", FakeStream(tty=True))
    monkeypatch.setenv("NO_COLOR", "1")

    assert resolve_render_mode() == "live"


# ---------------------------------------------------------------------------
# begin_run
# ---------------------------------------------------------------------------


def test_begin_run_when_tty_does_create_progress_reporter_with_live_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=True))

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "gymrat.cli.run_setup.ProgressReporter",
        _capturing_progress_reporter(captured),
    )

    flags = SharedFlags(bench="b", samples=7)

    begin_run(flags, target_count=3)

    assert captured["mode"] == "live"
    assert captured["target_count"] == 3
    assert captured["sample_count"] == 7


def test_begin_run_when_non_tty_does_create_progress_reporter_with_plain_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=False))

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "gymrat.cli.run_setup.ProgressReporter",
        _capturing_progress_reporter(captured),
    )

    flags = SharedFlags(bench="b", samples=5)

    begin_run(flags, target_count=1)

    assert captured["mode"] == "plain"


def test_begin_run_does_return_progress_reporter(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("sys.stderr", FakeStream(tty=False))

    result = begin_run(SharedFlags(bench="b", samples=1), target_count=1)

    assert isinstance(result, ProgressReporter)


# ---------------------------------------------------------------------------
# run_with_signal_abort
# ---------------------------------------------------------------------------


async def test_run_with_signal_abort_when_cleanup_invoked_kills_groups_before_setting_abort(
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


# ---------------------------------------------------------------------------
# flag dataclasses
# ---------------------------------------------------------------------------


def test_shared_flags_when_built_does_carry_config_set_plus_defaults():
    flags = SharedFlags(bench="my-bench", samples=5)

    assert flags.bench == "my-bench"
    assert flags.samples == 5
    assert flags.color is None
    assert flags.format == "text"
