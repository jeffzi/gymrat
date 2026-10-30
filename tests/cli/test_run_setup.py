"""Tests for the run setup the benchmarking commands share.

These cover the render-mode resolution and the abort event a termination
signal trips.
"""

import asyncio
from collections.abc import Callable

import pytest

from gymrat.cli.run_setup import resolve_render_mode, run_with_signal_abort
from gymrat.exec import release_contained, spawn_contained
from tests._process_helpers import SLEEPER_ARGV
from tests._rich import TERMINATION_SIGNAL
from tests._streams import FakeStream

# ---------------------------------------------------------------------------
# resolve_render_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tty", "force_color", "no_color", "expected"),
    [
        pytest.param(True, None, "1", "live", id="no-color-on-a-tty-still-live"),
        pytest.param(False, "1", None, "plain", id="force-color-off-a-tty-still-plain"),
    ],
)
def test_resolve_render_mode_when_color_env_varies_does_follow_the_tty_alone(
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
# run_with_signal_abort
# ---------------------------------------------------------------------------


async def test_run_with_signal_abort_when_termination_signal_received_does_kill_live_groups_setting_abort(
    raise_signal: Callable[[int], int],
):
    abort_after_cleanup: list[bool] = []
    children: list[asyncio.subprocess.Process] = []

    async def execute(abort: asyncio.Event) -> None:
        children.append(await spawn_contained(asyncio.create_subprocess_exec, *SLEEPER_ARGV))
        raise_signal(TERMINATION_SIGNAL)
        abort_after_cleanup.append(abort.is_set())

    await run_with_signal_abort(execute)

    exit_code = await asyncio.wait_for(children[0].wait(), timeout=5)
    release_contained(children[0].pid)
    assert exit_code != 0
    assert abort_after_cleanup == [True]
