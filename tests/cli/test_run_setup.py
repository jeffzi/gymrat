"""Tests for the run setup the benchmarking commands share.

These cover the render-mode resolution and the abort event a termination
signal trips.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

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


@pytest.fixture
async def children() -> AsyncIterator[list[asyncio.subprocess.Process]]:
    """The processes a body spawns, reaped in-loop and released whatever the test does."""
    spawned: list[asyncio.subprocess.Process] = []
    yield spawned
    for child in spawned:
        try:
            # Reaping here lets the loop close the child's transport; a transport
            # left to the garbage collector while the loop never saw the exit
            # warns that the process is still running.
            await asyncio.wait_for(child.wait(), timeout=5)
        finally:
            release_contained(child.pid)


def _signalled_body(
    raise_signal: Callable[[int], int],
    children: list[asyncio.subprocess.Process],
    abort_after_cleanup: list[bool],
) -> Callable[[asyncio.Event], Awaitable[None]]:
    """A body that spawns a sleeper, raises a termination signal, and records the abort state."""

    async def execute(abort: asyncio.Event) -> None:
        children.append(await spawn_contained(asyncio.create_subprocess_exec, *SLEEPER_ARGV))
        raise_signal(TERMINATION_SIGNAL)
        abort_after_cleanup.append(abort.is_set())

    return execute


async def test_run_with_signal_abort_when_termination_signal_received_does_kill_the_live_process_groups(
    raise_signal: Callable[[int], int],
    children: list[asyncio.subprocess.Process],
):
    execute = _signalled_body(raise_signal, children, [])

    await run_with_signal_abort(execute)

    exit_code = await asyncio.wait_for(children[0].wait(), timeout=5)
    assert exit_code != 0


async def test_run_with_signal_abort_when_termination_signal_received_does_set_the_abort_event(
    raise_signal: Callable[[int], int],
    children: list[asyncio.subprocess.Process],
):
    abort_after_cleanup: list[bool] = []
    execute = _signalled_body(raise_signal, children, abort_after_cleanup)

    await run_with_signal_abort(execute)

    assert abort_after_cleanup == [True]
