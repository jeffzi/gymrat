"""Hand-advanced clocks for the tests that pin elapsed time without waiting it out."""

import asyncio
import contextlib
import dataclasses
import time
import types
from collections.abc import Callable, Generator
from unittest.mock import create_autospec

import pytest

from gymrat.clock import monotonic_ms
from tests._rich import Clock


def install_monotonic_clock(
    monkeypatch: pytest.MonkeyPatch, start_ms: float = 1_000.0
) -> Clock[float]:
    """Stand a clock still in place of ``gymrat.clock.monotonic_ms``; it moves only on ``tick``.

    Args:
        monkeypatch: The fixture the clock is patched through.
        start_ms: The reading the clock starts at, in milliseconds.

    Returns:
        The installed clock, for the test to advance.
    """
    clock = Clock(start_ms)
    monkeypatch.setattr(
        "gymrat.clock.monotonic_ms", create_autospec(monotonic_ms, side_effect=clock)
    )
    return clock


@contextlib.contextmanager
def within_hang_guard(limit_s: float) -> Generator[None]:
    """Fail the block when it took ``limit_s`` real seconds or more.

    A wait on a fake clock costs no real time, so one that ran this long slept
    on real time instead of through the clock.

    Args:
        limit_s: Real seconds the block may take.

    Yields:
        Nothing: the block runs under the guard.

    Raises:
        AssertionError: The block took ``limit_s`` real seconds or more.
    """
    started = time.monotonic()
    yield
    assert time.monotonic() - started < limit_s, "the fake-clock wait took real time"


def _no_sleep_hook() -> None:
    pass


# Clock reads a wait may make between two sleeps. A poll loop reads the clock
# once or twice per sleep, so a loop that reads it this often has slept on real
# time between reads and would otherwise spin until the suite's own timeout.
_READS_PER_SLEEP_LIMIT = 100


@dataclasses.dataclass
class WaitClock:
    """Stand-in for the process-group wait clock, advancing only when a wait sleeps.

    Every sleep, blocking or async, is recorded in ``sleeps`` and moves ``now``
    on by its length, then runs ``on_sleep``, so a test can change the world a
    wait polls at a chosen sleep. A wait that reads the clock over and over
    without sleeping through it fails at once instead of spinning.
    """

    now: float = 0.0
    sleeps: list[float] = dataclasses.field(default_factory=list)
    on_sleep: Callable[[], None] = _no_sleep_hook
    _reads_since_sleep: int = dataclasses.field(default=0, init=False, repr=False)

    def monotonic(self) -> float:
        """Return the fake time: the sum of every sleep so far.

        Returns:
            The fake time, in seconds.

        Raises:
            AssertionError: The clock was read too often since the last sleep.
        """
        self._reads_since_sleep += 1
        if self._reads_since_sleep > _READS_PER_SLEEP_LIMIT:
            msg = "the wait kept reading the fake clock without sleeping through it"
            raise AssertionError(msg)
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the fake time by ``seconds`` without blocking."""
        self._reads_since_sleep = 0
        self.now += seconds
        self.sleeps.append(seconds)
        self.on_sleep()

    async def async_sleep(self, seconds: float) -> None:
        """Advance the fake time by ``seconds``, then yield once so other tasks still run."""
        self.sleep(seconds)
        await asyncio.sleep(0)


def install_wait_clock(monkeypatch: pytest.MonkeyPatch, module: types.ModuleType) -> WaitClock:
    """Run ``module``'s process-group waits on a fresh :class:`WaitClock`.

    Args:
        monkeypatch: The fixture the clock is patched through.
        module: ``gymrat.process_group``, or a private copy of it.

    Returns:
        The installed clock, for the test to read.
    """
    clock = WaitClock()
    monkeypatch.setattr(module, "monotonic", clock.monotonic)
    monkeypatch.setattr(module, "sleep", clock.sleep)
    monkeypatch.setattr(module, "async_sleep", clock.async_sleep)
    return clock
