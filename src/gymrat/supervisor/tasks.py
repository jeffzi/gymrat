"""Asyncio task helpers the supervisor uses for fire-and-forget work."""

import asyncio

from gymrat.supervisor.driver import DriverSession
from gymrat.warn import warn_to_stderr


def _warn_on_task_failure(finished: asyncio.Task[None], context: str) -> None:
    """Warn to stderr with ``context`` when ``finished`` raised; ignore cancellation."""
    if finished.cancelled():
        return
    error = finished.exception()
    if error is not None:
        warn_to_stderr(f"{context} failed: {error!s}")


def fire_and_report_interrupt(session: DriverSession) -> asyncio.Task[None] | None:
    """Interrupt the session, isolating any failure so grace setup continues.

    ``interrupt`` may throw synchronously or its coroutine may reject; either way
    the fallback recovery still runs, so the failure is warned, never raised.
    Returns the interrupt task so the caller can cancel it on teardown.

    Args:
        session: The driver session to interrupt.

    Returns:
        The interrupt task, or ``None`` when the interrupt could not be started.
    """
    try:
        pending = session.interrupt()
    except Exception as error:  # noqa: BLE001 - interrupt failure must not abort grace setup
        warn_to_stderr(f"session interrupt failed: {error!s}")
        return None

    task = asyncio.create_task(pending)

    def _report(finished: asyncio.Task[None]) -> None:
        _warn_on_task_failure(finished, "session interrupt")

    task.add_done_callback(_report)
    return task


def warn_unhandled(finished: asyncio.Task[None]) -> None:
    """Done-callback that surfaces exceptions from fire-and-forget tasks."""
    _warn_on_task_failure(finished, "background task")
