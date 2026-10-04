"""Run setup shared by the benchmarking commands.

The flags every command carries, the progress reporter a run prints through,
the render mode stderr's TTY status selects, and the abort event a termination
signal trips.
"""

import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from gymrat.cli import console
from gymrat.cli.progress import ProgressReporter
from gymrat.config import CliFlags
from gymrat.exec import kill_live_process_groups
from gymrat.report.style import is_tty
from gymrat.signals import install_termination_cleanup


@dataclass(frozen=True, slots=True)
class SharedFlags(CliFlags):
    """The flags every command carries: the config set plus how the report prints."""

    format: Literal["text", "json"] = "text"


def resolve_render_mode() -> Literal["live", "plain"]:
    """Map the stderr TTY status to the output strategy the progress reporter uses.

    A non-TTY stderr always renders plain; a TTY gets the rich-based live
    layout regardless of color — styling is handled by the console's own
    color resolution.

    Returns:
        ``"live"`` when stderr is a TTY, ``"plain"`` otherwise.
    """
    return "live" if is_tty(sys.stderr) else "plain"


def begin_run(
    flags: SharedFlags,
    target_count: int,
    *,
    command: str | None = None,
    target_labels: list[str] | None = None,
) -> ProgressReporter:
    """Build the progress reporter a run prints through, sized per the flags.

    Args:
        flags: The command's flags; ``samples`` sizes the progress display.
        target_count: How many targets the run measures.
        command: The command name the reporter labels its output with.
        target_labels: The display label of each target, in run order.

    Returns:
        A progress reporter writing to a stderr console in the render mode
        stderr's TTY status selects.
    """
    mode = resolve_render_mode()
    progress_console = console.stderr_console()
    return ProgressReporter(
        mode,
        progress_console,
        target_count,
        flags.samples,
        command=command,
        target_labels=target_labels,
    )


async def run_with_signal_abort[T](
    execute: Callable[[asyncio.Event], Awaitable[T]],
) -> T:
    """Run ``execute`` with an abort event a termination signal trips.

    ``execute`` receives an :class:`asyncio.Event` to hand the in-flight bench so
    a ``SIGINT`` / ``SIGTERM`` sets it and the current sample is abandoned rather
    than the process being torn down mid-command. The signal handler owns the
    exit itself (``128 +`` the signal number); this only wires the event and
    always removes the cleanup afterward, so a completed run leaves no handler
    behind.

    On the signal path the loop cannot resume to act on the abort before the
    process exits, so the cleanup kills any live exec-spawned group synchronously
    before setting the event; the event still drives the async abort race when
    the loop does keep running.

    Args:
        execute: An async callable that receives an abort event and returns the
            run result.

    Returns:
        The value returned by ``execute``.
    """
    abort = asyncio.Event()

    def terminate() -> None:
        kill_live_process_groups()
        abort.set()

    uninstall = install_termination_cleanup(terminate)
    try:
        return await execute(abort)
    finally:
        uninstall()
