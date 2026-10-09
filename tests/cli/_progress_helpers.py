"""Shared progress-event and renderer builders for the progress renderer and progress state tests."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Protocol

from gymrat.cli.iterate.progress import IterateRenderer
from gymrat.progress_events import PassFinished, PassStarted
from tests._rich import Clock, sealed_console, track

if TYPE_CHECKING:
    from typing import Literal

    from rich.console import Console

    from gymrat.progress_events import ProgressEvent


class _Reporting(Protocol):
    def report(self, event: ProgressEvent) -> None: ...


def ms_from_clock(clock: Clock[float]) -> int:
    """Return the clock's current time in milliseconds, for ``at_ms`` fields."""
    return int(clock.now * 1000)


def _pass_event[E: (PassStarted, PassFinished)](
    event_type: type[E],
    round_num: int,
    total_rounds: int,
    *,
    at_ms: int,
    target_count: int = 1,
    label: str = "bench",
    phase: Literal["measure", "confirm"] = "measure",
) -> E:
    return event_type(
        round=round_num,
        total_rounds=total_rounds,
        target_count=target_count,
        label=label,
        at_ms=at_ms,
        phase=phase,
    )


#: Build the event a bench pass emits as it starts, one target per pass unless overridden.
pass_started = partial(_pass_event, PassStarted)

#: Build the event a bench pass emits as it finishes, one target per pass unless overridden.
pass_finished = partial(_pass_event, PassFinished)


def report_full_pass(
    renderer: _Reporting,
    clock: Clock[float],
    round_num: int,
    total_rounds: int,
    *,
    duration_s: float,
    target_count: int = 1,
    label: str = "bench",
    phase: Literal["measure", "confirm"] = "measure",
) -> None:
    """Report one pass starting now and finishing ``duration_s`` later on ``clock``.

    Args:
        renderer: The renderer the two pass events are reported to.
        clock: The hand-advanced clock both events are stamped from; advanced
            by ``duration_s`` between them.
        round_num: The pass's round, counting from 1.
        total_rounds: The rounds the run makes.
        duration_s: How long the pass takes, in seconds.
        target_count: Targets sampled per pass.
        label: The target the pass samples.
        phase: Whether the pass measures or confirms.
    """
    renderer.report(
        pass_started(
            round_num,
            total_rounds,
            target_count=target_count,
            label=label,
            at_ms=ms_from_clock(clock),
            phase=phase,
        )
    )
    clock.tick(duration_s)
    renderer.report(
        pass_finished(
            round_num,
            total_rounds,
            target_count=target_count,
            label=label,
            at_ms=ms_from_clock(clock),
            phase=phase,
        )
    )


def iterate_renderer(
    mode: Literal["live", "plain"],
    *,
    console: Console | None = None,
    width: int = 80,
    height: int = 24,
    seq: int = 1,
    session_id: str = "test-session",
    sample_count: int = 5,
    metric_count: int = 3,
    primary_metric: str = "geomean",
    verbose: bool = False,
    checks_cmd: str | None = None,
    has_before_hook: bool = False,
    has_after_hook: bool = False,
) -> tuple[Console, Clock[float], IterateRenderer]:
    """Build an iterate renderer on a sealed console driven by a hand-advanced clock.

    The renderer is tracked, so the CLI tests' autouse teardown stops it.

    Args:
        mode: ``"live"`` for a rich live checklist, ``"plain"`` for milestone lines.
        console: The console to render to; a sealed ``width`` x ``height``
            console pinned to the clock when ``None``.
        width: Columns of the console built when ``console`` is ``None``.
        height: Rows of the console built when ``console`` is ``None``.
        seq: The iteration number shown.
        session_id: The session identifier shown.
        sample_count: Samples per pass.
        metric_count: Metrics the bench reports.
        primary_metric: The metric the verdict ranks by.
        verbose: Whether the renderer shows its verbose detail.
        checks_cmd: The checks command, or ``None`` when checks are off.
        has_before_hook: Whether a before hook runs.
        has_after_hook: Whether an after hook runs.

    Returns:
        The console, the clock, and the renderer.
    """
    clock = Clock(0.0)
    if console is None:
        console = sealed_console(width=width, height=height, get_time=clock)
    renderer = IterateRenderer(
        mode=mode,
        console=console,
        seq=seq,
        session_id=session_id,
        sample_count=sample_count,
        metric_count=metric_count,
        primary_metric=primary_metric,
        verbose=verbose,
        clock=clock,
        checks_cmd=checks_cmd,
        has_before_hook=has_before_hook,
        has_after_hook=has_after_hook,
    )
    return console, clock, track(renderer)
