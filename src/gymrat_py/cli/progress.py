"""Progress-line rendering and the live ETA countdown.

This layer owns the progress-specific text — ``prepare`` and ``sample`` lines,
the ETA suffix, and the pending-estimate label — and delegates the terminal
mechanics to :mod:`gymrat_py.cli.status_line`. The reporter records each step
into an :class:`~gymrat_py.eta.EtaTracker` and refreshes a live countdown
between emits.
"""

from collections.abc import Callable
from dataclasses import dataclass

from gymrat_py.cli.status_line import RenderMode, create_status_line
from gymrat_py.eta import EtaTracker, format_eta
from gymrat_py.progress_events import PassStarted, PrepareStarted, ProgressEvent, default_clock
from gymrat_py.report.style import RENDER_WIDTH, markup, render_lines

# Shown after a sample step until enough gaps have been measured for an ETA.
ETA_PENDING_LABEL = "estimating time left…"


def _identity(text: str) -> str:
    return text


@dataclass(frozen=True, slots=True)
class ProgressLineStyle:
    """Per-field presentation applied by :func:`render_progress_line`.

    ``finalize`` runs over the assembled line — it is where the styled variant
    renders its markup to ANSI, and where the plain variant does nothing.
    """

    label: Callable[[str], str]
    counter: Callable[[str], str]
    eta: Callable[[str], str]
    finalize: Callable[[str], str]


PLAIN_STYLE = ProgressLineStyle(
    label=_identity, counter=_identity, eta=_identity, finalize=_identity
)


def _finalize_styled(line: str) -> str:
    # STYLED is only used in spinner mode, which is chosen only when color is
    # allowed, so forcing color here is consistent with that gate.
    return render_lines(line, color=True, width=RENDER_WIDTH)


STYLED = ProgressLineStyle(
    label=lambda text: markup(text, "cyan"),
    counter=lambda text: markup(text, "bold"),
    eta=lambda text: markup(text, "dim"),
    finalize=_finalize_styled,
)


def render_progress_line(
    event: ProgressEvent, eta_ms: float | None, style: ProgressLineStyle
) -> str:
    """Assemble a progress line from ``event``, applying ``style`` to each field.

    ``PassStarted`` gains an ETA suffix when ``eta_ms`` is known, or the pending
    label until it is; ``PrepareStarted`` carries neither. All other event types
    produce an empty string.
    """
    if isinstance(event, PrepareStarted):
        label = style.label(event.label)
        return style.finalize(f"prepare · {label}")

    if isinstance(event, PassStarted):
        eta_suffix = format_eta(eta_ms) if eta_ms is not None else ETA_PENDING_LABEL

        label = style.label(event.label)
        counter = style.counter(f"{event.round}/{event.total_rounds}")
        line = f"sample {counter} · {label}"
        line += style.eta(f" · {eta_suffix}")
        return style.finalize(line)

    return ""


class ProgressReporter:
    """Single-use progress reporter: ``stop`` must be called once, after the run."""

    def __init__(
        self, mode: RenderMode, target_count: int, *, clock: Callable[[], float] | None = None
    ) -> None:
        self._eta = EtaTracker(target_count, clock)
        self._clock = clock if clock is not None else default_clock
        self._style = STYLED if mode == "spinner" else PLAIN_STYLE
        self._current_event: ProgressEvent | None = None
        self._emit_eta_ms: float | None = None
        self._emit_time: float | None = None
        on_tick = self._render_tick if mode != "plain" else None
        self._status_line = create_status_line(mode, on_tick)

    def _render_tick(self) -> str:
        if self._current_event is None:
            return ""
        if self._emit_eta_ms is None or self._emit_time is None:
            return render_progress_line(self._current_event, None, self._style)
        remaining = max(0.0, self._emit_eta_ms - (self._clock() - self._emit_time))
        return render_progress_line(self._current_event, remaining, self._style)

    def report(self, event: ProgressEvent) -> None:
        """Record ``event``, render its line, and write it to the status line.

        Only ``PrepareStarted`` and ``PassStarted`` events update the display.
        All other event types are silently ignored.
        """
        if not isinstance(event, PrepareStarted | PassStarted):
            return

        eta_ms = self._eta.record(event)
        self._current_event = event
        self._emit_eta_ms = eta_ms
        self._emit_time = self._clock() if eta_ms is not None else None
        self._status_line.write(render_progress_line(event, eta_ms, self._style))

    def warn(self, message: str) -> None:
        """Surface a warning through the status line without disturbing the run."""
        self._status_line.warn(message)

    def stop(self) -> None:
        """Stop the reporter and clear any live countdown."""
        self._current_event = None
        self._status_line.stop()


def create_progress_reporter(
    mode: RenderMode, target_count: int, *, clock: Callable[[], float] | None = None
) -> ProgressReporter:
    """Build the reporter a run streams its progress through for ``target_count`` targets."""
    return ProgressReporter(mode, target_count, clock=clock)
