"""Session and run span lifecycle for supervised sessions.

:func:`setup_tracing` opens the session and run spans, and
:func:`finalize_tracing` ends them. While the run is active,
:func:`create_run_span_observer` mirrors supervisor events onto the run span as
OpenTelemetry span events.

All ``opentelemetry`` imports live inside the functions so importing this module
never pulls the SDK into ``sys.modules``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gymrat.supervisor.events import combine_observers
from gymrat.telemetry.attributes import (
    RUN_COST_USD,
    RUN_DURATION_MS,
    RUN_END_REASON,
    RUN_ENDED_BY,
    RUN_SPAN,
    SESSION_BRANCH,
    SESSION_ID,
    SESSION_SPAN,
    SESSION_SPAN_KEY,
    run_attributes,
    run_event,
    run_span_key,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from gymrat.supervisor.driver import SessionPrompt
    from gymrat.supervisor.events import SessionEvent, SessionObserver
    from gymrat.supervisor.supervise import SupervisionResult

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session and run spans
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TracingState:
    """The spans held open across the session run; both ``None`` while tracing is off."""

    session_span: Span | None = None
    run_span: Span | None = None


def setup_tracing(  # noqa: PLR0913 — keyword-only tracing context from the session run
    *,
    session_id: str,
    branch: str,
    launch_at: int,
    head_sha: str,
    max_minutes: float,
    max_usd: float | None,
    effort: str | None,
    model: str | None,
    prompt: SessionPrompt,
    reporter_observer: SessionObserver,
) -> tuple[SessionPrompt, SessionObserver, TracingState]:
    """Configure tracing and open session/run spans when the endpoint is set.

    Args:
        session_id: Unique session identifier set as a span attribute.
        branch: Git branch name recorded on the session span.
        launch_at: Monotonic timestamp (ms) of the session launch, used as the
            run span key.
        head_sha: HEAD commit SHA recorded on the run span.
        max_minutes: Wall-clock cap recorded on the run span.
        max_usd: Spend cap recorded on the run span, or ``None`` when uncapped.
        effort: Agent effort level, or ``None`` when unset.
        model: Model name, or ``None`` when unset.
        prompt: The session prompt; a ``traceparent`` is injected when tracing
            activates.
        reporter_observer: The reporter's event observer, combined with the
            tracing observer when tracing activates.

    Returns:
        A three-tuple of ``(prompt, observer, state)``.  When no tracing
        endpoint is configured, or the spans it opens carry no valid trace
        context (as when the SDK is disabled), the prompt and observer are
        returned unchanged with an empty state and no spans left open; when
        tracing is active the prompt carries a ``traceparent`` and the observer
        fans out to both the reporter and the tracing observer.
    """
    from gymrat.telemetry.provider import configure_tracing, start_span  # noqa: PLC0415

    if not configure_tracing(session_id):
        return prompt, reporter_observer, TracingState()

    from opentelemetry.trace import set_span_in_context  # noqa: PLC0415

    from gymrat.telemetry.ids import format_traceparent  # noqa: PLC0415

    session_span = start_span(
        SESSION_SPAN,
        span_key=SESSION_SPAN_KEY,
        attributes={
            SESSION_ID: session_id,
            SESSION_BRANCH: branch,
        },
    )
    run_span = start_span(
        RUN_SPAN,
        span_key=run_span_key(launch_at),
        attributes=run_attributes(
            session_id=session_id,
            head_sha=head_sha,
            max_minutes=max_minutes,
            max_usd=max_usd,
            effort=effort,
            model=model,
        ),
        context=set_span_in_context(session_span),
    )

    # A disabled SDK (OTEL_SDK_DISABLED=true) still configures a provider but
    # hands out non-recording spans with no trace context to propagate.
    if not run_span.get_span_context().is_valid:
        run_span.end()
        session_span.end()
        return prompt, reporter_observer, TracingState()

    prompt = replace(prompt, traceparent=format_traceparent(run_span))
    observer = combine_observers(reporter_observer, create_run_span_observer(run_span))
    return prompt, observer, TracingState(session_span=session_span, run_span=run_span)


def finalize_tracing(
    state: TracingState,
    result: SupervisionResult | None,
) -> None:
    """End the run and session spans, set the run's outcome attributes, and flush.

    Args:
        state: The spans :func:`setup_tracing` opened; either may be ``None``
            when tracing never activated.
        result: The supervision outcome, or ``None`` when the run produced
            none, in which case the spans close without outcome attributes or
            an error status.
    """
    from gymrat.telemetry.provider import flush_tracing  # noqa: PLC0415

    run_span = state.run_span
    if result is not None and run_span is not None:
        run_span.set_attribute(RUN_COST_USD, result.outcome.cost_usd)
        run_span.set_attribute(RUN_ENDED_BY, result.ended_by)
        if result.end_reason is not None:
            run_span.set_attribute(RUN_END_REASON, result.end_reason)
        run_span.set_attribute(RUN_DURATION_MS, result.duration_ms)
        if result.outcome.reason == "error":
            from opentelemetry.trace import Status, StatusCode  # noqa: PLC0415

            run_span.set_status(Status(StatusCode.ERROR))
    if run_span is not None:
        run_span.end()
    if state.session_span is not None:
        state.session_span.end()
    flush_tracing()


# ---------------------------------------------------------------------------
# Run-span event mirroring
# ---------------------------------------------------------------------------


def create_run_span_observer(span: Span) -> SessionObserver:
    """Return a :data:`SessionObserver` that adds span events for each supervisor event."""

    def observe(event: SessionEvent) -> None:
        try:
            mirrored = run_event(event)
            if mirrored is not None:
                name, attributes = mirrored
                span.add_event(name, attributes=attributes, timestamp=event.at)
        except Exception as exc:  # noqa: BLE001 — telemetry must never crash the session
            _log.warning("span event mirroring failed: %s", exc, exc_info=True)

    return observe
