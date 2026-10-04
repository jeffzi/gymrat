"""Session and run span lifecycle for supervised sessions.

:func:`setup_tracing` opens the session and run spans, and
:func:`finalize_tracing` ends them. While the run is active,
:func:`create_run_span_observer` mirrors supervisor events onto the run span as
OpenTelemetry span events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    TurnEndEvent,
    combine_observers,
)
from gymrat.telemetry.attributes import (
    CAP_NAME,
    EVENT_CAP,
    EVENT_COMPACTION,
    EVENT_FOLLOW_UP,
    EVENT_TURN_END,
    FOLLOW_UP_ACTION,
    FOLLOW_UP_REASON,
    GEN_AI_MODEL,
    GEN_AI_PROVIDER,
    RUN_COST_USD,
    RUN_DURATION_MS,
    RUN_EFFORT,
    RUN_END_REASON,
    RUN_ENDED_BY,
    RUN_HEAD_SHA,
    RUN_MAX_MINUTES,
    RUN_MAX_USD,
    RUN_SPAN,
    SESSION_BRANCH,
    SESSION_ID,
    SESSION_SPAN,
    TURN_BUDGET_EXHAUSTED,
    TURN_ORIGIN,
    TURN_SESSION_COST_USD,
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
    """Mutable tracing context held across the session run."""

    active: bool = False
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
        returned unchanged with an inactive state and no spans left open; when tracing is active
        the prompt carries a ``traceparent`` and the observer fans out to
        both the reporter and the tracing observer.
    """
    from gymrat.telemetry.provider import configure_tracing, start_span  # noqa: PLC0415

    state = TracingState()
    if not configure_tracing(session_id):
        return prompt, reporter_observer, state

    from opentelemetry.trace import set_span_in_context  # noqa: PLC0415

    from gymrat.telemetry.ids import format_traceparent  # noqa: PLC0415

    state.active = True
    state.session_span = start_span(
        SESSION_SPAN,
        span_key="session",
        attributes={
            SESSION_ID: session_id,
            SESSION_BRANCH: branch,
        },
    )

    run_attrs: dict[str, object] = {
        SESSION_ID: session_id,
        RUN_HEAD_SHA: head_sha,
        RUN_MAX_MINUTES: max_minutes,
        GEN_AI_PROVIDER: "anthropic",
    }
    if max_usd is not None:
        run_attrs[RUN_MAX_USD] = max_usd
    if effort is not None:
        run_attrs[RUN_EFFORT] = effort
    if model is not None:
        run_attrs[GEN_AI_MODEL] = model

    session_ctx = set_span_in_context(state.session_span)
    state.run_span = start_span(
        RUN_SPAN,
        span_key=f"run:{launch_at}",
        attributes=run_attrs,
        context=session_ctx,
    )

    # A disabled SDK (OTEL_SDK_DISABLED=true) still configures a provider but
    # hands out non-recording spans with no trace context to propagate.
    if not state.run_span.get_span_context().is_valid:
        state.run_span.end()
        state.session_span.end()
        return prompt, reporter_observer, TracingState()

    prompt = replace(prompt, traceparent=format_traceparent(state.run_span))
    observer = combine_observers(reporter_observer, create_run_span_observer(state.run_span))
    return prompt, observer, state


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
        run_span.set_attribute(RUN_COST_USD, result.cost_usd)
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
            _mirror(span, event)
        except Exception as exc:  # noqa: BLE001 — telemetry must never crash the session
            _log.warning("span event mirroring failed: %s", exc, exc_info=True)

    return observe


def _mirror(span: Span, event: SessionEvent) -> None:
    if isinstance(event, TurnEndEvent):
        span.add_event(
            EVENT_TURN_END,
            attributes={
                TURN_SESSION_COST_USD: event.cost_usd,
                TURN_ORIGIN: event.origin,
                TURN_BUDGET_EXHAUSTED: event.budget_exhausted,
            },
            timestamp=event.at,
        )
    elif isinstance(event, FollowUpEvent):
        attrs: dict[str, str] = {FOLLOW_UP_ACTION: event.action}
        if event.reason is not None:
            attrs[FOLLOW_UP_REASON] = event.reason
        span.add_event(EVENT_FOLLOW_UP, attributes=attrs, timestamp=event.at)
    elif isinstance(event, CapEvent):
        span.add_event(
            EVENT_CAP,
            attributes={CAP_NAME: event.cap},
            timestamp=event.at,
        )
    elif isinstance(event, CompactionEvent):
        span.add_event(EVENT_COMPACTION, timestamp=event.at)
