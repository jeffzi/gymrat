"""Session and run span lifecycle for supervised sessions.

:func:`setup_tracing` opens the session and run spans, and
:func:`finalize_tracing` ends them. Both spans are built by
:func:`start_session_span` and :func:`start_run_span`, which replay uses too,
so an exported session carries the spans live tracing emitted. While the run is
active, :func:`create_run_span_observer` mirrors supervisor events onto the run
span as OpenTelemetry span events.

All ``opentelemetry`` imports live inside the functions so importing this module
never pulls the SDK into ``sys.modules``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    TurnEndEvent,
    combine_observers,
)
from gymrat.telemetry.provider import (
    RUN_DURATION_MS,
    RUN_END_REASON,
    RUN_ENDED_BY,
    RUN_SPAN,
    SESSION_BRANCH,
    SESSION_ID,
    SESSION_SPAN,
    SESSION_SPAN_KEY,
    Attrs,
    configure_tracing,
    format_traceparent,
    run_attributes,
    run_span_key,
    start_span,
)
from gymrat.telemetry.session_span import existing_session_span

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from gymrat.supervisor.driver import SessionPrompt
    from gymrat.supervisor.events import LaunchEvent, SessionEvent, SessionObserver
    from gymrat.supervisor.supervise import SupervisionResult

RUN_COST_USD = "gymrat.run.cost_usd"

TURN_SESSION_COST_USD = "gymrat.turn.session_cost_usd"
TURN_ORIGIN = "gymrat.turn.origin"
TURN_BUDGET_EXHAUSTED = "gymrat.turn.budget_exhausted"
FOLLOW_UP_ACTION = "gymrat.follow_up.action"
FOLLOW_UP_REASON = "gymrat.follow_up.reason"
CAP_NAME = "gymrat.cap.name"

EVENT_TURN_END = "gymrat.turn_end"
EVENT_FOLLOW_UP = "gymrat.follow_up"
EVENT_CAP = "gymrat.cap"
EVENT_COMPACTION = "gymrat.compaction"


# ---------------------------------------------------------------------------
# Session and run spans
# ---------------------------------------------------------------------------


def start_session_span(session_id: str, *, branch: str, start_time: int | None = None) -> Span:
    """Start the ``gymrat.session`` span, the root every run and command span hangs under.

    Shared by live tracing and replay, so both give the session span the same
    id and attributes.

    Args:
        session_id: The session the span stands for.
        branch: The session's git branch.
        start_time: When the session started, in epoch nanoseconds, or
            ``None`` for now.

    Returns:
        The started span.
    """
    return start_span(
        SESSION_SPAN,
        span_key=SESSION_SPAN_KEY,
        attributes={SESSION_ID: session_id, SESSION_BRANCH: branch},
        start_time=start_time,
    )


def start_run_span(launch: LaunchEvent, *, parent: Span, start_time: int | None = None) -> Span:
    """Start the ``gymrat.run`` span of one supervised run, under the session span.

    Shared by live tracing and replay, so both give the run span the same id
    and launch attributes.

    Args:
        launch: The run's launch event, whose timestamp keys the span id and
            whose options become the span's attributes.
        parent: The session span the run span starts under.
        start_time: When the run started, in epoch nanoseconds, or ``None``
            for now.

    Returns:
        The started span.
    """
    from opentelemetry.trace import set_span_in_context  # noqa: PLC0415 -- optional extra

    return start_span(
        RUN_SPAN,
        span_key=run_span_key(launch.at),
        attributes=run_attributes(launch),
        context=set_span_in_context(parent),
        start_time=start_time,
    )


@dataclass(frozen=True, slots=True)
class TracingState:
    """The spans held open across the session run; both ``None`` while tracing is off."""

    session_span: Span | None = None
    run_span: Span | None = None


def setup_tracing(
    launch: LaunchEvent,
    *,
    branch: str,
    prompt: SessionPrompt,
    reporter_observer: SessionObserver,
    resumed: bool = False,
) -> tuple[SessionPrompt, SessionObserver, TracingState]:
    """Configure tracing and open the run span, and the session span on the opening launch.

    The session span is emitted once per session, by the launch that opened
    it. A resumed launch opens only its run span, parented to the session span
    through the span context its deterministic ids reconstruct.

    Args:
        launch: The run's launch event: its session id goes on both spans, its
            timestamp keys the run span, and its HEAD commit, caps, effort and
            model go on the run span.
        branch: Git branch name recorded on the session span.
        prompt: The session prompt; a ``traceparent`` is injected when tracing
            activates.
        reporter_observer: The reporter's event observer, combined with the
            tracing observer when tracing activates.
        resumed: Whether the launch resumes a session an earlier launch
            opened, in which case no session span is started.

    Returns:
        A three-tuple of ``(prompt, observer, state)``.  When no tracing
        endpoint is configured, or the spans it opens carry no valid trace
        context (as when the SDK is disabled), the prompt and observer are
        returned unchanged with an empty state and no spans left open; when
        tracing is active the prompt carries a ``traceparent`` and the observer
        fans out to both the reporter and the tracing observer.
    """
    if not configure_tracing(launch.session_id):
        return prompt, reporter_observer, TracingState()

    session_span = None
    if resumed:
        parent = existing_session_span(launch.session_id)
    else:
        session_span = start_session_span(launch.session_id, branch=branch)
        parent = session_span
    run_span = start_run_span(launch, parent=parent)

    # A disabled SDK (OTEL_SDK_DISABLED=true) still configures a provider but
    # opens no span: it hands back the parent's non-recording span, whose
    # context is invalid under a fresh session span and valid under a
    # reconstructed one.
    if run_span is parent or not run_span.get_span_context().is_valid:
        run_span.end()
        if session_span is not None:
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
    # Looked up at call time, not bound at import: callers swap the provider's
    # flush_tracing to observe when the run's spans are flushed.
    from gymrat.telemetry.provider import flush_tracing  # noqa: PLC0415 -- bound at call time

    run_span = state.run_span
    if result is not None and run_span is not None:
        run_span.set_attribute(RUN_COST_USD, result.outcome.cost_usd)
        run_span.set_attribute(RUN_ENDED_BY, result.ended_by)
        if result.end_reason is not None:
            run_span.set_attribute(RUN_END_REASON, result.end_reason)
        run_span.set_attribute(RUN_DURATION_MS, result.duration_ms)
        if result.outcome.reason == "error":
            from opentelemetry.trace import Status, StatusCode  # noqa: PLC0415 -- optional extra

            run_span.set_status(Status(StatusCode.ERROR))
    if run_span is not None:
        run_span.end()
    if state.session_span is not None:
        state.session_span.end()
    flush_tracing()


# ---------------------------------------------------------------------------
# Run-span event mirroring
# ---------------------------------------------------------------------------


def run_event(event: SessionEvent) -> tuple[str, Attrs] | None:
    """Map a supervisor event to the span event a run span mirrors it as.

    Args:
        event: The supervisor event.

    Returns:
        The span event's name and attributes, or ``None`` for an event the run
        span does not mirror.
    """
    if isinstance(event, TurnEndEvent):
        return EVENT_TURN_END, {
            TURN_SESSION_COST_USD: event.cost_usd,
            TURN_ORIGIN: event.origin,
            TURN_BUDGET_EXHAUSTED: event.budget_exhausted,
        }
    if isinstance(event, FollowUpEvent):
        attrs: Attrs = {FOLLOW_UP_ACTION: event.action}
        if event.reason is not None:
            attrs[FOLLOW_UP_REASON] = event.reason
        return EVENT_FOLLOW_UP, attrs
    if isinstance(event, CapEvent):
        return EVENT_CAP, {CAP_NAME: event.cap}
    if isinstance(event, CompactionEvent):
        return EVENT_COMPACTION, {}
    return None


def create_run_span_observer(span: Span) -> SessionObserver:
    """Mirror each supervisor event onto the run span as a span event.

    A failure to mirror an event never reaches the caller, so telemetry cannot
    end the session: it is reported the way
    :func:`~gymrat.supervisor.events.combine_observers` reports any observer
    failure, as one :class:`RuntimeWarning`.

    Args:
        span: The run span the events are added to.

    Returns:
        The observer mirroring each supervisor event onto ``span``.
    """

    def observe(event: SessionEvent) -> None:
        mirrored = run_event(event)
        if mirrored is not None:
            name, attributes = mirrored
            span.add_event(name, attributes=attributes, timestamp=event.at)

    # Do not catch failures in ``observe`` and log them: ``combine_observers`` reports them as a
    # RuntimeWarning, and with no logging handler configured a logged traceback prints over the
    # supervise dashboard.
    return combine_observers(observe)
