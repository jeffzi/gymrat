"""Session and run span lifecycle for supervised sessions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gymrat.telemetry.attributes import (
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
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from gymrat.supervisor import SessionPrompt, SupervisionResult
    from gymrat.supervisor.events import SessionObserver


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

    When no tracing endpoint is configured, the inputs pass through unchanged
    and no spans are opened.

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
        A three-tuple of ``(prompt, observer, state)``.  When tracing is
        inactive the prompt and observer are returned unchanged; when active
        the prompt carries a ``traceparent`` and the observer fans out to
        both the reporter and the tracing observer.
    """
    from gymrat.telemetry.provider import configure_tracing, start_span  # noqa: PLC0415

    state = TracingState()
    if not configure_tracing(session_id):
        return prompt, reporter_observer, state

    from opentelemetry.trace import set_span_in_context  # noqa: PLC0415

    from gymrat.supervisor.events import combine_observers  # noqa: PLC0415
    from gymrat.supervisor.tracing import create_run_span_observer  # noqa: PLC0415
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
    state.session_span.__enter__()

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
    state.run_span.__enter__()

    prompt = replace(prompt, traceparent=format_traceparent(state.run_span))
    observer = combine_observers(reporter_observer, create_run_span_observer(state.run_span))
    return prompt, observer, state


def finalize_tracing(
    state: TracingState,
    result: SupervisionResult | None,
) -> None:
    """End the run and session spans, set final attributes, and flush."""
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
        run_span.__exit__(None, None, None)
    if state.session_span is not None:
        state.session_span.__exit__(None, None, None)
    flush_tracing()
