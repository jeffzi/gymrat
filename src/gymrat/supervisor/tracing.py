"""Run-span observer: mirrors supervisor events onto an OpenTelemetry span."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.trace import Span

from gymrat.supervisor.events import (
    CapEvent,
    CompactionEvent,
    FollowUpEvent,
    SessionEvent,
    SessionObserver,
    TurnEndEvent,
)
from gymrat.telemetry.attributes import (
    CAP_NAME,
    EVENT_CAP,
    EVENT_COMPACTION,
    EVENT_FOLLOW_UP,
    EVENT_TURN_END,
    FOLLOW_UP_ACTION,
    FOLLOW_UP_REASON,
    TURN_BUDGET_EXHAUSTED,
    TURN_COST_USD,
    TURN_ORIGIN,
)

_log = logging.getLogger(__name__)


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
                TURN_COST_USD: event.cost_usd,
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
