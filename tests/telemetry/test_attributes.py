"""Tests for record-to-attribute mapping and the command span a record starts."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.context import Context
from opentelemetry.trace import StatusCode

from gymrat.session.records import (
    CommandRecord,
    IterationPrimary,
    SessionLogRecord,
    record_event,
)
from gymrat.telemetry.command_span import command_attributes, start_command_span
from tests.session.records._fixtures import (
    SESSION_ID,
    baseline_record,
    command_record,
    committed_keep,
    iteration_record,
)
from tests.telemetry._fixtures import memory_tracing, span_by_name

# ---------------------------------------------------------------------------
# command_attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        pytest.param(
            command_record(
                name="measure",
                exit_code=0,
                duration_ms=1234,
                reason="budget-exceeded",
                seq=3,
                args={
                    "ref": "main",
                    "samples": 5,
                    "verbose": True,
                    "threshold": 0.05,
                    "paths": ["/a", "/b"],
                    "config": {"key": "val"},
                    "empty": None,
                },
            ),
            {
                "gymrat.session.id": SESSION_ID,
                "gymrat.command.name": "measure",
                "gymrat.command.exit_code": 0,
                "gymrat.command.duration_ms": 1234,
                "gymrat.command.reason": "budget-exceeded",
                "gymrat.iteration.seq": 3,
                "gymrat.command.args.ref": "main",
                "gymrat.command.args.samples": 5,
                "gymrat.command.args.verbose": True,
                "gymrat.command.args.threshold": 0.05,
            },
            id="every-field-scalar-args-only",
        ),
        pytest.param(
            command_record(name="measure", reason=None, seq=None, args={}),
            {
                "gymrat.session.id": SESSION_ID,
                "gymrat.command.name": "measure",
                "gymrat.command.exit_code": 1,
                "gymrat.command.duration_ms": 1840,
            },
            id="no-reason-seq-or-args",
        ),
    ],
)
def test_command_attributes_when_called_does_map_fields_and_scalar_args_only(
    record: CommandRecord, expected: dict[str, object]
):
    result = command_attributes(record, session_id=SESSION_ID)

    assert result == expected


# ---------------------------------------------------------------------------
# start_command_span
# ---------------------------------------------------------------------------


def _start_and_end_command_span(record: CommandRecord) -> Any:
    with memory_tracing(SESSION_ID) as exporter:
        start_command_span(
            record, session_id=SESSION_ID, line_number=2, context=Context(), start_time=1
        ).end()
    return span_by_name(exporter.get_finished_spans(), f"gymrat.command.{record.name}")


@pytest.mark.parametrize(
    ("exit_code", "reason", "expected"),
    [
        pytest.param(0, None, (StatusCode.OK, None), id="exit-0-ok"),
        pytest.param(2, "error", (StatusCode.ERROR, "error"), id="exit-2-error-with-its-reason"),
        pytest.param(1, "gating-block", (StatusCode.UNSET, None), id="exit-1-unset"),
    ],
)
def test_start_command_span_when_exit_code_varies_does_set_the_matching_status(
    exit_code: int, reason: str | None, expected: tuple[StatusCode, str | None]
):
    record = command_record(exit_code=exit_code, reason=reason)

    span = _start_and_end_command_span(record)

    assert (span.status.status_code, span.status.description) == expected


@pytest.mark.parametrize(
    ("traceparent", "expected_links"),
    [
        pytest.param(
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            [0xB7AD6B7169203331],
            id="given",
        ),
        pytest.param(None, [], id="absent"),
    ],
)
def test_start_command_span_when_traceparent_given_or_absent_does_link_only_a_given_one(
    traceparent: str | None, expected_links: list[int]
):
    record = command_record(traceparent=traceparent)

    span = _start_and_end_command_span(record)

    assert [link.context.span_id for link in span.links] == expected_links


# ---------------------------------------------------------------------------
# record_event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        pytest.param(
            iteration_record(
                seq=3, outcome="improved", primary=IterationPrimary(kind="geomean", delta_pct=-7.2)
            ),
            (
                "gymrat.iteration",
                {
                    "gymrat.iteration.seq": 3,
                    "gymrat.iteration.outcome": "improved",
                    "gymrat.iteration.delta_pct": -7.2,
                },
            ),
            id="iteration",
        ),
        pytest.param(
            iteration_record(primary=IterationPrimary(kind="geomean", delta_pct=None)),
            (
                "gymrat.iteration",
                {"gymrat.iteration.seq": 1, "gymrat.iteration.outcome": "improved"},
            ),
            id="iteration-without-delta",
        ),
        pytest.param(
            committed_keep(1),
            (
                "gymrat.keep",
                {
                    "gymrat.iteration.seq": 1,
                    "gymrat.keep.status": "committed",
                    "gymrat.keep.commit": "b" * 40,
                    "gymrat.keep.message": "cache the regex",
                },
            ),
            id="committed-keep",
        ),
        pytest.param(
            baseline_record(label="initial"),
            ("gymrat.baseline", {"gymrat.baseline.label": "initial"}),
            id="baseline-has-no-seq",
        ),
    ],
)
def test_record_event_when_called_does_return_the_event_name_and_its_scalar_attributes(
    record: SessionLogRecord, expected: tuple[str, dict[str, object]]
):
    result = record_event(record)

    assert result == expected
