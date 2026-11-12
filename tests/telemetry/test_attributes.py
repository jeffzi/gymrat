"""Tests for the attributes of a command span and the span a command record starts."""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.context import Context
from opentelemetry.trace import StatusCode

from gymrat.session.records import ITERATION_SEQ, CommandRecord
from gymrat.telemetry.command_span import (
    COMMAND_DURATION_MS,
    COMMAND_EXIT_CODE,
    COMMAND_NAME,
    COMMAND_REASON,
    command_attributes,
    start_command_span,
)
from gymrat.telemetry.provider import SESSION_ID as SESSION_ID_KEY
from tests.session.records._fixtures import SESSION_ID, command_record
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
                SESSION_ID_KEY: SESSION_ID,
                COMMAND_NAME: "measure",
                COMMAND_EXIT_CODE: 0,
                COMMAND_DURATION_MS: 1234,
                COMMAND_REASON: "budget-exceeded",
                ITERATION_SEQ: 3,
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
                SESSION_ID_KEY: SESSION_ID,
                COMMAND_NAME: "measure",
                COMMAND_EXIT_CODE: 1,
                COMMAND_DURATION_MS: 1840,
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
            record, session_id=SESSION_ID, position=2, context=Context(), start_time=1
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
