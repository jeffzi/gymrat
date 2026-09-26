"""Command-span export tests for the repository lock seam.

These cover the spans ``with_repo_lock`` emits when tracing is enabled: span
identity, attributes, status, events, trace-context parenting and linking, the
flush before return, and the delegation to ``command_span_inputs``. They share
the session-seeding helpers with ``test_lock``.
"""

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from gymrat.cli.lock import CommandTrace, with_repo_lock
from gymrat.errors import GymratError
from gymrat.session import append_record, read_records, session_jsonl_path
from gymrat.session.paths import repo_root
from tests.cli._lock_fixtures import (
    isolate_tracing_provider as _isolate_tracing_provider,  # noqa: F401 -- registers the autouse fixture
)
from tests.cli._lock_fixtures import (
    seeded_session as _seeded_session,
)
from tests.session.records._fixtures import iteration_record

# ---------------------------------------------------------------------------
# with_repo_lock — command span export (tracing enabled)
# ---------------------------------------------------------------------------


def _command_span(
    exporter: InMemorySpanExporter, name: str = "gymrat.command.measure"
) -> ReadableSpan:
    """Return the single finished span with the given name."""
    return next(s for s in exporter.get_finished_spans() if s.name == name)


async def _ok_body(trace: CommandTrace) -> str:
    """Trivial command body for tests that only inspect the exported span."""
    return "ok"


async def test_with_repo_lock_when_tracing_enabled_does_export_command_span(
    repo: str,
):
    from gymrat.telemetry.ids import trace_id_of
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    spans = exporter.get_finished_spans()
    command_spans = [s for s in spans if s.name == "gymrat.command.measure"]
    assert len(command_spans) == 1
    assert command_spans[0].context.trace_id == trace_id_of(header.session_id)  # pyrefly: ignore[missing-attribute]


async def test_with_repo_lock_when_tracing_enabled_does_use_deterministic_span_id(
    repo: str,
):
    from gymrat.telemetry.ids import span_id_of
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    records = read_records(session_jsonl_path(repo_root()))
    cmd_line = len(records)
    assert command_span.context.span_id == span_id_of(header.session_id, f"command:{cmd_line}")  # pyrefly: ignore[missing-attribute]


async def test_with_repo_lock_when_tracing_enabled_does_set_command_attributes(
    repo: str,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body, args={"samples": 5})

    command_span = _command_span(exporter)
    attrs = dict(command_span.attributes or {})
    assert attrs["gymrat.session.id"] == header.session_id
    assert attrs["gymrat.command.name"] == "measure"
    assert attrs["gymrat.command.exit_code"] == 0
    assert attrs["gymrat.command.args.samples"] == 5


async def test_with_repo_lock_when_exit_zero_does_set_span_status_ok(
    repo: str,
):
    from opentelemetry.trace import StatusCode

    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert command_span.status.status_code == StatusCode.OK


async def test_with_repo_lock_when_exit_two_does_set_span_status_error_with_reason(
    repo: str,
):
    from opentelemetry.trace import StatusCode

    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:

        async def body(trace: CommandTrace) -> str:
            msg = "boom"
            raise GymratError(msg, reason="no-session")

        with pytest.raises(GymratError):
            await with_repo_lock("measure", body)

    command_span = _command_span(exporter)
    assert command_span.status.status_code == StatusCode.ERROR
    assert command_span.status.description == "no-session"


async def test_with_repo_lock_when_exit_one_does_set_span_status_unset(
    repo: str,
):
    from opentelemetry.trace import StatusCode

    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:

        async def body(trace: CommandTrace) -> str:
            trace.gate = True
            trace.reason = "gating-regression"
            return "gated"

        await with_repo_lock("compare", body)

    command_span = _command_span(exporter, "gymrat.command.compare")
    assert command_span.status.status_code == StatusCode.UNSET


async def test_with_repo_lock_when_body_appends_records_does_add_span_events(
    repo: str,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:

        async def body(trace: CommandTrace) -> str:
            jsonl_path = session_jsonl_path(repo_root())
            iteration = iteration_record()
            append_record(jsonl_path, iteration)
            return "ok"

        await with_repo_lock("iterate", body)

    command_span = _command_span(exporter, "gymrat.command.iterate")
    event_names = [e.name for e in command_span.events]
    assert "gymrat.iteration" in event_names


async def test_with_repo_lock_when_gymrat_traceparent_set_does_use_as_parent(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)
    valid_traceparent = "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01"
    monkeypatch.setenv("GYMRAT_TRACEPARENT", valid_traceparent)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert command_span.parent is not None
    assert command_span.parent.span_id == 0x1112131415161718


async def test_with_repo_lock_when_gymrat_traceparent_absent_does_parent_under_session(
    repo: str,
):
    from gymrat.telemetry.ids import span_id_of, trace_id_of
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert command_span.parent is not None
    assert command_span.context.trace_id == trace_id_of(header.session_id)  # pyrefly: ignore[missing-attribute]
    assert command_span.parent.span_id == span_id_of(header.session_id, "session")


async def test_with_repo_lock_when_gymrat_traceparent_malformed_does_parent_under_session(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from gymrat.telemetry.ids import span_id_of, trace_id_of
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)
    monkeypatch.setenv("GYMRAT_TRACEPARENT", "not-a-valid-traceparent")

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert command_span.parent is not None
    assert command_span.context.trace_id == trace_id_of(header.session_id)  # pyrefly: ignore[missing-attribute]
    assert command_span.parent.span_id == span_id_of(header.session_id, "session")


async def test_with_repo_lock_when_traceparent_valid_does_add_link(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)
    valid_traceparent = "00-0102030405060708090a0b0c0d0e0f10-1112131415161718-01"
    monkeypatch.setenv("TRACEPARENT", valid_traceparent)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert len(command_span.links) == 1
    assert command_span.links[0].context.span_id == 0x1112131415161718


async def test_with_repo_lock_when_traceparent_malformed_does_not_add_link(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)
    monkeypatch.setenv("TRACEPARENT", "not-valid")

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

    command_span = _command_span(exporter)
    assert len(command_span.links) == 0


async def test_with_repo_lock_when_tracing_enabled_does_flush_before_return(
    repo: str,
):
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body)

        spans_before_exit = exporter.get_finished_spans()

    assert any(s.name == "gymrat.command.measure" for s in spans_before_exit)


# ---------------------------------------------------------------------------
# with_repo_lock — command_span_inputs delegation
# ---------------------------------------------------------------------------


async def test_with_repo_lock_when_tracing_enabled_does_delegate_to_command_span_inputs(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
):
    from gymrat.telemetry.attributes import command_span_inputs as real_command_span_inputs
    from tests.telemetry._fixtures import memory_tracing

    header = _seeded_session(repo)

    calls: list[dict[str, object]] = []

    def spy(record: object, *, session_id: str, line_number: int) -> object:
        calls.append({
            "name": getattr(record, "name", None),
            "session_id": session_id,
            "line_number": line_number,
        })
        return real_command_span_inputs(record, session_id=session_id, line_number=line_number)  # type: ignore[arg-type]

    monkeypatch.setattr("gymrat.telemetry.attributes.command_span_inputs", spy)

    with memory_tracing(header.session_id) as exporter:
        await with_repo_lock("measure", _ok_body, args={"samples": 5})

    assert len(calls) == 1
    assert calls[0]["name"] == "measure"
    assert calls[0]["session_id"] == header.session_id

    command_span = _command_span(exporter)
    assert command_span is not None
