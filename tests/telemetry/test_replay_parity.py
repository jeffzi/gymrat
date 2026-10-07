"""Tests that replaying a session log yields the spans live tracing emitted for it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from gymrat.telemetry.provider import span_id_of, trace_id_of
from gymrat.telemetry.replay import replay_session
from tests.session.records._fixtures import (
    append_records,
    baseline_record,
    hook_record,
    session_record,
    write_session_log,
)
from tests.telemetry._fixtures import (
    isolate_tracing_provider as _isolate_tracing_provider,  # noqa: F401 -- registers the autouse fixture
)
from tests.telemetry._fixtures import memory_tracing, span_by_name, spans_by_prefix
from tests.telemetry._replay_logs import T0, T5, launch_event, turn_end, write_supervisor_log

if TYPE_CHECKING:
    import pytest

    from gymrat.session.records import SessionRecord


def _span_signature(span: Any) -> tuple[Any, ...]:
    parent_span_id = span.parent.span_id if span.parent else None
    event_names = tuple(sorted(ev.name for ev in span.events))
    status_name = span.status.status_code.name
    return (
        span.name,
        span.context.trace_id,
        span.context.span_id,
        parent_span_id,
        status_name,
        event_names,
        dict(span.attributes),
        tuple(link.context.span_id for link in span.links),
    )


async def test_replay_session_when_two_commands_run_live_does_match_replayed_spans(
    repo: str,
):
    from gymrat.command_run import CommandTrace, with_repo_lock
    from gymrat.session.paths import session_jsonl_path

    header = session_record()
    write_session_log(repo, header)

    jsonl_path = session_jsonl_path(repo)

    with memory_tracing(header.session_id) as live_exporter:

        async def body_measure(trace: CommandTrace) -> str:
            return "ok"

        async def body_compare(trace: CommandTrace) -> str:
            return "ok"

        await with_repo_lock("measure", body_measure)
        await with_repo_lock("compare", body_compare)

    live_spans = live_exporter.get_finished_spans()
    live_cmd_spans = spans_by_prefix(live_spans, "gymrat.command.")

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(jsonl_path, [])

    replay_spans = replay_exporter.get_finished_spans()
    replay_cmd_spans = spans_by_prefix(replay_spans, "gymrat.command.")

    assert len(live_cmd_spans) == len(replay_cmd_spans), (
        f"live={[s.name for s in live_cmd_spans]} vs replay={[s.name for s in replay_cmd_spans]}"
    )

    live_sigs = {s.name: _span_signature(s) for s in live_cmd_spans}
    replay_sigs = {s.name: _span_signature(s) for s in replay_cmd_spans}

    assert live_sigs == replay_sigs


def _event_records(span: Any) -> list[tuple[str, dict[str, Any]]]:
    return [(ev.name, dict(ev.attributes)) for ev in span.events]


async def test_replay_session_when_later_command_appends_record_live_does_match_replayed_events(
    repo: str,
):
    from gymrat.command_run import CommandTrace, with_repo_lock
    from gymrat.session.paths import session_jsonl_path

    header = session_record()
    write_session_log(repo, header)

    async def body_measure(trace: CommandTrace) -> str:
        return "ok"

    async def body_compare(trace: CommandTrace) -> str:
        append_records(repo, hook_record())
        return "ok"

    with memory_tracing(header.session_id) as live_exporter:
        await with_repo_lock("measure", body_measure)
        await with_repo_lock("compare", body_compare)

    live_spans = live_exporter.get_finished_spans()

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(session_jsonl_path(repo), [])

    replay_spans = replay_exporter.get_finished_spans()
    live_compare = span_by_name(live_spans, "gymrat.command.compare")
    replay_compare = span_by_name(replay_spans, "gymrat.command.compare")
    live_measure = span_by_name(live_spans, "gymrat.command.measure")
    replay_measure = span_by_name(replay_spans, "gymrat.command.measure")
    assert "gymrat.hook" in [ev.name for ev in live_compare.events]
    assert _event_records(replay_compare) == _event_records(live_compare)
    assert _event_records(replay_measure) == _event_records(live_measure)


def _replace_session_log(previous_session_id: str, fresh: SessionRecord) -> None:
    """Archive the open session log and write a fresh one holding ``fresh`` and a baseline."""
    from gymrat.session.paths import archived_session_path, repo_root, session_jsonl_path

    root = repo_root()
    archive = Path(archived_session_path(root, previous_session_id))
    archive.parent.mkdir(parents=True, exist_ok=True)
    Path(session_jsonl_path(root)).rename(archive)
    write_session_log(root, fresh, (baseline_record(),))


async def test_replay_session_when_command_opens_session_with_baseline_live_does_match_replayed_events(
    repo: str,
):
    from gymrat.command_run import CommandTrace, with_repo_lock
    from gymrat.session.paths import session_jsonl_path

    previous = session_record()
    write_session_log(repo, previous)
    fresh = session_record(session_id="20260809-090000-b7e4")

    async def body_supervise(trace: CommandTrace) -> str:
        _replace_session_log(previous.session_id, fresh)
        return "ok"

    with memory_tracing(fresh.session_id) as live_exporter:
        await with_repo_lock("supervise", body_supervise)

    live_cmd = span_by_name(live_exporter.get_finished_spans(), "gymrat.command.supervise")

    with memory_tracing(fresh.session_id) as replay_exporter:
        replay_session(session_jsonl_path(repo), [])

    replay_cmd = span_by_name(replay_exporter.get_finished_spans(), "gymrat.command.supervise")
    live_events = [ev.name for ev in live_cmd.events]
    assert "gymrat.baseline" in live_events
    assert [ev.name for ev in replay_cmd.events] == live_events


async def test_replay_session_when_gymrat_traceparent_set_live_does_match_replayed_spans(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
):
    from gymrat.command_run import CommandTrace, with_repo_lock
    from gymrat.session.paths import session_jsonl_path

    header = session_record()
    write_session_log(repo, header)

    # Build a GYMRAT_TRACEPARENT from deterministic IDs so live and replay agree
    run_trace = trace_id_of(header.session_id)
    run_span = span_id_of(header.session_id, f"run:{T0}")
    run_traceparent = f"00-{run_trace:032x}-{run_span:016x}-01"
    monkeypatch.setenv("GYMRAT_TRACEPARENT", run_traceparent)

    jsonl_path = session_jsonl_path(repo)

    # Write a supervisor log that covers the command's time
    sup_dir = tmp_path_factory.mktemp("parity-sup")
    sup_log = str(sup_dir / "run.jsonl")
    write_supervisor_log(
        sup_log,
        [launch_event(session_id=header.session_id, at=T0), turn_end(at=T5)],
    )

    with memory_tracing(header.session_id) as live_exporter:

        async def body(trace: CommandTrace) -> str:
            return "ok"

        await with_repo_lock("measure", body)

    live_spans = live_exporter.get_finished_spans()
    live_cmd = next(s for s in live_spans if s.name.startswith("gymrat.command."))

    with memory_tracing(header.session_id) as replay_exporter:
        replay_session(jsonl_path, [sup_log])

    replay_spans = replay_exporter.get_finished_spans()
    replay_cmd = next(s for s in replay_spans if s.name.startswith("gymrat.command."))

    assert live_cmd.name == replay_cmd.name
    assert live_cmd.context is not None
    assert replay_cmd.context is not None
    assert live_cmd.context.trace_id == replay_cmd.context.trace_id
    assert live_cmd.context.span_id == replay_cmd.context.span_id
    assert live_cmd.parent is not None
    assert replay_cmd.parent is not None
    assert live_cmd.parent.span_id == replay_cmd.parent.span_id
