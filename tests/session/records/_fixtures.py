"""Canonical session-record builders shared by the store tests.

Each builder returns a fully-populated record with the same defaults the real
session writes, and takes keyword overrides for the fields a test cares about.
Optional fields default to ``None`` on the underlying dataclasses, so passing a
keyword of ``None`` erases the builder's default for that field.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.session.records._fixtures``.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gymrat.session.paths import baseline_worktree_dir, experiment_worktree_dir, session_jsonl_path
from gymrat.session.records import (
    BaselineRecord,
    CommandRecord,
    DiscardRecord,
    FinalizeRecord,
    HookRecord,
    IterationPrimary,
    IterationRecord,
    KeepChecks,
    KeepRecord,
    MetricVerdict,
    PairedSamples,
    SessionConfig,
    SessionLogRecord,
    SessionRecord,
    StopRecord,
)
from gymrat.session.store import SessionState, append_record, read_records
from gymrat.session.workspace import BaselineRef, Worktrees

#: The instant every fixture record in this file was written at (nanoseconds since epoch).
AT = 1_786_198_530_000_000_000

#: A commit SHA fixture records point at; not a real commit.
COMMIT = "b" * 40

#: The squash commit SHA finalize fixtures point at; distinct from COMMIT.
SQUASH_COMMIT = "c" * 40

#: The session id every fixture record belongs to.
SESSION_ID = "20260808-141530-a3f2"

#: The unterminated JSON prefix a writer killed mid-append leaves as the log's tail.
TORN_PREFIX: bytes = b'{"type":"iter'


def _overridden[T: BaseModel](default: T, overrides: dict[str, Any]) -> T:
    return default.model_copy(update=overrides) if overrides else default


def tear_final_line(path: str | Path) -> None:
    """Append ``TORN_PREFIX`` so the log ends on an unterminated final line."""
    with Path(path).open("ab") as handle:
        handle.write(TORN_PREFIX)


def worktrees_at(root: str) -> Worktrees:
    """The worktree pair a session started under ``root`` records."""
    return Worktrees(experiment=experiment_worktree_dir(root), baseline=baseline_worktree_dir(root))


def session_record(**overrides: Any) -> SessionRecord:
    """The session header a started session writes, with every field overridable.

    ``session_id`` drives the default ``branch``, so a caller overriding just
    the id still gets a matching branch; a caller after a divergent branch
    overrides both explicitly.
    """
    session_id = overrides.get("session_id", SESSION_ID)
    # pyrefly: ignore[missing-argument] -- validate_by_name accepts schema_version
    default = SessionRecord(
        type="session",
        schema_version=1,
        session_id=session_id,
        at=AT,
        baseline=BaselineRef(ref="main", sha="a" * 40),
        branch=f"gymrat/{session_id}",
        worktrees=Worktrees(
            experiment="/repo/.gymrat/worktrees/experiment",
            baseline="/repo/.gymrat/worktrees/baseline",
        ),
        config=SessionConfig(
            bench="npm run bench",
            adapter="metric-lines",
            samples=10,
            timeout_seconds=1800,
            primary="geomean",
        ),
    )
    return _overridden(default, overrides)


def baseline_record(**overrides: Any) -> BaselineRecord:
    """A one-round baseline measurement of ``main`` that timed nothing, every field overridable."""
    default = BaselineRecord(type="baseline", at=AT, label="main", samples=({"total_ms": 15200},))
    return _overridden(default, overrides)


def metric_verdict(**overrides: Any) -> MetricVerdict:
    """A metric verdict the engine produces, improved and gating unless overridden."""
    default = MetricVerdict(
        delta_pct=-7.2,
        verdict="improved",
        method="permutation",
        p=0.002,
        noise_pct=1.4,
        gating=True,
        confirmed=False,
    )
    return _overridden(default, overrides)


def iteration_record(**overrides: Any) -> IterationRecord:
    """A measured iteration numbered 1, improved unless overridden."""
    default = IterationRecord(
        type="iteration",
        seq=1,
        at=AT,
        samples=PairedSamples(
            experiment=({"total_ms": 14100},),
            baseline=({"total_ms": 15200},),
        ),
        metrics={"total_ms": metric_verdict()},
        primary=IterationPrimary(kind="geomean", delta_pct=-7.2),
        outcome="improved",
        target_reached=False,
    )
    return _overridden(default, overrides)


def make_iteration(delta_pct: float | None, outcome: str, seq: int = 1) -> IterationRecord:
    """An iteration whose only reader-visible fields are its primary delta and outcome."""
    return iteration_record(
        seq=seq,
        primary=IterationPrimary(kind="geomean", delta_pct=delta_pct),
        outcome=outcome,
    )


def committed_keep(seq: int, **overrides: Any) -> KeepRecord:
    """A keep that committed the iteration numbered ``seq``, every field overridable."""
    default = KeepRecord(
        type="keep",
        seq=seq,
        at=AT,
        status="committed",
        checks=KeepChecks(configured=True, passed=True),
        commit=COMMIT,
        message="cache the regex",
    )
    return _overridden(default, overrides)


def blocked_keep(seq: int, **overrides: Any) -> KeepRecord:
    """A keep the checks gate refused, leaving the iteration numbered ``seq`` uncommitted.

    ``reason`` defaults to ``"checks-failed"``; pass ``reason=None`` to erase it,
    or a settling reason such as ``"gating-regression"`` to override it.
    """
    default = KeepRecord(
        type="keep",
        seq=seq,
        at=AT,
        status="blocked",
        checks=KeepChecks(configured=True, passed=False),
        reason="checks-failed",
    )
    return _overridden(default, overrides)


def discard_record(seq: int) -> DiscardRecord:
    return DiscardRecord(type="discard", seq=seq, at=AT)


def hook_record(**overrides: Any) -> HookRecord:
    """The hook a ``before`` stage runs, with every field overridable."""
    default = HookRecord(
        type="hook",
        at=AT,
        stage="before",
        seq=1,
        exit_code=0,
        duration_ms=120,
        stdout_bytes=80,
        timed_out=False,
    )
    return _overridden(default, overrides)


def finalize_record(**overrides: Any) -> FinalizeRecord:
    """The record that closes a session, with every field overridable."""
    default = FinalizeRecord(
        type="finalize",
        at=AT,
        branch=f"gymrat/{SESSION_ID}-final",
        commit=SQUASH_COMMIT,
        message="squash 1 kept iteration",
    )
    return _overridden(default, overrides)


def stop_record(**overrides: Any) -> StopRecord:
    """A stop record requesting the loop halt, with every field overridable."""
    default = StopRecord(
        type="stop",
        at=AT,
        message="user requested stop",
    )
    return _overridden(default, overrides)


def command_record(**overrides: Any) -> CommandRecord:
    """A command record for a failed iterate, with every field overridable."""
    default = CommandRecord(
        type="command",
        at=AT,
        name="iterate",
        args={},
        exit_code=1,
        reason="budget-exceeded",
        duration_ms=1840,
        seq=3,
    )
    return _overridden(default, overrides)


def empty_session_state() -> SessionState:
    """The state of a session log that holds no records yet (``session=None``)."""
    return SessionState(
        session=None,
        iteration_count=0,
        last_iteration=None,
        unsettled=False,
        keep_count=0,
        discard_count=0,
        target_reached_and_kept=False,
        last_seq=0,
        last_kept_commit=None,
        ends_on_gating_block=False,
        ends_on_stop=False,
        finalized=None,
    )


def session_state(**changes: Any) -> SessionState:
    """The empty session state with the named fields overridden.

    Args:
        **changes: Field values that replace the empty state's.

    Returns:
        The empty session state with ``changes`` applied.
    """
    return replace(empty_session_state(), **changes)


def write_session_log(
    root: str,
    header: SessionRecord,
    history: tuple[SessionLogRecord, ...] = (),
) -> None:
    """Append *header* then every record in *history* to the session JSONL log."""
    append_records(root, header, *history)


def append_records(root: str, *records: SessionLogRecord) -> None:
    """Append every record, in order, to the session JSONL log under ``root``."""
    jsonl_path = session_jsonl_path(root)
    for record in records:
        append_record(jsonl_path, record)


def log_records(root: str) -> list[SessionLogRecord]:
    """Every record the session JSONL log under ``root`` currently holds."""
    return read_records(session_jsonl_path(root))


def session_header_of(root: str) -> SessionRecord:
    """The session header ``root``'s log opens with, failing when there is none."""
    records = log_records(root)
    assert records, f"expected a session header in {session_jsonl_path(root)}"
    first = records[0]
    assert isinstance(first, SessionRecord), f"expected a header in {session_jsonl_path(root)}"
    return first
