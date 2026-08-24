"""Canonical session-record builders shared by the store tests.

Each builder returns a fully-populated record with the same defaults the real
session writes, and takes keyword overrides for the fields a test cares about.
Optional fields default to ``None`` on the underlying dataclasses, so passing a
keyword of ``None`` erases the builder's default for that field.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.session._records``.
"""

from dataclasses import replace
from typing import Any

from gymrat_py.session import (
    BaselineRef,
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
    Worktrees,
    session_jsonl_path,
)
from gymrat_py.session.store import append_record

#: The instant every fixture record in this file was written at.
AT = "2026-08-08T14:15:30.000Z"

#: A commit SHA fixture records point at; not a real commit.
COMMIT = "b" * 40

#: The squash commit SHA finalize fixtures point at; distinct from COMMIT.
SQUASH_COMMIT = "c" * 40

#: The session id every fixture record belongs to.
SESSION_ID = "20260808-141530-a3f2"


def session_record(**overrides: Any) -> SessionRecord:
    """The session header a started session writes, with every field overridable.

    ``session_id`` drives the default ``branch``, so a caller overriding just
    the id still gets a matching branch; a caller after a divergent branch
    overrides both explicitly.
    """
    session_id = overrides.get("session_id", SESSION_ID)
    default = SessionRecord(
        type="session",
        schema_version=1,
        session_id=session_id,
        created_at=AT,
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
    return replace(default, **overrides) if overrides else default


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
        metrics={
            "total_ms": MetricVerdict(
                delta_pct=-7.2,
                verdict="improved",
                method="signed-rank",
                p=0.002,
                noise_pct=1.4,
                gating=True,
                confirmed=False,
            )
        },
        primary=IterationPrimary(kind="geomean", delta_pct=-7.2),
        outcome="improved",
        target_reached=False,
    )
    return replace(default, **overrides) if overrides else default


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
    return replace(default, **overrides) if overrides else default


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
    return replace(default, **overrides) if overrides else default


def discard_record(seq: int) -> DiscardRecord:
    """A discard of the iteration numbered ``seq``."""
    return DiscardRecord(type="discard", seq=seq, at=AT)


def hook_record(**overrides: Any) -> HookRecord:
    """The hook a ``before`` stage runs, with every field overridable."""
    default = HookRecord(
        type="hook",
        stage="before",
        seq=1,
        exit_code=0,
        duration_ms=120,
        stdout_bytes=80,
        timed_out=False,
    )
    return replace(default, **overrides) if overrides else default


def finalize_record(**overrides: Any) -> FinalizeRecord:
    """The record that closes a session, with every field overridable."""
    default = FinalizeRecord(
        type="finalize",
        at=AT,
        branch=f"gymrat/{SESSION_ID}-final",
        commit=SQUASH_COMMIT,
        message="squash 1 kept iteration",
    )
    return replace(default, **overrides) if overrides else default


def write_session_log(
    root: str,
    header: SessionRecord,
    history: tuple[SessionLogRecord, ...] = (),
) -> None:
    """Write a session log opening on ``header`` and holding ``history`` after it.

    The header is appended first, then each history record in order.
    """
    jsonl_path = session_jsonl_path(root)
    for record in (header, *history):
        append_record(jsonl_path, record)
