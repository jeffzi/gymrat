"""Canonical session-record builders shared across the test suite.

Each builder returns a fully-populated record with the same defaults the real
session writes, and takes keyword overrides for the fields a test cares about.
Optional fields default to ``None`` on the underlying pydantic models, so passing
a keyword of ``None`` erases the builder's default for that field.
"""

from dataclasses import replace
from pathlib import Path
from types import UnionType
from typing import Any, Literal, overload

from pydantic import BaseModel

from gymrat.session.paths import (
    archived_session_path,
    baseline_worktree_dir,
    experiment_worktree_dir,
    repo_root,
    session_jsonl_path,
)
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
from gymrat.session.schema import KeepReason
from gymrat.session.store import SessionState, append_record, read_records
from gymrat.session.workspace import BaselineRef, Worktrees

#: The instant every fixture record in this file was written at (nanoseconds since epoch).
AT = 1_786_198_530_000_000_000

#: A commit SHA fixture records point at; not a real commit.
COMMIT = "b" * 40

#: A 40-hex baseline SHA whose first seven characters are recognizable on their own.
RECOGNIZABLE_BASELINE_SHA = "a1b2c3d" + "e" * 33

#: A 40-hex keep-commit SHA whose first seven characters are recognizable on their own.
RECOGNIZABLE_KEEP_COMMIT = "b1b2b3b" + "c" * 33

#: The squash commit SHA finalize fixtures point at; distinct from COMMIT.
SQUASH_COMMIT = "c" * 40

#: The session id every fixture record belongs to.
SESSION_ID = "20260808-141530-a3f2"

#: The session id the supervised-run fixtures (launch events, dashboards) carry.
SUPERVISED_SESSION_ID = "20260813-125044-34ec"

#: The baseline commit SHA fixture records pin; not a real commit.
BASELINE_SHA = "a" * 40

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

    Args:
        **overrides: ``SessionRecord`` fields to set in place of the defaults.

    Returns:
        The session header with ``overrides`` applied.
    """
    session_id = overrides.get("session_id", SESSION_ID)
    # pyrefly: ignore[missing-argument] -- validate_by_name accepts schema_version
    default = SessionRecord(
        type="session",
        schema_version=1,
        session_id=session_id,
        at=AT,
        baseline=BaselineRef(ref="main", sha=BASELINE_SHA),
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

    Args:
        seq: The iteration the refused keep settles.
        **overrides: ``KeepRecord`` fields to set in place of the defaults. ``reason``
            defaults to ``"checks-failed"``; pass ``reason=None`` to erase it, or a
            settling reason such as ``"gating-regression"`` to override it.

    Returns:
        The blocked keep record with ``overrides`` applied.
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


def gate_block(seq: int, reason: KeepReason) -> KeepRecord:
    """Build a keep a gate refused before the configured checks ever ran.

    Args:
        seq: The iteration the refused keep settles.
        reason: The gate that refused it, such as ``"gating-regression"`` or
            ``"nothing-measured"``.

    Returns:
        The blocked keep record, its checks configured but not run.
    """
    return blocked_keep(seq, reason=reason, checks=KeepChecks(configured=True))


def discard_record(seq: int) -> DiscardRecord:
    """A discard that settles the iteration numbered ``seq``."""
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


def archive_and_reopen_session_log(previous: SessionRecord, fresh: SessionRecord) -> None:
    """Archive the open session log, as a new session does, and write a fresh one.

    The fresh log holds ``fresh`` and one baseline record. The repository is the
    one the process runs in.

    Args:
        previous: The header of the session whose log is archived.
        fresh: The header the fresh log opens with.
    """
    root = repo_root()
    archive = Path(archived_session_path(root, previous.session_id))
    archive.parent.mkdir(parents=True, exist_ok=True)
    Path(session_jsonl_path(root)).rename(archive)
    write_session_log(root, fresh, (baseline_record(),))


@overload
def records_of_type[R: SessionLogRecord](
    root: str, record_type: type[R], *, matching: Literal[True] = True
) -> list[R]: ...


@overload
def records_of_type(
    root: str, record_type: type[SessionLogRecord] | UnionType, *, matching: bool
) -> list[SessionLogRecord]: ...


def records_of_type(
    root: str, record_type: type[SessionLogRecord] | UnionType, *, matching: bool = True
) -> list[SessionLogRecord]:
    """The records of ``root``'s session log that are, or are not, of ``record_type``.

    Args:
        root: The repository whose session log is read.
        record_type: A record class, or a union of record classes.
        matching: ``False`` keeps the records that are *not* of ``record_type``.

    Returns:
        The selected records, in file order.
    """
    return [record for record in log_records(root) if isinstance(record, record_type) is matching]


def last_command_record(root: str) -> CommandRecord:
    """Read the session log and return the last ``CommandRecord``.

    Args:
        root: The repository whose session log is read.

    Returns:
        The last command record in the log.

    Raises:
        AssertionError: The log holds no command record.
    """
    for record in reversed(log_records(root)):
        if isinstance(record, CommandRecord):
            return record
    msg = "no CommandRecord found in session log"
    raise AssertionError(msg)


def session_header_of(root: str) -> SessionRecord:
    """The session header ``root``'s log opens with, failing when there is none."""
    records = log_records(root)
    assert records, f"expected a session header in {session_jsonl_path(root)}"
    first = records[0]
    assert isinstance(first, SessionRecord), f"expected a header in {session_jsonl_path(root)}"
    return first
