"""The session JSONL store and the fold that summarizes it.

A session log is an append-only file of one record per line. This module owns
the two ends of that contract:

- :func:`append_record` writes a record only after proving it reads back, and
  recovers a torn final line before appending so a crash mid-write never leaves
  the log unparseable.
- :func:`read_records` reads the log back in file order, and :func:`fold_session`
  folds the records into the :class:`SessionState` every loop command consults
  before acting.

:func:`require_session` and :func:`require_open_session` wrap read-and-fold with
the guards a command needs: a session must exist, and — for the writers — it
must not already be finalized.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import assert_never

from gymrat.errors import GymratError, hint_of
from gymrat.finite_json import null_non_finite
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import (
    BaselineRecord,
    DiscardRecord,
    FinalizeRecord,
    HookRecord,
    IterationRecord,
    KeepRecord,
    SessionLogRecord,
    SessionRecord,
    parse_record,
    record_to_wire,
)

__all__ = [
    "RequiredSession",
    "SessionState",
    "append_record",
    "fold_session",
    "last_kept_position",
    "read_records",
    "require_open_session",
    "require_session",
]


@dataclass(frozen=True, slots=True)
class SessionState:
    """What a session log adds up to: the state every loop command reads before acting."""

    #: The header opening the log, absent while no session has been started.
    session: SessionRecord | None
    #: How many edits have been measured.
    iteration_count: int
    #: The most recently measured edit, absent while nothing has been measured.
    last_iteration: IterationRecord | None
    #: Whether a measured edit is waiting to be kept or discarded.
    unsettled: bool
    #: How many edits were committed.
    keep_count: int
    #: How many edits were reverted.
    discard_count: int
    #: Whether the last committed keep settled an edit that reached the target metric.
    target_reached_and_kept: bool
    #: The highest number any iteration or settling record has taken, ``0`` while
    #: the log has none. A refusal that settles nothing still claims a number, so
    #: this is a high-water mark: the next record to need a number takes
    #: ``last_seq + 1`` and cannot alias one the log already carries.
    last_seq: int
    #: The commit made by the last keep that committed, absent when none did. The
    #: baseline worktree advances to every kept commit, so this — not the header's
    #: pinned SHA — is where the baseline stands once a keep has landed.
    last_kept_commit: str | None
    #: Whether the log ends on a keep the loop blocked for a gating regression.
    #: The block settles the iteration it refused, but the edit it would not
    #: commit still stands in the experiment worktree, so ``discard`` accepts this
    #: as the one settled state it may still revert. Any iteration, keep, or
    #: discard after the block supersedes it — except a keep refused for want of a
    #: measurement, which must not wedge the edit in place.
    ends_on_gating_block: bool
    #: The record that closed the session, absent while it is still open.
    finalized: FinalizeRecord | None


@dataclass(frozen=True, slots=True)
class RequiredSession:
    """An open session, with everything reading its log already produced."""

    #: The header the log opens with.
    session: SessionRecord
    #: What the whole log folds to, ``session`` included.
    state: SessionState
    #: The log the session was read from.
    jsonl_path: str
    #: Every record the log holds, in file order — the same ones ``state`` folds.
    records: list[SessionLogRecord]


def last_kept_position(state: SessionState, baseline_sha: str) -> str:
    """The commit the experiment worktree should stand at after the last keep.

    Returns the commit from the most recent committed keep, falling back to
    ``baseline_sha`` when the session has kept nothing. Both ``discard_session``
    and ``finalize_session`` need this position: discard resets the worktree to
    it, and finalize refuses when the worktree has drifted past it.
    """
    return state.last_kept_commit or baseline_sha


def append_record(jsonl_path: str, record: SessionLogRecord) -> None:
    """Append ``record`` to the session log at ``jsonl_path``, creating its directory.

    One record is one write of one line, so a reader can treat the log as
    append-only truth rather than a file that may be caught half-written.

    Raises:
        GymratError: When ``record`` would not read back. The log is left
            untouched, and no directory is created for a log that does not exist
            yet — serialization is proven before any I/O.
    """
    line = _serialize_record(record)
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    _truncate_torn_tail(jsonl_path)
    payload = f"{line}\n".encode()
    fd = os.open(jsonl_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def _serialize_record(record: SessionLogRecord) -> str:
    """The line ``record`` is written as, proven to parse back into the same record.

    A record can satisfy the type checker and still not survive JSON: ``NaN`` and
    ``Infinity`` are floats in Python but JSON ``null`` under JavaScript's
    ``JSON.stringify``, and any measurement an adapter hands back unchecked can
    carry one. Non-finite floats are lowered to ``None`` — matching that wire
    form — and the serialized line is run back through :func:`parse_record`, the
    very parser :func:`read_records` uses. Nothing is written until that round
    trip succeeds, which stops a single bad measurement from leaving the whole
    session log unreadable.
    """
    try:
        wire = record_to_wire(record)
        line = json.dumps(null_non_finite(wire))
        parse_record(json.loads(line))
    except (GymratError, ValueError, TypeError) as error:
        message = f"Refusing to log an unreadable {record.type} record: {error!s}"
        hint = (
            "Nothing was written. A metric that is NaN or Infinity becomes "
            "null in JSON and no longer reads back."
        )
        raise GymratError(message, hint=hint) from error
    return line


def _truncate_torn_tail(jsonl_path: str) -> None:
    """Remove an unterminated last line left by a torn write.

    The reader already skips such a line, so the data is already lost; truncating
    it here keeps the next append from landing on the same line and making the
    log unparseable on re-read.
    """
    try:
        data = Path(jsonl_path).read_bytes()
    except FileNotFoundError:
        return
    # The byte a completed record line ends with; a file not ending on it was torn.
    if not data or data[-1] == ord(b"\n"):
        return
    # rfind returns -1 when the torn line was the only content, so the log
    # truncates to zero bytes and the caller appends the first clean line.
    last_newline = data.rfind(b"\n")
    with Path(jsonl_path).open("rb+") as handle:
        handle.truncate(last_newline + 1)


def read_records(jsonl_path: str) -> list[SessionLogRecord]:
    """Read every record from the session log at ``jsonl_path``, in file order.

    A log that does not exist reads as no session — an empty list — because the
    loop commands distinguish "no session" from "corrupt session" and only the
    latter is a failure.

    Raises:
        GymratError: When a line is not JSON, when a line matches no record
            schema, or when the first record is not a session header. Every
            message names the log and the 1-based line at fault.
    """
    try:
        raw = Path(jsonl_path).read_bytes()
    except FileNotFoundError:
        return []

    # Split on newline bytes. The final element is whatever follows the last
    # newline — either empty (when the file ends on \n) or the torn tail of a
    # write that never finished.
    raw_lines = raw.split(b"\n")
    # A file whose last byte is not b"\n" ends on a line the writer never
    # finished — a torn append or a crash mid-flush. Each record is written as
    # a single newline-terminated write, so a line lacking the terminator was
    # never completed and must not be trusted.
    last_unterminated = bool(raw) and raw[-1] != ord(b"\n")

    records: list[SessionLogRecord] = []
    for index, raw_line in enumerate(raw_lines):
        if raw_line.strip() == b"":
            continue

        is_last_line = index == len(raw_lines) - 1
        if is_last_line and last_unterminated:
            break

        at = f"{jsonl_path}:{index + 1}"

        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as error:
            message = f"Corrupt session log at {at}"
            raise GymratError(
                message, hint=f"Line {index + 1} contains invalid UTF-8 bytes."
            ) from error

        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            message = f"Invalid JSON at {at}"
            raise GymratError(message, hint=f"Line {index + 1} is not a JSON object.") from error

        try:
            record = parse_record(value)
        except GymratError as error:
            message = f"{error!s} (at {at})"
            raise GymratError(message, hint=hint_of(error)) from error

        if not records and record.type != "session":
            message = f"Expected session header at {at}, got a {record.type} record"
            raise GymratError(
                message, hint="The session log is corrupt; start a new session."
            ) from None
        records.append(record)

    return records


@dataclass(slots=True)
class _FoldState:
    """Mutable accumulator for :func:`fold_session`.

    Mirrors :class:`SessionState`, plus a scratch map of which iteration numbers
    reached the target — needed so a committed keep can settle whether the edit
    it commits was on target. The scratch never leaves the fold.
    """

    session: SessionRecord | None = None
    iteration_count: int = 0
    last_iteration: IterationRecord | None = None
    unsettled: bool = False
    keep_count: int = 0
    discard_count: int = 0
    target_reached_and_kept: bool = False
    last_seq: int = 0
    last_kept_commit: str | None = None
    ends_on_gating_block: bool = False
    finalized: FinalizeRecord | None = None
    target_reached_by_seq: dict[int, bool] = field(default_factory=dict)


def _fold_iteration(state: _FoldState, record: IterationRecord) -> None:
    state.iteration_count += 1
    state.last_seq = max(state.last_seq, record.seq)
    state.last_iteration = record
    state.unsettled = True
    state.ends_on_gating_block = False
    state.target_reached_by_seq[record.seq] = record.target_reached


def _fold_keep(state: _FoldState, record: KeepRecord) -> None:
    state.last_seq = max(state.last_seq, record.seq)
    if record.status == "committed":
        state.unsettled = False
        state.keep_count += 1
        # Assignment, not accumulation: a committed keep of a non-target seq
        # resets the flag the way a target one sets it.
        state.target_reached_and_kept = state.target_reached_by_seq.get(record.seq, False)
        if record.commit is not None:
            state.last_kept_commit = record.commit
    elif record.reason is not None and record.reason not in ("checks-failed", "nothing-measured"):
        # A blocked keep settles the iteration — except "checks-failed" (fix and
        # retry) and an absent reason (nothing says the edit is beyond recovery).
        state.unsettled = False
    # A "nothing-measured" refusal commits and settles nothing, so it leaves the
    # standing edit — and the gating-block window — as it found them.
    if record.reason != "nothing-measured":
        state.ends_on_gating_block = (
            record.status == "blocked" and record.reason == "gating-regression"
        )


def _fold_discard(state: _FoldState, record: DiscardRecord) -> None:
    state.last_seq = max(state.last_seq, record.seq)
    state.unsettled = False
    state.discard_count += 1
    state.ends_on_gating_block = False


def fold_session(records: list[SessionLogRecord]) -> SessionState:
    """Fold ``records`` into the state they describe.

    Folds whatever it is given: validating the log — that it parses and opens
    with a session header — belongs to :func:`read_records`.
    """
    state = _FoldState()
    for record in records:
        match record:
            case SessionRecord():
                if state.session is None:
                    state.session = record
            case IterationRecord():
                _fold_iteration(state, record)
            case KeepRecord():
                _fold_keep(state, record)
            case DiscardRecord():
                _fold_discard(state, record)
            case FinalizeRecord():
                state.finalized = record
            case BaselineRecord() | HookRecord():
                pass
            case _ as unreachable:
                assert_never(unreachable)

    return SessionState(
        session=state.session,
        iteration_count=state.iteration_count,
        last_iteration=state.last_iteration,
        unsettled=state.unsettled,
        keep_count=state.keep_count,
        discard_count=state.discard_count,
        target_reached_and_kept=state.target_reached_and_kept,
        last_seq=state.last_seq,
        last_kept_commit=state.last_kept_commit,
        ends_on_gating_block=state.ends_on_gating_block,
        finalized=state.finalized,
    )


def require_session(root: str, verb: str) -> RequiredSession:
    """The session open in ``root``, or the error telling the caller to open one.

    ``verb`` names what the caller was about to do — "measuring an edit" — and
    becomes the thing the hint says no session was open for, so every loop
    command refuses in its own words while sharing one guard.

    Raises:
        GymratError: When no session has been started, or when the log is
            corrupt — every parse failure names the log and the line at fault.
    """
    jsonl_path = session_jsonl_path(root)
    records = read_records(jsonl_path)
    state = fold_session(records)

    if state.session is None:
        message = f"No session in {root}"
        raise GymratError(message, hint=f"Run gymrat start to open one before {verb}.")

    return RequiredSession(
        session=state.session, state=state, jsonl_path=jsonl_path, records=records
    )


def require_open_session(root: str, verb: str) -> RequiredSession:
    """The *open* session in ``root``, or the error telling the caller why there is none.

    Every command that writes to the log goes through this rather than
    :func:`require_session`: a finalized session has had its kept work collapsed
    and its worktrees removed, so appending to it would record work no branch
    carries. Reading a closed session stays on the unguarded path.

    Raises:
        GymratError: When no session has been started, when the log is corrupt,
            or when the session was already finalized.
    """
    required = require_session(root, verb)
    finalized = required.state.finalized

    if finalized is not None:
        message = f"Session {required.session.session_id} was finalized onto {finalized.branch}"
        raise GymratError(message, hint=f"Run gymrat start to open a new session before {verb}.")

    return required
