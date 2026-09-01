"""Create a repository's optimization session, or resume the one it already has.

Resuming is what makes a second ``start`` safe: an existing log is never appended
to, and only a worktree that went missing is put back. A finalized session is the
one exception — it is closed, so its log is moved aside and a fresh session takes
its place rather than the agent being refused until it deletes something. Holding
the repository lock across the call is the caller's job — this touches ``.gymrat/``
and the repository's branches, so two concurrent runs must not reach it.
"""

import contextlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from gymrat.config import ResolvedConfig
from gymrat.errors import GymratError
from gymrat.session.clock import format_iso
from gymrat.session.paths import archived_session_path, session_jsonl_path
from gymrat.session.records import SCHEMA_VERSION, SessionConfig, SessionHooks, SessionRecord
from gymrat.session.store import SessionState, append_record, fold_session, read_records
from gymrat.session.workspace import BaselineRef, create_workspace, recreate_workspace
from gymrat.targets import RefTarget, resolve_target

# Ref the baseline is pinned to when the caller names none.
DEFAULT_BASELINE_REF = "HEAD"

# Bytes of entropy distinguishing two sessions started within the same second.
SESSION_ID_ENTROPY_BYTES = 2


@dataclass(frozen=True, slots=True)
class StartResult:
    """A session ready to iterate in, together with everything its log already holds."""

    #: The header the session's log opens with, newly written or read back.
    session: SessionRecord
    #: What the log adds up to, so a resumed session can report its history.
    state: SessionState
    #: Whether the session was already on disk.
    resumed: bool
    #: The id of the finalized session whose log was moved aside for this one.
    archived: str | None = None
    #: The path the finalized session's log was moved to.
    archived_path: str | None = None


def start_session(root: str, ref: str | None, config: ResolvedConfig) -> StartResult:
    """Create the repository's optimization session, or resume the one it already has.

    Args:
        root: The repository the session lives in.
        ref: The ref the baseline is pinned to, or ``None`` to pin at ``HEAD``.
        config: The resolved run settings the header snapshots as provenance.

    Raises:
        GymratError: When ``ref`` names no commit or a directory, when the log is
            corrupt, or when git refuses to create or recreate the workspace.
    """
    jsonl_path = session_jsonl_path(root)
    state = fold_session(read_records(jsonl_path))
    session = state.session
    baseline_ref = ref if ref is not None else DEFAULT_BASELINE_REF

    if session is None:
        return _create_session(root, jsonl_path, baseline_ref, config)

    if state.finalized is not None:
        # Renamed rather than copied: the new log must not exist until the closed
        # one is out of the way, or a start interrupted mid-archive would leave two
        # sessions claiming the same file. A start that then fails renames it back,
        # so the closed session stays where ``status`` reads it — safe because the
        # new header lands last, leaving nothing at ``jsonl_path`` to collide with.
        archived_path = archived_session_path(root, session.session_id)
        Path(jsonl_path).rename(archived_path)
        try:
            created = _create_session(root, jsonl_path, baseline_ref, config)
        except BaseException:
            _restore_archived_log(archived_path, jsonl_path)
            raise
        return StartResult(
            session=created.session,
            state=created.state,
            resumed=False,
            archived=session.session_id,
            archived_path=archived_path,
        )

    # Every keep moves the baseline onto the commit it made, so a baseline worktree
    # put back at the header's pinned SHA would have the next iteration measure the
    # whole session's diff instead of the edit in front of it. recreate_workspace
    # already no-ops when both worktrees stand, so no on-disk check is needed here.
    recreate_workspace(
        root,
        session.branch,
        state.last_kept_commit if state.last_kept_commit is not None else session.baseline.sha,
    )
    return StartResult(session=session, state=state, resumed=True)


def _restore_archived_log(archived_path: str, jsonl_path: str) -> None:
    """Move a closed session's log back from the archive after a start that failed.

    Best-effort: the caller is re-raising the failure that broke the start, and a
    rename that cannot run must not speak in its place — the closed session's
    records are still on disk under its own id either way.
    """
    # Swallowed by contract — see above.
    with contextlib.suppress(OSError):
        Path(archived_path).rename(jsonl_path)


def _create_session(root: str, jsonl_path: str, ref: str, config: ResolvedConfig) -> StartResult:
    """Pin the baseline, build the workspace, and write the header that opens the log.

    The header lands last: a session the log claims exists but whose branch git
    never created would send every later command looking for a workspace that is
    not there.
    """
    sha = _resolve_baseline_sha(ref, root)
    now = datetime.now(UTC)
    session_id = _new_session_id(now)
    workspace = create_workspace(root, session_id, BaselineRef(ref=ref, sha=sha))

    session = SessionRecord(
        type="session",
        schema_version=SCHEMA_VERSION,
        session_id=session_id,
        created_at=format_iso(now),
        baseline=workspace.baseline,
        branch=workspace.branch,
        worktrees=workspace.worktrees,
        config=_snapshot_config(config),
    )
    append_record(jsonl_path, session)

    return StartResult(session=session, state=fold_session([session]), resumed=False)


def _resolve_baseline_sha(ref: str, root: str) -> str:
    """The commit ``ref`` names, peeled through the same resolution ``compare`` uses.

    A baseline is always a ref: the session's own worktrees are where benchmarks
    run, so a directory — which :func:`resolve_target` otherwise accepts, and
    prefers over a ref of the same name — has nothing to pin a branch at.

    Raises:
        GymratError: When ``ref`` resolves to a directory or to no commit at all.
    """
    target = resolve_target(ref, root)
    if not isinstance(target, RefTarget):
        raise _directory_baseline_error(ref)
    return target.resolved_sha


def _directory_baseline_error(ref: str) -> GymratError:
    """The failure a baseline that resolves to a directory is reported as."""
    return GymratError(
        f"Cannot start a session at '{ref}': it names a directory, not a git ref",
        hint="Pass a branch, tag, or commit the session's baseline is pinned to.",
    )


def _new_session_id(now: datetime) -> str:
    """``<YYYYMMDD-HHMMSS>-<4 hex>`` in UTC.

    The timestamp sorts sessions the way they were started and reads back as a
    date; the random suffix keeps two sessions started in the same second — and
    therefore their branches — apart.
    """
    suffix = secrets.token_hex(SESSION_ID_ENTROPY_BYTES)
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{suffix}"


def _snapshot_config(config: ResolvedConfig) -> SessionConfig:
    """The settings the header records as provenance.

    Only the keys the session schema declares survive, and an optional that was
    never set stays absent rather than becoming an explicit null. Every command
    re-reads its config for the settings it acts on, so this snapshot answers
    "what was this session started with", never "what runs next".
    """
    hooks = config.hooks
    return SessionConfig(
        bench=config.bench,
        adapter=config.adapter,
        samples=config.samples,
        timeout_seconds=config.timeout_seconds,
        primary=config.primary,
        prepare=config.prepare,
        filter=config.filter,
        hooks=SessionHooks(before=hooks.before, after=hooks.after) if hooks is not None else None,
    )
