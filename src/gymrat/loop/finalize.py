"""Collapse a session's kept commits into one and close the session.

The squash is built with git plumbing (``commit-tree`` against the session
branch's tree), so nothing is ever checked out: the user's own working copy stays
on whatever branch it was on, which a ``merge --squash`` could not promise. The
session branch is left in place too, so the iteration-by-iteration history that
earned the squash stays readable.
"""

from dataclasses import dataclass
from pathlib import Path

from gymrat.errors import GymratError
from gymrat.git import try_git
from gymrat.plural import pluralize
from gymrat.report.loop import SHORT_SHA_LENGTH
from gymrat.session.clock import now_iso
from gymrat.session.records import FinalizeRecord, KeepRecord, SessionLogRecord, SessionRecord
from gymrat.session.store import (
    SessionState,
    append_record,
    last_kept_position,
    require_open_session,
)
from gymrat.session.workspace import (
    is_worktree_dirty,
    remove_worktrees,
    run_git_step,
    worktree_head,
)

#: The hint a refusal points at whenever the fix is to settle the last iteration.
_SETTLE_FIRST_HINT = "Run gymrat keep or gymrat discard before closing the session."

#: The body line a committed keep gets when it names neither a message nor a commit.
_UNNAMED_KEEP_LINE = "(no message)"


@dataclass(frozen=True, slots=True)
class FinalizeOptions:
    """What a caller can hand a finalize beyond the repository it runs in."""

    #: The squash commit's message; absent, one is generated from the kept commits.
    message: str | None = None
    #: The branch to point at the squash commit; absent, ``<session branch>-final``.
    branch: str | None = None


@dataclass(frozen=True, slots=True)
class FinalizeResult:
    """One closed session: what was written to the log, and what to print about it."""

    #: The record appended to the session log.
    record: FinalizeRecord
    #: The finalize as the agent reads it: the branch, the commit, and any cleanup left.
    report: str


def _validate_finalize(
    root: str,
    opts: FinalizeOptions,
    session: SessionRecord,
    state: SessionState,
) -> str:
    """Guard the session against all finalize refusals and return the expected baseline position."""
    if state.keep_count == 0:
        message = f"Finalize refused: session {session.session_id} has kept nothing to squash."
        hint = "Run gymrat keep on a measured edit before closing the session."
        raise GymratError(message, hint=hint)
    if state.unsettled:
        message = (
            f"Finalize refused: iteration {state.last_seq} has been neither kept nor discarded."
        )
        raise GymratError(message, hint=_SETTLE_FIRST_HINT)
    if is_worktree_dirty(session.worktrees.experiment):
        message = (
            f"Finalize refused: the experiment worktree at {session.worktrees.experiment} "
            "carries uncommitted work."
        )
        raise GymratError(message, hint=_SETTLE_FIRST_HINT)

    expected_position = last_kept_position(state, session.baseline.sha)
    if state.last_kept_commit is not None and Path(session.worktrees.experiment).is_dir():
        head = worktree_head(session.worktrees.experiment)
        if head != expected_position:
            message = (
                "Finalize refused: the experiment worktree has commits that were "
                "neither kept nor discarded."
            )
            raise GymratError(message, hint=_SETTLE_FIRST_HINT)

    if opts.branch is not None and opts.branch.startswith("-"):
        message = (
            f"Finalize refused: '{opts.branch}' starts with a dash, so git reads it as a flag "
            "rather than as a branch name."
        )
        hint = (
            f"Name a branch that does not start with a dash, such as "
            f"--branch {session.branch}-final"
        )
        raise GymratError(message, hint=hint)

    branch = opts.branch if opts.branch is not None else f"{session.branch}-final"
    if try_git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], root) is None:
        message = f"Finalize refused: the branch '{branch}' already exists."
        hint = f"Name another with --branch <name>, or delete it with: git branch -D {branch}"
        raise GymratError(message, hint=hint)

    return expected_position


def finalize_session(root: str, options: FinalizeOptions | None = None) -> FinalizeResult:
    """Collapse a session's kept commits into one commit on the pinned baseline and close it.

    The squash is built with plumbing — ``commit-tree`` against the session
    branch's tree — so nothing is ever checked out: the user's own working copy
    stays on whatever branch it was on, which a ``merge --squash`` could not
    promise. The session branch is left in place too, so the per-iteration history
    that earned the squash stays readable.

    Holding the repository lock across the call is the caller's job: the record
    and the worktree removal it explains must not be separable by another run.

    Raises:
        GymratError: When no open session exists, when the session kept nothing,
            when an iteration is still unsettled, when the experiment worktree
            carries uncommitted work, when the caller's branch name would read as
            a git flag, when the target branch already exists, or when git refuses
            to build the squash commit.
    """
    opts = options if options is not None else FinalizeOptions()
    required = require_open_session(root, "closing the session")
    session, state = required.session, required.state

    expected_position = _validate_finalize(root, opts, session, state)
    branch = opts.branch if opts.branch is not None else f"{session.branch}-final"

    message = opts.message if opts.message is not None else _generated_message(required.records)
    commit = _squash_onto_baseline(root, expected_position, session.baseline.sha, message)
    # `--` ends git's options, so the branch name can never be read as one whatever
    # the leading-dash check let through.
    run_git_step(
        ["branch", "--", branch, commit],
        root,
        f"Cannot point '{branch}' at the squash commit {commit}",
        "Inspect the branches this repository has with: git branch --list",
    )

    record = FinalizeRecord(
        type="finalize",
        at=now_iso(),
        branch=branch,
        commit=commit,
        message=message,
    )
    # Record the squash before clearing the worktrees: the branch already carries
    # the work, so a removal that fails must not leave the session open on a log
    # that never mentions the commit the agent is about to be told to look at.
    append_record(required.jsonl_path, record)

    warnings = remove_worktrees(root, session.worktrees)

    return FinalizeResult(
        record=record,
        report=_finalize_report(record, state.keep_count, session.branch, warnings),
    )


def _squash_onto_baseline(
    root: str,
    tree_source: str,
    baseline_sha: str,
    message: str,
) -> str:
    """Build one commit carrying ``tree_source``'s tree onto the pinned baseline.

    ``tree_source`` is the last-kept position — the commit whose tree the squash
    should carry. Reading the tree and writing the commit are both plumbing, so
    neither needs — or moves — a checkout. The single parent is the baseline the
    session started from, which is what makes the result a squash rather than a
    merge.
    """
    tree = run_git_step(
        ["rev-parse", f"{tree_source}^{{tree}}"],
        root,
        f"Cannot read the tree at {tree_source[:SHORT_SHA_LENGTH]}",
        f"Check that the commit is still there: git cat-file -t {tree_source}",
    ).strip()

    build_hint = (
        f"Check that {baseline_sha} is a commit this repository has: git cat-file -t {baseline_sha}"
    )
    return run_git_step(
        ["commit-tree", tree, "-p", baseline_sha, "-m", message],
        root,
        f"Cannot build the squash commit from {tree_source[:SHORT_SHA_LENGTH]} onto {baseline_sha}",
        build_hint,
    ).strip()


def _generated_message(records: list[SessionLogRecord]) -> str:
    """The squash message written when the caller supplied none.

    The subject counts what was collapsed and the body lists the kept commits in
    the order they landed, so the one commit that reaches the user's branch still
    says what the session did — the per-iteration history stays on the session
    branch, which nothing but the reader's curiosity brings back.

    A keep whose ``message`` the log omits — which gymrat never writes, but a
    hand-edited or older log can hold — falls back to its short commit rather than
    dropping its line and leaving the subject claiming more than the body shows.
    """
    kept = [
        record.message
        or (record.commit[:SHORT_SHA_LENGTH] if record.commit else None)
        or _UNNAMED_KEEP_LINE
        for record in records
        if isinstance(record, KeepRecord) and record.status == "committed"
    ]

    subject = f"gymrat: squash {pluralize(len(kept), 'kept iteration')}"
    return f"{subject}\n\n" + "\n".join(kept)


def _finalize_report(
    record: FinalizeRecord,
    keep_count: int,
    session_branch: str,
    warnings: list[str],
) -> str:
    """The finalize as the agent reads it, with anything git would not clean up spelled out."""
    lines = [
        f"Finalized onto {record.branch} as {record.commit[:SHORT_SHA_LENGTH]}",
        f"  squashed {pluralize(keep_count, 'kept iteration')} into one commit",
        f"  the session is closed; {session_branch} is left in place for its history",
        *warnings,
    ]
    return "\n".join(lines)
