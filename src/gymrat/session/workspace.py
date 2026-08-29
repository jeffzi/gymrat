"""Git orchestration for a session's branch and worktrees.

A session edits on its own ``gymrat/<id>`` branch inside an *experiment*
worktree and measures against a *baseline* worktree detached at the commit the
session pinned. This module owns the git plumbing that creates, resumes, and
tears those down; it holds no session records and reads no JSON.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from gymrat.errors import GymratError, stderr_text_of
from gymrat.git import repository_lookup_error, run_git, try_git
from gymrat.session.paths import (
    SESSION_DIR_NAME,
    baseline_worktree_dir,
    experiment_worktree_dir,
)

# Prefix of the branch a session's experiment worktree sits on.
BRANCH_PREFIX = "gymrat/"

# Hint repeated by every git step whose failure leaves the worktree worth
# inspecting as-is.
INSPECT_STATUS_HINT = "Inspect what is standing there with: git status"


@dataclass(frozen=True, slots=True)
class BaselineRef:
    """A git ref together with the commit it resolved to when the session started."""

    ref: str
    sha: str


@dataclass(frozen=True, slots=True)
class Worktrees:
    """The two worktree directories a session runs in."""

    experiment: str
    baseline: str


@dataclass(frozen=True, slots=True)
class WorkspaceResult:
    """A session's git state: the branch it edits on, its worktrees, and its pinned baseline."""

    branch: str
    worktrees: Worktrees
    baseline: BaselineRef


# ---------------------------------------------------------------------------
# Workspace lifecycle
# ---------------------------------------------------------------------------


def create_workspace(root: str, session_id: str, baseline: BaselineRef) -> WorkspaceResult:
    """Create the branch and both worktrees a session runs in.

    The experiment worktree is checked out *on* ``gymrat/<session_id>`` so edits
    and commits land on the session's own branch; the baseline worktree is
    detached at ``baseline.sha`` so it keeps measuring the same commit no matter
    what the ref it came from does afterwards.

    Either all of the branch and both worktrees exist afterwards, or nothing this
    call made does: a worktree git refuses takes the branch and whatever this
    attempt had already checked out down with it, so a retry starts from the
    state it found instead of tripping over its own leftovers. A worktree
    directory that was already standing survives — see :func:`_unwind_workspace`.

    Raises:
        GymratError: When ``root`` is not a git repository, or when git refuses
            to prune, to create the branch, or to create either worktree.
    """
    branch = f"{BRANCH_PREFIX}{session_id}"

    ensure_git_exclude(root)
    _prune_stale_worktrees(root)

    run_git_step(
        ["branch", branch, baseline.sha],
        root,
        f"Cannot create the session branch '{branch}'",
        f"A crashed session may have left it behind. Delete it with: git branch -D {branch}",
    )
    # Read before the first add, so the unwind can tell a directory this attempt
    # checked out from one that was already there.
    standing = [
        directory
        for directory in (experiment_worktree_dir(root), baseline_worktree_dir(root))
        if _is_directory(directory)
    ]

    try:
        _add_experiment_worktree(root, branch)
        _add_baseline_worktree(root, baseline.sha)
    except GymratError:
        _unwind_workspace(root, branch, standing)
        raise

    return WorkspaceResult(
        branch=branch,
        worktrees=Worktrees(
            experiment=experiment_worktree_dir(root),
            baseline=baseline_worktree_dir(root),
        ),
        baseline=baseline,
    )


def ensure_git_exclude(root: str) -> None:
    """Keep git from reporting the session directory as untracked.

    The line goes in ``.git/info/exclude`` rather than ``.gitignore`` because it
    is this checkout's business, not the project's: nothing gymrat writes should
    show up in a commit the agent under test prepares.

    Raises:
        GymratError: When ``root`` is not a git repository.
    """
    exclude_file = Path(_git_common_dir(root)) / "info" / "exclude"
    line = f"{SESSION_DIR_NAME}/"
    try:
        existing = exclude_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""

    if any(entry.strip() == line for entry in existing.split("\n")):
        return

    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    separator = "" if existing == "" or existing.endswith("\n") else "\n"
    with exclude_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{separator}{line}\n")


def recreate_workspace(root: str, branch: str, baseline_sha: str) -> None:
    """Put back whichever of a session's worktrees is no longer on disk.

    Resuming has to survive a worktree the user deleted, so this is a no-op when
    both are present — an experiment worktree carrying uncommitted work is never
    re-checked-out.

    Raises:
        GymratError: When git refuses to prune or to add a worktree.
    """
    needs_experiment = not _is_directory(experiment_worktree_dir(root))
    needs_baseline = not _is_directory(baseline_worktree_dir(root))

    if not needs_experiment and not needs_baseline:
        return

    _prune_stale_worktrees(root)

    if needs_experiment:
        _add_experiment_worktree(root, branch)
    if needs_baseline:
        _add_baseline_worktree(root, baseline_sha)


def _prune_stale_worktrees(root: str) -> None:
    """Remove the session's own worktree entries whose directories are gone.

    Only the session's experiment and baseline paths are touched; a user's
    temporarily-absent worktree stays registered.
    """
    for directory in (experiment_worktree_dir(root), baseline_worktree_dir(root)):
        if not _is_directory(directory):
            try_git(["worktree", "remove", "--force", directory], root)


def _add_experiment_worktree(root: str, branch: str) -> None:
    directory = experiment_worktree_dir(root)
    run_git_step(
        ["worktree", "add", directory, branch],
        root,
        f"Cannot create the experiment worktree at {directory}",
        f"Check whether {branch} is already checked out elsewhere: git worktree list",
    )


def _add_baseline_worktree(root: str, sha: str) -> None:
    directory = baseline_worktree_dir(root)
    run_git_step(
        ["worktree", "add", "--detach", directory, sha],
        root,
        f"Cannot create the baseline worktree at {directory}",
        f"Check that {sha} is a commit this repository has: git cat-file -t {sha}",
    )


def _unwind_workspace(root: str, branch: str, standing: list[str]) -> None:
    """Take back the branch and worktrees a failed create attempt had made.

    A directory listed in ``standing`` was there before the attempt began — what
    a session whose log was lost leaves behind — so it stays, uncommitted work
    and all. The error the caller is about to raise names the path, which is the
    only notice the user gets that something is in the way.

    The worktrees go before the branch: git refuses to delete a branch one of
    them still has checked out. Every step is best-effort — the caller is about
    to surface the git step that broke the session, and a cleanup that cannot
    finish must not speak in its place.
    """
    for directory in (experiment_worktree_dir(root), baseline_worktree_dir(root)):
        if _is_directory(directory) and directory not in standing:
            try_git(["worktree", "remove", "--force", directory], root)
    try_git(["branch", "-D", branch], root)


# ---------------------------------------------------------------------------
# Experiment tree
# ---------------------------------------------------------------------------


def commit_workspace(experiment_dir: str, message: str) -> str:
    """Commit everything standing in the experiment worktree, and report the commit.

    Staging is ``add -A`` so the commit holds what the agent produced whether it
    tracked its new files or not — the worktree belongs to the session, and a
    keep that left half an edit behind would put the next iteration's baseline
    out of step with the code that earned it.

    Raises:
        GymratError: When git refuses to stage or to commit — a worktree with
            nothing to commit included.
    """
    run_git_step(
        ["add", "-A"],
        experiment_dir,
        f"Cannot stage the experiment worktree at {experiment_dir}",
        INSPECT_STATUS_HINT,
    )
    run_git_step(
        ["commit", "-m", message],
        experiment_dir,
        f"Cannot commit the experiment worktree at {experiment_dir}",
        INSPECT_STATUS_HINT,
    )
    return run_git_step(
        ["rev-parse", "HEAD"],
        experiment_dir,
        f"Cannot read the commit just made in {experiment_dir}",
        "Inspect the branch with: git log -1",
    ).strip()


def revert_workspace(experiment_dir: str, *, target: str | None = None) -> None:
    """Reset the experiment worktree to ``target`` (or HEAD) and drop untracked files.

    Destructive by contract, and safe because the directory is one gymrat owns:
    the reset covers tracked edits — staged or not — and moves the branch when
    ``target`` names a commit behind HEAD, and the clean covers the files the
    agent added, which a reset alone would leave behind to be picked up by the
    next keep.

    Raises:
        GymratError: When git refuses to reset or to clean.
    """
    sha = target or "HEAD"
    run_git_step(
        ["reset", "--hard", sha],
        experiment_dir,
        f"Cannot revert the experiment worktree at {experiment_dir}",
        INSPECT_STATUS_HINT,
    )
    run_git_step(
        ["clean", "-fd"],
        experiment_dir,
        f"Cannot remove the untracked files in {experiment_dir}",
        "Inspect what is standing there with: git status --ignored",
    )


# ---------------------------------------------------------------------------
# Worktree queries
# ---------------------------------------------------------------------------


def worktree_head(directory: str) -> str:
    """The commit ``directory`` currently has checked out, as a full hex SHA.

    Raises:
        GymratError: When git refuses to read the HEAD.
    """
    return run_git_step(
        ["rev-parse", "HEAD"],
        directory,
        f"Cannot read the HEAD of the worktree at {directory}",
        "Inspect what is standing there with: git log -1",
    ).strip()


def is_worktree_dirty(directory: str) -> bool:
    """Whether ``directory`` holds work git has not committed — untracked files included.

    A directory that is not there reads as clean: a worktree the user deleted
    carries no uncommitted work anyone can still act on, and refusing to finalize
    over a directory that cannot be inspected would strand the session.
    """
    if not _is_directory(directory):
        return False
    return (
        run_git_step(
            ["status", "--porcelain"],
            directory,
            f"Cannot read the status of the worktree at {directory}",
            INSPECT_STATUS_HINT,
        ).strip()
        != ""
    )


def advance_baseline(baseline_dir: str, sha: str) -> None:
    """Move the baseline worktree onto ``sha``, detached as it was created.

    Detached is not incidental: ``sha`` is a commit on the session branch, which
    the experiment worktree has checked out, and git refuses to check the same
    branch out twice.

    Raises:
        GymratError: When git refuses the checkout.
    """
    run_git_step(
        ["checkout", "--detach", sha],
        baseline_dir,
        f"Cannot move the baseline worktree at {baseline_dir} to {sha}",
        f"Check that {sha} is a commit this repository has: git cat-file -t {sha}",
    )


def remove_worktrees(root: str, worktrees: Worktrees) -> list[str]:
    """Take both of a session's worktrees off disk and out of git's bookkeeping.

    Each worktree is named by path rather than swept for, because git clears the
    entry of a directory that vanished behind its back when it is asked for that
    path — which leaves a worktree of the user's own that is only temporarily
    absent registered. A refusal over a directory that is already gone stays
    quiet: nothing is standing there for the user to clear by hand.

    Removal failures are returned instead of raised because this runs after the
    finalize record is written — the session is closed either way, and the
    caller's job is to tell the user which directory to clear by hand.

    Returns:
        One warning per worktree git left standing, empty when both went.
    """
    warnings: list[str] = []

    for directory in (worktrees.experiment, worktrees.baseline):
        error = try_git(["worktree", "remove", "--force", directory], root)
        if error is not None and _is_directory(directory):
            warnings.append(
                f"Could not remove the worktree at {directory}: {error}\n"
                f"  remove it by hand with: git worktree remove --force {directory}"
            )

    return warnings


# ---------------------------------------------------------------------------
# Git plumbing
# ---------------------------------------------------------------------------


def _git_common_dir(root: str) -> str:
    """Absolute path of the repository's shared git directory.

    ``info/exclude`` lives in the common directory, so a linked worktree — whose
    ``.git`` is a file pointing elsewhere — excludes through the same file the
    main checkout uses. git prints the path relative to the working directory
    when it sits inside it, hence the resolve against ``root``.
    """
    try:
        printed = run_git(["rev-parse", "--git-common-dir"], root).strip()
    except (subprocess.SubprocessError, OSError) as error:
        raise repository_lookup_error(root, error) from error
    # An absolute path git prints (a linked worktree's common dir) stands on its
    # own; a relative one (``.git`` in the main checkout) resolves against root.
    return str(Path(root, printed))


def _is_directory(directory: str) -> bool:
    return Path(directory).is_dir()


def run_git_step(args: list[str], cwd: str, message: str, hint: str) -> str:
    """Run git in ``cwd``, turning a non-zero exit into a ``GymratError``.

    The error carries git's own diagnostics after ``message`` so the reader sees
    the real reason, and ``hint`` for what to do next.
    """
    try:
        return run_git(args, cwd)
    except (subprocess.SubprocessError, OSError) as error:
        detail = f"{message}: {stderr_text_of(error)}"
        raise GymratError(detail, hint=hint) from error
