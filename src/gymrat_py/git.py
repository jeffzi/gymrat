"""Git subprocess helpers and repository-lookup classification.

Every git call runs with an argv list rather than a shell string, so refs and
paths containing shell metacharacters are treated as literal git arguments. The
child env is scrubbed of the repo-targeting ``GIT_*`` variables (so an outer git
process cannot redirect the call away from ``cwd``) and pinned to ``LC_ALL=C``
for stable, locale-independent diagnostics that classification can key on.
"""

import os
import re
import signal
import subprocess
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager

from gymrat_py.errors import GymratError, stderr_text_of
from gymrat_py.signals import TERMINATION_SIGNALS

# Env vars an outer git process exports to point child git at a specific repo.
# Removing them forces this call to resolve the repository from ``cwd`` alone.
_REPO_TARGETING_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
)

# Git's wording when it places a directory outside every repository.
#
# Anchored to ``^fatal:`` so a path that embeds the phrase (e.g.
# ``/tmp/not a git repository/config``) cannot skip the classification. The
# ``re.MULTILINE`` flag lets ``^`` match at line boundaries within multi-line
# stderr. ``LC_ALL=C`` in :func:`run_git` stabilizes the wording across locales,
# so a case-insensitive flag is not needed.
_NOT_A_REPOSITORY_RE = re.compile(r"^fatal: not a git repository", re.MULTILINE)

# POSIX-only seam for blocking signals. ``None`` on platforms without
# ``pthread_sigmask`` (win32), where run_git falls back to running git unmasked.
# Kept as a module-level reference so the fallback branch stays testable.
_pthread_sigmask: Callable[[int, Iterable[int]], list[int]] | None = getattr(
    signal, "pthread_sigmask", None
)


@contextmanager
def _deferring_termination_signals() -> Iterator[None]:
    """Block termination signals for the duration of the wrapped git call.

    A termination signal delivered while git is running must not fire the
    process's termination cleanup mid-call: that cleanup sweeps stranded
    worktrees, and a ``git worktree add`` interrupted partway through is exactly
    the state the sweep must never see. Blocking the signals keeps them pending
    until git returns, so a registered handler runs only after git has fully
    materialized its work, then still exits with ``128 + signal_number``.

    On platforms without ``pthread_sigmask`` (win32) this is a no-op and git
    runs unmasked.

    The git child forked while the mask is held inherits the blocked mask for
    its own lifetime — exec does not reset it. This is an accepted divergence:
    git sets its own signal dispositions, and unblocking the mask in the forked
    child would trade that small divergence for a fork-time hazard.
    """
    if _pthread_sigmask is None or not TERMINATION_SIGNALS:
        yield
        return

    previous = _pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
    try:
        yield
    finally:
        _pthread_sigmask(signal.SIG_SETMASK, previous)


def run_git(args: Sequence[str], cwd: str) -> str:
    """Run git in ``cwd`` and return its untrimmed stdout.

    Args:
        args: Git arguments passed as an argv list, never a shell string.
        cwd: Working directory the git call resolves the repository from.

    Returns:
        The command's stdout, exactly as git wrote it (not trimmed).

    Raises:
        subprocess.CalledProcessError: When git exits non-zero. It carries
            ``.stderr`` and ``.returncode`` for callers to mine.
    """
    env = os.environ.copy()
    for key in _REPO_TARGETING_ENV_VARS:
        env.pop(key, None)
    env["LC_ALL"] = "C"

    with _deferring_termination_signals():
        completed = subprocess.run(  # noqa: S603 -- argv is a fixed list, not shell-injected
            ["git", *args],  # noqa: S607 -- argv is a fixed list, not shell-injected
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            # Paths, refs, and author names in git output can contain bytes that
            # are not valid UTF-8; the classification and parsing that consumes
            # this output needs a string to work with, not a crash.
            errors="replace",
            env=env,
            # Close stdin so a git command that wants user input (credential
            # fill, interactive rebase) fails immediately with its own
            # diagnostic instead of hanging while the signal mask is held.
            stdin=subprocess.DEVNULL,
        )
    return completed.stdout


def try_git(args: Sequence[str], cwd: str) -> str | None:
    """Run a git command, reporting success or failure instead of raising.

    Args:
        args: Git arguments passed as an argv list.
        cwd: Working directory the git call resolves the repository from.

    Returns:
        ``None`` on success, or the failure's stderr text on failure — callers
        decide whether a failure is a warning, a silent swallow, or an error.
        Never raises: a non-zero exit, a timeout, or a git binary that cannot be
        found or run all surface as text.
    """
    try:
        run_git(args, cwd)
    except (subprocess.SubprocessError, OSError) as error:
        # SubprocessError covers CalledProcessError and TimeoutExpired; OSError
        # covers a git binary that is missing or cannot be executed
        # (FileNotFoundError, PermissionError). try_git reports every failure as
        # text, never raises.
        return stderr_text_of(error)
    return None


class NotAGitRepositoryError(GymratError):
    """A directory git placed outside every repository.

    Its own class because callers act on the distinction: standing outside a
    repository is a supported way to run gymrat, while a git that merely
    declined to answer says nothing about where the directory sits and must
    never be read as "no repository here".
    """


def repository_lookup_error(directory: str, cause: object) -> GymratError:
    """Classify a failed repository lookup by what git said.

    Git reporting that ``directory`` is outside every repository is an answer
    callers can act on. Every other failure — dubious ownership, an unreadable
    ``.git``, a git that cannot run at all — is git declining to answer, and
    carries git's own diagnostics so the reader sees the real reason rather than
    a wrong one.

    Args:
        directory: The directory whose repository lookup failed.
        cause: The failure to classify — typically the git exception.

    Returns:
        A :class:`NotAGitRepositoryError` when git's stderr opens with its
        not-a-repository diagnostic, otherwise a plain :class:`GymratError`
        carrying git's diagnostics.
    """
    diagnostics = stderr_text_of(cause)

    if _NOT_A_REPOSITORY_RE.search(diagnostics):
        return NotAGitRepositoryError(
            f"Not a git repository: {directory}",
            hint="Run gymrat from inside a git repository.",
        )

    return GymratError(f"Cannot determine the git repository at {directory}: {diagnostics}")
