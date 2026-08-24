"""Git subprocess helpers and repository-lookup classification.

Every git call runs with an argv list rather than a shell string, so refs and
paths containing shell metacharacters are treated as literal git arguments. The
child env is scrubbed of the repo-targeting ``GIT_*`` variables (so an outer git
process cannot redirect the call away from ``cwd``) and pinned to ``LC_ALL=C``
for stable, locale-independent diagnostics that classification can key on.
"""

import os
import re
import subprocess
from collections.abc import Sequence

from gymrat_py.errors import GymratError, stderr_text_of

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

    completed = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
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
