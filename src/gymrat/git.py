"""Git subprocess helpers.

Every git call runs with an argv list rather than a shell string, so refs and
paths containing shell metacharacters are treated as literal git arguments. The
child env is scrubbed of the repo-targeting ``GIT_*`` variables (so an outer git
process cannot redirect the call away from ``cwd``) and pinned to ``LC_ALL=C``
for stable, locale-independent diagnostics that classification can key on.
"""

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence

from gymrat.child_signal_mask import unblock_termination_signals
from gymrat.errors import GymratError
from gymrat.signals import deferring_termination_signals, pthread_sigmask
from gymrat.utils import stderr_text_of

# Env vars an outer git process exports to point child git at a specific repo.
# Removing them forces this call to resolve the repository from ``cwd`` alone.
_REPO_TARGETING_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


def run_git(
    args: Sequence[str],
    cwd: str,
    *,
    env: Mapping[str, str] | None = None,
    allow_interrupt: bool = False,
) -> str:
    """Run git in ``cwd`` and return its untrimmed stdout.

    This process defers its own termination-signal handling until the call
    returns, whatever ``allow_interrupt`` says.

    Args:
        args: Git arguments passed as an argv list, never a shell string.
        cwd: Working directory the git call resolves the repository from.
        env: Extra environment variables merged into the child process after the
            repository-targeting scrub, so a caller can set keys like
            ``GIT_INDEX_FILE`` without them being removed. ``None`` (the
            default) leaves the scrubbed environment unchanged.
        allow_interrupt: Whether a termination signal sent to the process group
            may end git and anything it spawns. Keep it ``False`` for a call
            that must not be cut short, such as a cleanup step: on POSIX that
            keeps the signals blocked in the child. Win32 has no signal mask,
            so the flag changes nothing there.

    Returns:
        The command's stdout, exactly as git wrote it (not trimmed).

    Raises:
        subprocess.CalledProcessError: When git exits non-zero. It carries
            ``.stderr`` and ``.returncode`` for callers to mine.
        OSError: When git is missing from ``PATH`` or cannot be executed.
    """
    child_env = os.environ.copy()
    for key in _REPO_TARGETING_ENV_VARS:
        child_env.pop(key, None)
    child_env["LC_ALL"] = "C"
    if env is not None:
        child_env.update(env)
    # Without ``pthread_sigmask`` (win32) there is no mask to lift, and
    # ``preexec_fn`` is rejected there outright.
    preexec_fn: Callable[[], None] | None = (
        unblock_termination_signals if allow_interrupt and pthread_sigmask is not None else None
    )

    with deferring_termination_signals():
        completed = subprocess.run(  # noqa: S603 -- argv is a fixed list, not shell-injected
            ["git", *args],  # noqa: S607 -- git is resolved from PATH by design; the user's git is the one to run
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            # Paths, refs, and author names in git output can contain bytes that
            # are not valid UTF-8; the classification and parsing that consumes
            # this output needs a string to work with, not a crash.
            errors="replace",
            env=child_env,
            # Close stdin so a git command that wants user input (credential
            # fill, interactive rebase) fails immediately with its own
            # diagnostic instead of hanging while this process defers
            # termination signals.
            stdin=subprocess.DEVNULL,
            preexec_fn=preexec_fn,
        )
    return completed.stdout


def try_git(args: Sequence[str], cwd: str, *, allow_interrupt: bool = False) -> str | None:
    """Run a git command, reporting success or failure instead of raising.

    Args:
        args: Git arguments passed as an argv list.
        cwd: Working directory the git call resolves the repository from.
        allow_interrupt: Whether a termination signal sent to the process group
            may end git, as for :func:`run_git`.

    Returns:
        ``None`` on success, or the failure's stderr text on failure — callers
        decide whether a failure is a warning, a silent swallow, or an error.
        Never raises: a non-zero exit, a subprocess failure, or a git binary that
        cannot be found or run all surface as text.
    """
    try:
        run_git(args, cwd, allow_interrupt=allow_interrupt)
    except (subprocess.SubprocessError, OSError) as error:
        # SubprocessError covers CalledProcessError and TimeoutExpired; OSError
        # covers a git binary that is missing or cannot be executed
        # (FileNotFoundError, PermissionError).
        return stderr_text_of(error)
    return None


def run_git_step(
    args: Sequence[str],
    cwd: str,
    message: str,
    hint: str | None = None,
    *,
    allow_interrupt: bool = False,
) -> str:
    """Run git in ``cwd``, turning any failure into a ``GymratError``.

    The error carries git's own diagnostics after ``message`` so the reader sees
    the real reason, and ``hint`` for what to do next.

    Args:
        args: The git command-line arguments to run.
        cwd: Working directory to run the command in.
        message: The error message prefix used if the command fails.
        hint: The hint attached to the raised error, or ``None`` for none.
        allow_interrupt: Whether a termination signal sent to the process group
            may end git, as for :func:`run_git`.

    Returns:
        The captured stdout from the git command.

    Raises:
        GymratError: When the git command exits non-zero, or when the git
            binary is missing or cannot be executed.
    """
    try:
        return run_git(args, cwd, allow_interrupt=allow_interrupt)
    except (subprocess.SubprocessError, OSError) as error:
        detail = f"{message}: {stderr_text_of(error)}"
        raise GymratError(detail, hint=hint) from error
