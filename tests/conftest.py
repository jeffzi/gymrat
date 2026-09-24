"""Shared scratch-repository fixtures for target and worktree tests.

These fixtures build throwaway git repositories in the system temp directory,
each in its own ``tempfile.mkdtemp`` slot resolved through ``os.path.realpath``
(so macOS ``/var`` → ``/private/var`` matches what git reports in
``worktree list``). Every repository is order-independent and safe under
``pytest-xdist`` / ``pytest-randomly``.

The helpers expose a common fixture surface so later worktree and driver
tests can reuse the same building blocks.
"""

import contextlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from filelock import FileLock

from gymrat.cli.shared import set_color_override
from gymrat.session.clock import now_iso
from gymrat.session.lock import _os_lock_file
from gymrat.session.paths import lockfile_path, supervise_lockfile_path
from gymrat.signals import TERMINATION_SIGNALS
from gymrat.signals import reset as signals_reset
from tests._git import run_git as _run_git

#: Every environment variable a test must not inherit from the developer's
#: shell — the GYMRAT_* set the config resolver reads, plus the OTLP endpoint
#: configure_tracing reads.
SCRUBBED_ENV_VARS = (
    "GYMRAT_BENCH",
    "GYMRAT_PREPARE",
    "GYMRAT_ADAPTER",
    "GYMRAT_SAMPLES",
    "GYMRAT_TIMEOUT",
    "GYMRAT_CONFIG",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)


def hold_lock(
    lock_path: str, command: str = "measure", *, holder: dict[str, object] | None = None
) -> FileLock:
    """Acquire a real OS lock on ``lock_path`` and stamp it with holder JSON.

    Simulates another live process holding the repository lock, so a rival
    ``acquire_lock`` call sees contention.  The OS lock lives on
    ``lock_path + ".lock"`` — the same layout ``acquire_lock`` uses — so the
    holder JSON at ``lock_path`` stays readable on Windows where ``LockFileEx``
    blocks reads through a separate handle.

    Returns the acquired ``FileLock`` so the caller can release it during
    teardown.  Pass ``holder`` to stamp an exact record; otherwise one is
    built from ``command`` and the current process.
    """
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(_os_lock_file(lock_path), timeout=0)
    lock.acquire()
    if holder is None:
        holder = {"pid": os.getpid(), "command": command, "at": now_iso()}
    Path(lock_path).write_text(json.dumps(holder), encoding="utf-8")
    return lock


@pytest.fixture(autouse=True)
def _restore_signal_dispositions() -> Iterator[None]:
    """Restore termination-signal dispositions after every test."""
    saved = {sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS}
    yield
    if any(signal.getsignal(sig) is not handler for sig, handler in saved.items()):
        # Un-wiring the OS dispositions alone would strand the module's
        # installed-signals bookkeeping: the next install would then no-op and
        # leave a real signal on the default handler. reset() is the sanctioned
        # seam that clears that state alongside the registry.
        signals_reset()
        for sig, handler in saved.items():
            signal.signal(sig, handler)


@pytest.fixture(autouse=True)
def _clear_gymrat_env() -> Iterator[None]:
    """Remove every GYMRAT_* and OTLP env var for the duration of the test."""
    # A private MonkeyPatch context rather than the `monkeypatch` fixture: an
    # autouse dependency on `monkeypatch` would reorder its teardown after
    # module-level autouse cleanups, running them under still-active patches.
    with pytest.MonkeyPatch.context() as patcher:
        for var in SCRUBBED_ENV_VARS:
            patcher.delenv(var, raising=False)
        yield


@pytest.fixture(autouse=True)
def _reset_color() -> Iterator[None]:
    """Reset the color override before and after every test."""
    set_color_override(None)
    yield
    set_color_override(None)


@pytest.fixture
def stray_process_ids() -> Iterator[list[int]]:
    """Collect PIDs a test spawned and SIGKILL any still alive on teardown."""
    process_ids: list[int] = []
    yield process_ids

    for pid in process_ids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def _init_scratch_repo() -> str:
    """Create one temporary git repo on ``main`` with a single committed file."""
    directory = os.path.realpath(tempfile.mkdtemp(prefix="gymrat-test-"))
    try:
        _run_git(["init", "-b", "main"], directory)
        for key, value in (
            ("user.name", "Test User"),
            ("user.email", "test@example.com"),
            ("commit.gpgsign", "false"),
            ("core.autocrlf", "false"),
        ):
            _run_git(["config", key, value], directory)
        (Path(directory) / "README.md").write_text("# Test Repo\n", encoding="utf-8")
        _run_git(["add", "README.md"], directory)
        _run_git(["commit", "-m", "Initial commit"], directory)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return directory


def _list_worktree_dirs(repo_dir: str, *, include_main: bool = True) -> list[str]:
    """Directories git currently lists as worktrees of ``repo_dir``.

    ``git worktree remove`` clears a worktree's registry entry itself, so this
    only reveals pruning behavior when a directory vanished behind git's back.
    With ``include_main=False`` the main worktree's own directory is dropped;
    git prints resolved paths, so the main directory is matched through
    ``os.path.realpath``.
    """
    output = _run_git(["worktree", "list", "--porcelain"], repo_dir)
    dirs = [
        os.path.normpath(line[len("worktree ") :])
        for line in output.split("\n")
        if line.startswith("worktree ")
    ]
    if not include_main:
        main_dir = os.path.realpath(repo_dir)
        return [directory for directory in dirs if directory != main_dir]
    return dirs


def _wait_for_worktrees(repo_dir: str, count: int, timeout_s: float = 30.0) -> list[str]:
    """Poll until ``repo_dir`` lists at least ``count`` linked worktrees.

    Only for polling while another process is still adding or removing
    worktrees: ``git worktree list`` reads each registry entry non-atomically
    and exits 128 when it meets one half-written by a concurrent ``git worktree
    add`` or half-cleared by a ``remove`` — a file of the entry it expects is
    not there. Such a read counts as "not there yet". Once no writer runs,
    call :func:`_list_worktree_dirs` directly so a real failure stays loud.

    Args:
        repo_dir: The main worktree whose registry is polled.
        count: The minimum number of linked worktrees to wait for.
        timeout_s: How long to poll before giving up.

    Returns:
        The linked worktree directories of the first read that reached ``count``.

    Raises:
        AssertionError: When ``count`` is not reached within ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    listed: list[str] = []
    while True:
        with contextlib.suppress(subprocess.CalledProcessError):
            listed = _list_worktree_dirs(repo_dir, include_main=False)
        if len(listed) >= count:
            return listed
        if time.monotonic() > deadline:
            message = f"expected >= {count} worktrees within {timeout_s}s, saw {listed}"
            raise AssertionError(message)
        time.sleep(0.05)


def _remove_stranded_worktrees(repo_dir: str) -> None:
    """Delete worktree directories a run stranded, keeping temp dirs clean."""
    try:
        stranded = _list_worktree_dirs(repo_dir, include_main=False)
    except subprocess.CalledProcessError:
        return
    for directory in stranded:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def create_scratch_repo() -> Iterator[Callable[[], str]]:
    """Factory yielding fresh scratch repositories, all cleaned up on teardown."""
    created: list[str] = []
    original_cwd = Path.cwd()

    def factory() -> str:
        directory = _init_scratch_repo()
        created.append(directory)
        return directory

    try:
        yield factory
    finally:
        if Path.cwd() != original_cwd and original_cwd.is_dir():
            os.chdir(original_cwd)
        for directory in created:
            _remove_stranded_worktrees(directory)
            shutil.rmtree(directory, ignore_errors=True)
            # Lock files persist after release (filelock preserves the file);
            # every process using them is dead by cleanup time.
            for lock in (lockfile_path(directory), supervise_lockfile_path(directory)):
                Path(lock).unlink(missing_ok=True)


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch) -> str:
    """A fresh scratch repository, chdir'd into so the command runs there."""
    root = create_scratch_repo()
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def supervise_lock(repo: str) -> Iterator[None]:
    """Hold the real supervise lock for ``repo`` for the duration of the test."""
    lock = hold_lock(supervise_lockfile_path(repo), "supervise")
    yield
    lock.release()


@pytest.fixture
def list_worktree_dirs() -> Callable[..., list[str]]:
    """Expose the worktree-listing helper to tests."""
    return _list_worktree_dirs


@pytest.fixture
def wait_for_worktrees() -> Callable[..., list[str]]:
    """Expose the race-tolerant worktree poller to tests."""
    return _wait_for_worktrees


@pytest.fixture
def kill_git_during_worktree_add() -> Callable[[str], None]:
    """Install a post-checkout hook that kills git once a worktree is on disk."""

    def install(repo_dir: str) -> None:
        hook_path = Path(repo_dir) / ".git" / "hooks" / "post-checkout"
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text(
            '#!/bin/sh\nexec >/dev/null 2>&1\nkill -9 "$PPID"\nsleep 1\n',  # cspell:disable-line
            encoding="utf-8",
        )
        hook_path.chmod(0o755)

    return install


@pytest.fixture
def register_absent_worktree() -> Callable[[str], str]:
    """Register a worktree of a repo the way a user would, then delete its dir."""

    def register(repo_dir: str) -> str:
        directory = str(Path(os.path.realpath(repo_dir)) / "absent-user-worktree")
        _run_git(["worktree", "add", "--detach", directory, "HEAD"], repo_dir)
        shutil.rmtree(directory, ignore_errors=True)
        return directory

    return register


@pytest.fixture
def create_in_place_target_dir() -> Callable[[str, str, str], str]:
    """Write a bench script into a plain subdirectory of a repo."""

    def create(repo_dir: str, name: str, bench_script: str) -> str:
        target = Path(repo_dir) / name
        target.mkdir()
        (target / "bench.sh").write_text(bench_script, encoding="utf-8")
        return str(target)

    return create
