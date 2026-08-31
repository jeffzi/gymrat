"""Shared scratch-repository fixtures for target and worktree tests.

These fixtures build throwaway git repositories in the system temp directory,
each in its own ``tempfile.mkdtemp`` slot resolved through ``os.path.realpath``
(so macOS ``/var`` → ``/private/var`` matches what git reports in
``worktree list``). Every repository is order-independent and safe under
``pytest-xdist`` / ``pytest-randomly``.

The helpers expose a common fixture surface so later worktree and driver
tests can reuse the same building blocks.
"""

import json
import os
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from filelock import FileLock

from gymrat.cli.shared import set_stderr_color_override
from gymrat.session.clock import now_iso
from gymrat.session.paths import lockfile_path, supervise_lockfile_path
from gymrat.signals import TERMINATION_SIGNALS
from gymrat.signals import reset as signals_reset
from tests._git import run_git as _run_git

#: Every GYMRAT_* variable the config resolver reads; cleared before each
#: test so ambient developer environment never leaks into expectations.
GYMRAT_ENV_VARS = (
    "GYMRAT_BENCH",
    "GYMRAT_PREPARE",
    "GYMRAT_ADAPTER",
    "GYMRAT_SAMPLES",
    "GYMRAT_TIMEOUT",
    "GYMRAT_CONFIG",
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
    lock = FileLock(lock_path + ".lock", timeout=0)
    lock.acquire()
    if holder is None:
        holder = {"pid": os.getpid(), "command": command, "at": now_iso()}
    Path(lock_path).write_text(json.dumps(holder), encoding="utf-8")
    return lock


@pytest.fixture(autouse=True)
def _restore_signal_dispositions() -> Iterator[None]:
    """Restore termination-signal dispositions after every test.

    Any test that reaches the real ``install_termination_cleanup`` — directly
    or through a live renderer — leaves the gymrat handler installed for the
    rest of the worker's life; a later Ctrl-C would then ``os._exit`` the
    worker and skip every remaining teardown. Restoring the saved dispositions
    keeps pytest's own interrupt handling in charge between tests.
    """
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
    # A private MonkeyPatch context rather than the `monkeypatch` fixture: an
    # autouse dependency on `monkeypatch` would reorder its teardown after
    # module-level autouse cleanups, running them under still-active patches.
    with pytest.MonkeyPatch.context() as patcher:
        for var in GYMRAT_ENV_VARS:
            patcher.delenv(var, raising=False)
        yield


@pytest.fixture(autouse=True)
def _reset_stderr_color() -> Iterator[None]:
    set_stderr_color_override(None)
    yield
    set_stderr_color_override(None)


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
    """Factory yielding fresh scratch repositories, all cleaned up on teardown.

    The returned callable can be invoked several times; every repository it
    hands back is torn down together, and any worktree stranded in the system
    temp directory is swept before the repositories are removed. If a test
    changed into a repository, the working directory is restored first (a
    directory that is a process's cwd cannot always be removed).
    """
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
def list_worktree_dirs() -> Callable[..., list[str]]:
    """Expose the worktree-listing helper to tests."""
    return _list_worktree_dirs


@pytest.fixture
def kill_git_during_worktree_add() -> Callable[[str], None]:
    """Install a post-checkout hook that kills git once a worktree is on disk.

    Reproduces the one state a run can strand: a worktree on disk whose
    ``git worktree add`` never returned success. ``post-checkout`` fires after
    git has laid the worktree down and registered it, so what survives the kill
    is a complete worktree, not a half-written one. POSIX-only.
    """

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
    """Register a worktree of a repo the way a user would, then delete its dir.

    Stands in for a user's own worktree that is only temporarily absent. Git
    keeps listing it until ``git worktree prune`` runs, which makes its entry a
    probe for whether a sweep reached past the worktrees it was asked about.
    Returns the resolved path git lists the worktree under.
    """

    def register(repo_dir: str) -> str:
        directory = str(Path(os.path.realpath(repo_dir)) / "absent-user-worktree")
        _run_git(["worktree", "add", "--detach", directory, "HEAD"], repo_dir)
        shutil.rmtree(directory, ignore_errors=True)
        return directory

    return register


@pytest.fixture
def create_in_place_target_dir() -> Callable[[str, str, str], str]:
    """Write a bench script into a plain subdirectory of a repo.

    ``resolve_target`` sees a plain directory and returns an in-place target,
    so benching it never creates a worktree. Returns the created directory.
    """

    def create(repo_dir: str, name: str, bench_script: str) -> str:
        target = Path(repo_dir) / name
        target.mkdir()
        (target / "bench.sh").write_text(bench_script, encoding="utf-8")
        return str(target)

    return create
