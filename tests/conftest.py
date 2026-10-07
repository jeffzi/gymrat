"""Shared test environment baseline and scratch-repository fixtures.

Every test runs against one explicit environment. ``pytest_configure``
rebuilds ``os.environ`` from an allowlist before any test module is imported,
in the controller and in every ``pytest-xdist`` worker, so nothing the
developer's shell exports reaches code under test or the children it spawns;
an autouse fixture puts that baseline back after every test.

The scratch-repository fixtures build throwaway git repositories in the system
temp directory, each in its own ``tempfile.mkdtemp`` slot resolved through
``os.path.realpath`` (so macOS ``/var`` → ``/private/var`` matches what git
reports in ``worktree list``). Every repository is order-independent and safe
under ``pytest-xdist`` / ``pytest-randomly``.
"""

import contextlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NoReturn

import pytest

from gymrat import signals
from gymrat.cli.console import set_color_override, set_debug_mode
from gymrat.exec import reset as exec_reset
from gymrat.signals import TERMINATION_SIGNALS
from gymrat.signals import reset as signals_reset
from gymrat.telemetry.provider import reset_tracing
from tests._git import head_of, init_scratch_repo, list_worktree_dirs
from tests._lock import remove_lock_files

#: Names a test may inherit from the launching shell, value unchanged. An
#: allowlisted name the shell does not export stays absent. ``PATH`` and
#: ``VIRTUAL_ENV`` come from ``uv run`` and make ``sys.executable`` children
#: and ``git`` resolvable; ``CI`` selects Hypothesis's ``ci`` profile and
#: pytest's untruncated assertion output; the last four are the Windows
#: process set a child Python, ``expanduser``, ``shell=True`` and
#: ``shutil.which`` need.
BASELINE_ENV_NAMES = frozenset({
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USER",
    "LOGNAME",
    "SHELL",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "CI",
    "SYSTEMROOT",
    "USERPROFILE",
    "PATHEXT",
    "COMSPEC",
})

#: Prefixes of inherited names kept alongside ``BASELINE_ENV_NAMES``: uv's
#: own settings, pytest's and its plugins' (xdist marks its workers with
#: ``PYTEST_XDIST_*``), and coverage's.
BASELINE_ENV_PREFIXES = ("UV_", "PYTEST_", "COVERAGE_")

#: Names every test sees with a fixed value, whatever the shell exports. The
#: git pair hides the developer's global and system config, so a setting such
#: as ``tag.gpgSign`` cannot change what a plain ``git tag`` does in a scratch
#: repo.
PINNED_ENV = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _live_environ() -> dict[str, str]:
    """Return the process's real environment block.

    ``os.environ`` is a snapshot taken at interpreter start, and native code can
    write the real environment behind it: pytest imports ``readline`` before any
    ``pytest_configure``, and GNU readline then exports ``LINES`` and
    ``COLUMNS``. A child inherits the real block, so it reports those names too.
    """
    probe = "import json, os; print(json.dumps(dict(os.environ)))"
    output = subprocess.run(  # noqa: S603 -- argv is sys.executable with a fixed probe, not shell-injected
        [sys.executable, "-I", "-S", "-c", probe], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(output)


def pytest_configure() -> None:
    """Rebuild the environment as the allowlisted baseline before collection.

    Runs in the controller and again in every xdist worker, which inherits the
    rebuilt environment; a second rebuild keeps it unchanged. ``os.environ`` is
    first brought in line with the real environment, so names written behind it
    are cleared with the rest and no child process inherits them. Also imports
    ``typer.rich_utils`` under that baseline, so typer's once-per-process
    terminal detection never sees the shell or an earlier test's variables.
    """
    os.environ.update({**_live_environ(), **os.environ})
    kept = {
        name: value
        for name, value in os.environ.items()
        if name in BASELINE_ENV_NAMES or name.startswith(BASELINE_ENV_PREFIXES)
    }
    os.environ.clear()
    os.environ.update(kept | PINNED_ENV)
    # typer.rich_utils decides once, at first import, whether to force a
    # terminal from FORCE_COLOR, PY_COLORS and GITHUB_ACTIONS. Importing it
    # under the baseline keeps a test that sets FORCE_COLOR before that first
    # import from forcing a terminal for every later test in its worker.
    importlib.import_module("typer.rich_utils")


@pytest.fixture(autouse=True)
def _restore_signal_dispositions() -> Iterator[None]:
    """Restore termination-signal dispositions after every test."""
    saved = {sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS}
    yield
    # Un-wiring the OS dispositions alone would strand the module's
    # installed-signals bookkeeping: the next install would then no-op and
    # leave a real signal on the default handler. reset() is the sanctioned
    # seam that clears that state alongside the registry, and it also drops a
    # cleanup a test registered without installing the handlers.
    signals_reset()
    for sig, handler in saved.items():
        signal.signal(sig, handler)


class _ProcessExitedError(BaseException):
    """Raised by the stubbed exit seam so a handler unwinds where it would exit.

    A ``BaseException``, like the ``SystemExit`` a real exit is closest to, so
    no ``except Exception`` between the exit and the test swallows it.
    """

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"exit_process({code})")


@pytest.fixture
def forbid_direct_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test when the process exit is reached other than through a replaced seam."""
    # The termination handler must exit through `signals.exit_process`, the seam
    # tests replace. A handler that calls `os._exit` itself, or that kept the
    # real seam bound from before the replacement, would otherwise take the test
    # worker down with no assertion naming the bypass.

    def bypassed(code: int) -> NoReturn:
        pytest.fail(f"os._exit({code}) reached without going through the replaced exit seam")

    monkeypatch.setattr(os, "_exit", bypassed)


@pytest.fixture
def raise_signal(monkeypatch: pytest.MonkeyPatch, forbid_direct_exit: None) -> Callable[[int], int]:
    """Stub the process exit and return a helper that runs the installed signal handler."""
    # Emitting a real signal would take the test runner down, so the helper
    # fetches the handler via signal.getsignal and calls it directly. With the
    # exit seam stubbed to raise, the handler unwinds exactly where the real one
    # would exit, and the helper reports the exit code.
    #
    # A signal delivered while another is being handled (a call made from
    # inside a cleanup) ends the process for both: its exit unwinds through the
    # outer handler, and the outermost call reports the code.
    depth = 0

    def fake_exit(code: int) -> None:
        raise _ProcessExitedError(code)

    monkeypatch.setattr(signals, "exit_process", fake_exit)

    def _raise(signal_number: int) -> int:
        nonlocal depth
        handler = signal.getsignal(signal_number)
        if not callable(handler):
            pytest.fail(f"no handler installed for signal {signal_number}")
        depth += 1
        try:
            handler(signal_number, None)
        except _ProcessExitedError as exited:
            if depth > 1:
                raise
            return exited.code
        finally:
            depth -= 1
        pytest.fail("handler returned instead of exiting")

    return _raise


@pytest.fixture(autouse=True)
def _restore_env_baseline() -> Iterator[None]:
    """Put ``os.environ`` back to its pre-test contents after every test."""
    # A plain snapshot rather than a MonkeyPatch context, which would only undo
    # its own calls and miss a direct `os.environ` write. No dependency on the
    # `monkeypatch` fixture either: that would reorder its teardown after
    # module-level autouse cleanups, running them under still-active patches.
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _reset_color() -> Iterator[None]:
    """Reset the color override before and after every test."""
    set_color_override(None)
    yield
    set_color_override(None)


@pytest.fixture(autouse=True)
def isolate_tracing_provider() -> Iterator[None]:
    """Start and end every test with no tracing provider."""
    reset_tracing()
    yield
    reset_tracing()


@pytest.fixture(autouse=True)
def isolate_live_groups() -> Iterator[None]:
    """Start and end every test with an empty live process-group registry."""
    exec_reset()
    yield
    exec_reset()


@pytest.fixture(autouse=True)
def _reset_debug() -> Iterator[None]:
    """Turn debug mode off before and after every test."""
    set_debug_mode(False)
    yield
    set_debug_mode(False)


@pytest.fixture
def stdin_holding_text() -> Iterator[None]:
    """Point this process's stdin at a pipe holding text, so a child inheriting it reads that text."""
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"banana\n")
    os.close(write_fd)
    saved_stdin = os.dup(0)
    os.dup2(read_fd, 0)
    os.close(read_fd)
    yield
    os.dup2(saved_stdin, 0)
    os.close(saved_stdin)


@pytest.fixture
def stray_process_ids() -> Iterator[list[int]]:
    """Collect PIDs a test spawned and SIGKILL any still alive on teardown."""
    process_ids: list[int] = []
    yield process_ids

    for pid in process_ids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


@pytest.fixture
def reap_groups() -> Iterator[list[int]]:
    """Track process-group leaders and hard-kill any survivor on teardown."""
    leaders: list[int] = []
    try:
        yield leaders
    finally:
        for pid in leaders:
            # A leader that already exited still names its group, so a group whose
            # leader died while a child lives on is reaped by the leader's pid.
            try:
                group = os.getpgid(pid)
            except ProcessLookupError:
                group = pid
            with contextlib.suppress(OSError):
                os.killpg(group, signal.SIGKILL)


_SCRATCH_PREFIX = "gymrat-test-"


def _remove_stranded_worktrees(repo_dir: str) -> None:
    """Delete worktree directories a run stranded, keeping temp dirs clean.

    A test may move or delete its scratch repository; git then has no registry
    left to list, so the repository is skipped rather than failing teardown for
    every repository after it. Starting git in the vanished directory fails with
    ``FileNotFoundError`` on POSIX and ``NotADirectoryError`` on Windows.
    """
    try:
        stranded = list_worktree_dirs(repo_dir, include_main=False)
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return
    for directory in stranded:
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def create_scratch_repo() -> Iterator[Callable[..., str]]:
    """Factory yielding fresh scratch repositories, all cleaned up on teardown."""
    created: list[str] = []
    original_cwd = Path.cwd()

    def factory(prefix: str = _SCRATCH_PREFIX) -> str:
        directory = init_scratch_repo(prefix)
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
            # Every process using the locks is dead by cleanup time.
            remove_lock_files(directory)


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch) -> str:
    """A fresh scratch repository, chdir'd into so the command runs there."""
    root = create_scratch_repo()
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def repo_head(repo: str) -> str:
    """The commit SHA ``repo`` starts at."""
    return head_of(repo)
