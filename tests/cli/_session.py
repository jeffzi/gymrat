"""Shared helpers for the CLI command test files.

Builders and stubs used by more than one ``tests/cli`` module: the loop-command
repos and tty stand-ins, the session-log readers, and the ``measure`` seam
stubs.  This is test-support code, not a test module: it carries no test
functions.  Its fixtures (``stop_repo``, ``sync_repo`` and ``_in_non_repo``)
are registered for the directory by ``tests/cli/conftest.py``.
"""

import contextlib
import errno
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, override

import pytest
import tomli_w
from typer.testing import CliRunner

from gymrat.config.types import ResolvedConfig
from gymrat.loop.finalize import finalize_session
from gymrat.loop.start import start_session
from gymrat.measure import MeasureOptions
from gymrat.report.types import MeasurementResult
from gymrat.session.paths import experiment_worktree_dir, session_jsonl_path
from gymrat.session.records import CommandRecord, SessionRecord
from gymrat.session.store import append_record, read_records
from tests._ansi import SGR_RE
from tests._streams import RaisingStream
from tests.loop._probe import install_measure
from tests.loop._settle import git, head_of, iteration, start_with
from tests.loop.iterate._fixtures import resolved_config
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    committed_keep,
    iteration_record,
    session_record,
    write_session_log,
)

runner = CliRunner()

#: How each platform reports a stdout reader that has gone: ``(error, sys.platform)``.
CLOSED_STDOUT_ERRORS = [
    pytest.param(BrokenPipeError(errno.EPIPE, "Broken pipe"), "linux", id="posix-broken-pipe"),
    pytest.param(OSError(errno.EINVAL, "Invalid argument"), "win32", id="windows-einval"),
]


def closed_stdout_error() -> OSError:
    """Build the error a POSIX stdout write raises once its reader has gone.

    Returns:
        A fresh ``BrokenPipeError`` carrying ``EPIPE``.
    """
    return BrokenPipeError(errno.EPIPE, "Broken pipe")


def disk_full_error() -> OSError:
    """Build the error a stdout write raises when the disk is full.

    Returns:
        A fresh ``OSError`` carrying ``ENOSPC`` and its system message.
    """
    return OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


class FailingStdoutRunner(CliRunner):
    """A ``CliRunner`` whose isolated ``sys.stdout`` fails every write with ``error``.

    The runner still captures stderr, so a test can check nothing was reported.

    Args:
        error: The exception every stdout write raises.
    """

    def __init__(self, error: OSError) -> None:
        super().__init__()
        self._error = error

    @override
    @contextlib.contextmanager
    def isolation(self, *args: Any, **kwargs: Any) -> Generator[Any]:
        with super().isolation(*args, **kwargs) as streams:
            # Keep the runner's wrapper alive: collecting it closes the buffer it captures into.
            captured_stdout = sys.stdout
            sys.stdout = RaisingStream(self._error)
            try:
                yield streams
            finally:
                sys.stdout = captured_stdout


class ResolverRecorder:
    """A stand-in for a config resolver recording ``(flags, base_dir)`` per call."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[object, str | Path | None]] = []

    def __call__(self, flags: object, base_dir: str | Path | None = None) -> object:
        self.calls.append((flags, base_dir))
        return self.result


def stub_resolve_config(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> object:
    """Pin what ``start`` reads by replacing its ``resolve_config`` with a fixed config."""
    config = resolved_config(**overrides)

    def fake(*_a: object, **_k: object) -> object:
        return config

    monkeypatch.setattr("gymrat.cli.session_cmds.resolve_config", fake)
    return config


def stub_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ``measure`` command's config resolution with a fixed config."""

    def fake(*_a: object, **_k: object) -> ResolvedConfig:
        # A config the fake ``measure`` never actually benches against.
        return ResolvedConfig(
            bench="sh bench.sh",
            prepare=None,
            adapter="metric-lines",
            samples=5,
            timeout_seconds=30,
            unstable_noise_pct=2.0,
            primary="time",
        )

    monkeypatch.setattr("gymrat.cli.measure_cmd.resolve_config", fake)


def capture_measure(
    monkeypatch: pytest.MonkeyPatch, result: MeasurementResult | None = None
) -> list[MeasureOptions]:
    """Stub the ``measure`` seam and capture the options of each call.

    The fake lets a test pin the label and raw rounds a recording is built from,
    or assert the seam was never reached by checking the returned list stayed
    empty.

    Args:
        monkeypatch: The fixture that installs the fake.
        result: What the fake hands back; a default clean run when ``None``.

    Returns:
        The options of every call, in order; empty if the seam was never reached.
    """
    handed_back = create_measurement_result() if result is None else result
    return install_measure(monkeypatch, handed_back).calls


def stub_measure(
    monkeypatch: pytest.MonkeyPatch, result: MeasurementResult | None = None
) -> list[MeasureOptions]:
    """Stub both config resolution and the ``measure`` seam; return captured options."""
    stub_resolve(monkeypatch)
    return capture_measure(monkeypatch, result)


def plain_lines(text: str) -> list[str]:
    """The non-blank lines of ``text``, stripped of color and surrounding space."""
    return [SGR_RE.sub("", line).strip() for line in text.split("\n") if line.strip()]


def always_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its interactive path."""
    return True


def never_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its non-interactive path."""
    return False


def make_discard_repo(repo: str) -> str:
    """Set up ``repo`` with an open session and one unsettled iteration to discard."""
    start_session(repo, "main", resolved_config())
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    return repo


@pytest.fixture
def _in_non_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run from a directory that is not a git repo, so the command benches lock-free."""
    monkeypatch.chdir(tmp_path)


def open_session(repo: str) -> None:
    """Open a session in ``repo`` so a command has a session log to write to."""
    write_session_log(repo, session_record())


@pytest.fixture
def stop_repo(repo: str) -> str:
    """A repository with a settled, configured session ready for the stop command."""
    start_with(repo, (iteration(1), committed_keep(1)))
    write_config(repo)
    return repo


@pytest.fixture
def sync_repo(repo: str) -> str:
    """A repository with an open session, ready for sync tests."""
    start_session(repo, "main", resolved_config())
    return repo


def open_session_with_one_keep(root: str) -> SessionRecord:
    """Open a session, commit and log one kept iteration, and return the session header."""
    start_session(root, "main", resolved_config())
    worktree = experiment_worktree_dir(root)
    (Path(worktree) / "step.txt").write_text("cache the regex\n", encoding="utf-8")
    git(["add", "-A"], worktree)
    git(["commit", "-m", "cache the regex"], worktree)
    commit = head_of(worktree)
    append_record(session_jsonl_path(root), iteration_record(seq=1))
    append_record(session_jsonl_path(root), committed_keep(1, commit=commit))
    header = read_records(session_jsonl_path(root))[0]
    assert isinstance(header, SessionRecord)
    return header


def close_session_with_one_keep(root: str) -> str:
    """Open a session with one kept commit, finalize it, and return its closed id."""
    header = open_session_with_one_keep(root)
    finalize_session(root)
    return header.session_id


def last_command_record(root: str) -> CommandRecord:
    """Read the session log and return the last ``CommandRecord``.

    Raises ``AssertionError`` when the log contains no command record.
    """
    records = read_records(session_jsonl_path(root))
    for record in reversed(records):
        if isinstance(record, CommandRecord):
            return record
    msg = "no CommandRecord found in session log"
    raise AssertionError(msg)


def records_of(repo: str, *, commands: bool) -> list[object]:
    """The session-log records that are (or are not) command traces."""
    return [
        r
        for r in read_records(session_jsonl_path(repo))
        if isinstance(r, CommandRecord) is commands
    ]


def write_config(root: str, **extra: object) -> None:
    """Write the implicit ``gymrat.toml`` at the repository root."""
    payload: dict[str, object] = {"bench": "npm run bench", **extra}
    (Path(root) / "gymrat.toml").write_text(tomli_w.dumps(payload), encoding="utf-8")
