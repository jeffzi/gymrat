"""Shared helpers for the CLI command test files.

Builders and stubs used by more than one ``tests/cli`` module: the loop-command
repos and tty stand-ins, the session-log readers, and the ``measure`` and
``compare`` seam stubs.  This is test-support code, not a test module: it carries no test
functions.
"""

import contextlib
import errno
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, override

import pytest
from typer.testing import CliRunner

from gymrat.config import ResolvedConfig
from gymrat.loop.finalize import finalize_session
from gymrat.loop.start import start_session
from gymrat.measure import MeasureOptions
from gymrat.report.types import ComparisonResult, MeasurementResult
from gymrat.session.paths import experiment_worktree_dir
from gymrat.session.records import CommandRecord, SessionRecord
from tests._config import resolved_config
from tests._git import commit_all
from tests._streams import RaisingStream
from tests.config._toml import write_config
from tests.loop._probe import install_measure
from tests.loop._settle import (
    keep_iteration,
    start_with,
)
from tests.report._comparisons import create_comparison_result
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    append_records,
    iteration_record,
    log_records,
    session_header_of,
    session_record,
    write_session_log,
)

runner = CliRunner()


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

    monkeypatch.setattr("gymrat.cli.commands.session.resolve_config", fake)
    return config


def stub_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ``measure`` command's config resolution with a fixed config."""

    def fake(*_a: object, **_k: object) -> ResolvedConfig:
        # A config the fake ``measure`` never actually benches against.
        return resolved_config(
            bench="sh bench.sh",
            samples=5,
            timeout_seconds=30,
            unstable_noise_pct=2.0,
            primary="time",
        )

    monkeypatch.setattr("gymrat.cli.commands.measure.resolve_config", fake)


def capture_measure(
    monkeypatch: pytest.MonkeyPatch, result: MeasurementResult | None = None
) -> list[MeasureOptions]:
    """Stub the ``measure`` seam and capture the options of each call.

    The fake lets a test pin the label and raw rounds a recording is built from.

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


def stub_compare(monkeypatch: pytest.MonkeyPatch, result: ComparisonResult | None = None) -> None:
    """Replace the ``compare`` seam with a fake that returns a fixed comparison.

    Args:
        monkeypatch: The fixture that installs the fake.
        result: What the fake hands back; a comparison with no regressions when ``None``.
    """
    handed_back = create_comparison_result() if result is None else result

    async def fake_compare(_options: object) -> ComparisonResult:
        return handed_back

    monkeypatch.setattr("gymrat.compare.compare", fake_compare)


def never_tty(_stream: object) -> bool:
    """Stand in for ``is_tty`` so the discard command takes its non-interactive path."""
    return False


def make_discard_repo(repo: str) -> str:
    """Set up ``repo`` with an open session and one unsettled iteration to discard."""
    start_session(repo, "main", resolved_config())
    append_records(repo, iteration_record(seq=1))
    return repo


def open_stop_ready_session(repo: str) -> None:
    """Open a configured session with one settled iteration, ready for the stop command."""
    start_with(repo)
    keep_iteration(repo, 1)
    write_bench_config(repo)


def open_session(repo: str) -> None:
    """Open a session in ``repo`` so a command has a session log to write to."""
    write_session_log(repo, session_record())


def open_session_with_one_keep(root: str) -> SessionRecord:
    """Open a session, commit and log one kept iteration, and return the session header."""
    start_session(root, "main", resolved_config())
    commit = commit_all(experiment_worktree_dir(root), "cache the regex", file="step.txt")
    keep_iteration(root, 1, commit=commit)
    return session_header_of(root)


def close_session_with_one_keep(root: str) -> str:
    """Open a session with one kept commit, finalize it, and return its closed id."""
    header = open_session_with_one_keep(root)
    finalize_session(root)
    return header.session_id


def last_command_record(root: str) -> CommandRecord:
    """Read the session log and return the last ``CommandRecord``.

    Args:
        root: The repository whose session log is read.

    Returns:
        The last command record in the log.

    Raises:
        AssertionError: The log holds no command record.
    """
    records = log_records(root)
    for record in reversed(records):
        if isinstance(record, CommandRecord):
            return record
    msg = "no CommandRecord found in session log"
    raise AssertionError(msg)


def write_bench_config(root: str, **extra: object) -> None:
    """Write the implicit ``gymrat.toml`` at the repository root, naming a bench command."""
    write_config(Path(root), {"bench": "npm run bench", **extra})
