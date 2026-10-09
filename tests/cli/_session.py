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
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any, override
from unittest.mock import create_autospec

import pytest
from typer.testing import CliRunner

from gymrat.compare import compare
from gymrat.config import KindEntry, MetricEntry, ResolvedConfig, resolve_config
from gymrat.loop.finalize import finalize_session
from gymrat.loop.start import start_session
from gymrat.measure import MeasureOptions
from gymrat.progress_events import ProgressEvent
from gymrat.report.types import ComparisonResult, MeasurementResult
from gymrat.session.paths import experiment_worktree_dir
from gymrat.session.records import CommandRecord, SessionLogRecord, SessionRecord
from tests._config import resolved_config
from tests._git import commit_all
from tests._streams import RaisingStream
from tests.config._toml import write_config
from tests.loop._probe import BASELINE_SAMPLES, MeasureRecorder, install_measure, measurement
from tests.loop._settle import (
    CHECKS,
    edit_experiment,
    keep_iteration,
    start_with,
)
from tests.report._comparisons import create_comparison_result
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    append_records,
    baseline_record,
    committed_keep,
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


#: The config the stubbed ``measure`` command resolves; its fake engine never benches against it.
MEASURE_CONFIG = resolved_config(
    bench="sh bench.sh",
    samples=5,
    timeout_seconds=30,
    unstable_noise_pct=2.0,
    primary="time",
)

#: The config the stubbed ``compare`` command resolves; its fake engine never benches against it.
COMPARE_CONFIG = resolved_config(
    bench="sh bench.sh",
    prepare="npm ci",
    samples=5,
    timeout_seconds=30,
    unstable_noise_pct=2.0,
    primary="time",
    metrics={"decode/time": MetricEntry(direction="higher")},
    kinds={"memory": KindEntry(gating=False)},
)


def stub_config(
    monkeypatch: pytest.MonkeyPatch, command: str, config: ResolvedConfig
) -> ResolvedConfig:
    """Replace a command's config resolution with one that hands back ``config``.

    The stand-in keeps the real resolver's signature, so a call the real
    ``resolve_config`` would reject fails the test.

    Args:
        monkeypatch: The fixture that installs the stand-in.
        command: The module under ``gymrat.cli.commands`` whose resolver is replaced.
        config: What every resolution hands back.

    Returns:
        ``config``, for a test that asserts against it.
    """
    monkeypatch.setattr(
        f"gymrat.cli.commands.{command}.resolve_config",
        create_autospec(resolve_config, return_value=config),
    )
    return config


def stub_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ``measure`` command's config resolution with ``MEASURE_CONFIG``."""
    stub_config(monkeypatch, "measure", MEASURE_CONFIG)


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


def stub_compare_command(
    monkeypatch: pytest.MonkeyPatch, result: ComparisonResult | None = None
) -> None:
    """Stub both config resolution and the ``compare`` seam, so invoking ``compare`` succeeds.

    Args:
        monkeypatch: The fixture that installs the fakes.
        result: What the fake ``compare`` hands back; a comparison with no
            regressions when ``None``.
    """
    stub_config(monkeypatch, "compare", COMPARE_CONFIG)
    stub_compare(monkeypatch, result)


def stub_compare(monkeypatch: pytest.MonkeyPatch, result: ComparisonResult | None = None) -> None:
    """Replace the ``compare`` seam with a fake that returns a fixed comparison.

    Args:
        monkeypatch: The fixture that installs the fake.
        result: What the fake hands back; a comparison with no regressions when ``None``.
    """
    handed_back = create_comparison_result() if result is None else result
    monkeypatch.setattr(
        "gymrat.compare.compare", create_autospec(compare, return_value=handed_back)
    )


def leave_as_is(_repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange nothing: the no-op row of an ``(repo, monkeypatch)`` arrange table."""


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


def write_settled_session(repo: str, *trailing_records: SessionLogRecord) -> None:
    """Log a configured session with one kept iteration, followed by ``trailing_records``."""
    write_session_log(
        repo, session_record(), (iteration_record(seq=1), committed_keep(1), *trailing_records)
    )
    write_bench_config(repo)


def open_probe_session(repo: str, **config: object) -> None:
    """Open a session on a recorded baseline and write the bench config, ready for probe.

    Args:
        repo: The repository the session opens in.
        **config: Extra ``gymrat.toml`` keys written beside the bench command.
    """
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    write_bench_config(repo, **config)


def stub_probe_measure(
    monkeypatch: pytest.MonkeyPatch, progress: Sequence[ProgressEvent] = ()
) -> MeasureRecorder:
    """Replace the measurement engine with a recorder answering a metric-lines measurement.

    Args:
        monkeypatch: The fixture the engine is patched through.
        progress: Events each call reports through the progress callback it was handed.

    Returns:
        The installed recorder.
    """
    return install_measure(monkeypatch, measurement(adapter="metric-lines"), progress=progress)


def start_edited_session(
    root: str,
    history: tuple[SessionLogRecord, ...] = (iteration_record(seq=1),),
    **config: object,
) -> None:
    """Open a session on ``history``, edit the experiment, and write the config.

    Args:
        root: The repository root.
        history: The records logged after the session header; one unsettled
            iteration by default.
        **config: Extra ``gymrat.toml`` entries beside the bench.
    """
    start_with(root, history)
    edit_experiment(root)
    write_bench_config(root, **config)


def open_unedited_session(root: str) -> None:
    """Open a session with one unsettled iteration and configured checks, and edit nothing."""
    start_with(root, (iteration_record(seq=1),))
    write_bench_config(root, checks=CHECKS)


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
