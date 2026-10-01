"""Shared helpers for the CLI command test files.

Builders and stubs used by more than one ``tests/cli`` module: the loop-command
repos and tty stand-ins, the session-log readers, and the ``measure`` seam
stubs.  This is test-support code, not a test module: it carries no test
functions or pytest fixtures of its own.
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

from gymrat.config import ResolvedConfig
from gymrat.measure import MeasureOptions
from gymrat.report.types import MeasurementResult
from gymrat.session import CommandRecord, read_records, session_jsonl_path
from tests._ansi import SGR_RE, strip_ansi
from tests._streams import RaisingStream
from tests.report._inputs import create_measurement_result

__all__ = [
    "CLOSED_STDOUT_ERRORS",
    "always_tty",
    "capture_measure",
    "closed_stdout_error",
    "closed_stdout_runner",
    "disk_full_error",
    "last_command_record",
    "make_discard_repo",
    "make_stop_repo",
    "never_tty",
    "plain_lines",
    "records_of",
    "runner",
    "strip_ansi",
    "stub_measure",
    "stub_resolve",
    "write_config",
]

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


class _FailingStdoutRunner(CliRunner):
    """A ``CliRunner`` whose isolated ``sys.stdout`` fails every write with ``error``."""

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


def closed_stdout_runner(error: OSError) -> CliRunner:
    """Build a runner whose commands see every stdout write fail with ``error``.

    The runner still captures stderr, so a test can check nothing was reported.

    Args:
        error: The exception every stdout write raises.

    Returns:
        A ``CliRunner`` whose isolated ``sys.stdout`` raises ``error`` on write.
    """
    return _FailingStdoutRunner(error)


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
    captured: list[MeasureOptions] = []
    handed_back = create_measurement_result() if result is None else result

    async def fake_measure(options: MeasureOptions) -> MeasurementResult:
        captured.append(options)
        return handed_back

    monkeypatch.setattr("gymrat.measure.measure", fake_measure)
    return captured


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
    from gymrat.loop.start import start_session
    from gymrat.session import append_record, session_jsonl_path
    from tests.loop.iterate._fixtures import resolved_config
    from tests.session.records._fixtures import iteration_record

    start_session(repo, "main", resolved_config())
    append_record(session_jsonl_path(repo), iteration_record(seq=1))
    return repo


def make_stop_repo(repo: str) -> str:
    """Set up ``repo`` with a settled, configured session ready for the stop command."""
    from tests.loop._settle import iteration, start_with
    from tests.session.records._fixtures import committed_keep

    start_with(repo, (iteration(1), committed_keep(1)))
    write_config(repo)
    return repo


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
