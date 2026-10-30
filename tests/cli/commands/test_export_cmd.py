"""Tests for the ``gymrat export`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` against a
local OTLP collector, on session logs written to a temporary directory. They cover
the argument/option contract, error exits, and the success path with its printed
summary. Which session-log lines and supervisor logs replay skips is pinned in
``tests/telemetry/test_replay.py``.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from gymrat.cli.app import app
from gymrat.cli.exit import BUGS_URL
from gymrat.session.paths import session_jsonl_path
from gymrat.utils import ENDPOINT_ENV
from tests._cli import combined_output, stderr_text
from tests.cli._runner import (
    runner,
)
from tests.session.records._fixtures import SESSION_ID
from tests.telemetry._collector import otlp_collector
from tests.telemetry._fixtures import arm_placeholder_endpoint, hide_otlp_exporter
from tests.telemetry._replay_logs import write_measure_command_run

if TYPE_CHECKING:
    import pytest

_TIMEOUT_ENV = "OTEL_EXPORTER_OTLP_TIMEOUT"
_SHORT_TIMEOUT_SECONDS = "0.5"
_UNREACHABLE_ENDPOINT = "http://127.0.0.1:1"


def _populate_session_dir(root: str, *, session_id: str = SESSION_ID) -> str:
    """Write a session log and a matching supervisor log under ``root``.

    Args:
        root: The repository root the session log goes under.
        session_id: The session both logs belong to.

    Returns:
        The session log path.
    """
    session_log = session_jsonl_path(root)
    sup_log = str(Path(session_log).parent / "supervisor-001.jsonl")
    write_measure_command_run(session_log, sup_log, session_id=session_id)
    return session_log


# ---------------------------------------------------------------------------
# missing SDK
# ---------------------------------------------------------------------------


def test_export_when_tracing_package_not_importable_does_exit_two_naming_otel_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    arm_placeholder_endpoint(monkeypatch)
    hide_otlp_exporter(monkeypatch)

    result = runner.invoke(app, ["export", session_log])

    output = stderr_text(result)
    assert result.exit_code == 2, output
    assert "'gymrat[otel]'" in output


# ---------------------------------------------------------------------------
# tracer records nothing
# ---------------------------------------------------------------------------


def test_export_when_tracer_records_nothing_does_exit_two_without_reporting_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    with otlp_collector() as collector:
        monkeypatch.setenv(ENDPOINT_ENV, collector.endpoint)

        result = runner.invoke(app, ["export", session_log])

    output = stderr_text(result)
    assert result.exit_code == 2, output
    assert "No spans recorded" in output


# ---------------------------------------------------------------------------
# missing endpoint
# ---------------------------------------------------------------------------


def test_export_when_endpoint_missing_does_exit_two_naming_env_var(tmp_path: Path):
    session_log = _populate_session_dir(str(tmp_path))

    result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 2
    assert ENDPOINT_ENV in stderr_text(result)


# ---------------------------------------------------------------------------
# missing or unreadable session log
# ---------------------------------------------------------------------------


def test_export_when_session_log_missing_does_exit_two_reporting_no_session_without_bug_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    arm_placeholder_endpoint(monkeypatch)
    session_log = session_jsonl_path(str(tmp_path))

    result = runner.invoke(app, ["export", session_log])

    first_line = stderr_text(result).splitlines()[0]
    assert result.exit_code == 2
    assert "No session found" in first_line
    assert session_log in first_line
    assert BUGS_URL not in combined_output(result)


#: What the OS says when a directory is opened as a file: Windows denies access.
_DIRECTORY_OPEN_REASON = os.strerror(errno.EACCES if sys.platform == "win32" else errno.EISDIR)


def test_export_when_session_log_unreadable_does_exit_two_naming_path_and_os_reason_without_bug_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_log = _populate_session_dir(str(tmp_path))
    arm_placeholder_endpoint(monkeypatch)
    Path(session_log).unlink()
    Path(session_log).mkdir()

    result = runner.invoke(app, ["export", session_log])

    error = stderr_text(result)
    output = combined_output(result)
    assert result.exit_code == 2
    assert session_log in error
    assert _DIRECTORY_OPEN_REASON in error
    assert BUGS_URL not in output


# ---------------------------------------------------------------------------
# failed export
# ---------------------------------------------------------------------------


def test_export_when_collector_unreachable_does_exit_two_naming_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv(ENDPOINT_ENV, _UNREACHABLE_ENDPOINT)
    monkeypatch.setenv(_TIMEOUT_ENV, _SHORT_TIMEOUT_SECONDS)

    result = runner.invoke(app, ["export", session_log])

    output = stderr_text(result)
    assert result.exit_code == 2, output
    assert _UNREACHABLE_ENDPOINT in output


# ---------------------------------------------------------------------------
# default session log
# ---------------------------------------------------------------------------


def test_export_when_no_session_log_argument_does_use_repo_session_path(
    monkeypatch: pytest.MonkeyPatch, repo: str
):
    _populate_session_dir(repo)

    with otlp_collector() as collector:
        monkeypatch.setenv(ENDPOINT_ENV, collector.endpoint)

        result = runner.invoke(app, ["export"])

    (exported,) = [line for line in result.stderr.splitlines() if SESSION_ID in line]
    assert result.exit_code == 0, combined_output(result)
    assert "3" in exported.split()
    assert collector.endpoint in exported
