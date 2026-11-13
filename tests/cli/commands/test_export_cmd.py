"""Tests for the ``gymrat export`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` against a
local OTLP collector, on session logs written to a temporary directory. They cover
the argument/option contract, error exits, and the success path with its printed
summary. Which session-log lines and supervisor logs replay skips is pinned in
``tests/telemetry/test_replay.py``.
"""

from __future__ import annotations

from http import HTTPStatus
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
from tests.telemetry._fixtures import (
    arm_placeholder_endpoint,
    disable_otel_sdk,
    hide_otlp_exporter,
)
from tests.telemetry._replay_logs import write_measure_command_run

if TYPE_CHECKING:
    import pytest

_TIMEOUT_ENV = "OTEL_EXPORTER_OTLP_TIMEOUT"
_SHORT_TIMEOUT_SECONDS = "0.5"
_UNREACHABLE_ENDPOINT = "http://127.0.0.1:1"


def _populate_session_dir(root: str) -> str:
    """Write a session log and a matching supervisor log under ``root``.

    Args:
        root: The repository root the session log goes under.

    Returns:
        The session log path.
    """
    session_log = session_jsonl_path(root)
    sup_log = str(Path(session_log).parent / "supervisor-001.jsonl")
    write_measure_command_run(session_log, sup_log, session_id=SESSION_ID)
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


def test_export_when_tracer_records_nothing_does_exit_two_naming_the_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    disable_otel_sdk(monkeypatch)

    result = runner.invoke(app, ["export", session_log])

    output = stderr_text(result)
    assert result.exit_code == 2, output
    assert len([line for line in output.splitlines() if SESSION_ID in line]) == 1


# ---------------------------------------------------------------------------
# missing endpoint
# ---------------------------------------------------------------------------


def test_export_when_endpoint_missing_does_exit_two_naming_env_var(tmp_path: Path):
    session_log = _populate_session_dir(str(tmp_path))

    result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 2
    assert ENDPOINT_ENV in stderr_text(result)


# ---------------------------------------------------------------------------
# missing session log
# ---------------------------------------------------------------------------


def test_export_when_session_log_missing_does_exit_two_reporting_no_session_without_bug_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    arm_placeholder_endpoint(monkeypatch)
    session_log = session_jsonl_path(str(tmp_path))

    result = runner.invoke(app, ["export", session_log])

    first_line = stderr_text(result).splitlines()[0]
    assert result.exit_code == 2
    assert session_log in first_line
    assert BUGS_URL not in combined_output(result)


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


def test_export_when_spans_are_dropped_before_the_exporter_does_exit_two_naming_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv("OTEL_BSP_MAX_QUEUE_SIZE", "1")
    monkeypatch.setenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "1")
    # The 503 makes the exporter back off and retry, so the one-span queue
    # overflows while the replay keeps ending spans.
    with otlp_collector(statuses=[HTTPStatus.SERVICE_UNAVAILABLE]) as collector:
        monkeypatch.setenv(ENDPOINT_ENV, collector.endpoint)

        result = runner.invoke(app, ["export", session_log])

    output = stderr_text(result)
    assert result.exit_code == 2, output
    assert collector.endpoint in output


def test_export_when_only_traces_variable_set_does_post_to_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    session_log = _populate_session_dir(str(tmp_path))

    with otlp_collector() as collector:
        monkeypatch.delenv(ENDPOINT_ENV, raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"{collector.endpoint}/custom")

        result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 0, combined_output(result)
    assert {export.path for export in collector.received} == {"/custom"}


def test_export_when_base_and_traces_variables_set_without_flag_does_post_to_the_traces_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    session_log = _populate_session_dir(str(tmp_path))

    with otlp_collector() as collector:
        monkeypatch.setenv(ENDPOINT_ENV, _UNREACHABLE_ENDPOINT)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"{collector.endpoint}/custom")

        result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 0, combined_output(result)
    assert {export.path for export in collector.received} == {"/custom"}
    assert f"to {collector.endpoint}/custom" in stderr_text(result)


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

    assert result.exit_code == 0, combined_output(result)
    (exported,) = [line for line in result.stderr.splitlines() if SESSION_ID in line]
    assert "3" in exported.split()
    assert collector.endpoint in exported
