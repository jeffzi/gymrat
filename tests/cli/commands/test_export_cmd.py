"""Tests for the ``gymrat export`` command wiring.

These drive the assembled app through :class:`typer.testing.CliRunner` against a
local OTLP collector, on session logs written to a temporary directory. They cover
the argument/option contract, error exits, supervisor log selection, and the
success path with its printed summary.
"""

from __future__ import annotations

import errno
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import create_autospec

import pytest

from gymrat.cli.app import app
from gymrat.session.paths import repo_root, session_jsonl_path
from gymrat.session.records import record_to_wire
from gymrat.utils import ENDPOINT_ENV
from tests._cli import no_color_env, run_cli
from tests._mode_bits import needs_mode_bits
from tests.cli._session import runner
from tests.session.records._fixtures import SESSION_ID, command_record
from tests.telemetry._collector import otlp_collector
from tests.telemetry._fixtures import arm_placeholder_endpoint, hide_otel_sdk, hide_otlp_exporter
from tests.telemetry._replay_logs import (
    T0,
    write_measure_command_run,
    write_standard_run,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from typer.testing import Result


def _output(result: Result) -> str:
    """Combine stdout and stderr the way CliRunner splits them across streams."""
    return result.stdout + result.stderr


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


@pytest.mark.parametrize(
    "hide_package",
    [
        pytest.param(hide_otel_sdk, id="sdk"),
        pytest.param(hide_otlp_exporter, id="otlp-exporter"),
    ],
)
def test_export_when_tracing_package_not_importable_does_exit_two_naming_otel_extra(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hide_package: Callable[[pytest.MonkeyPatch], None],
):
    session_log = _populate_session_dir(str(tmp_path))
    arm_placeholder_endpoint(monkeypatch)
    hide_package(monkeypatch)

    result = runner.invoke(app, ["export", session_log])

    output = _output(result)
    assert result.exit_code == 2, output
    assert "OpenTelemetry SDK or OTLP exporter not available" in output
    assert "'gymrat[otel]'" in output


# ---------------------------------------------------------------------------
# tracer records nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        pytest.param("OTEL_SDK_DISABLED", "true", id="sdk-disabled"),
        pytest.param("OTEL_TRACES_SAMPLER", "always_off", id="sampler-always-off"),
    ],
)
def test_export_when_tracer_records_nothing_does_exit_two_without_reporting_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variable: str,
    value: str,
):
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv(variable, value)

    with otlp_collector() as collector:
        monkeypatch.setenv(ENDPOINT_ENV, collector.endpoint)

        result = runner.invoke(app, ["export", session_log])

    output = _output(result)
    assert result.exit_code == 2, output
    assert "No spans recorded" in output
    assert "Unset OTEL_SDK_DISABLED and set OTEL_TRACES_SAMPLER" in output
    assert "exported" not in output
    assert collector.received == []


# ---------------------------------------------------------------------------
# missing endpoint
# ---------------------------------------------------------------------------


def _endpoint_args(monkeypatch: pytest.MonkeyPatch, source: str | None, endpoint: str) -> list[str]:
    """Supply ``endpoint`` through ``--endpoint`` or the environment, per ``source``.

    Args:
        monkeypatch: Sets the endpoint environment variable when ``source`` is ``"env"``.
        source: ``"flag"`` to pass ``--endpoint``, ``None`` to supply no endpoint
            at all; any other value sets the environment variable.
        endpoint: The endpoint value to supply.

    Returns:
        The extra command-line arguments the invocation needs.
    """
    if source is None:
        return []
    if source == "flag":
        return ["--endpoint", endpoint]
    monkeypatch.setenv(ENDPOINT_ENV, endpoint)
    return []


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(None, id="no-flag-and-no-env"),
        pytest.param("flag", id="whitespace-only-flag"),
        pytest.param("env", id="whitespace-only-env"),
    ],
)
def test_export_when_endpoint_missing_or_blank_does_exit_two_naming_env_var(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str | None,
):
    session_log = _populate_session_dir(str(tmp_path))
    args = _endpoint_args(monkeypatch, source, " \t ")

    result = runner.invoke(app, ["export", session_log, *args])

    assert result.exit_code == 2
    assert "No endpoint: pass --endpoint or set OTEL_EXPORTER_OTLP_ENDPOINT" in _output(result)


# ---------------------------------------------------------------------------
# missing or corrupt session log
# ---------------------------------------------------------------------------


def _blank_log(path: Path) -> None:
    """Write a session log that holds only a blank line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n", encoding="utf-8")


def _no_log(path: Path) -> None:
    """Leave the session log path empty."""


@pytest.mark.parametrize(
    "arrange",
    [
        pytest.param(_no_log, id="missing"),
        pytest.param(_blank_log, id="blank"),
    ],
)
def test_export_when_session_log_missing_or_blank_does_exit_two_reporting_no_session(
    arrange: Callable[[Path], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    arm_placeholder_endpoint(monkeypatch)
    session_log = session_jsonl_path(str(tmp_path))
    arrange(Path(session_log))

    result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 2
    assert f"No session found in {session_log}" in _output(result)


@pytest.mark.parametrize(
    ("first_line", "expected_fragments"),
    [
        pytest.param(
            b"not json at all\n",
            ("Invalid JSON at {path}:1", "Line 1 is not a JSON object."),
            id="malformed-json",
        ),
        pytest.param(
            b"\xff\xff\n",
            ("Corrupt session log at {path}:1", "Line 1 contains invalid UTF-8 bytes."),
            id="undecodable-bytes",
        ),
    ],
)
def test_export_when_first_line_corrupt_does_exit_two_naming_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_line: bytes,
    expected_fragments: tuple[str, ...],
):
    arm_placeholder_endpoint(monkeypatch)
    session_log = session_jsonl_path(str(tmp_path))
    Path(session_log).parent.mkdir(parents=True, exist_ok=True)
    Path(session_log).write_bytes(first_line)

    result = runner.invoke(app, ["export", session_log])

    output = _output(result)
    assert result.exit_code == 2
    expected = [fragment.format(path=session_log) for fragment in expected_fragments]
    assert [fragment for fragment in expected if fragment not in output] == []


def test_export_when_first_record_not_session_does_exit_two_naming_its_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    arm_placeholder_endpoint(monkeypatch)
    session_log = session_jsonl_path(str(tmp_path))
    command_line = json.dumps(record_to_wire(command_record(name="measure", at=T0)))
    Path(session_log).parent.mkdir(parents=True, exist_ok=True)
    Path(session_log).write_text(f"{command_line}\n", encoding="utf-8")

    result = runner.invoke(app, ["export", session_log])

    assert (result.exit_code, _output(result).splitlines()) == (
        2,
        [
            f"Error: Expected session header at {session_log}:1, got a command record",
            "The session log is corrupt; start a new session.",
        ],
    )


@needs_mode_bits
def test_export_when_session_log_unreadable_does_exit_two_naming_path_and_os_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_log = _populate_session_dir(str(tmp_path))
    arm_placeholder_endpoint(monkeypatch)
    Path(session_log).chmod(0o000)

    result = runner.invoke(app, ["export", session_log])

    output = _output(result)
    assert result.exit_code == 2
    assert session_log in output
    assert "Permission denied" in output
    assert "No session found" not in output


# ---------------------------------------------------------------------------
# success path
# ---------------------------------------------------------------------------


def test_export_when_endpoint_flag_given_does_send_spans_to_the_flag_over_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv(ENDPOINT_ENV, _UNREACHABLE_ENDPOINT)
    monkeypatch.setenv(_TIMEOUT_ENV, _SHORT_TIMEOUT_SECONDS)

    with otlp_collector() as collector:
        result = runner.invoke(app, ["export", session_log, "--endpoint", collector.endpoint])

    assert result.exit_code == 0, _output(result)
    assert len(collector.span_names) == 3
    assert f"exported 3 spans for session {SESSION_ID} to {collector.endpoint}" in result.stderr


@pytest.mark.parametrize("source", ["flag", "env"])
def test_export_when_endpoint_padded_does_send_spans_to_trimmed_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
):
    session_log = _populate_session_dir(str(tmp_path))

    with otlp_collector() as collector:
        padded = f"  {collector.endpoint} "
        args = _endpoint_args(monkeypatch, source, padded)

        result = runner.invoke(app, ["export", session_log, *args])

    success_lines = [line for line in result.stderr.splitlines() if line.startswith("exported ")]
    assert result.exit_code == 0, _output(result)
    assert {export.path for export in collector.received} == {"/v1/traces"}
    assert len(collector.span_names) == 3
    assert success_lines == [
        f"exported 3 spans for session {SESSION_ID} to {collector.endpoint}"
    ], _output(result)


def test_export_when_final_session_line_is_torn_utf8_does_skip_only_that_line(
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    with Path(session_log).open("ab") as log:
        log.write(b'{"type": "iteration", "note": "caf\xc3')
    env = no_color_env()

    with otlp_collector() as collector:
        env[ENDPOINT_ENV] = collector.endpoint

        result = run_cli(
            ["export", session_log],
            tmp_path,
            check=False,
            timeout=60,
            env=env,
        )

    assert result.returncode == 0, result.stderr
    assert f"session log {session_log}: skipping line 3 (invalid JSON)" in result.stderr
    assert f"exported 3 spans for session {SESSION_ID} to {collector.endpoint}" in result.stderr
    assert len(collector.span_names) == 3


# ---------------------------------------------------------------------------
# failed export
# ---------------------------------------------------------------------------


# Bound on a failed export under the short timeout above. The exporter's
# default 10 s timeout would first sleep through 1 + 2 + 4 s of retry backoff,
# so an export that ignored the short timeout overruns it.
_EXPORT_DEADLINE_S = 5.0


def _failed_export_error(endpoint: str) -> str:
    return (
        f"Could not export spans to {endpoint}: the collector is unreachable or rejected a batch."
    )


def test_export_when_collector_unreachable_does_exit_two_naming_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv(ENDPOINT_ENV, _UNREACHABLE_ENDPOINT)
    monkeypatch.setenv(_TIMEOUT_ENV, _SHORT_TIMEOUT_SECONDS)
    started = time.monotonic()

    result = runner.invoke(app, ["export", session_log])

    elapsed = time.monotonic() - started
    output = _output(result)
    assert result.exit_code == 2, output
    assert _failed_export_error(_UNREACHABLE_ENDPOINT) in output
    assert "exported" not in output
    assert elapsed < _EXPORT_DEADLINE_S


@pytest.mark.parametrize(
    ("batch_size", "expected_exports"),
    [
        pytest.param("512", 1, id="the-only-batch"),
        pytest.param("1", 3, id="the-first-of-three-batches"),
    ],
)
def test_export_when_collector_rejects_a_batch_does_exit_two_naming_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    batch_size: str,
    expected_exports: int,
):
    session_log = _populate_session_dir(str(tmp_path))
    monkeypatch.setenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", batch_size)

    with otlp_collector(statuses=[400]) as collector:
        monkeypatch.setenv(ENDPOINT_ENV, collector.endpoint)

        result = runner.invoke(app, ["export", session_log])

    output = _output(result)
    assert result.exit_code == 2, output
    assert len(collector.received) == expected_exports
    assert _failed_export_error(collector.endpoint) in output
    assert "exported" not in output


# ---------------------------------------------------------------------------
# default session log
# ---------------------------------------------------------------------------


def test_export_when_no_session_log_argument_does_use_repo_session_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _populate_session_dir(str(tmp_path))
    monkeypatch.setattr(
        "gymrat.cli.commands.export.repo_root",
        create_autospec(repo_root, return_value=str(tmp_path)),
    )

    with otlp_collector() as collector:
        monkeypatch.setenv(ENDPOINT_ENV, collector.endpoint)

        result = runner.invoke(app, ["export"])

    assert result.exit_code == 0, _output(result)
    assert f"exported 3 spans for session {SESSION_ID} to {collector.endpoint}" in result.stderr


# ---------------------------------------------------------------------------
# supervisor log filtering
# ---------------------------------------------------------------------------


def _first_line_writer(first_line: bytes) -> Callable[[Path], None]:
    """Build a writer that creates a supervisor log holding only ``first_line``."""

    def write(path: Path) -> None:
        path.write_bytes(first_line)

    return write


def _other_session_log(path: Path) -> None:
    """Write a well-formed supervisor log whose launch line names another session."""
    write_standard_run(str(path), session_id="other-session-id")


def _dangling_symlink(path: Path) -> None:
    """Create an entry the listing sees but whose target is gone by the time it is read."""
    path.symlink_to(path.with_name("vanished.jsonl"))


@pytest.mark.parametrize(
    "make_entry",
    [
        pytest.param(_other_session_log, id="a-log-of-another-session"),
        pytest.param(_first_line_writer(b"{not json\n"), id="a-first-line-that-is-not-json"),
        pytest.param(_first_line_writer(b"\xff\xff\n"), id="a-first-line-that-is-not-utf8"),
        pytest.param(_first_line_writer(b"[1, 2]\n"), id="a-first-line-that-is-not-an-object"),
        pytest.param(
            _dangling_symlink,
            id="a-log-that-vanished-after-listing",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="creating a symlink needs extra rights on Windows"
            ),
        ),
    ],
)
def test_export_when_supervisor_log_not_a_launch_of_this_session_does_skip_it_silently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_entry: Callable[[Path], None],
):
    session_log = _populate_session_dir(str(tmp_path))
    make_entry(Path(session_log).parent / "supervisor-002.jsonl")

    with otlp_collector() as collector:
        monkeypatch.setenv(ENDPOINT_ENV, collector.endpoint)

        result = runner.invoke(app, ["export", session_log])

    assert result.exit_code == 0, _output(result)
    assert len(collector.span_names) == 3
    assert f"exported 3 spans for session {SESSION_ID}" in result.stderr
    assert "warning: " not in result.stderr


def _deny_read(path: Path) -> None:
    """Write a supervisor log for the session, then remove every permission on it."""
    write_standard_run(str(path))
    path.chmod(0o000)


def _make_directory(path: Path) -> None:
    """Create a directory whose name matches the supervisor log pattern."""
    path.mkdir()


@pytest.mark.parametrize(
    ("make_unreadable", "reason"),
    [
        pytest.param(
            _deny_read,
            os.strerror(errno.EACCES),
            id="read-denied",
            marks=needs_mode_bits,
        ),
        pytest.param(
            _make_directory,
            os.strerror(errno.EISDIR),
            id="a-directory",
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="Windows reports a directory as access denied"
            ),
        ),
    ],
)
def test_export_when_supervisor_log_unreadable_does_skip_it_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_unreadable: Callable[[Path], None],
    reason: str,
):
    session_log = _populate_session_dir(str(tmp_path))
    unreadable = Path(session_log).parent / "supervisor-002.jsonl"
    make_unreadable(unreadable)

    with otlp_collector() as collector:
        monkeypatch.setenv(ENDPOINT_ENV, collector.endpoint)

        result = runner.invoke(app, ["export", session_log])

    warning_lines = [line for line in result.stderr.splitlines() if line.startswith("warning: ")]
    assert result.exit_code == 0, _output(result)
    assert len(collector.span_names) == 3
    assert len(warning_lines) == 1
    assert str(unreadable) in warning_lines[0]
    assert reason in warning_lines[0]
    assert f"exported 3 spans for session {SESSION_ID}" in result.stderr
