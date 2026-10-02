"""Tests for the ``gymrat measure`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``measure`` and ``resolve_config`` seams replaced. They cover the optional
target defaulting to ``.``, the report going to stdout, the missing-bench error
routing to exit 2, the rejection of ``--verbose``/``--fail-on`` and unknown
options as usage errors, the ``--record`` flag that appends the run to an
open session log as a baseline (including elapsed duration), budget time-left
reporting in text and JSON output, and duration warnings when the budget is
tight.
"""

import errno
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.measure import MeasureOptions
from gymrat.report.types import MeasurementResult
from gymrat.sampling import TargetSpec
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import BaselineRecord, CommandRecord
from gymrat.session.store import append_record, read_records
from tests._rich import unwrap_panel
from tests.cli._budget import install_budget, install_tight_budget
from tests.cli._session import (
    capture_measure,
    closed_stdout_error,
    closed_stdout_runner,
    disk_full_error,
    last_command_record,
    runner,
    stub_measure,
    stub_resolve,
)
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    finalize_record,
    iteration_record,
    session_record,
    write_session_log,
)


@pytest.fixture
def _in_non_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run from a directory that is not a git repo, so the command benches lock-free."""
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# target defaulting
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_no_target_given_does_default_to_current_directory(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert captured[0].target == TargetSpec(label=None, target=".")


# ---------------------------------------------------------------------------
# report to stdout
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_run_does_render_report_to_stdout(monkeypatch: pytest.MonkeyPatch):
    stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert "main" in result.stdout


# ---------------------------------------------------------------------------
# missing bench
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_bench_missing_does_exit_two_with_message_on_stderr():
    result = runner.invoke(app, ["measure"])

    assert result.exit_code == 2
    assert "bench is required" in result.stderr
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# rejected options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "option",
    [
        pytest.param(["--verbose"], id="verbose"),
        pytest.param(["--fail-on", "regressed"], id="fail-on"),
        pytest.param(["--bogus"], id="unknown"),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_unsupported_option_given_does_exit_two(option: list[str]):
    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh", *option])

    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# --record
# ---------------------------------------------------------------------------


def _open_session(repo: str) -> None:
    """Open a session in ``repo`` so a ``--record`` run has somewhere to write."""
    write_session_log(repo, session_record())


def _finalize_session(repo: str) -> None:
    """Open then close a session in ``repo``, leaving it finalized."""
    write_session_log(repo, session_record(), (finalize_record(),))


@pytest.fixture
def record_repo(monkeypatch: pytest.MonkeyPatch, create_scratch_repo: Callable[[], str]) -> str:
    """A scratch git repo, chdir'd into, with ``resolve_config`` stubbed for ``--record`` tests."""
    repo = create_scratch_repo()
    monkeypatch.chdir(repo)
    stub_resolve(monkeypatch)
    return repo


@pytest.mark.parametrize(
    ("positional", "label"),
    [
        pytest.param("main", "main", id="bare-ref"),
        pytest.param("build=main", "build", id="label=ref"),
    ],
)
def test_measure_when_record_and_open_session_does_append_baseline_and_print_report_note(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
    positional: str,
    label: str,
):
    _open_session(record_repo)
    rounds: list[dict[str, float]] = [{"latency": 41}, {"latency": 43}]
    capture_measure(monkeypatch, create_measurement_result(label=label, rounds=rounds))

    result = runner.invoke(app, ["measure", positional, "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    baselines = [
        r for r in read_records(session_jsonl_path(record_repo)) if isinstance(r, BaselineRecord)
    ]
    assert len(baselines) == 1
    recorded = baselines[0]
    assert recorded.type == "baseline"
    assert isinstance(recorded.at, int)
    assert recorded.at > 0
    assert recorded.label == label
    assert recorded.samples == tuple(rounds)
    assert label in result.stdout
    assert re.search(r"recorded to session", result.stdout, re.IGNORECASE)


def test_measure_when_record_and_json_format_does_route_note_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _open_session(record_repo)
    capture_measure(monkeypatch, create_measurement_result(rounds=[{"latency": 42}]))

    result = runner.invoke(
        app, ["measure", "main", "--bench", "sh bench.sh", "--record", "--format", "json"]
    )

    assert result.exit_code == 0
    assert "recorded to session" not in result.stdout
    assert re.search(r"recorded to session", result.stderr, re.IGNORECASE)


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param(["--format", "text"], id="text"),
        pytest.param(["--format", "json"], id="json"),
        pytest.param(["--record"], id="record-note"),
    ],
)
def test_measure_when_stdout_reader_closed_does_exit_zero_without_stderr(
    monkeypatch: pytest.MonkeyPatch, record_repo: str, extra_args: list[str]
):
    _open_session(record_repo)
    capture_measure(monkeypatch, create_measurement_result(rounds=[{"latency": 42}]))

    result = closed_stdout_runner(closed_stdout_error()).invoke(
        app, ["measure", "main", "--bench", "sh bench.sh", *extra_args]
    )

    assert (result.exit_code, result.stderr) == (0, "")


@pytest.mark.usefixtures("repo")
def test_measure_when_stdout_write_fails_otherwise_does_report_the_error(
    monkeypatch: pytest.MonkeyPatch,
):
    stub_measure(monkeypatch)

    result = closed_stdout_runner(disk_full_error()).invoke(
        app, ["measure", "main", "--bench", "sh bench.sh"]
    )

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert os.strerror(errno.ENOSPC) in unwrap_panel(result.stderr)


@pytest.mark.usefixtures("record_repo")
def test_measure_when_record_and_no_session_does_exit_two_without_benching(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = capture_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 2
    assert captured == []
    assert "gymrat start" in result.stderr


def test_measure_when_record_and_finalized_session_does_exit_two_without_benching(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _finalize_session(record_repo)
    captured = capture_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 2
    assert captured == []
    assert session_record().session_id in result.stderr
    assert "gymrat start" in result.stderr


def test_measure_when_no_record_flag_does_leave_open_session_untouched(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _open_session(record_repo)
    capture_measure(monkeypatch, create_measurement_result(rounds=[{"latency": 42}]))

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    non_command = [
        r for r in read_records(session_jsonl_path(record_repo)) if not isinstance(r, CommandRecord)
    ]
    assert non_command == [session_record()]
    assert "recorded to session" not in result.stdout


# ---------------------------------------------------------------------------
# --record duration
# ---------------------------------------------------------------------------


def test_measure_when_record_does_write_duration_ms_to_baseline(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _open_session(record_repo)
    capture_measure(monkeypatch, create_measurement_result(rounds=[{"latency": 42}]))
    ticks = iter([1_000.0, 1_500.0, 2_000.0, 2_500.0])
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: next(ticks))

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    baselines = [
        r for r in read_records(session_jsonl_path(record_repo)) if isinstance(r, BaselineRecord)
    ]
    assert len(baselines) == 1
    assert baselines[0].duration_ms == 500


# ---------------------------------------------------------------------------
# budget time-left line (text) and key (JSON) on measure
# ---------------------------------------------------------------------------


def test_measure_when_budget_active_does_end_text_with_time_left_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    stub_measure(monkeypatch)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    lines = [line.strip() for line in result.stdout.split("\n") if line.strip()]
    assert re.search(r"left of 30m", lines[-1])


def test_measure_when_no_budget_does_omit_time_left_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert "left of" not in result.stdout


def test_measure_when_format_json_and_budget_active_does_include_budget_object(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    stub_measure(monkeypatch)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_measure_when_format_json_and_no_budget_does_omit_budget_key(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" not in doc


# ---------------------------------------------------------------------------
# budget absent on error exits
# ---------------------------------------------------------------------------


def test_measure_when_error_and_budget_active_does_not_include_budget(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["measure"])

    assert result.exit_code == 2
    assert "left of" not in result.stdout
    assert "left of" not in result.stderr


# ---------------------------------------------------------------------------
# duration warnings
# ---------------------------------------------------------------------------


def test_measure_when_budget_tight_and_estimate_known_does_warn_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _open_session(record_repo)
    capture_measure(monkeypatch)
    install_tight_budget(record_repo, monkeypatch)
    append_record(session_jsonl_path(record_repo), iteration_record(duration_ms=720_000))

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert "warning" in result.stderr.lower()


def test_measure_when_estimate_unknown_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _open_session(record_repo)
    capture_measure(monkeypatch)
    install_tight_budget(record_repo, monkeypatch)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert "warning" not in result.stderr.lower()


# ---------------------------------------------------------------------------
# command trace — args and exit recording
# ---------------------------------------------------------------------------


def test_measure_when_success_does_record_trace_with_target_and_record_false(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _open_session(repo)
    stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "measure"
    assert cmd.args["target"] == "main"
    assert cmd.args["record"] is False
    assert cmd.exit_code == 0
    assert cmd.reason is None
    for key in ("prepare", "adapter", "samples", "timeout", "config"):
        assert key not in cmd.args


def test_measure_when_default_target_does_record_dot_in_trace_args(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _open_session(repo)
    stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.args["target"] == "."


def test_measure_when_config_overrides_given_does_include_them_in_trace_args(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _open_session(repo)
    stub_measure(monkeypatch)

    result = runner.invoke(
        app,
        [
            "measure",
            "main",
            "--bench",
            "sh bench.sh",
            "--prepare",
            "make",
            "--adapter",
            "mitata",
            "--samples",
            "7",
            "--timeout",
            "42",
            "--config",
            "gymrat.json",
        ],
    )

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.args["bench"] == "sh bench.sh"
    assert cmd.args["prepare"] == "make"
    assert cmd.args["adapter"] == "mitata"
    assert cmd.args["samples"] == 7
    assert cmd.args["timeout"] == 42
    assert cmd.args["config"] == "gymrat.json"


def test_measure_when_record_success_does_record_trace_with_record_true(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _open_session(record_repo)
    capture_measure(monkeypatch, create_measurement_result(rounds=[{"latency": 42}]))

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    cmd = last_command_record(record_repo)
    assert cmd.args["target"] == "main"
    assert cmd.args["record"] is True
    assert cmd.exit_code == 0
    assert cmd.reason is None


def test_measure_when_record_and_finalized_session_does_record_trace_with_exit_two_finalized(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _finalize_session(record_repo)
    capture_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 2
    cmd = last_command_record(record_repo)
    assert cmd.args["target"] == "main"
    assert cmd.args["record"] is True
    assert cmd.exit_code == 2
    assert cmd.reason == "finalized"


def test_measure_when_bench_fails_does_record_trace_with_exit_two_error(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _open_session(repo)
    stub_resolve(monkeypatch)

    async def failing_measure(_options: MeasureOptions) -> MeasurementResult:
        msg = "bench exploded"
        raise GymratError(msg)

    monkeypatch.setattr("gymrat.measure.measure", failing_measure)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert cmd.args["target"] == "main"
    assert cmd.exit_code == 2
    assert cmd.reason == "error"
