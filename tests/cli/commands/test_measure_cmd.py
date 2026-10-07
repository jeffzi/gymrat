"""Tests for the ``gymrat measure`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``measure`` and ``resolve_config`` seams replaced. They cover the optional
target defaulting to ``.``, the report going to stdout, the missing-bench error
routing to exit 2, the ``--record`` flag that appends the run to an open
session log as a baseline (including elapsed duration), the absent time-left
line without a budget, and the command trace. The budget time-left line and
the tight-budget warning are pinned with every other command's in
``test_session_cmds``.
"""

import re
from collections.abc import Callable

import pytest

from gymrat.cli.app import app
from gymrat.errors import GymratError
from gymrat.measure import MeasureOptions
from gymrat.report.types import MeasurementResult
from gymrat.sampling import TargetSpec
from gymrat.session.records import BaselineRecord, CommandRecord
from tests.cli._session import (
    FailingStdoutRunner,
    capture_measure,
    closed_stdout_error,
    last_command_record,
    open_session,
    runner,
    stub_measure,
    stub_resolve,
)
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    finalize_record,
    records_of_type,
    session_record,
    write_session_log,
)

# ---------------------------------------------------------------------------
# target defaulting and report to stdout
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_no_target_given_does_report_the_current_directory_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert captured[0].target == TargetSpec(label=None, target=".")
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
# --record
# ---------------------------------------------------------------------------


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
def test_measure_when_record_and_open_session_does_record_the_run_as_a_baseline(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
    positional: str,
    label: str,
):
    open_session(record_repo)
    rounds: list[dict[str, float]] = [{"latency": 41}, {"latency": 43}]
    capture_measure(monkeypatch, create_measurement_result(label=label, rounds=rounds))

    result = runner.invoke(app, ["measure", positional, "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    baselines = records_of_type(record_repo, BaselineRecord)
    assert len(baselines) == 1
    recorded = baselines[0]
    assert recorded.at > 0
    assert recorded.label == label
    assert recorded.samples == tuple(rounds)
    assert label in result.stdout
    assert re.search(r"recorded to session", result.stdout, re.IGNORECASE)


def test_measure_when_record_and_json_format_does_route_note_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    open_session(record_repo)
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
    open_session(record_repo)
    capture_measure(monkeypatch, create_measurement_result(rounds=[{"latency": 42}]))

    result = FailingStdoutRunner(closed_stdout_error()).invoke(
        app, ["measure", "main", "--bench", "sh bench.sh", *extra_args]
    )

    assert (result.exit_code, result.stderr) == (0, "")


def _no_session(repo: str) -> None:
    """Leave ``repo`` without a session log."""


@pytest.mark.parametrize(
    ("arrange", "named"),
    [
        pytest.param(_no_session, ["gymrat start"], id="no-session"),
        pytest.param(
            _finalize_session, [session_record().session_id, "gymrat start"], id="finalized"
        ),
    ],
)
def test_measure_when_record_without_an_open_session_does_exit_two_without_benching(
    arrange: Callable[[str], None],
    named: list[str],
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    arrange(record_repo)
    captured = capture_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 2
    assert captured == []
    assert [fragment for fragment in named if fragment not in result.stderr] == []


def test_measure_when_no_record_flag_does_leave_open_session_untouched(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    open_session(record_repo)
    capture_measure(monkeypatch, create_measurement_result(rounds=[{"latency": 42}]))

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    non_command = records_of_type(record_repo, CommandRecord, matching=False)
    assert non_command == [session_record()]
    assert "recorded to session" not in result.stdout


# ---------------------------------------------------------------------------
# --record duration
# ---------------------------------------------------------------------------


def test_measure_when_record_does_write_duration_ms_to_baseline(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    open_session(record_repo)
    now_ms = [1_000.0]
    monkeypatch.setattr("gymrat.clock.monotonic_ms", lambda: now_ms[0])
    measured = create_measurement_result(rounds=[{"latency": 42}])

    async def measure_for_half_a_second(_options: MeasureOptions) -> MeasurementResult:
        now_ms[0] += 500
        return measured

    monkeypatch.setattr("gymrat.measure.measure", measure_for_half_a_second)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    baselines = records_of_type(record_repo, BaselineRecord)
    assert len(baselines) == 1
    assert baselines[0].duration_ms == 500


# ---------------------------------------------------------------------------
# budget time-left line
# ---------------------------------------------------------------------------


def test_measure_when_no_budget_does_omit_time_left_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert "left of" not in result.stdout


# ---------------------------------------------------------------------------
# command trace — args and exit recording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "target", "record"),
    [
        pytest.param(["main"], "main", False, id="bare-ref"),
        pytest.param(["build=main"], "build", False, id="labeled-target"),
        pytest.param([], ".", False, id="default-target"),
        pytest.param(["build=main", "--record"], "build", True, id="recorded"),
    ],
)
def test_measure_when_success_does_record_trace_with_target_and_record_flag(
    argv: list[str],
    target: str,
    record: bool,
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    open_session(repo)
    stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", *argv, "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("measure", 0, None)
    assert (cmd.args["target"], cmd.args["record"], cmd.args["bench"]) == (
        target,
        record,
        "sh bench.sh",
    )


def test_measure_when_bench_fails_does_record_trace_with_exit_two_error(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    open_session(repo)
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
