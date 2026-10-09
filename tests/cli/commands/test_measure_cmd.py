"""Tests for the ``gymrat measure`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``measure`` and ``resolve_config`` seams replaced. They cover the optional
target defaulting to ``.``, the report going to stdout, the missing-bench error
routing to exit 2 for ``measure`` and ``compare`` alike, the ``--record`` flag that appends the run to an open
session log as a baseline (including elapsed duration), and the command trace.
The budget time-left line comes from the shared ``emit_report`` path, pinned in
``test_session_cmds`` with and without a budget; the tight-budget warning is
pinned there for measure.
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
from tests._clock import install_monotonic_clock
from tests.cli._session import (
    capture_measure,
    last_command_record,
    open_session,
    runner,
    stub_measure,
    stub_resolve,
)
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    records_of_type,
    session_record,
)

# ---------------------------------------------------------------------------
# target defaulting
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_no_target_given_does_measure_the_current_directory(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert captured[0].target == TargetSpec(label=None, target=".")


# ---------------------------------------------------------------------------
# missing bench
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["measure"], id="measure"),
        pytest.param(["compare", "main", "cand"], id="compare"),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_benching_command_when_bench_missing_does_exit_two_with_message_on_stderr(
    argv: list[str],
):
    result = runner.invoke(app, argv)

    assert result.exit_code == 2
    assert "bench is required" in result.stderr
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# --record
# ---------------------------------------------------------------------------


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


def test_measure_when_record_without_a_session_does_exit_two_without_benching(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    captured = capture_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 2
    assert captured == []
    assert "gymrat start" in result.stderr


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
    clock = install_monotonic_clock(monkeypatch)
    measured = create_measurement_result(rounds=[{"latency": 42}])

    async def measure_for_half_a_second(_options: MeasureOptions) -> MeasurementResult:
        clock.tick(500.0)
        return measured

    monkeypatch.setattr("gymrat.measure.measure", measure_for_half_a_second)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    baselines = records_of_type(record_repo, BaselineRecord)
    assert len(baselines) == 1
    assert baselines[0].duration_ms == 500


# ---------------------------------------------------------------------------
# command trace — args and exit recording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "target", "record"),
    [
        pytest.param(["main"], "main", False, id="bare-ref"),
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
