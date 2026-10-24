"""Tests for the ``gymrat measure`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``measure`` and ``resolve_config`` seams replaced. They cover the optional
target defaulting to ``.``, the report going to stdout, the missing-bench error
routing to exit 2 for ``measure`` and ``compare`` alike, the ``--record`` flag that appends the run to an open
session log as a baseline (with its elapsed duration), and the command trace.
The budget time-left line comes from the shared ``emit_report`` path, pinned in
``test_session_cmds`` with and without a budget; the tight-budget warning is
pinned there for measure.
"""

import re
from unittest.mock import create_autospec

import pytest

from gymrat.cli.app import app
from gymrat.errors import GymratError
from gymrat.measure import MeasureOptions, measure
from gymrat.report.types import MeasurementResult
from gymrat.sampling import TargetSpec
from gymrat.session.records import BaselineRecord, CommandRecord
from tests._clock import install_monotonic_clock
from tests.cli._command_stubs import (
    capture_measure,
    stub_measure,
    stub_resolve,
)
from tests.cli._runner import (
    runner,
)
from tests.cli._session import (
    open_session,
)
from tests.loop._probe import install_measure
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    last_command_record,
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
def record_repo(repo: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """A scratch git repo, chdir'd into, with ``resolve_config`` stubbed for ``--record`` tests."""
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
    clock = install_monotonic_clock(monkeypatch)
    install_measure(
        monkeypatch,
        create_measurement_result(label=label, rounds=rounds),
        on_call=lambda: clock.tick(500.0),
    )

    result = runner.invoke(app, ["measure", positional, "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    baselines = records_of_type(record_repo, BaselineRecord)
    assert len(baselines) == 1
    recorded = baselines[0]
    assert recorded.at > 0
    assert recorded.label == label
    assert recorded.samples == tuple(rounds)
    assert recorded.duration_ms == 500
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
# command trace — args and exit recording
# ---------------------------------------------------------------------------


_BENCH_TRACE = {"bench": "sh bench.sh"}
"""The config override every traced measure below is invoked with."""


@pytest.mark.parametrize(
    ("argv", "traced"),
    [
        pytest.param(["main"], {**_BENCH_TRACE, "target": "main"}, id="bare-ref"),
        pytest.param([], {**_BENCH_TRACE, "target": "."}, id="default-target"),
        pytest.param(
            ["build=main", "--record"],
            {**_BENCH_TRACE, "target": "build", "record": True},
            id="recorded",
        ),
    ],
)
def test_measure_when_success_does_record_trace_with_only_the_args_given(
    argv: list[str],
    traced: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    open_session(repo)
    stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", *argv, "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("measure", 0, None)
    assert cmd.args == traced


def test_measure_when_bench_fails_does_record_trace_with_exit_two_error(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    open_session(repo)
    stub_resolve(monkeypatch)

    async def failing_measure(_options: MeasureOptions) -> MeasurementResult:
        msg = "bench exploded"
        raise GymratError(msg)

    monkeypatch.setattr(
        "gymrat.measure.measure", create_autospec(measure, side_effect=failing_measure)
    )

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == 2
    cmd = last_command_record(repo)
    assert cmd.args["target"] == "main"
    assert cmd.exit_code == 2
    assert cmd.reason == "error"
