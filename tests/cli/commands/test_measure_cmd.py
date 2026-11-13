"""Tests for the ``gymrat measure`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``measure`` and ``resolve_config`` seams replaced. They cover the optional
target defaulting to ``.``, the ``--record`` flag that appends the run to an
open session log as a baseline, and the command trace.
The budget time-left line and the tight-budget warning come from the shared
``emit_report`` path, pinned in ``tests/cli/test_budget_report.py``.
"""

import pytest

from gymrat.cli.app import app
from gymrat.sampling import TargetSpec
from gymrat.session.records import BaselineRecord, CommandRecord
from tests.cli._command_stubs import (
    capture_measure,
    stub_measure,
    stub_resolve,
)
from tests.cli._runner import (
    runner,
)
from tests.cli._session import (
    close_session_with_one_keep,
    open_session,
    write_bench_config,
)
from tests.loop._probe import install_measure
from tests.loop._settle import CHECKS
from tests.report._measurements import create_measurement_result
from tests.session.records._fixtures import (
    SESSION_ID,
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
# --record
# ---------------------------------------------------------------------------


@pytest.fixture
def record_repo(repo: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """A scratch git repo, chdir'd into, with ``resolve_config`` stubbed for ``--record`` tests."""
    stub_resolve(monkeypatch)
    return repo


def test_measure_when_record_and_open_session_does_record_the_run_as_a_baseline(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    open_session(record_repo)
    install_measure(monkeypatch, create_measurement_result(label="build"))

    result = runner.invoke(app, ["measure", "build=main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    assert len(records_of_type(record_repo, BaselineRecord)) == 1


def test_measure_when_record_and_text_format_does_print_note_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    open_session(record_repo)
    install_measure(monkeypatch, create_measurement_result(label="build"))

    result = runner.invoke(app, ["measure", "build=main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    assert SESSION_ID in result.stdout


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
    assert SESSION_ID not in result.stdout
    assert SESSION_ID in result.stderr


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


def test_measure_when_record_and_session_finalized_does_refuse_as_finalized(repo: str):
    close_session_with_one_keep(repo)
    write_bench_config(repo, checks=CHECKS)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 2
    assert last_command_record(repo).reason == "finalized"


# ---------------------------------------------------------------------------
# command trace — args and exit recording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "traced"),
    [
        pytest.param(["main"], {"target": "main"}, id="bare-ref"),
        pytest.param(
            ["build=main", "--record"],
            {"target": "build", "record": True},
            id="recorded",
        ),
    ],
)
def test_measure_when_success_does_record_the_resolved_target_with_only_the_flags_given(
    argv: list[str],
    traced: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    open_session(repo)
    stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", *argv])

    assert result.exit_code == 0
    assert last_command_record(repo).args == traced
