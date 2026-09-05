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

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.config import ResolvedConfig
from gymrat.measure import MeasureOptions
from gymrat.report.types import MeasurementResult
from gymrat.sampling import TargetSpec
from gymrat.session import BaselineRecord, append_record, read_records, session_jsonl_path
from tests.cli._budget import install_budget, install_tight_budget
from tests.report._inputs import create_measurement_result
from tests.session.records._fixtures import (
    finalize_record,
    iteration_record,
    session_record,
    write_session_log,
)

runner = CliRunner()

# An ISO-8601 timestamp at millisecond precision, ``Z``-suffixed — the shape the
# session writer stamps every record with.
ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def _resolved() -> ResolvedConfig:
    """A resolved config the fake ``measure`` never actually benches against."""
    return ResolvedConfig(
        bench="sh bench.sh",
        prepare=None,
        adapter="metric-lines",
        samples=5,
        timeout_seconds=30,
        unstable_noise_pct=2.0,
        primary="time",
    )


@pytest.fixture
def _in_non_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run from a directory that is not a git repo, so the command benches lock-free."""
    monkeypatch.chdir(tmp_path)


def _stub_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*_a: object, **_k: object) -> ResolvedConfig:
        return _resolved()

    monkeypatch.setattr("gymrat.cli.measure_cmd.resolve_config", fake)


def _capture_measure(
    monkeypatch: pytest.MonkeyPatch, result: MeasurementResult | None = None
) -> list[MeasureOptions]:
    """Replace the ``measure`` seam with a fake that records the options it received.

    The fake hands back ``result`` (a default clean run when omitted), so a test
    can pin the label and raw rounds a recording is built from, or assert the
    seam was never reached by checking the returned list stayed empty.
    """
    captured: list[MeasureOptions] = []
    handed_back = create_measurement_result() if result is None else result

    async def fake_measure(options: MeasureOptions) -> MeasurementResult:
        captured.append(options)
        return handed_back

    monkeypatch.setattr("gymrat.measure.measure", fake_measure)
    return captured


def _stub_measure(
    monkeypatch: pytest.MonkeyPatch, result: MeasurementResult | None = None
) -> list[MeasureOptions]:
    _stub_resolve(monkeypatch)
    return _capture_measure(monkeypatch, result)


# ---------------------------------------------------------------------------
# target defaulting
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_no_target_given_does_default_to_current_directory(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert captured[0].target == TargetSpec(label=None, target=".")


# ---------------------------------------------------------------------------
# report to stdout
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_run_does_render_report_to_stdout(monkeypatch: pytest.MonkeyPatch):
    _stub_measure(monkeypatch)

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
    _stub_resolve(monkeypatch)
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
    _capture_measure(monkeypatch, create_measurement_result(label=label, rounds=rounds))

    result = runner.invoke(app, ["measure", positional, "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    recorded = read_records(session_jsonl_path(record_repo))[-1]
    assert isinstance(recorded, BaselineRecord)
    assert recorded.type == "baseline"
    assert ISO_PATTERN.match(recorded.at)
    assert recorded.label == label
    assert recorded.samples == tuple(rounds)
    assert label in result.stdout
    assert re.search(r"recorded to session", result.stdout, re.IGNORECASE)


def test_measure_when_record_and_json_format_does_route_note_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _open_session(record_repo)
    _capture_measure(monkeypatch, create_measurement_result(rounds=[{"latency": 42}]))

    result = runner.invoke(
        app, ["measure", "main", "--bench", "sh bench.sh", "--record", "--format", "json"]
    )

    assert result.exit_code == 0
    assert "recorded to session" not in result.stdout
    assert re.search(r"recorded to session", result.stderr, re.IGNORECASE)


@pytest.mark.usefixtures("record_repo")
def test_measure_when_record_and_no_session_does_exit_two_without_benching(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = _capture_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 2
    assert captured == []
    assert "gymrat start" in result.stderr


def test_measure_when_record_and_finalized_session_does_exit_two_without_benching(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _finalize_session(record_repo)
    captured = _capture_measure(monkeypatch)

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
    _capture_measure(monkeypatch, create_measurement_result(rounds=[{"latency": 42}]))

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert read_records(session_jsonl_path(record_repo)) == [session_record()]
    assert "recorded to session" not in result.stdout


# ---------------------------------------------------------------------------
# --record duration
# ---------------------------------------------------------------------------


def test_measure_when_record_does_write_duration_ms_to_baseline(
    monkeypatch: pytest.MonkeyPatch,
    record_repo: str,
):
    _open_session(record_repo)
    _capture_measure(monkeypatch, create_measurement_result(rounds=[{"latency": 42}]))
    ticks = iter([1_000.0, 1_500.0])
    monkeypatch.setattr("gymrat.session.clock.monotonic_ms", lambda: next(ticks))

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", "--record"])

    assert result.exit_code == 0
    recorded = read_records(session_jsonl_path(record_repo))[-1]
    assert isinstance(recorded, BaselineRecord)
    assert recorded.duration_ms == 500


# ---------------------------------------------------------------------------
# budget time-left line (text) and key (JSON) on measure
# ---------------------------------------------------------------------------


def test_measure_when_budget_active_does_end_text_with_time_left_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_measure(monkeypatch)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    lines = [line.strip() for line in result.stdout.split("\n") if line.strip()]
    assert re.search(r"left of 30m", lines[-1])


def test_measure_when_no_budget_does_omit_time_left_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert "left of" not in result.stdout


def test_measure_when_format_json_and_budget_active_does_include_budget_object(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_measure(monkeypatch)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["capMinutes"] == 30
    assert isinstance(doc["budget"]["remainingSeconds"], int)


def test_measure_when_format_json_and_no_budget_does_omit_budget_key(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_measure(monkeypatch)

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
    """When half the estimated duration exceeds budget remaining, measure warns."""
    _open_session(record_repo)
    _capture_measure(monkeypatch)
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
    _capture_measure(monkeypatch)
    install_tight_budget(record_repo, monkeypatch)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert "warning" not in result.stderr.lower()
