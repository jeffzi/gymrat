"""Tests for the ``gymrat measure`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``measure`` and ``resolve_config`` seams replaced. They cover the optional
target defaulting to ``.``, the report going to stdout, the missing-bench error
routing to exit 2, and the rejection of ``--verbose``/``--fail-on`` — options
the compare command owns but measure does not — as usage errors.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat_py.cli.app import app
from gymrat_py.config import ResolvedConfig
from gymrat_py.measure import MeasureOptions
from gymrat_py.report.types import MeasurementResult
from gymrat_py.sampling import TargetSpec
from tests.report._inputs import create_measurement_result

runner = CliRunner()


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

    monkeypatch.setattr("gymrat_py.cli.measure_cmd.resolve_config", fake)


def _capture_measure(monkeypatch: pytest.MonkeyPatch) -> list[MeasureOptions]:
    """Replace the ``measure`` seam with a fake that records the options it received."""
    captured: list[MeasureOptions] = []

    async def fake_measure(options: MeasureOptions) -> MeasurementResult:
        captured.append(options)
        return create_measurement_result()

    monkeypatch.setattr("gymrat_py.measure.measure", fake_measure)
    return captured


# ---------------------------------------------------------------------------
# target defaulting
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_no_target_given_does_default_to_current_directory(
    monkeypatch: pytest.MonkeyPatch,
):
    _stub_resolve(monkeypatch)
    captured = _capture_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert captured[0].target == TargetSpec(label=None, target=".")


# ---------------------------------------------------------------------------
# report to stdout
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_run_does_render_report_to_stdout(monkeypatch: pytest.MonkeyPatch):
    _stub_resolve(monkeypatch)
    _capture_measure(monkeypatch)

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
        pytest.param(["--record"], id="record"),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_measure_when_compare_only_option_given_does_exit_two(option: list[str]):
    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh", *option])

    assert result.exit_code == 2
