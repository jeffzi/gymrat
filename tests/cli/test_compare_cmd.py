"""Tests for the ``gymrat compare`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``compare`` and ``resolve_config`` seams replaced, so no real bench runs. They
cover flag parsing into the config resolver, the text and JSON report going to
stdout, the missing-bench error routing to exit 2 on stderr, and the fail-on
gate tripping to exit 1 only after the report is printed.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.config import CliFlags, ResolvedConfig
from gymrat.report.types import ComparisonResult
from tests.report._inputs import (
    create_candidate,
    create_comparison_result,
    permutation_metric,
)

runner = CliRunner()


def _resolved(bench: str = "sh bench.sh") -> ResolvedConfig:
    """A resolved config the fake ``compare`` never actually benches against."""
    return ResolvedConfig(
        bench=bench,
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


def _patch_compare(monkeypatch: pytest.MonkeyPatch, result: ComparisonResult) -> None:
    """Replace the ``compare`` seam with a fake returning ``result``."""

    async def fake_compare(_options: object) -> ComparisonResult:
        return result

    monkeypatch.setattr("gymrat.compare.compare", fake_compare)


def _stub_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``resolve_config`` with one that returns a fixed resolved config."""

    def fake(*_a: object, **_k: object) -> ResolvedConfig:
        return _resolved()

    monkeypatch.setattr("gymrat.cli.compare_cmd.resolve_config", fake)


def _regressed_result() -> ComparisonResult:
    """A comparison whose single gating metric regressed, so a fail-on gate trips."""
    return create_comparison_result(
        metrics={"m/time": permutation_metric(verdict="regressed", delta=4, gating=True)},
        candidates=[create_candidate()],
    )


# ---------------------------------------------------------------------------
# flag parsing → resolve_config
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_flags_given_does_feed_them_to_resolve_config(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[CliFlags] = []

    def spy_resolve(flags: CliFlags, base_dir: object = None) -> ResolvedConfig:
        captured.append(flags)
        return _resolved()

    monkeypatch.setattr("gymrat.cli.compare_cmd.resolve_config", spy_resolve)
    _patch_compare(monkeypatch, create_comparison_result())

    result = runner.invoke(
        app,
        [
            "compare",
            "main",
            "cand",
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
    assert len(captured) == 1
    flags = captured[0]
    assert flags.bench == "sh bench.sh"
    assert flags.prepare == "make"
    assert flags.adapter == "mitata"
    assert flags.samples == 7
    assert flags.timeout == 42
    assert flags.config == "gymrat.json"


# ---------------------------------------------------------------------------
# report to stdout
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_format_text_does_render_report_to_stdout(monkeypatch: pytest.MonkeyPatch):
    _stub_resolve(monkeypatch)
    _patch_compare(monkeypatch, create_comparison_result())

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--format", "text"]
    )

    assert result.exit_code == 0
    assert "main" in result.stdout


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_format_json_does_render_json_document_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
):
    _stub_resolve(monkeypatch)
    _patch_compare(monkeypatch, create_comparison_result())

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--format", "json"]
    )

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["baseline"] == "main"


# ---------------------------------------------------------------------------
# missing bench
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_bench_missing_does_exit_two_with_message_on_stderr():
    result = runner.invoke(app, ["compare", "main", "cand"])

    assert result.exit_code == 2
    assert "bench is required" in result.stderr
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# fail-on gate
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_fail_on_trips_does_exit_one_after_printing_report(
    monkeypatch: pytest.MonkeyPatch,
):
    _stub_resolve(monkeypatch)
    _patch_compare(monkeypatch, _regressed_result())

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--fail-on", "regressed"]
    )

    assert result.exit_code == 1
    assert "main" in result.stdout


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_fail_on_does_not_trip_does_exit_zero(monkeypatch: pytest.MonkeyPatch):
    _stub_resolve(monkeypatch)
    _patch_compare(monkeypatch, create_comparison_result())

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--fail-on", "regressed"]
    )

    assert result.exit_code == 0
