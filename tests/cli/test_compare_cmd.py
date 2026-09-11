"""Tests for the ``gymrat compare`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``compare`` and ``resolve_config`` seams replaced, so no real bench runs. They
cover flag parsing into the config resolver, the text and JSON report going to
stdout, the missing-bench error routing to exit 2 on stderr, the fail-on gate
tripping to exit 1 only after the report is printed, budget time-left reporting
in text and JSON output (including on gate-refusal), and duration warnings when
the budget is tight.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.cli.compare_cmd import _serialize_fail_on
from gymrat.config import CliFlags, ResolvedConfig
from gymrat.report.types import (
    ComparisonResult,
    FailOnCondition,
    GeomeanFailOn,
    RegressedFailOn,
)
from gymrat.session import append_record, session_jsonl_path
from tests.cli._budget import install_budget, install_tight_budget
from tests.cli._loop_cmds import last_command_record
from tests.report._inputs import (
    create_candidate,
    create_comparison_result,
    permutation_metric,
)
from tests.session.records._fixtures import iteration_record, session_record, write_session_log

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


def _stub_compare(monkeypatch: pytest.MonkeyPatch, result: ComparisonResult | None = None) -> None:
    """Stub ``resolve_config`` and ``compare`` so invoking the command succeeds.

    ``result`` becomes the comparison the fake ``compare`` returns; defaults to
    a comparison with no regressions.
    """
    _stub_resolve(monkeypatch)
    _patch_compare(monkeypatch, create_comparison_result() if result is None else result)


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
    _stub_compare(monkeypatch)

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--format", "text"]
    )

    assert result.exit_code == 0
    assert "main" in result.stdout


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_format_json_does_render_json_document_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
):
    _stub_compare(monkeypatch)

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
    _stub_compare(monkeypatch, _regressed_result())

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--fail-on", "regressed"]
    )

    assert result.exit_code == 1
    assert "main" in result.stdout


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_fail_on_does_not_trip_does_exit_zero(monkeypatch: pytest.MonkeyPatch):
    _stub_compare(monkeypatch)

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--fail-on", "regressed"]
    )

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# budget time-left line (text) and key (JSON) on compare
# ---------------------------------------------------------------------------


def test_compare_when_budget_active_does_end_text_with_time_left_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_compare(monkeypatch)
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["compare", "main", "cand", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    lines = [line.strip() for line in result.stdout.split("\n") if line.strip()]
    assert re.search(r"left of 30m", lines[-1])


def test_compare_when_no_budget_does_omit_time_left_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_compare(monkeypatch)

    result = runner.invoke(app, ["compare", "main", "cand", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert "left of" not in result.stdout


def test_compare_when_format_json_and_budget_active_does_include_budget_object(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_compare(monkeypatch)
    install_budget(repo, monkeypatch)

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--format", "json"]
    )

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30
    assert isinstance(doc["budget"]["remaining_seconds"], int)


def test_compare_when_format_json_and_no_budget_does_omit_budget_key(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_compare(monkeypatch)

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--format", "json"]
    )

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert "budget" not in doc


# ---------------------------------------------------------------------------
# budget on gate-refusal output
# ---------------------------------------------------------------------------


def test_compare_when_fail_on_trips_and_budget_active_does_include_time_left_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_compare(monkeypatch, _regressed_result())
    install_budget(repo, monkeypatch)

    result = runner.invoke(
        app,
        ["compare", "main", "cand", "--bench", "sh bench.sh", "--fail-on", "regressed"],
    )

    assert result.exit_code == 1
    assert re.search(r"left of 30m", result.stdout)


def test_compare_when_fail_on_trips_and_format_json_and_budget_active_does_include_budget_key(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_compare(monkeypatch, _regressed_result())
    install_budget(repo, monkeypatch)

    result = runner.invoke(
        app,
        [
            "compare",
            "main",
            "cand",
            "--bench",
            "sh bench.sh",
            "--fail-on",
            "regressed",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    doc = json.loads(result.stdout)
    assert "budget" in doc
    assert doc["budget"]["cap_minutes"] == 30


# ---------------------------------------------------------------------------
# budget absent on error exits
# ---------------------------------------------------------------------------


def test_compare_when_error_and_budget_active_does_not_include_budget(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    install_budget(repo, monkeypatch)

    result = runner.invoke(app, ["compare", "main", "cand"])

    assert result.exit_code == 2
    assert "left of" not in result.stdout
    assert "left of" not in result.stderr


# ---------------------------------------------------------------------------
# duration warnings
# ---------------------------------------------------------------------------


def _write_session_with_duration(repo: str, duration_ms: float) -> None:
    """Write a session log with one iteration record carrying a known duration."""
    write_session_log(repo, session_record())
    append_record(session_jsonl_path(repo), iteration_record(duration_ms=duration_ms))


def test_compare_when_budget_tight_and_estimate_known_does_warn_on_stderr(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_compare(monkeypatch)
    install_tight_budget(repo, monkeypatch)
    _write_session_with_duration(repo, 720_000)

    result = runner.invoke(app, ["compare", "main", "cand", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert "warning" in result.stderr.lower()


def test_compare_when_estimate_unknown_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _stub_compare(monkeypatch)
    install_tight_budget(repo, monkeypatch)

    result = runner.invoke(app, ["compare", "main", "cand", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert "warning" not in result.stderr.lower()


# ---------------------------------------------------------------------------
# command trace — args and exit recording
# ---------------------------------------------------------------------------


def _open_session(repo: str) -> None:
    """Open a session in ``repo`` so the command trace has somewhere to write."""
    write_session_log(repo, session_record())


def test_compare_when_success_does_record_trace_with_baseline_candidates_fail_on(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _open_session(repo)
    _stub_compare(monkeypatch)

    result = runner.invoke(app, ["compare", "main", "cand", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.name == "compare"
    assert cmd.args["baseline"] == "main"
    assert cmd.args["candidates"] == ["cand"]
    assert cmd.args["fail_on"] == ""
    assert cmd.exit_code == 0
    assert cmd.reason is None
    for key in ("prepare", "adapter", "samples", "timeout", "config"):
        assert key not in cmd.args


def test_compare_when_config_overrides_given_does_include_them_in_trace_args(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _open_session(repo)
    _stub_compare(monkeypatch)

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
    cmd = last_command_record(repo)
    assert cmd.args["bench"] == "sh bench.sh"
    assert cmd.args["prepare"] == "make"
    assert cmd.args["adapter"] == "mitata"
    assert cmd.args["samples"] == 7
    assert cmd.args["timeout"] == 42
    assert cmd.args["config"] == "gymrat.json"


def test_compare_when_multiple_candidates_does_record_all_in_trace_args(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _open_session(repo)
    _stub_compare(monkeypatch)

    result = runner.invoke(app, ["compare", "main", "cand1", "cand2", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.args["candidates"] == ["cand1", "cand2"]


def test_compare_when_fail_on_trips_does_record_exit_one_with_gate_reason(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _open_session(repo)
    _stub_compare(monkeypatch, _regressed_result())

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--fail-on", "regressed"]
    )

    assert result.exit_code == 1
    cmd = last_command_record(repo)
    assert cmd.exit_code == 1
    assert cmd.reason == "fail-on"


def test_compare_when_fail_on_does_not_trip_does_record_trace_with_baseline_and_exit_zero(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    _open_session(repo)
    _stub_compare(monkeypatch)

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--fail-on", "regressed"]
    )

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert cmd.args["baseline"] == "main"
    assert cmd.args["fail_on"] == "regressed"
    assert cmd.exit_code == 0
    assert cmd.reason is None


# ---------------------------------------------------------------------------
# _serialize_fail_on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("conditions", "expected"),
    [
        pytest.param((RegressedFailOn(),), "regressed", id="regressed-only"),
        pytest.param((GeomeanFailOn(pct=3.5),), "geomean:3.5", id="geomean-only"),
        pytest.param(
            (RegressedFailOn(), GeomeanFailOn(pct=2.5)),
            "regressed,geomean:2.5",
            id="regressed-and-geomean",
        ),
    ],
)
def test_serialize_fail_on_when_known_conditions_does_render_csv(
    conditions: tuple[FailOnCondition, ...],
    expected: str,
) -> None:
    result = _serialize_fail_on(conditions)

    assert result == expected


def test_serialize_fail_on_when_unknown_condition_does_raise() -> None:
    bogus = cast("FailOnCondition", object())

    with pytest.raises(AssertionError, match="Expected code to be unreachable"):
        _serialize_fail_on((bogus,))
