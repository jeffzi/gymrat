"""Tests for the ``gymrat compare`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``compare`` and ``resolve_config`` seams replaced, so no real bench runs. They
cover flag parsing into the config resolver, the text and JSON report going to
stdout, the missing-bench error routing to exit 2 on stderr, the fail-on gate
tripping to exit 1 only after the report is printed, the empty-geomean warning,
and the command trace. The fail-on gate evaluation is also driven directly. The
budget time-left line and the tight-budget warning are pinned with every other
command's in ``test_session_cmds``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gymrat.compare import CompareOptions

import pytest

from gymrat.cli.app import app
from gymrat.cli.commands.compare import should_fail_gate
from gymrat.config import CliFlags, KindEntry, MetricEntry, ResolvedConfig
from gymrat.report.text.render import render_report
from gymrat.report.types import (
    ComparisonResult,
    GeomeanFailOn,
    RegressedFailOn,
    ReportOptions,
)
from gymrat.sampling import RunOptions, SamplingOptions
from tests._config import resolved_config
from tests.cli._session import (
    last_command_record,
    open_session,
    runner,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    other_kind,
    permutation_metric,
    without_gated_geomean,
)


def _resolved(bench: str = "sh bench.sh") -> ResolvedConfig:
    """A resolved config the fake ``compare`` never actually benches against."""
    return resolved_config(
        bench=bench,
        prepare="npm ci",
        samples=5,
        timeout_seconds=30,
        unstable_noise_pct=2.0,
        primary="time",
        metrics={"decode/time": MetricEntry(direction="higher")},
        kinds={"memory": KindEntry(gating=False)},
    )


def _patch_compare(monkeypatch: pytest.MonkeyPatch, result: ComparisonResult) -> None:
    """Replace the ``compare`` seam with a fake returning ``result``."""

    async def fake_compare(_options: object) -> ComparisonResult:
        return result

    monkeypatch.setattr("gymrat.compare.compare", fake_compare)


def _stub_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``resolve_config`` with one that returns a fixed resolved config."""

    def fake(*_a: object, **_k: object) -> ResolvedConfig:
        return _resolved()

    monkeypatch.setattr("gymrat.cli.commands.compare.resolve_config", fake)


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

    monkeypatch.setattr("gymrat.cli.commands.compare.resolve_config", spy_resolve)
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
# run options → CompareOptions
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_run_options_built_does_forward_every_field_to_compare_options(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[CompareOptions] = []

    async def spy_compare(options: CompareOptions) -> ComparisonResult:
        captured.append(options)
        return create_comparison_result()

    _stub_resolve(monkeypatch)
    monkeypatch.setattr("gymrat.compare.compare", spy_compare)

    result = runner.invoke(app, ["compare", "main", "cand", "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    assert len(captured) == 1
    resolved = _resolved()
    run = captured[0].run
    sampling = run.sampling
    assert run == RunOptions(
        sampling=SamplingOptions(
            bench=resolved.bench,
            prepare=resolved.prepare,
            samples=resolved.samples,
            timeout_seconds=resolved.timeout_seconds,
            on_progress=sampling.on_progress,
            warn=sampling.warn,
        ),
        adapter=resolved.adapter,
        config_metrics=resolved.metrics,
        config_kinds=resolved.kinds,
    )


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


@pytest.mark.parametrize(
    ("fail_on", "warnings"),
    [
        pytest.param(
            "geomean:2",
            ['warning: geomean gate for "cand-empty" had no stable gating metrics to measure'],
            id="geomean-gate-names-the-empty-candidate",
        ),
        pytest.param("regressed", [], id="no-geomean-gate-stays-silent"),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_a_candidate_has_no_stable_gating_metric_does_warn_only_under_a_geomean_gate(
    monkeypatch: pytest.MonkeyPatch, fail_on: str, warnings: list[str]
):
    _stub_compare(
        monkeypatch,
        create_comparison_result(
            candidates=[
                create_candidate(label="cand-empty", kinds=[other_kind(5.0, 0)]),
                create_candidate(label="cand-ok", kinds=[other_kind(5.0, 3)]),
            ]
        ),
    )

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--fail-on", fail_on]
    )

    assert [line for line in result.stderr.splitlines() if line.startswith("warning: ")] == warnings


# ---------------------------------------------------------------------------
# should_fail_gate
# ---------------------------------------------------------------------------


_REGRESSED_GATING = {"m/time": permutation_metric(verdict="regressed", delta=4, gating=True)}


@pytest.mark.parametrize(
    ("conditions", "result_args", "expected"),
    [
        pytest.param(
            (),
            {"metrics": _REGRESSED_GATING, "candidates": [create_candidate()]},
            False,
            id="no-conditions",
        ),
        pytest.param(
            (RegressedFailOn(),),
            {"metrics": _REGRESSED_GATING, "candidates": [create_candidate()]},
            True,
            id="regressed-gating-metric",
        ),
        pytest.param(
            (RegressedFailOn(),),
            {
                "metrics": {
                    "m/time": permutation_metric(verdict="regressed", delta=4, gating=False)
                },
                "candidates": [create_candidate()],
            },
            False,
            id="regression-non-gating",
        ),
        *(
            pytest.param(
                (GeomeanFailOn(pct=2.0),),
                {"candidates": [create_candidate(kinds=[other_kind(geomean, 3)])]},
                expected,
                id=f"geomean-{label}",
            )
            for geomean, expected, label in (
                (5.0, True, "above-threshold"),
                (2.0, True, "exactly-on-threshold"),
                (1.0, False, "below-threshold"),
            )
        ),
        pytest.param(
            (GeomeanFailOn(pct=2.0),),
            {"candidates": [create_candidate(kinds=[other_kind(5.0, 0)])]},
            False,
            id="gated-geomean-without-samples",
        ),
        pytest.param(
            (GeomeanFailOn(pct=2.0),),
            {"candidates": [create_candidate(kinds=[without_gated_geomean(other_kind(5.0, 3))])]},
            False,
            id="kind-non-gating",
        ),
        pytest.param(
            (RegressedFailOn(), GeomeanFailOn(pct=99.0)),
            {
                "metrics": _REGRESSED_GATING,
                "candidates": [create_candidate(kinds=[other_kind(1.0, 3)])],
            },
            True,
            id="any-of-several-conditions",
        ),
        pytest.param(
            (GeomeanFailOn(pct=99.0), RegressedFailOn()),
            {
                "metrics": _REGRESSED_GATING,
                "candidates": [create_candidate(kinds=[other_kind(1.0, 3)])],
            },
            True,
            id="tripping-condition-after-a-quiet-one",
        ),
    ],
)
def test_should_fail_gate_when_evaluated_does_trip_only_on_a_matching_gating_condition(
    conditions: tuple[RegressedFailOn | GeomeanFailOn, ...],
    result_args: dict[str, Any],
    expected: bool,
):
    result = create_comparison_result(**result_args)

    assert should_fail_gate(conditions, result) is expected


# ---------------------------------------------------------------------------
# command trace — args and exit recording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "baseline", "candidates", "fail_on"),
    [
        pytest.param(["main", "cand"], "main", ["cand"], "", id="no-fail-on"),
        pytest.param(
            ["main", "cand", "--fail-on", "regressed"],
            "main",
            ["cand"],
            "regressed",
            id="fail-on-not-tripped",
        ),
        pytest.param(
            ["main", "cand", "--fail-on", "geomean:99.5"],
            "main",
            ["cand"],
            "geomean:99.5",
            id="geomean-fail-on",
        ),
        pytest.param(
            ["main", "cand", "--fail-on", "regressed", "--fail-on", "geomean:99.5"],
            "main",
            ["cand"],
            "regressed,geomean:99.5",
            id="regressed-and-geomean-fail-on",
        ),
        pytest.param(["before=main", "after=cand"], "before", ["after"], "", id="labeled-targets"),
        pytest.param(
            ["main", "cand1", "cand2"], "main", ["cand1", "cand2"], "", id="two-candidates"
        ),
    ],
)
def test_compare_when_success_does_record_trace_with_baseline_candidates_and_fail_on(
    *,
    argv: list[str],
    baseline: str,
    candidates: list[str],
    fail_on: str,
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    open_session(repo)
    _stub_compare(monkeypatch)

    result = runner.invoke(app, ["compare", *argv, "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("compare", 0, None)
    assert (cmd.args["baseline"], cmd.args["candidates"]) == (baseline, candidates)
    assert (cmd.args["fail_on"], cmd.args["bench"]) == (fail_on, "sh bench.sh")


def test_compare_when_fail_on_trips_does_exit_one_as_a_fail_on_gate_trip(
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    open_session(repo)
    _stub_compare(monkeypatch, _regressed_result())

    result = runner.invoke(
        app, ["compare", "main", "cand", "--bench", "sh bench.sh", "--fail-on", "regressed"]
    )

    assert result.exit_code == 1
    assert result.stdout == (
        render_report(_regressed_result(), ReportOptions(fail_on=(RegressedFailOn(),), color=False))
        + "\n"
    )
    cmd = last_command_record(repo)
    assert (cmd.exit_code, cmd.reason) == (1, "fail-on")


# ---------------------------------------------------------------------------
# -v short form works like --verbose
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_short_verbose_flag_does_succeed(monkeypatch: pytest.MonkeyPatch):
    _stub_compare(monkeypatch)

    result = runner.invoke(app, ["compare", "main", "cand", "--bench", "sh bench.sh", "-v"])

    assert result.exit_code == 0
