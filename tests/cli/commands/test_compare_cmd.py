"""Tests for the ``gymrat compare`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``compare`` seam replaced, so no real bench runs, and config resolution stubbed
except where flags and an explicit config file are read for real. They cover
those flags reaching the engine's options, the text and JSON report going to
stdout, the fail-on gate tripping to exit 1 only after the report is printed,
the empty-geomean warning, and the command trace. The fail-on gate evaluation is
also driven directly. The missing-bench error is pinned with ``measure``'s in
``test_measure_cmd``. The budget time-left line comes from the shared
``emit_report`` path, pinned through ``probe`` in ``test_session_cmds``; the
tight-budget warning is pinned there for compare.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.cli.commands.compare import should_fail_gate
from gymrat.config import KindEntry, MetricEntry
from gymrat.report.json_doc import render_json
from gymrat.report.text.render import render_report
from gymrat.report.types import (
    ComparisonResult,
    GeomeanFailOn,
    RegressedFailOn,
    ReportOptions,
)
from gymrat.sampling import RunOptions, SamplingOptions
from tests.cli._session import (
    open_session,
    runner,
    stub_compare,
    stub_compare_command,
)
from tests.config._toml import write_config
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    other_kind,
    permutation_metric,
    without_gated_geomean,
)
from tests.session.records._fixtures import last_command_record


def _regressed_result() -> ComparisonResult:
    """A comparison whose single gating metric regressed, so a fail-on gate trips."""
    return create_comparison_result(
        metrics={"m/time": permutation_metric(verdict="regressed", delta=4, gating=True)},
        candidates=[create_candidate()],
    )


# ---------------------------------------------------------------------------
# flags and config file → CompareOptions
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_flags_and_config_file_given_does_forward_them_to_compare_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    write_config(
        tmp_path,
        {
            "metrics": {"decode/time": {"direction": "higher"}},
            "kinds": {"memory": {"gating": False}},
        },
        name="compare.toml",
    )
    fake_compare = stub_compare(monkeypatch)

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
            "compare.toml",
        ],
    )

    assert result.exit_code == 0
    (options,) = fake_compare.call_args.args
    sampling = options.run.sampling
    assert options.run == RunOptions(
        sampling=SamplingOptions(
            bench="sh bench.sh",
            prepare="make",
            samples=7,
            timeout_seconds=42,
            on_progress=sampling.on_progress,
            warn=sampling.warn,
        ),
        adapter="mitata",
        config_metrics={"decode/time": MetricEntry(direction="higher")},
        config_kinds={"memory": KindEntry(gating=False)},
    )


# ---------------------------------------------------------------------------
# report to stdout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra_argv", "comparison", "render"),
    [
        pytest.param(
            ["--format", "text"],
            create_comparison_result(),
            partial(render_report, options=ReportOptions(color=False)),
            id="text",
        ),
        pytest.param(
            ["-v"],
            _regressed_result(),
            partial(render_report, options=ReportOptions(verbose=True, color=False)),
            id="short-verbose-flag",
        ),
        pytest.param(["--format", "json"], create_comparison_result(), render_json, id="json"),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_report_rendered_does_write_it_to_stdout(
    monkeypatch: pytest.MonkeyPatch,
    extra_argv: list[str],
    comparison: ComparisonResult,
    render: Callable[[ComparisonResult], str],
):
    stub_compare_command(monkeypatch, comparison)

    result = runner.invoke(app, ["compare", "main", "cand", "--bench", "sh bench.sh", *extra_argv])

    assert result.exit_code == 0
    assert result.stdout == render(comparison) + "\n"


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
    stub_compare_command(
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
    stub_compare_command(monkeypatch)

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
    stub_compare_command(monkeypatch, _regressed_result())

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
