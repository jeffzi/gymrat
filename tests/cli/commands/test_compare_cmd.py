"""Tests for the ``gymrat compare`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` with the
``compare`` seam replaced, so no real bench runs. They cover the fail-on gate
tripping to exit 1, the empty-geomean warning, and the command trace. The
fail-on gate evaluation is also driven directly. The budget time-left line and
the tight-budget warning come from the shared ``emit_report`` path, pinned in
``tests/cli/test_budget_report.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from gymrat.cli.app import app
from gymrat.cli.commands.compare import should_fail_gate
from gymrat.report.types import (
    ComparisonResult,
    GeomeanFailOn,
    RegressedFailOn,
)
from tests._ansi import warning_lines
from tests.cli._command_stubs import (
    stub_compare_command,
)
from tests.cli._runner import (
    runner,
)
from tests.cli._session import (
    open_session,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    other_kind,
    permutation_metric,
)
from tests.session.records._fixtures import last_command_record


def _regressed_result() -> ComparisonResult:
    """A comparison whose single gating metric regressed, so a fail-on gate trips."""
    return create_comparison_result(
        metrics={"m/time": permutation_metric(verdict="regressed", delta=4, gating=True)},
        candidates=[create_candidate()],
    )


# ---------------------------------------------------------------------------
# fail-on gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fail_on", "named"),
    [
        pytest.param("geomean:2", ["cand-empty"], id="geomean-gate-names-the-empty-candidate"),
        pytest.param("regressed", [], id="no-geomean-gate-stays-silent"),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_compare_when_a_candidate_has_no_stable_gating_metric_does_warn_only_under_a_geomean_gate(
    monkeypatch: pytest.MonkeyPatch, fail_on: str, named: list[str]
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

    warnings = warning_lines(result.stderr)
    assert len(warnings) == len(named)
    assert all(f'"{label}"' in warning for label, warning in zip(named, warnings, strict=True))


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
        pytest.param(
            (GeomeanFailOn(pct=2.0),),
            {"candidates": [create_candidate(kinds=[other_kind(5.0, 3)])]},
            True,
            id="geomean-above-threshold",
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


_BENCH_TRACE = {"bench": "sh bench.sh"}
"""The config override every traced compare below is invoked with."""


@pytest.mark.parametrize(
    ("argv", "traced"),
    [
        pytest.param(
            ["main", "cand"],
            {**_BENCH_TRACE, "baseline": "main", "candidates": ["cand"]},
            id="no-fail-on",
        ),
        pytest.param(
            ["main", "cand", "--fail-on", "regressed", "--fail-on", "geomean:99.5"],
            {
                **_BENCH_TRACE,
                "baseline": "main",
                "candidates": ["cand"],
                "fail_on": "regressed,geomean:99.5",
            },
            id="regressed-and-geomean-fail-on",
        ),
        pytest.param(
            ["before=main", "after=cand"],
            {**_BENCH_TRACE, "baseline": "before", "candidates": ["after"]},
            id="labeled-targets",
        ),
    ],
)
def test_compare_when_success_does_record_trace_with_only_the_args_given(
    *,
    argv: list[str],
    traced: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
):
    open_session(repo)
    stub_compare_command(monkeypatch)

    result = runner.invoke(app, ["compare", *argv, "--bench", "sh bench.sh"])

    assert result.exit_code == 0
    cmd = last_command_record(repo)
    assert (cmd.exit_code, cmd.reason) == (0, None)
    assert cmd.args == traced


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
    cmd = last_command_record(repo)
    assert (cmd.exit_code, cmd.reason) == (1, "fail-on")
