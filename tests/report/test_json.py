"""Tests for the JSON report document builders.

These cover the compare document (``render_json``) and the measure document
(``render_measure_json``), including
their schema shapes, per-metric and per-candidate serialization, worktree
sections, non-finite handling, and the no-ANSI guarantee under forced color.
"""

from __future__ import annotations

import json
import re

import pytest

from gymrat_py.model import Effect, Exclusion, MetricUnit, PermutationVerdict
from gymrat_py.report import render_json, render_measure_json
from gymrat_py.report.types import (
    CandidateMetric,
    ComparisonResult,
    MetricComparison,
)
from gymrat_py.targets import WorktreeRemovalFailure
from gymrat_py.verdict import GroupAggregate, KindAggregate
from tests.report._inputs import (
    NWayCandidate,
    band_metric,
    create_candidate,
    create_comparison_result,
    create_measurement_result,
    exact_metric,
    geomean_of,
    kind_metric,
    measured_metric,
    metric_meta,
    n_way_metric,
    other_kind,
    permutation_metric,
    single_sample_result,
    two_kind_measurement,
)

_ANSI_ESCAPE = re.compile("\x1b\\[")


def _two_kind_with_exclusions() -> ComparisonResult:
    """A run spanning a gating ``time`` kind and an informational ``memory`` kind.

    ``time`` holds a grouped metric and lost two metrics to exclusion rules, so
    its aggregate exercises groups and exclusions at once; ``memory`` holds one
    ungrouped metric and gates nothing, so it pins the no-groups, no-gated case.
    """
    time_geomean = geomean_of(
        -3.2,
        2,
        excluded=[
            Exclusion(metric="jittery/time", reason="unstable"),
            Exclusion(metric="broken/ratio", reason="undefined-ratio"),
        ],
    )
    return create_comparison_result(
        candidates=[
            create_candidate(
                label="experiment",
                kinds=[
                    KindAggregate(
                        kind="time",
                        geomean=time_geomean,
                        groups=(GroupAggregate(group="entity", geomean=geomean_of(-3.1, 2)),),
                        gated_geomean=geomean_of(-3.2, 2),
                    ),
                    KindAggregate(
                        kind="memory",
                        geomean=geomean_of(-7, 1),
                        groups=(),
                        gated_geomean=None,
                    ),
                ],
            ),
        ],
        metrics={
            "entity.alive_check/time": kind_metric(
                kind="time",
                short_name="entity.alive_check",
                verdict="improved",
                delta=-10,
            ),
            "encode/heap": kind_metric(
                kind="memory",
                short_name="encode",
                verdict="improved",
                delta=-7,
                gating=False,
                unit="bytes",
            ),
        },
    )


# ---------------------------------------------------------------------------
# render_json — schema shape
# ---------------------------------------------------------------------------


def test_render_json_when_single_candidate_does_produce_schema_version_2_shape():
    result = create_comparison_result(
        baseline_label="main",
        candidates=[create_candidate(label="experiment")],
        samples=10,
        adapter="mitata",
        metrics={"decode/time": permutation_metric(verdict="improved", delta=-10)},
    )

    doc = json.loads(render_json(result))

    assert doc["schemaVersion"] == 2
    assert doc["baseline"] == "main"
    assert doc["candidates"] == ["experiment"]
    assert doc["samples"] == 10
    assert doc["adapter"] == "mitata"
    assert "metrics" in doc
    assert "perCandidate" in doc
    assert "worktrees" in doc


def test_render_json_when_top_level_keys_does_order_them_canonically():
    result = create_comparison_result(
        metrics={"decode/time": permutation_metric(verdict="improved", delta=-10)},
    )

    doc = json.loads(render_json(result))

    assert list(doc.keys()) == [
        "schemaVersion",
        "baseline",
        "candidates",
        "samples",
        "adapter",
        "metrics",
        "perCandidate",
        "worktrees",
    ]


# ---------------------------------------------------------------------------
# render_json — multi-candidate support
# ---------------------------------------------------------------------------


def test_render_json_when_several_candidates_does_include_all_in_order():
    result = create_comparison_result(
        candidates=[
            create_candidate(label="alpha"),
            create_candidate(label="beta"),
            create_candidate(label="gamma"),
        ],
        metrics={
            "decode/time": n_way_metric(
                [
                    NWayCandidate(verdict="improved", delta=-10, median=90),
                    NWayCandidate(verdict="regressed", delta=5, median=105),
                    NWayCandidate(verdict="no-signal", delta=0.1, median=100.1),
                ],
            ),
        },
    )

    doc = json.loads(render_json(result))

    assert doc["candidates"] == ["alpha", "beta", "gamma"]
    assert len(doc["perCandidate"]) == 3
    assert [entry["label"] for entry in doc["perCandidate"]] == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# render_json — metric verdict methods
# ---------------------------------------------------------------------------


def test_render_json_when_permutation_verdict_does_set_p_and_null_band():
    result = create_comparison_result(
        metrics={"decode/time": permutation_metric(verdict="improved", delta=-10, p=0.003)},
    )

    candidate = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][0]

    assert candidate["method"] == "permutation"
    assert candidate["p"] == 0.003
    assert candidate["band"] is None


def test_render_json_when_band_verdict_does_set_band_and_null_p():
    result = create_comparison_result(
        metrics={"decode/time": band_metric(verdict="no-signal", delta=-1, noise_pct=3.5)},
    )

    candidate = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][0]

    assert candidate["method"] == "band"
    assert candidate["band"] == 3.5
    assert candidate["p"] is None


def test_render_json_when_single_pair_does_store_no_signal_band_verdict():
    candidate = json.loads(render_json(single_sample_result()))["metrics"]["decode/time"][
        "candidates"
    ][0]

    assert candidate["verdict"] == "no-signal"
    assert candidate["method"] == "band"
    assert candidate["noisePct"] == 0.5
    assert candidate["band"] == 0.5


def test_render_json_when_exact_verdict_does_null_noise_p_and_band():
    result = create_comparison_result(metrics={"alloc/heap": exact_metric(delta=-7.9)})

    candidate = json.loads(render_json(result))["metrics"]["alloc/heap"]["candidates"][0]

    assert candidate["method"] == "exact"
    assert candidate["noisePct"] is None
    assert candidate["p"] is None
    assert candidate["band"] is None


def test_render_json_when_delta_is_non_finite_does_render_null():
    metric = MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=(
            CandidateMetric(
                median=90.0,
                spread=1.0,
                verdict=PermutationVerdict(
                    method="permutation",
                    verdict="improved",
                    p=0.01,
                    noise_pct=2.5,
                    noise_abs=3.5,
                    delta=Effect(value=float("inf"), unit="percent"),
                    n=10,
                ),
            ),
        ),
        meta=metric_meta("decode/time", unit="ns"),
    )
    result = create_comparison_result(metrics={"decode/time": metric})

    candidate = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][0]

    assert candidate["delta"] is None


# ---------------------------------------------------------------------------
# render_json — metric metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "expected"),
    [(None, None), ("ns", "ns")],
)
def test_render_json_when_metric_unit_does_serialize_unit(
    unit: MetricUnit | None,
    expected: str | None,
):
    result = create_comparison_result(
        metrics={"decode/time": permutation_metric(verdict="improved", delta=-5, unit=unit)},
    )

    doc = json.loads(render_json(result))

    assert doc["metrics"]["decode/time"]["unit"] == expected


def test_render_json_when_metric_meta_does_include_direction_and_gating():
    result = create_comparison_result(
        metrics={"decode/time": permutation_metric(verdict="improved", delta=-5, gating=False)},
    )

    metric = json.loads(render_json(result))["metrics"]["decode/time"]

    assert metric["direction"] == "lower"
    assert metric["gating"] is False


def test_render_json_when_baseline_measured_does_include_median_and_spread():
    result = create_comparison_result(
        metrics={
            "decode/time": permutation_metric(
                verdict="improved",
                delta=-10,
                baseline_median=200,
                baseline_spread=3.5,
            ),
        },
    )

    baseline = json.loads(render_json(result))["metrics"]["decode/time"]["baseline"]

    assert baseline["median"] == 200
    assert baseline["spreadPct"] == 3.5


# ---------------------------------------------------------------------------
# render_json — per-candidate kinds
# ---------------------------------------------------------------------------


def test_render_json_when_candidate_spans_kinds_does_carry_one_entry_per_kind():
    doc = json.loads(render_json(_two_kind_with_exclusions()))

    assert doc["perCandidate"][0]["kinds"] == [
        {
            "kind": "time",
            "hasGating": True,
            "geomean": {
                "value": -3.2,
                "n": 2,
                "excluded": [
                    {"metric": "jittery/time", "reason": "unstable"},
                    {"metric": "broken/ratio", "reason": "undefined-ratio"},
                ],
                "band": 0,
            },
            "groups": [
                {"group": "entity", "geomean": {"value": -3.1, "n": 2, "excluded": [], "band": 0}},
            ],
            "gatedGeomean": {"value": -3.2, "n": 2, "excluded": [], "band": 0},
        },
        {
            "kind": "memory",
            "hasGating": False,
            "geomean": {"value": -7, "n": 1, "excluded": [], "band": 0},
            "groups": [],
            "gatedGeomean": None,
        },
    ]


def test_render_json_when_candidate_spans_kinds_does_leave_no_blended_geomean():
    doc = json.loads(render_json(_two_kind_with_exclusions()))

    assert "geomean" not in doc["perCandidate"][0]


def test_render_json_when_single_kind_does_use_same_shape_with_one_entry():
    result = create_comparison_result(
        candidates=[create_candidate(label="experiment", kinds=[other_kind(-3.2, 2)])],
        metrics={"decode/time": permutation_metric(verdict="improved", delta=-10)},
    )

    doc = json.loads(render_json(result))

    assert doc["perCandidate"][0]["kinds"] == [
        {
            "kind": "other",
            "hasGating": True,
            "geomean": {"value": -3.2, "n": 2, "excluded": [], "band": 0},
            "groups": [],
            "gatedGeomean": {"value": -3.2, "n": 2, "excluded": [], "band": 0},
        },
    ]


# ---------------------------------------------------------------------------
# render_json — metric kind and group
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "kind", "group"),
    [
        ("entity.alive_check/time", "time", "entity"),
        ("encode/heap", "memory", None),
    ],
)
def test_render_json_when_reporting_metric_does_carry_kind_and_group(
    metric_name: str,
    kind: str,
    group: str | None,
):
    doc = json.loads(render_json(_two_kind_with_exclusions()))

    assert doc["metrics"][metric_name]["kind"] == kind
    assert doc["metrics"][metric_name]["group"] == group


# ---------------------------------------------------------------------------
# render_json — verdict counts
# ---------------------------------------------------------------------------


def test_render_json_when_metrics_vary_does_tally_verdict_counts_per_candidate():
    result = create_comparison_result(
        metrics={
            "faster/time": permutation_metric(verdict="improved", delta=-10),
            "also-faster/time": permutation_metric(verdict="improved", delta=-5),
            "slower/time": permutation_metric(verdict="regressed", delta=8),
            "jittery/time": permutation_metric(verdict="unstable", delta=5),
            "flat/time": permutation_metric(verdict="no-signal", delta=0.2),
        },
    )

    counts = json.loads(render_json(result))["perCandidate"][0]["verdictCounts"]

    assert counts == {"improved": 2, "regressed": 1, "unstable": 1, "noSignal": 1}


# ---------------------------------------------------------------------------
# render_json — missing metric data
# ---------------------------------------------------------------------------


def test_render_json_when_baseline_unmeasured_does_render_null_baseline_fields():
    metric = MetricComparison(
        baseline_median=None,
        baseline_spread=None,
        candidates=(
            CandidateMetric(
                verdict=PermutationVerdict(
                    method="permutation",
                    verdict="improved",
                    p=0.01,
                    noise_pct=2.5,
                    noise_abs=3.5,
                    delta=Effect(value=-5, unit="percent"),
                    n=10,
                ),
            ),
        ),
        meta=metric_meta("sparse/time", unit="ns"),
    )
    result = create_comparison_result(
        candidates=[create_candidate(label="alpha")],
        metrics={"sparse/time": metric},
    )

    serialized = json.loads(render_json(result))["metrics"]["sparse/time"]

    assert serialized["baseline"]["median"] is None
    assert serialized["baseline"]["spreadPct"] is None
    assert serialized["candidates"][0]["median"] is None
    assert serialized["candidates"][0]["spreadPct"] is None


def test_render_json_when_candidate_has_no_metric_data_does_render_all_nulls():
    metric = MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=(
            CandidateMetric(
                median=90.0,
                spread=1.0,
                verdict=PermutationVerdict(
                    method="permutation",
                    verdict="improved",
                    p=0.01,
                    noise_pct=2.5,
                    noise_abs=3.5,
                    delta=Effect(value=-10, unit="percent"),
                    n=10,
                ),
            ),
            CandidateMetric(),
        ),
        meta=metric_meta("decode/time", unit="ns"),
    )
    result = create_comparison_result(
        candidates=[create_candidate(label="alpha"), create_candidate(label="beta")],
        metrics={"decode/time": metric},
    )

    beta = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][1]

    assert beta["label"] == "beta"
    assert beta["median"] is None
    assert beta["spreadPct"] is None
    assert beta["verdict"] is None
    assert beta["method"] is None
    assert beta["delta"] is None
    assert beta["noisePct"] is None
    assert beta["p"] is None
    assert beta["band"] is None


def test_render_json_when_candidate_measured_but_unpaired_does_keep_measurements():
    metric = MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=(
            CandidateMetric(
                median=90.0,
                spread=1.0,
                verdict=PermutationVerdict(
                    method="permutation",
                    verdict="improved",
                    p=0.01,
                    noise_pct=2.5,
                    noise_abs=3.5,
                    delta=Effect(value=-10, unit="percent"),
                    n=10,
                ),
            ),
            CandidateMetric(median=95.0, spread=3.0),
        ),
        meta=metric_meta("decode/time", unit="ns"),
    )
    result = create_comparison_result(
        candidates=[create_candidate(label="alpha"), create_candidate(label="beta")],
        metrics={"decode/time": metric},
    )

    beta = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][1]

    assert beta["median"] == 95
    assert beta["spreadPct"] == 3
    assert beta["verdict"] is None


# ---------------------------------------------------------------------------
# render_json — worktrees section
# ---------------------------------------------------------------------------


def test_render_json_when_cleanup_clean_does_report_no_issues():
    result = create_comparison_result(
        worktrees_removed=2,
        worktrees_left_behind=[],
        worktree_prune_error=None,
    )

    worktrees = json.loads(render_json(result))["worktrees"]

    assert worktrees == {"removed": 2, "leftBehind": [], "pruneError": None}


def test_render_json_when_cleanup_has_failures_does_report_left_behind_and_prune_error():
    result = create_comparison_result(
        worktrees_removed=1,
        worktrees_left_behind=[
            WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="contains modified files"),
        ],
        worktree_prune_error="fatal: prune failed",
    )

    worktrees = json.loads(render_json(result))["worktrees"]

    assert worktrees["removed"] == 1
    assert worktrees["leftBehind"] == [
        {"path": "/tmp/gymrat-abc", "reason": "contains modified files"},
    ]
    assert worktrees["pruneError"] == "fatal: prune failed"


# ---------------------------------------------------------------------------
# render_json — JSON validity and ANSI
# ---------------------------------------------------------------------------


def test_render_json_when_output_parsed_again_matches_identically():
    result = create_comparison_result(
        metrics={
            "decode/time": permutation_metric(verdict="improved", delta=-10, unit="ns"),
            "alloc/heap": exact_metric(delta=-5, unit="bytes"),
        },
        worktrees_removed=1,
        worktrees_left_behind=[WorktreeRemovalFailure(dir="/tmp/gymrat-x", error="locked")],
        worktree_prune_error="prune failed",
    )

    output = render_json(result)

    assert json.dumps(json.loads(output), indent=2) == output


def test_render_json_when_environment_forces_color_does_emit_no_ansi(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    assert not _ANSI_ESCAPE.search(render_json(_two_kind_with_exclusions()))


# ---------------------------------------------------------------------------
# render_measure_json — schema shape
# ---------------------------------------------------------------------------


def test_render_measure_json_when_single_run_does_use_schema_version_1_shape():
    result = create_measurement_result(
        label="experiment",
        samples=10,
        adapter="mitata",
        metrics={"decode/time": measured_metric(unit="ns")},
    )

    doc = json.loads(render_measure_json(result))

    assert doc["schemaVersion"] == 1
    assert doc["label"] == "experiment"
    assert doc["samples"] == 10
    assert doc["adapter"] == "mitata"
    assert "metrics" in doc
    assert "worktrees" in doc


def test_render_measure_json_when_single_run_does_omit_comparison_only_sections():
    doc = json.loads(render_measure_json(two_kind_measurement()))

    assert "baseline" not in doc
    assert "candidates" not in doc
    assert "perCandidate" not in doc


def test_render_measure_json_when_top_level_keys_does_order_them_canonically():
    doc = json.loads(render_measure_json(two_kind_measurement()))

    assert list(doc.keys()) == [
        "schemaVersion",
        "label",
        "samples",
        "adapter",
        "metrics",
        "worktrees",
    ]


# ---------------------------------------------------------------------------
# render_measure_json — metric entries
# ---------------------------------------------------------------------------


def test_render_measure_json_when_grouped_metric_does_carry_measurement_and_metadata():
    doc = json.loads(render_measure_json(two_kind_measurement()))

    assert doc["metrics"]["entity.alive_check/time"] == {
        "median": 100,
        "spreadPct": 1,
        "unit": "ns",
        "direction": "lower",
        "gating": True,
        "exact": False,
        "kind": "time",
        "group": "entity",
    }


def test_render_measure_json_when_metric_names_no_group_does_report_null_group():
    doc = json.loads(render_measure_json(two_kind_measurement()))

    assert doc["metrics"]["encode/heap"]["group"] is None
    assert doc["metrics"]["encode/heap"]["kind"] == "memory"


@pytest.mark.parametrize(
    ("field", "median", "spread"),
    [("median", None, 1.0), ("spreadPct", 100.0, None)],
)
def test_render_measure_json_when_field_absent_does_render_null(
    field: str,
    median: float | None,
    spread: float | None,
):
    result = create_measurement_result(
        metrics={"sparse/time": measured_metric(median=median, spread=spread, unit="ns")},
    )

    doc = json.loads(render_measure_json(result))

    assert doc["metrics"]["sparse/time"][field] is None


def test_render_measure_json_when_metric_has_no_unit_does_render_null_unit():
    result = create_measurement_result(
        metrics={"throughput/ops": measured_metric()},
    )

    doc = json.loads(render_measure_json(result))

    assert doc["metrics"]["throughput/ops"]["unit"] is None


# ---------------------------------------------------------------------------
# render_measure_json — worktrees section
# ---------------------------------------------------------------------------


def test_render_measure_json_when_cleanup_clean_does_report_no_issues():
    doc = json.loads(render_measure_json(create_measurement_result(worktrees_removed=2)))

    assert doc["worktrees"] == {"removed": 2, "leftBehind": [], "pruneError": None}


def test_render_measure_json_when_cleanup_has_failures_does_report_left_behind_and_prune_error():
    result = create_measurement_result(
        worktrees_removed=1,
        worktrees_left_behind=[
            WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="contains modified files"),
        ],
        worktree_prune_error="fatal: prune failed",
    )

    doc = json.loads(render_measure_json(result))

    assert doc["worktrees"] == {
        "removed": 1,
        "leftBehind": [{"path": "/tmp/gymrat-abc", "reason": "contains modified files"}],
        "pruneError": "fatal: prune failed",
    }


# ---------------------------------------------------------------------------
# render_measure_json — JSON validity and ANSI
# ---------------------------------------------------------------------------


def test_render_measure_json_when_environment_forces_color_does_emit_no_ansi(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")

    assert not _ANSI_ESCAPE.search(render_measure_json(two_kind_measurement()))


def test_render_measure_json_when_nesting_fields_does_indent_two_spaces_per_level():
    result = two_kind_measurement(
        worktrees_removed=1,
        worktrees_left_behind=[WorktreeRemovalFailure(dir="/tmp/gymrat-x", error="locked")],
        worktree_prune_error="prune failed",
    )

    lines = render_measure_json(result).split("\n")
    worktrees_line = next(i for i, line in enumerate(lines) if line.strip() == '"worktrees": {')

    assert re.match(r'^ {2}"schemaVersion": 1,$', lines[1])
    assert re.match(r'^ {2}"worktrees": \{$', lines[worktrees_line])
    assert re.match(r'^ {4}"removed": 1,$', lines[worktrees_line + 1])
