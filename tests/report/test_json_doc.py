"""Tests for the JSON report document builders.

These cover the compare document (``render_json``), the measure document
(``render_measure_json``), and the probe document (``render_probe_json``),
including their schema shapes, per-metric and per-candidate serialization,
worktree sections, non-finite handling, and the no-ANSI guarantee under forced
color. They also cover the loop commands' start, keep, finalize, and sync
documents (``render_start_json``, ``render_keep_json``,
``render_finalize_json``, ``render_sync_json``): their schema shapes, the
resumed and archived start variants, the runbook and budget fields, and the
keep record's checks object.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import pytest

from gymrat.loop.finalize import FinalizeResult
from gymrat.loop.keep import KeepResult
from gymrat.loop.start import StartResult
from gymrat.loop.sync import SyncResult
from gymrat.model import Exclusion, PermutationVerdict
from gymrat.report.json_doc import (
    BudgetSummary,
    render_finalize_json,
    render_json,
    render_keep_json,
    render_measure_json,
    render_probe_json,
    render_start_json,
    render_sync_json,
)
from gymrat.report.types import (
    CandidateMetric,
    ComparisonResult,
    MetricComparison,
)
from gymrat.session.records import KeepChecks
from gymrat.targets import WorktreeRemovalFailure
from gymrat.verdict import GroupAggregate, KindAggregate
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    create_comparison_result,
    exact_metric,
    kind_metric,
    metric_meta,
    n_way_metric,
    permutation_metric,
    two_kind_result,
)
from tests.report._measurements import (
    create_measurement_result,
    measured_metric,
    two_kind_measurement,
)
from tests.report._probes import probe_metric, probe_result
from tests.report._verdicts import band_metric, geomean_of, permutation_verdict
from tests.session.records._fixtures import (
    committed_keep,
    empty_session_state,
    finalize_record,
    session_record,
    session_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from syrupy.assertion import SnapshotAssertion


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
            Exclusion(metric="jittery/time#time", reason="unstable"),
            Exclusion(metric="broken/ratio#time", reason="undefined-ratio"),
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
            "entity/alive_check#time": kind_metric(
                kind="time",
                short_name="alive_check",
                verdict="improved",
                delta=-10,
            ),
            "encode#memory": kind_metric(
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
    assert len(doc["per_candidate"]) == 3
    assert [entry["label"] for entry in doc["per_candidate"]] == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# render_json — metric verdict methods
# ---------------------------------------------------------------------------


def test_render_json_when_band_verdict_does_set_band_and_null_p():
    result = create_comparison_result(
        metrics={"decode/time": band_metric(verdict="no-signal", delta=-1, noise_pct=3.5)},
    )

    candidate = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][0]

    assert candidate["method"] == "band"
    assert candidate["band"] == 3.5
    assert candidate["p"] is None


def test_render_json_when_exact_verdict_does_null_noise_p_and_band():
    result = create_comparison_result(metrics={"alloc/heap": exact_metric(delta=-7.9)})

    candidate = json.loads(render_json(result))["metrics"]["alloc/heap"]["candidates"][0]

    assert candidate["method"] == "exact"
    assert candidate["noise_pct"] is None
    assert candidate["p"] is None
    assert candidate["band"] is None


def _paired_candidate(delta: float = -10.0) -> CandidateMetric:
    """A candidate that measured and paired against the baseline, with an "improved" verdict."""
    return CandidateMetric(
        median=90.0,
        spread=1.0,
        verdict=permutation_verdict(verdict="improved", delta=delta, p=0.01, noise_abs=3.5),
    )


def _first_candidate_delta(doc: dict[str, Any]) -> object:
    """The ``decode/time`` metric's first candidate delta in a compare document."""
    return doc["metrics"]["decode/time"]["candidates"][0]["delta"]


def _first_kind_geomean(doc: dict[str, Any]) -> object:
    """The first candidate's first kind geomean value in a compare document."""
    return doc["per_candidate"][0]["kinds"][0]["geomean"]["value"]


@pytest.mark.parametrize(
    ("result", "extract"),
    [
        pytest.param(
            create_comparison_result(
                metrics={
                    "decode/time": MetricComparison(
                        baseline_median=100.0,
                        baseline_spread=1.0,
                        candidates=(_paired_candidate(delta=float("inf")),),
                        meta=metric_meta("decode/time", unit="ns"),
                    ),
                },
            ),
            _first_candidate_delta,
            id="delta",
        ),
        pytest.param(
            create_comparison_result(
                candidates=[
                    create_candidate(
                        kinds=[
                            KindAggregate(kind="time", geomean=geomean_of(math.nan, 0), groups=())
                        ],
                    ),
                ],
            ),
            _first_kind_geomean,
            id="geomean",
        ),
    ],
)
def test_render_json_when_value_is_non_finite_does_render_null(
    result: ComparisonResult, extract: Callable[[dict[str, Any]], object]
):
    doc = json.loads(render_json(result))

    assert extract(doc) is None


# ---------------------------------------------------------------------------
# render_json and render_measure_json — metric metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("render", "result", "name"),
    [
        pytest.param(
            render_json,
            create_comparison_result(
                metrics={
                    "decode/time": permutation_metric(verdict="improved", delta=-5, unit=None)
                },
            ),
            "decode/time",
            id="compare",
        ),
        pytest.param(
            render_measure_json,
            create_measurement_result(metrics={"throughput/ops": measured_metric()}),
            "throughput/ops",
            id="measure",
        ),
    ],
)
def test_render_document_json_when_metric_has_no_unit_does_serialize_null_unit(
    render: Callable[..., str], result: object, name: str
):
    doc = json.loads(render(result))

    assert doc["metrics"][name]["unit"] is None


# ---------------------------------------------------------------------------
# render_json — per-candidate kinds
# ---------------------------------------------------------------------------


def test_render_json_when_candidate_spans_kinds_does_carry_one_entry_per_kind_in_field_order():
    doc = json.loads(render_json(_two_kind_with_exclusions()))

    kinds = doc["per_candidate"][0]["kinds"]
    geomeans = [kinds[0]["geomean"], kinds[0]["groups"][0]["geomean"], kinds[0]["gated_geomean"]]
    assert kinds == [
        {
            "kind": "time",
            "has_gating": True,
            "geomean": {
                "value": -3.2,
                "n": 2,
                "excluded": [
                    {"metric": "jittery/time#time", "reason": "unstable"},
                    {"metric": "broken/ratio#time", "reason": "undefined-ratio"},
                ],
                "band": 0,
            },
            "groups": [
                {"group": "entity", "geomean": {"value": -3.1, "n": 2, "excluded": [], "band": 0}},
            ],
            "gated_geomean": {"value": -3.2, "n": 2, "excluded": [], "band": 0},
        },
        {
            "kind": "memory",
            "has_gating": False,
            "geomean": {"value": -7, "n": 1, "excluded": [], "band": 0},
            "groups": [],
            "gated_geomean": None,
        },
    ]
    assert [list(geomean) for geomean in geomeans] == [["value", "n", "band", "excluded"]] * 3
    assert [list(entry) for entry in kinds[0]["geomean"]["excluded"]] == [["metric", "reason"]] * 2


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
                    delta=-5,
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
    assert serialized["baseline"]["spread_pct"] is None
    assert serialized["candidates"][0]["median"] is None
    assert serialized["candidates"][0]["spread_pct"] is None


#: A candidate row with no measurement behind it, in the key order the document writes.
_UNMEASURED_ROW: dict[str, object] = {
    "label": "beta",
    "median": None,
    "spread_pct": None,
    "verdict": None,
    "method": None,
    "delta": None,
    "noise_pct": None,
    "p": None,
    "band": None,
}


@pytest.mark.parametrize(
    "candidates",
    [
        pytest.param((_paired_candidate(), CandidateMetric()), id="empty-candidate-slice"),
        pytest.param((_paired_candidate(),), id="fewer-slices-than-candidates"),
    ],
)
def test_render_json_when_candidate_has_no_metric_data_does_render_an_all_null_row(
    candidates: tuple[CandidateMetric, ...],
):
    metric = MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=candidates,
        meta=metric_meta("decode/time", unit="ns"),
    )
    result = create_comparison_result(
        candidates=[create_candidate(label="alpha"), create_candidate(label="beta")],
        metrics={"decode/time": metric},
    )

    beta = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][1]

    assert list(beta.items()) == list(_UNMEASURED_ROW.items())


def test_render_json_when_candidate_measured_but_unpaired_does_keep_measurements():
    metric = MetricComparison(
        baseline_median=100.0,
        baseline_spread=1.0,
        candidates=(_paired_candidate(), CandidateMetric(median=95.0, spread=3.0)),
        meta=metric_meta("decode/time", unit="ns"),
    )
    result = create_comparison_result(
        candidates=[create_candidate(label="alpha"), create_candidate(label="beta")],
        metrics={"decode/time": metric},
    )

    beta = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][1]

    assert beta["median"] == 95
    assert beta["spread_pct"] == 3
    assert beta["verdict"] is None


# ---------------------------------------------------------------------------
# render_json and render_measure_json — worktrees section
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("render", "result"),
    [
        pytest.param(
            render_json,
            create_comparison_result(
                worktrees_removed=1,
                worktrees_left_behind=[
                    WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="contains modified files"),
                ],
                worktree_prune_error="fatal: prune failed",
            ),
            id="compare",
        ),
        pytest.param(
            render_measure_json,
            create_measurement_result(
                worktrees_removed=1,
                worktrees_left_behind=[
                    WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="contains modified files"),
                ],
                worktree_prune_error="fatal: prune failed",
            ),
            id="measure",
        ),
    ],
)
def test_render_document_json_when_cleanup_has_failures_does_report_them_in_worktrees(
    render: Callable[..., str], result: object
):
    doc = json.loads(render(result))

    assert doc["worktrees"] == {
        "removed": 1,
        "left_behind": [{"path": "/tmp/gymrat-abc", "reason": "contains modified files"}],
        "prune_error": "fatal: prune failed",
    }


# ---------------------------------------------------------------------------
# render_json — JSON forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "fragment"),
    [
        pytest.param(
            create_comparison_result(candidates=[create_candidate(label="café")]),
            '"café"',
            id="non-ascii-raw",
        ),
        pytest.param(
            create_comparison_result(
                metrics={"decode/time": permutation_metric(verdict="improved", delta=-10, p=1e-7)},
            ),
            '"p": 1e-7',
            id="float-exponent-without-leading-zero",
        ),
    ],
)
def test_render_json_when_serializing_does_write_compact_json_forms(
    result: ComparisonResult, fragment: str
):
    output = render_json(result)

    assert fragment in output


# ---------------------------------------------------------------------------
# render_measure_json — metric entries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "median", "spread"),
    [("median", None, 1.0), ("spread_pct", 100.0, None)],
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


# ---------------------------------------------------------------------------
# render_probe_json — schema shape
# ---------------------------------------------------------------------------


def test_render_probe_json_when_names_given_does_report_the_scope_and_names():
    result = probe_result(names=("total_ms", "decode"), samples=3)

    doc = json.loads(render_probe_json(result))

    assert (doc["scoped"], doc["names"], doc["samples"]) == (True, ["total_ms", "decode"], 3)


# ---------------------------------------------------------------------------
# render_probe_json — metric entries
# ---------------------------------------------------------------------------


def test_render_probe_json_when_metric_fields_absent_does_carry_them_as_null():
    result = probe_result(
        metrics=[
            probe_metric(
                "alloc_bytes",
                median=None,
                spread=None,
                reference_median=None,
                delta_pct=None,
                unit="ns",
            )
        ]
    )

    doc = json.loads(render_probe_json(result))

    assert doc["metrics"]["alloc_bytes"] == {
        "median": None,
        "spread": None,
        "reference_median": None,
        "delta_pct": None,
    }


# ---------------------------------------------------------------------------
# whole documents — golden
# ---------------------------------------------------------------------------


def _golden_probe_json() -> str:
    return render_probe_json(
        probe_result(
            metrics=[
                probe_metric("total_ns", unit="ns", kind="time"),
                probe_metric("cold_start_ns", reference_median=None, delta_pct=None),
            ]
        )
    )


@pytest.mark.parametrize(
    "render",
    [
        pytest.param(lambda: render_json(two_kind_result()), id="compare"),
        pytest.param(lambda: render_measure_json(two_kind_measurement()), id="measure"),
        pytest.param(_golden_probe_json, id="probe"),
    ],
)
def test_render_document_json_when_rendered_does_match_its_golden(
    render: Callable[[], str], snapshot: SnapshotAssertion
):
    document = render()

    assert document.split("\n") == snapshot


# ---------------------------------------------------------------------------
# helpers — start / finalize / sync
# ---------------------------------------------------------------------------


#: The session header every start document in this file is rendered from.
_SESSION = session_record()


def _fresh_start() -> StartResult:
    """A brand-new session (not resumed, nothing archived)."""
    return StartResult(
        session=_SESSION,
        state=empty_session_state(),
        resumed=False,
    )


# ---------------------------------------------------------------------------
# render_start_json — fresh, resumed, and runbook
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "runbook", "expected"),
    [
        pytest.param(
            _fresh_start(),
            None,
            {
                "session_id": _SESSION.session_id,
                "branch": _SESSION.branch,
                "baseline": {"ref": _SESSION.baseline.ref, "sha": _SESSION.baseline.sha},
                "worktrees": {
                    "experiment": _SESSION.worktrees.experiment,
                    "baseline": _SESSION.worktrees.baseline,
                },
                "resumed": False,
                "iteration_count": 0,
                "keep_count": 0,
                "runbook": None,
            },
            id="fresh",
        ),
        pytest.param(
            StartResult(
                session=session_record(),
                state=session_state(iteration_count=5, keep_count=3),
                resumed=True,
            ),
            None,
            {"resumed": True, "iteration_count": 5, "keep_count": 3, "runbook": None},
            id="resumed",
        ),
        pytest.param(
            _fresh_start(),
            "my-runbook.yml",
            {"resumed": False, "iteration_count": 0, "keep_count": 0, "runbook": "my-runbook.yml"},
            id="runbook",
        ),
    ],
)
def test_render_start_json_when_start_varies_does_reflect_state_and_runbook(
    result: StartResult, runbook: str | None, expected: dict[str, object]
):
    doc = json.loads(render_start_json(result, runbook=runbook))

    assert {key: doc[key] for key in expected} == expected


# ---------------------------------------------------------------------------
# render_start_json — archived start
# ---------------------------------------------------------------------------


def test_render_start_json_when_archived_does_include_archived_object():
    result = StartResult(
        session=session_record(),
        state=empty_session_state(),
        resumed=False,
        archived="20260701-120000-beef",
        archived_path="/repo/.gymrat/archive/20260701-120000-beef",
    )

    doc = json.loads(render_start_json(result))

    assert doc["archived"] == {
        "session_id": "20260701-120000-beef",
        "path": "/repo/.gymrat/archive/20260701-120000-beef",
    }


# ---------------------------------------------------------------------------
# render_keep_json — checks object
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        pytest.param(
            KeepChecks(configured=True, passed=False, stdout_bytes=120, stderr_bytes=45),
            [("configured", True), ("passed", False), ("stdout_bytes", 120), ("stderr_bytes", 45)],
            id="checks-ran",
        ),
        pytest.param(
            KeepChecks(configured=False),
            [
                ("configured", False),
                ("passed", None),
                ("stdout_bytes", None),
                ("stderr_bytes", None),
            ],
            id="no-checks-configured",
        ),
    ],
)
def test_render_keep_json_when_checks_recorded_does_serialize_every_checks_field_in_order(
    checks: KeepChecks, expected: list[tuple[str, object]]
):
    result = KeepResult(record=committed_keep(1, checks=checks), report="keep report")

    doc = json.loads(render_keep_json(result))

    assert list(doc["checks"].items()) == expected


# ---------------------------------------------------------------------------
# render_finalize_json — schema shape
# ---------------------------------------------------------------------------


def test_render_finalize_json_when_rendered_does_produce_expected_keys():
    record = finalize_record()
    result = FinalizeResult(record=record, report="final report text")

    doc = json.loads(render_finalize_json(result))

    assert doc["branch"] == record.branch
    assert doc["commit"] == record.commit
    assert doc["message"] == record.message
    assert doc["at"] == record.at


# ---------------------------------------------------------------------------
# render_sync_json — schema shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            ("src/main.py", "src/lib.py"), ["src/main.py", "src/lib.py"], id="files-synced"
        ),
        pytest.param((), [], id="no-files"),
    ],
)
def test_render_sync_json_when_files_vary_does_list_them(
    files: tuple[str, ...], expected: list[str]
):
    result = SyncResult(files=files)

    doc = json.loads(render_sync_json(result))

    assert doc["files"] == expected


# ---------------------------------------------------------------------------
# render_*_json — budget key
# ---------------------------------------------------------------------------


_ABSENT = object()


@pytest.mark.parametrize(
    ("render", "expected"),
    [
        pytest.param(
            lambda: render_start_json(
                _fresh_start(), budget=BudgetSummary(cap_minutes=30, remaining_seconds=900)
            ),
            {"cap_minutes": 30, "remaining_seconds": 900},
            id="budget-given",
        ),
        pytest.param(
            lambda: render_finalize_json(
                FinalizeResult(record=finalize_record(), report="report"), budget=None
            ),
            _ABSENT,
            id="no-budget",
        ),
    ],
)
def test_render_document_json_when_budget_varies_does_reflect_the_budget_key(
    render: Callable[[], str], expected: object
):
    doc = json.loads(render())

    assert doc.get("budget", _ABSENT) == expected
