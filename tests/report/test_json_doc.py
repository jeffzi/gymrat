"""Tests for the JSON report document builders.

These cover the compare document (``render_json``), the measure document
(``render_measure_json``), and the probe document (``render_probe_json``),
including their schema shapes, per-metric and per-candidate serialization,
worktree sections, and non-finite handling. They also cover the loop commands' start, keep, finalize, and sync
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
from gymrat.model import Exclusion
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
)
from gymrat.sampling import CleanupResult
from gymrat.session.records import KeepChecks
from gymrat.worktree_failure import WorktreeRemovalFailure
from tests.report._comparisons import (
    NWayCandidate,
    create_candidate,
    create_comparison_result,
    exact_metric,
    informational_kind,
    n_way_metric,
    permutation_candidate,
    permutation_metric,
    shared_baseline_metric,
    two_kind_result,
)
from tests.report._measurements import two_kind_measurement
from tests.report._probes import golden_probe, probe_result
from tests.report._verdicts import band_metric, geomean_of
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


def test_render_json_when_band_verdict_does_serialize_band_method_fields():
    result = create_comparison_result(
        metrics={"decode/time": band_metric(verdict="no-signal", delta=-1, noise_pct=4.0)},
    )

    candidate = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][0]

    assert candidate["method"] == "band"
    assert candidate["band"] == 4.0
    assert candidate["noise_pct"] == 4.0
    assert candidate["p"] is None


def test_render_json_when_exact_verdict_does_null_statistical_fields():
    result = create_comparison_result(metrics={"alloc/heap": exact_metric(delta=-7.9)})

    candidate = json.loads(render_json(result))["metrics"]["alloc/heap"]["candidates"][0]

    assert candidate["method"] == "exact"
    assert candidate["noise_pct"] is None
    assert candidate["p"] is None
    assert candidate["band"] is None


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
                    "decode/time": shared_baseline_metric(
                        [
                            permutation_candidate(
                                verdict="improved", delta=float("inf"), median=90.0
                            )
                        ],
                        name="decode/time",
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
                        kinds=[informational_kind("time", geomean_of(math.nan, 0))],
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
# render_json — per-candidate kinds
# ---------------------------------------------------------------------------


def test_render_json_when_geomean_has_exclusions_does_list_them_in_field_order():
    excluded = [
        Exclusion(metric="jittery/time#time", reason="unstable"),
        Exclusion(metric="broken/ratio#time", reason="undefined-ratio"),
    ]
    result = create_comparison_result(
        candidates=[
            create_candidate(
                kinds=[informational_kind("time", geomean_of(-3.2, 2, excluded=excluded))],
            ),
        ],
    )

    kinds = json.loads(render_json(result))["per_candidate"][0]["kinds"]

    assert [list(entry.items()) for entry in kinds[0]["geomean"]["excluded"]] == [
        [("metric", "jittery/time#time"), ("reason", "unstable")],
        [("metric", "broken/ratio#time"), ("reason", "undefined-ratio")],
    ]


# ---------------------------------------------------------------------------
# render_json — missing metric data
# ---------------------------------------------------------------------------


#: A candidate that measured and paired against the baseline, with an "improved" verdict.
_PAIRED = permutation_candidate(verdict="improved", delta=-10.0, median=90.0)

#: A candidate row with no measurement behind it.
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
    ("candidates", "expected"),
    [
        pytest.param(
            (_PAIRED, CandidateMetric()),
            _UNMEASURED_ROW,
            id="empty-candidate-slice",
        ),
        pytest.param((_PAIRED,), _UNMEASURED_ROW, id="fewer-slices-than-candidates"),
        pytest.param(
            (_PAIRED, CandidateMetric(median=95.0, spread=3.0)),
            {**_UNMEASURED_ROW, "median": 95.0, "spread_pct": 3.0},
            id="measured-unpaired",
        ),
    ],
)
def test_render_json_when_candidate_has_no_verdict_does_null_its_verdict_fields(
    candidates: tuple[CandidateMetric, ...], expected: dict[str, object]
):
    metric = shared_baseline_metric(candidates, name="decode/time")
    result = create_comparison_result(
        candidates=[create_candidate(label="alpha"), create_candidate(label="beta")],
        metrics={"decode/time": metric},
    )

    beta = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][1]

    assert beta == expected


# ---------------------------------------------------------------------------
# render_json — worktrees section
# ---------------------------------------------------------------------------


def test_render_json_when_cleanup_has_failures_does_report_them_in_worktrees():
    result = create_comparison_result(
        cleanup=CleanupResult(
            removed=1,
            failures=(
                WorktreeRemovalFailure(dir="/tmp/gymrat-abc", error="contains modified files"),
            ),
            prune_error="fatal: prune failed",
        )
    )

    doc = json.loads(render_json(result))

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
def test_render_json_when_text_or_float_serialized_does_write_its_unescaped_short_form(
    result: ComparisonResult, fragment: str
):
    output = render_json(result)

    assert fragment in output


# ---------------------------------------------------------------------------
# render_probe_json — schema shape
# ---------------------------------------------------------------------------


def test_render_probe_json_when_names_given_does_report_a_scoped_probe():
    result = probe_result(names=("total_ms", "decode"))

    doc = json.loads(render_probe_json(result))

    assert (doc["scoped"], doc["names"]) == (True, ["total_ms", "decode"])


# ---------------------------------------------------------------------------
# whole documents — golden
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "render",
    [
        pytest.param(lambda: render_json(two_kind_result()), id="compare"),
        pytest.param(lambda: render_measure_json(two_kind_measurement()), id="measure"),
        pytest.param(lambda: render_probe_json(golden_probe()), id="probe"),
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
# render_start_json — fresh, resumed, runbook, and budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "runbook", "budget", "expected"),
    [
        pytest.param(
            _fresh_start(),
            None,
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
                "archived": None,
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
            None,
            {"resumed": True, "iteration_count": 5, "keep_count": 3, "runbook": None},
            id="resumed",
        ),
        pytest.param(
            _fresh_start(),
            "my-runbook.yml",
            None,
            {"resumed": False, "iteration_count": 0, "keep_count": 0, "runbook": "my-runbook.yml"},
            id="runbook",
        ),
        pytest.param(
            StartResult(
                session=session_record(),
                state=empty_session_state(),
                resumed=False,
                archived="20260701-120000-beef",
                archived_path="/repo/.gymrat/archive/20260701-120000-beef",
            ),
            None,
            None,
            {
                "archived": {
                    "session_id": "20260701-120000-beef",
                    "path": "/repo/.gymrat/archive/20260701-120000-beef",
                },
            },
            id="archived",
        ),
        pytest.param(
            _fresh_start(),
            None,
            BudgetSummary(cap_minutes=30, remaining_seconds=900),
            {"budget": {"cap_minutes": 30, "remaining_seconds": 900}},
            id="budget",
        ),
    ],
)
def test_render_start_json_when_start_varies_does_reflect_the_start_state(
    result: StartResult,
    runbook: str | None,
    budget: BudgetSummary | None,
    expected: dict[str, object],
):
    doc = json.loads(render_start_json(result, runbook=runbook, budget=budget))

    assert {key: doc[key] for key in expected} == expected


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


def test_render_finalize_json_when_rendered_does_copy_the_record_fields():
    record = finalize_record()
    result = FinalizeResult(record=record, report="final report text")

    doc = json.loads(render_finalize_json(result))

    assert (doc["branch"], doc["commit"], doc["message"], doc["at"]) == (
        record.branch,
        record.commit,
        record.message,
        record.at,
    )


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
