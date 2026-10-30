"""Tests for the JSON report document builders.

These cover the compare document (``render_json``), the measure document
(``render_measure_json``), and the probe document (``render_probe_json``),
including their per-metric and per-candidate serialization and worktree
sections. They also cover the iterate document's confirm object and
the start document's archived and budget fields (``render_start_json``), and the
discard document's ``seq`` and ``measured`` fields (``render_discard_json``).
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from gymrat.loop.discard import DiscardResult
from gymrat.loop.iterate.run import IterateResult
from gymrat.report.json_doc import (
    BudgetSummary,
    render_discard_json,
    render_iterate_json,
    render_json,
    render_measure_json,
    render_probe_json,
    render_start_json,
)
from gymrat.report.types import (
    CandidateMetric,
)
from gymrat.sampling import CleanupResult, WorktreeRemovalFailure
from gymrat.session.records import (
    Confirm,
    PairedSamples,
)
from tests.report._comparisons import (
    create_candidate,
    create_comparison_result,
    permutation_candidate,
    shared_baseline_metric,
    two_kind_result,
)
from tests.report._measurements import two_kind_measurement
from tests.report._probes import golden_probe, probe_result
from tests.report._verdicts import band_metric, permutation_verdict
from tests.session.records._fixtures import (
    ARCHIVED_SESSION_ID,
    ARCHIVED_SESSION_PATH,
    discard_record,
    iteration_record,
    start_result,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from syrupy.assertion import SnapshotAssertion

    from gymrat.loop.start import StartResult


# ---------------------------------------------------------------------------
# render_json — metric verdict methods
# ---------------------------------------------------------------------------


def test_render_json_when_band_verdict_does_serialize_the_band_width():
    result = create_comparison_result(
        metrics={"decode/time": band_metric(verdict="no-signal", delta=-1, noise_pct=4.0)},
    )

    candidate = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][0]

    assert candidate["band_pct"] == 4.0


def test_render_json_when_approximate_verdict_withheld_does_serialize_its_reason():
    verdict = permutation_verdict(delta=0.4, reason="resolution")
    metric = shared_baseline_metric(
        [CandidateMetric(median=100.4, spread=1.0, verdict=verdict)], name="decode/time"
    )
    result = create_comparison_result(metrics={"decode/time": metric})

    candidate = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][0]

    assert candidate["reason"] == "resolution"


def test_render_json_when_approximate_verdict_carries_a_detectable_change_does_serialize_it():
    verdict = replace(permutation_verdict(delta=0.4), detectable_pct=2.1)
    metric = shared_baseline_metric(
        [CandidateMetric(median=100.4, spread=1.0, verdict=verdict)], name="decode/time"
    )
    result = create_comparison_result(metrics={"decode/time": metric})

    candidate = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][0]

    assert candidate["detectable_pct"] == 2.1


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
    "delta_pct": None,
    "noise_pct": None,
    "detectable_pct": None,
    "p": None,
    "band_pct": None,
    "reason": None,
}


def test_render_json_when_candidate_unmeasured_does_null_every_field_but_its_label():
    metric = shared_baseline_metric((_PAIRED,), name="decode/time")
    result = create_comparison_result(
        candidates=[create_candidate(label="alpha"), create_candidate(label="beta")],
        metrics={"decode/time": metric},
    )

    beta = json.loads(render_json(result))["metrics"]["decode/time"]["candidates"][1]

    assert beta == _UNMEASURED_ROW


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
# render_probe_json — schema shape
# ---------------------------------------------------------------------------


def test_render_probe_json_when_names_given_does_report_a_scoped_probe():
    result = probe_result(names=("total_ms", "decode"))

    doc = json.loads(render_probe_json(result))

    assert doc["scoped"] is True


# ---------------------------------------------------------------------------
# render_iterate_json — optional fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("confirm", "expected"),
    [
        pytest.param(None, None, id="no-rerun"),
        pytest.param(
            Confirm(
                ran=True,
                filtered=("total_ms", "alloc_bytes"),
                samples=PairedSamples(
                    experiment=({"total_ms": 16700},), baseline=({"total_ms": 15200},)
                ),
            ),
            {"ran": True, "filtered": ["total_ms", "alloc_bytes"], "absent": None},
            id="confirmed-on-rerun",
        ),
    ],
)
def test_render_iterate_json_when_rerun_varies_does_serialize_the_confirm_object(
    confirm: Confirm | None, expected: dict[str, object] | None
):
    result = IterateResult(record=iteration_record(confirm=confirm), report="iteration report")

    doc = json.loads(render_iterate_json(result))

    assert doc["confirm"] == expected


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
def test_render_json_documents_when_rendered_does_match_their_snapshots(
    render: Callable[[], str], snapshot: SnapshotAssertion
):
    document = render()

    assert document.split("\n") == snapshot


# ---------------------------------------------------------------------------
# render_discard_json — measured or not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        pytest.param(
            DiscardResult(record=discard_record(3), report="", at=7),
            {"seq": 3, "measured": True},
            id="measured",
        ),
        pytest.param(
            DiscardResult(record=None, report="", at=7),
            {"seq": None, "measured": False},
            id="unmeasured",
        ),
    ],
)
def test_render_discard_json_when_record_present_or_not_does_report_whether_the_discard_was_measured(
    result: DiscardResult, expected: dict[str, object]
):
    doc = json.loads(render_discard_json(result))

    assert {"seq": doc["seq"], "measured": doc["measured"]} == expected


# ---------------------------------------------------------------------------
# render_start_json — fresh, archived, and budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "budget", "expected"),
    [
        pytest.param(start_result(), None, {"archived": None}, id="fresh"),
        pytest.param(
            start_result(archived=ARCHIVED_SESSION_ID, archived_path=ARCHIVED_SESSION_PATH),
            None,
            {"archived": {"session_id": ARCHIVED_SESSION_ID, "path": ARCHIVED_SESSION_PATH}},
            id="archived",
        ),
        pytest.param(
            start_result(),
            BudgetSummary(cap_minutes=30, remaining_seconds=900),
            {"budget": {"cap_minutes": 30, "remaining_seconds": 900}},
            id="budget",
        ),
    ],
)
def test_render_start_json_when_start_varies_does_reflect_the_start_state(
    result: StartResult, budget: BudgetSummary | None, expected: dict[str, object]
):
    doc = json.loads(render_start_json(result, runbook=None, budget=budget))

    assert {key: doc[key] for key in expected} == expected
