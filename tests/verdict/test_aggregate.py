"""Behavioral tests for hierarchical kind/group aggregation.

Drives behavior through the public ``compute_kind_aggregates`` and
``infer_group`` API. Bucketing order, per-kind grouping, and the per-subset
exclusion taxonomy are all exercised here.
"""

import math
from collections.abc import Sequence

import pytest

from gymrat.model import (
    Exclusion,
    ExclusionReason,
    GeomeanResult,
    MetricVerdict,
)
from gymrat.verdict import (
    KindAggregate,
    compute_kind_aggregates,
    infer_group,
)
from tests.verdict._inputs import (
    MetricSpec,
    build_inputs,
    exact_verdict,
    unstable_band_verdict,
)


def kinds_of(specs: Sequence[MetricSpec]) -> list[KindAggregate]:
    """The kind aggregates for a spec list — ``build_inputs`` fed into ``compute_kind_aggregates``."""
    verdicts, metric_meta = build_inputs(specs)
    return compute_kind_aggregates(verdicts, metric_meta)


def kind_named(aggregates: Sequence[KindAggregate], kind: str) -> KindAggregate:
    """The aggregate for ``kind``, or a failure naming the kinds produced."""
    for aggregate in aggregates:
        if aggregate.kind == kind:
            return aggregate
    names = ", ".join(aggregate.kind for aggregate in aggregates)
    pytest.fail(f'no aggregate for kind "{kind}", only: {names}')


def only_kind(aggregates: Sequence[KindAggregate]) -> KindAggregate:
    """The first aggregate, or a failure — most specs describe a single kind."""
    if not aggregates:
        pytest.fail("expected one kind aggregate but got none")
    return aggregates[0]


def mixed_gating_kind() -> list[MetricSpec]:
    """One kind holding one group whose metrics differ only in whether they gate.

    rho(gating) = 0.9 and rho(non-gating) = 0.95, so a geomean over both is
    (0.9 * 0.95)^(1/2) - 1 ~= -7.54%, and one over the gating metric alone is -10%.
    """
    return [
        MetricSpec(name="decode/time#time", short_name="decode.time", gating=True, delta=-10.0),
        MetricSpec(name="decode/alloc#time", short_name="decode.alloc", gating=False, delta=-5.0),
    ]


# ---------------------------------------------------------------------------
# infer_group
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("node/access.get_1field#time", "node", id="two-segment-path"),
        pytest.param("node/access/get_1field#time", "node/access", id="three-segment-path"),
        pytest.param("fib#time", None, id="one-segment-no-group"),
        pytest.param("fib", None, id="one-segment-no-kind"),
    ],
)
def test_infer_group_when_given_metric_name_does_return_path_minus_last_segment(
    name: str,
    expected: str | None,
):
    assert infer_group(name) == expected


# ---------------------------------------------------------------------------
# Kind aggregate shape and empty inputs
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_single_non_gating_metric_does_carry_kind_geomean_no_gate():
    result = kinds_of(
        [MetricSpec(name="warmup", short_name="warmup", gating=False, delta=0.0)],
    )

    assert result == [
        KindAggregate(
            kind="time",
            geomean=GeomeanResult(value=0.0, n=1, band=0.0, excluded=()),
            groups=(),
            gated_geomean=None,
        ),
    ]


def test_compute_kind_aggregates_when_nothing_measured_does_return_no_aggregates():
    assert kinds_of([]) == []


# ---------------------------------------------------------------------------
# Grouping by metric name contract (path minus last segment)
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_multi_segment_names_does_group_by_path_prefix():
    groups = only_kind(
        kinds_of(
            [
                MetricSpec(name="decode/time#time", short_name="decode.time", delta=-10.0),
                MetricSpec(name="decode/alloc#time", short_name="decode.alloc", delta=-5.0),
                MetricSpec(name="encode/time#time", short_name="encode.time", delta=-10.0),
            ],
        ),
    ).groups

    assert [group.group for group in groups] == ["decode", "encode"]
    assert groups[0].geomean.n == 2
    assert groups[1].geomean.n == 1


def test_compute_kind_aggregates_when_deeper_path_does_use_full_prefix_as_group():
    groups = only_kind(
        kinds_of(
            [
                MetricSpec(
                    name="node/access/get_1field#time", short_name="access.get_1field", delta=-10.0
                ),
                MetricSpec(
                    name="node/access/get_2field#time", short_name="access.get_2field", delta=-10.0
                ),
            ],
        ),
    ).groups

    assert [group.group for group in groups] == ["node/access"]
    assert groups[0].geomean.n == 2


def test_compute_kind_aggregates_when_single_segment_name_does_count_in_kind_not_group():
    kind = only_kind(
        kinds_of(
            [
                MetricSpec(name="decode/time#time", short_name="decode.time", delta=-10.0),
                MetricSpec(name="warmup#time", short_name="warmup", delta=-10.0),
            ],
        ),
    )

    assert [group.group for group in kind.groups] == ["decode"]
    assert kind.groups[0].geomean.n == 1
    assert kind.geomean.n == 2


def test_compute_kind_aggregates_when_all_single_segment_does_give_kind_no_groups():
    groups = only_kind(
        kinds_of(
            [
                MetricSpec(name="alpha#time", short_name="alpha", delta=-10.0),
                MetricSpec(name="beta#time", short_name="beta", delta=-5.0),
            ],
        ),
    ).groups

    assert groups == ()


def test_compute_kind_aggregates_when_grouped_name_in_one_kind_does_leave_other_kind_flat():
    result = kinds_of(
        [
            MetricSpec(name="decode/time#time", kind="time", short_name="decode.time", delta=-10.0),
            MetricSpec(name="heap#memory", kind="memory", short_name="heap", delta=-10.0),
        ],
    )

    assert [group.group for group in kind_named(result, "time").groups] == ["decode"]
    assert kind_named(result, "memory").groups == ()


# ---------------------------------------------------------------------------
# Ordering by first mention
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_many_kinds_and_groups_does_order_by_first_mention():
    result = kinds_of(
        [
            MetricSpec(name="encode/time#time", kind="time", short_name="encode.time", delta=-10.0),
            MetricSpec(
                name="encode/heap#memory", kind="memory", short_name="encode.heap", delta=-10.0
            ),
            MetricSpec(name="decode/time#time", kind="time", short_name="decode.time", delta=-10.0),
        ],
    )

    assert [aggregate.kind for aggregate in result] == ["time", "memory"]
    assert [group.group for group in kind_named(result, "time").groups] == ["encode", "decode"]


# ---------------------------------------------------------------------------
# Kind geomean scope and per-subset exclusions
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_kind_mixes_gating_does_geomean_over_all_its_metrics():
    geomean = only_kind(kinds_of(mixed_gating_kind())).geomean

    assert geomean.n == 2
    assert geomean.value == pytest.approx(-7.54, abs=5e-2)


@pytest.mark.parametrize(
    ("bad_verdict", "reason"),
    [
        pytest.param(unstable_band_verdict(), "unstable", id="unstable"),
        pytest.param(exact_verdict(math.nan), "undefined-ratio", id="undefined-ratio"),
        pytest.param(exact_verdict(-150.0), "infinite-rho", id="infinite-rho"),
    ],
)
def test_compute_kind_aggregates_when_metric_excluded_does_report_it_against_the_kind_subset(
    bad_verdict: MetricVerdict,
    reason: ExclusionReason,
):
    geomean = only_kind(
        kinds_of(
            [
                MetricSpec(name="bad", short_name="bad", verdict=bad_verdict),
                MetricSpec(name="good", short_name="good", delta=-5.0),
            ],
        ),
    ).geomean

    assert geomean.n == 1
    assert geomean.excluded == (Exclusion(metric="bad", reason=reason),)
    assert geomean.value == pytest.approx(-5.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Gated geomean
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_kind_mixes_gating_does_gate_over_only_gating_metrics():
    kind = only_kind(kinds_of(mixed_gating_kind()))

    assert kind.gated_geomean is not None
    assert kind.gated_geomean.n == 1
    assert kind.gated_geomean.value == pytest.approx(-10.0, abs=1e-5)


def test_compute_kind_aggregates_when_gating_differs_across_kinds_does_decide_gate_per_kind():
    result = kinds_of(
        [
            MetricSpec(name="m1", kind="time", short_name="time", gating=True, delta=-10.0),
            MetricSpec(name="m2", kind="memory", short_name="heap", gating=False, delta=-10.0),
        ],
    )

    assert kind_named(result, "time").gated_geomean is not None
    assert kind_named(result, "memory").gated_geomean is None


# ---------------------------------------------------------------------------
# Group geomean
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_group_mixes_gating_does_geomean_over_all_group_metrics():
    groups = only_kind(kinds_of(mixed_gating_kind())).groups

    assert groups[0].geomean.n == 2
    assert groups[0].geomean.value == pytest.approx(-7.54, abs=5e-2)
