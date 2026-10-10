"""Behavioral tests for geomean aggregation over verdict records.

Drives behavior through the public ``compute_geomean`` and
``compute_kind_aggregates`` APIs. Exclusion ordering, ratio normalization and
noise-band propagation are exercised on the geomean itself; bucketing order,
per-kind grouping and the per-subset exclusion taxonomy on the kind aggregates
built from it.

A compact ``MetricSpec`` describes one metric's contribution to a run, and
:func:`build_inputs` turns a list of specs into the ``(verdicts, metric_meta)``
pair both APIs consume, preserving spec order.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from gymrat.model import (
    BandVerdict,
    Direction,
    Exclusion,
    GeomeanResult,
    MetricVerdict,
    ResolvedMetricMeta,
)
from gymrat.verdict import KindAggregate, compute_geomean, compute_kind_aggregates
from tests.report._verdicts import band_verdict, exact_verdict, metric_meta

# ---------------------------------------------------------------------------
# Metric specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """One metric's contribution to a run: what it is, and how it moved."""

    name: str
    """Full metric name — the key the verdict and exclusion lists report it under."""

    direction: Direction = "lower"
    """Defaults to ``"lower"``, so a directionless spec list is lower-is-better."""

    gating: bool = True
    """Defaults to ``True``, matching every case that isn't explicitly non-gating."""

    kind: str = "time"
    """Defaults to ``"time"``, so a kindless spec list describes a single-kind run."""

    delta: float | None = None
    """Percentage delta behind an exact verdict. Ignored when ``verdict`` is given."""

    verdict: MetricVerdict | None = None
    """A full verdict object, for band and unstable cases :func:`exact_verdict` cannot express."""

    no_verdict: bool = False
    """True to add ``name`` to ``metric_meta`` without a matching verdict — the no-verdict case."""


def _resolve_verdict(spec: MetricSpec) -> MetricVerdict | None:
    """The verdict a spec contributes, or ``None`` for the no-verdict case."""
    if spec.no_verdict:
        return None
    if spec.verdict is not None:
        return spec.verdict
    return exact_verdict(delta=spec.delta or 0.0, n=1)


def build_inputs(
    specs: Sequence[MetricSpec],
) -> tuple[dict[str, MetricVerdict], dict[str, ResolvedMetricMeta]]:
    """Verdicts and metadata keyed by metric name, in the order the specs are listed."""
    verdicts: dict[str, MetricVerdict] = {}
    meta_by_name: dict[str, ResolvedMetricMeta] = {}

    for spec in specs:
        verdict = _resolve_verdict(spec)
        if verdict is not None:
            verdicts[spec.name] = verdict

        meta_by_name[spec.name] = metric_meta(
            spec.name,
            direction=spec.direction,
            gating=spec.gating,
            exact=verdict is not None and verdict.method == "exact",
            kind=spec.kind,
        )

    return verdicts, meta_by_name


def unstable_band_verdict() -> BandVerdict:
    """A band verdict too noisy to judge, regardless of its ratio."""
    return band_verdict(
        verdict="unstable", usable_n=4, noise_pct=250.0, noise_abs=25.0, delta=-50.0, n=4
    )


# ---------------------------------------------------------------------------
# Multiple gating metrics
# ---------------------------------------------------------------------------


def test_compute_geomean_when_directions_differ_does_respect_each_metric():
    verdicts, meta = build_inputs(
        [
            MetricSpec(name="metric1", direction="lower", delta=-10.0),
            MetricSpec(name="metric2", direction="higher", delta=10.0),
        ],
    )

    result = compute_geomean(verdicts, meta)

    assert result.n == 2
    assert result.excluded == ()
    assert result.value == pytest.approx((math.sqrt(0.9 / 1.1) - 1) * 100, abs=1e-6)


def test_compute_geomean_when_one_metric_invalid_does_keep_other_ratio():
    verdicts, meta = build_inputs(
        [
            MetricSpec(name="metric1", delta=math.nan),
            MetricSpec(name="metric2", delta=-5.0),
        ],
    )

    result = compute_geomean(verdicts, meta)

    assert result.n == 1
    assert result.excluded == (Exclusion(metric="metric1", reason="undefined-ratio"),)
    assert result.value == pytest.approx(-5.0, abs=1e-5)


def test_compute_geomean_when_several_metrics_excluded_does_list_them_in_meta_order():
    # Neither names nor reasons are in sorted order, so only the metric_meta
    # order explains the expected sequence.
    verdicts, meta = build_inputs(
        [
            MetricSpec(name="metric2", delta=math.nan),
            MetricSpec(name="metric1", no_verdict=True),
            MetricSpec(name="metric3", delta=-5.0),
        ],
    )

    result = compute_geomean(verdicts, meta)

    assert (result.n, result.excluded) == (
        1,
        (
            Exclusion(metric="metric2", reason="undefined-ratio"),
            Exclusion(metric="metric1", reason="no-verdict"),
        ),
    )


# ---------------------------------------------------------------------------
# Unstable exclusion
# ---------------------------------------------------------------------------


def test_compute_geomean_when_metric_unstable_does_exclude_it():
    verdicts, meta = build_inputs(
        [
            MetricSpec(name="noisy", verdict=unstable_band_verdict()),
            MetricSpec(
                name="stable",
                verdict=band_verdict(
                    verdict="improved",
                    usable_n=4,
                    noise_pct=4.0,
                    noise_abs=2.0,
                    delta=-5.0,
                    n=4,
                ),
            ),
        ],
    )

    result = compute_geomean(verdicts, meta)

    assert (result.n, result.excluded) == (1, (Exclusion(metric="noisy", reason="unstable"),))
    assert (result.value, result.band) == pytest.approx((-5.0, 4.0), abs=1e-5)


def test_compute_geomean_when_unstable_delta_nan_does_report_unstable_over_undefined():
    verdicts, meta = build_inputs(
        [
            MetricSpec(
                name="noisy",
                verdict=band_verdict(
                    verdict="unstable",
                    usable_n=4,
                    noise_pct=300.0,
                    noise_abs=30.0,
                    delta=math.nan,
                    n=4,
                ),
            ),
        ],
    )

    result = compute_geomean(verdicts, meta)

    assert result == GeomeanResult(
        value=0.0,
        n=0,
        band=0.0,
        excluded=(Exclusion(metric="noisy", reason="unstable"),),
    )


# ---------------------------------------------------------------------------
# Propagated noise band
# ---------------------------------------------------------------------------


def test_compute_geomean_when_exact_metric_beside_noisy_one_does_add_no_noise_to_band():
    # An exact verdict carries no noise figure, but it is judged, so the geomean
    # takes it and the band halves the noisy metric's noise across both.
    verdicts, meta = build_inputs([
        MetricSpec(name="metric1", verdict=exact_verdict(delta=-50.0, n=4)),
        MetricSpec(
            name="metric2",
            verdict=band_verdict(
                verdict="improved",
                usable_n=4,
                noise_pct=6.0,
                noise_abs=3.0,
                delta=-50.0,
                n=4,
            ),
        ),
    ])

    result = compute_geomean(verdicts, meta)

    assert result.band == pytest.approx(3.0, abs=1e-10)


def kinds_of(specs: Sequence[MetricSpec]) -> list[KindAggregate]:
    """The kind aggregates for a spec list — ``build_inputs`` fed into ``compute_kind_aggregates``."""
    verdicts, meta = build_inputs(specs)
    return compute_kind_aggregates(verdicts, meta)


def kind_named(aggregates: Sequence[KindAggregate], kind: str) -> KindAggregate:
    """The aggregate for ``kind``, or a failure naming the kinds produced."""
    for aggregate in aggregates:
        if aggregate.kind == kind:
            return aggregate
    names = ", ".join(aggregate.kind for aggregate in aggregates)
    pytest.fail(f'no aggregate for kind "{kind}", only: {names}')


# ---------------------------------------------------------------------------
# Kind aggregate shape and empty inputs
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_single_non_gating_metric_does_carry_kind_geomean_no_gate():
    result = kinds_of(
        [MetricSpec(name="warmup", gating=False, delta=0.0)],
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
    result = kinds_of([])

    assert result == []


# ---------------------------------------------------------------------------
# Grouping by metric name contract (path minus last segment)
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_multi_segment_names_does_group_by_path_prefix():
    [kind] = kinds_of(
        [
            MetricSpec(name="decode/time#time", delta=-10.0),
            MetricSpec(name="decode/alloc#time", delta=-5.0),
            MetricSpec(name="encode/time#time", delta=-10.0),
        ],
    )

    assert [group.group for group in kind.groups] == ["decode", "encode"]
    assert kind.groups[0].geomean.n == 2
    assert kind.groups[1].geomean.n == 1


def test_compute_kind_aggregates_when_single_segment_name_does_count_in_kind_not_group():
    [kind] = kinds_of(
        [
            MetricSpec(name="decode/time#time", delta=-10.0),
            MetricSpec(name="warmup#time", delta=-10.0),
        ],
    )

    assert [group.group for group in kind.groups] == ["decode"]
    assert kind.groups[0].geomean.n == 1
    assert kind.geomean.n == 2


def test_compute_kind_aggregates_when_grouped_name_in_one_kind_does_leave_other_kind_flat():
    result = kinds_of(
        [
            MetricSpec(name="decode/time#time", kind="time", delta=-10.0),
            MetricSpec(name="heap#memory", kind="memory", delta=-10.0),
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
            MetricSpec(name="encode/time#time", kind="time", delta=-10.0),
            MetricSpec(name="encode/heap#memory", kind="memory", delta=-10.0),
            MetricSpec(name="decode/time#time", kind="time", delta=-10.0),
        ],
    )

    assert [aggregate.kind for aggregate in result] == ["time", "memory"]
    assert [group.group for group in kind_named(result, "time").groups] == ["encode", "decode"]


# ---------------------------------------------------------------------------
# Geomean scope: kind, gated and group
# ---------------------------------------------------------------------------


def test_compute_kind_aggregates_when_metrics_differ_only_in_gating_does_gate_over_gating_alone():
    [kind] = kinds_of(
        [
            MetricSpec(name="decode/time#time", gating=True, delta=-10.0),
            MetricSpec(name="decode/alloc#time", gating=False, delta=-5.0),
        ],
    )

    assert kind.geomean.n == 2
    assert kind.gated_geomean is not None
    assert (kind.gated_geomean.n, kind.gated_geomean.value) == (1, pytest.approx(-10.0, abs=1e-5))
    assert kind.groups[0].geomean.n == 2


def test_compute_kind_aggregates_when_metric_unstable_does_exclude_it_only_where_it_belongs():
    [kind] = kinds_of(
        [
            MetricSpec(name="decode/bad#time", verdict=unstable_band_verdict()),
            MetricSpec(name="decode/good#time", delta=-5.0),
            MetricSpec(name="encode/fine#time", delta=-5.0),
        ],
    )

    excluded = (Exclusion(metric="decode/bad#time", reason="unstable"),)
    assert (kind.geomean.n, kind.geomean.excluded) == (2, excluded)
    assert [(group.group, group.geomean.n, group.geomean.excluded) for group in kind.groups] == [
        ("decode", 1, excluded),
        ("encode", 1, ()),
    ]
