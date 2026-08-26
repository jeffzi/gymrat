import dataclasses
from typing import assert_never

import pytest

from gymrat_py.model import (
    BAND_DESCRIPTOR,
    DEFAULT_UNSTABLE_NOISE_PCT,
    EXACT_DESCRIPTOR,
    NOISE_FLOOR_PCT,
    NOISE_K,
    PERMUTATION_DESCRIPTOR,
    Aggregate,
    BandVerdict,
    Effect,
    ExactVerdict,
    Exclusion,
    GeomeanResult,
    Method,
    MethodDescriptor,
    MetricMeta,
    MetricVerdict,
    PermutationVerdict,
    ResolvedMetricMeta,
    Verdict,
)

# ---------------------------------------------------------------------------
# Effect
# ---------------------------------------------------------------------------


def test_effect_when_constructed_does_store_value_and_unit():
    effect = Effect(value=12.5, unit="percent")

    assert effect.value == 12.5
    assert effect.unit == "percent"


def test_effect_when_field_assigned_does_raise_frozen_instance_error():
    effect = Effect(value=1.0, unit="percent")

    with pytest.raises(dataclasses.FrozenInstanceError):
        # The write is rejected at runtime; the type checker flags it statically, so the
        # suppression documents the intentional frozen-field violation under test.
        effect.value = 2.0  # pyrefly: ignore


def test_effect_when_values_equal_does_compare_equal():
    assert Effect(value=1.0, unit="percent") == Effect(value=1.0, unit="percent")


def test_effect_when_values_differ_does_compare_unequal():
    assert Effect(value=1.0, unit="percent") != Effect(value=2.0, unit="percent")


# ---------------------------------------------------------------------------
# MetricMeta
# ---------------------------------------------------------------------------


def test_metric_meta_when_constructed_does_store_four_fields():
    meta = MetricMeta(direction="lower", gating=True, exact=False, unit="ns")

    assert meta.direction == "lower"
    assert meta.gating is True
    assert meta.exact is False
    assert meta.unit == "ns"


def test_metric_meta_when_inspected_does_have_exactly_four_named_fields():
    names = [field.name for field in dataclasses.fields(MetricMeta)]

    assert names == ["direction", "gating", "exact", "unit"]


def test_metric_meta_when_field_assigned_does_raise_frozen_instance_error():
    meta = MetricMeta(direction="higher", gating=False, exact=True, unit=None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        # The write is rejected at runtime; the type checker flags it statically, so the
        # suppression documents the intentional frozen-field violation under test.
        meta.gating = True  # pyrefly: ignore


# ---------------------------------------------------------------------------
# ResolvedMetricMeta
# ---------------------------------------------------------------------------


@pytest.fixture
def resolved_meta() -> ResolvedMetricMeta:
    return ResolvedMetricMeta(
        direction="lower",
        gating=True,
        exact=False,
        unit="ns",
        kind="time",
        short_name="decode",
    )


def test_resolved_metric_meta_when_constructed_does_store_six_fields(
    resolved_meta: ResolvedMetricMeta,
):
    assert resolved_meta.direction == "lower"
    assert resolved_meta.gating is True
    assert resolved_meta.exact is False
    assert resolved_meta.unit == "ns"
    assert resolved_meta.kind == "time"
    assert resolved_meta.short_name == "decode"


def test_resolved_metric_meta_when_constructed_does_satisfy_metric_meta(
    resolved_meta: ResolvedMetricMeta,
):
    assert isinstance(resolved_meta, MetricMeta)


def test_resolved_metric_meta_when_inspected_does_have_exactly_six_named_fields():
    names = [field.name for field in dataclasses.fields(ResolvedMetricMeta)]

    assert names == ["direction", "gating", "exact", "unit", "kind", "short_name"]


def test_resolved_metric_meta_when_field_assigned_does_raise_frozen_instance_error(
    resolved_meta: ResolvedMetricMeta,
):
    with pytest.raises(dataclasses.FrozenInstanceError):
        # The write is rejected at runtime; the type checker flags it statically, so the
        # suppression documents the intentional frozen-field violation under test.
        resolved_meta.kind = "mem"  # pyrefly: ignore


# ---------------------------------------------------------------------------
# Method descriptors and noise constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("descriptor", "method", "min_n", "p_threshold"),
    [
        (PERMUTATION_DESCRIPTOR, "permutation", 6, 0.05),
        (BAND_DESCRIPTOR, "band", 2, None),
        (EXACT_DESCRIPTOR, "exact", 1, None),
    ],
)
def test_method_descriptor_when_defined_does_expose_statistical_floors(
    descriptor: MethodDescriptor,
    method: Method,
    min_n: int,
    p_threshold: float | None,
):
    assert descriptor.method == method
    assert descriptor.min_n == min_n
    assert descriptor.p_threshold == p_threshold


def test_noise_constants_when_referenced_does_match_model_defaults():
    assert NOISE_K == 1.5
    assert NOISE_FLOOR_PCT == 0.5
    assert DEFAULT_UNSTABLE_NOISE_PCT == 200


# ---------------------------------------------------------------------------
# Verdict records
# ---------------------------------------------------------------------------


def _percent(value: float) -> Effect:
    """Build an ``Effect`` with an arbitrary, unit-under-test-irrelevant "percent" unit."""
    return Effect(value=value, unit="percent")


def test_permutation_verdict_when_constructed_does_store_fields():
    verdict = PermutationVerdict(
        method="permutation",
        verdict="improved",
        p=0.01,
        noise_pct=1.2,
        noise_abs=3.4,
        delta=_percent(5.5),
        n=8,
    )

    assert verdict.method == "permutation"
    assert verdict.verdict == "improved"
    assert verdict.p == 0.01
    assert verdict.noise_pct == 1.2
    assert verdict.noise_abs == 3.4
    assert verdict.delta == _percent(5.5)
    assert verdict.n == 8


def test_band_verdict_when_constructed_does_store_fields():
    verdict = BandVerdict(
        method="band",
        verdict="unstable",
        usable_n=4,
        noise_pct=2.0,
        noise_abs=5.0,
        delta=_percent(-7.0),
        n=6,
    )

    assert verdict.method == "band"
    assert verdict.verdict == "unstable"
    assert verdict.usable_n == 4
    assert verdict.noise_pct == 2.0
    assert verdict.noise_abs == 5.0
    assert verdict.delta == _percent(-7.0)
    assert verdict.n == 6


def test_exact_verdict_when_constructed_does_store_fields():
    verdict = ExactVerdict(
        method="exact",
        verdict="regressed",
        delta=_percent(-1.5),
        n=10,
    )

    assert verdict.method == "exact"
    assert verdict.verdict == "regressed"
    assert verdict.delta == _percent(-1.5)
    assert verdict.n == 10


def test_exact_verdict_when_inspected_does_omit_noise_fields():
    names = {field.name for field in dataclasses.fields(ExactVerdict)}

    assert "noise_pct" not in names
    assert "noise_abs" not in names


def _accept_verdict(value: Verdict) -> Verdict:
    """Type-checked sink: an ``ExactVerdict.verdict`` must satisfy the non-approximate ``Verdict``."""
    return value


def test_exact_verdict_verdict_when_passed_to_verdict_sink_does_round_trip():
    verdict = ExactVerdict(
        method="exact",
        verdict="no-signal",
        delta=_percent(0.0),
        n=3,
    )

    assert _accept_verdict(verdict.verdict) == "no-signal"


def describe(verdict: MetricVerdict) -> str:
    """Exhaustive match over the discriminant; the ``assert_never`` arm pins the union."""
    match verdict.method:
        case "permutation":
            return "permutation"
        case "band":
            return "band"
        case "exact":
            return "exact"
        case _ as unreachable:
            assert_never(unreachable)


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (
            PermutationVerdict(
                method="permutation",
                verdict="improved",
                p=0.01,
                noise_pct=1.0,
                noise_abs=2.0,
                delta=_percent(1.0),
                n=6,
            ),
            "permutation",
        ),
        (
            BandVerdict(
                method="band",
                verdict="unstable",
                usable_n=3,
                noise_pct=1.0,
                noise_abs=2.0,
                delta=_percent(2.0),
                n=4,
            ),
            "band",
        ),
        (
            ExactVerdict(
                method="exact",
                verdict="improved",
                delta=_percent(1.0),
                n=5,
            ),
            "exact",
        ),
    ],
)
def test_describe_when_given_each_variant_does_return_method_tag(
    verdict: MetricVerdict,
    expected: str,
):
    assert describe(verdict) == expected


# ---------------------------------------------------------------------------
# Aggregate protocol and exclusion taxonomy
# ---------------------------------------------------------------------------


def _accept_aggregate(aggregate: Aggregate) -> Aggregate:
    """Type-checked sink proving ``GeomeanResult`` structurally satisfies ``Aggregate``."""
    return aggregate


def test_geomean_result_when_passed_to_aggregate_sink_does_satisfy_protocol():
    result = GeomeanResult(value=1.0, n=3, band=0.5, excluded=())

    aggregate: Aggregate = _accept_aggregate(result)

    assert aggregate.value == 1.0


def test_geomean_result_when_given_all_exclusion_reasons_does_round_trip_fields():
    excluded = (
        Exclusion(metric="a", reason="no-verdict"),
        Exclusion(metric="b", reason="unstable"),
        Exclusion(metric="c", reason="undefined-ratio"),
        Exclusion(metric="d", reason="infinite-rho"),
    )

    result = GeomeanResult(value=2.5, n=4, band=0.1, excluded=excluded)

    assert result.value == 2.5
    assert result.n == 4
    assert result.band == 0.1
    assert result.excluded == excluded
    assert [entry.reason for entry in result.excluded] == [
        "no-verdict",
        "unstable",
        "undefined-ratio",
        "infinite-rho",
    ]
