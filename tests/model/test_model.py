import dataclasses
from typing import assert_never

import pytest

from gymrat.model import (
    BAND_MIN_N,
    DEFAULT_UNSTABLE_NOISE_PCT,
    NOISE_FLOOR_PCT,
    NOISE_K,
    PERMUTATION_MIN_N,
    PERMUTATION_P_THRESHOLD,
    BandVerdict,
    ExactVerdict,
    Exclusion,
    GeomeanResult,
    MetricMeta,
    MetricVerdict,
    PairResult,
    PermutationVerdict,
    ResolvedMetricMeta,
    Verdict,
)

_DELTA = 1.0

_VALUE_RECORDS = [
    pytest.param(
        MetricMeta(direction="higher", gating=False, exact=True, unit=None),
        "gating",
        id="metric-meta",
    ),
    pytest.param(
        ResolvedMetricMeta(
            direction="lower",
            gating=True,
            exact=False,
            unit="ns",
            kind="time",
            short_name="decode",
        ),
        "kind",
        id="resolved-metric-meta",
    ),
    pytest.param(
        PermutationVerdict(
            method="permutation",
            verdict="improved",
            p=0.01,
            noise_pct=1.0,
            noise_abs=0.1,
            delta=_DELTA,
            n=6,
        ),
        "verdict",
        id="permutation-verdict",
    ),
    pytest.param(
        BandVerdict(
            method="band",
            verdict="no-signal",
            usable_n=2,
            noise_pct=1.0,
            noise_abs=0.1,
            delta=_DELTA,
            n=2,
        ),
        "verdict",
        id="band-verdict",
    ),
    pytest.param(
        ExactVerdict(method="exact", verdict="regressed", delta=_DELTA, n=1),
        "verdict",
        id="exact-verdict",
    ),
    pytest.param(Exclusion(metric="a", reason="unstable"), "reason", id="exclusion"),
    pytest.param(
        GeomeanResult(value=1.1, n=2, band=0.5, excluded=()), "value", id="geomean-result"
    ),
    pytest.param(PairResult(left=(1.0,), right=(2.0,), dropped=0), "dropped", id="pair-result"),
]


@pytest.mark.parametrize(("instance", "field"), _VALUE_RECORDS)
def test_value_record_when_field_assigned_does_raise_frozen_instance_error(
    instance: object, field: str
):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, None)


@pytest.mark.parametrize(("instance", "_field"), _VALUE_RECORDS)
def test_value_record_when_instantiated_does_carry_no_instance_dict(instance: object, _field: str):
    assert not hasattr(instance, "__dict__")


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


# ---------------------------------------------------------------------------
# ResolvedMetricMeta
# ---------------------------------------------------------------------------


@pytest.fixture
def resolved_meta() -> ResolvedMetricMeta:
    """A fully-populated ResolvedMetricMeta for reuse across assertions."""
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


# ---------------------------------------------------------------------------
# Method floors and noise constants
# ---------------------------------------------------------------------------


def test_method_floors_when_referenced_does_match_model_defaults():
    assert PERMUTATION_MIN_N == 6
    assert PERMUTATION_P_THRESHOLD == 0.05
    assert BAND_MIN_N == 2


def test_noise_constants_when_referenced_does_match_model_defaults():
    assert NOISE_K == 1.5
    assert NOISE_FLOOR_PCT == 0.5
    assert DEFAULT_UNSTABLE_NOISE_PCT == 200


# ---------------------------------------------------------------------------
# Verdict records
# ---------------------------------------------------------------------------


def test_permutation_verdict_when_constructed_does_store_fields():
    verdict = PermutationVerdict(
        method="permutation",
        verdict="improved",
        p=0.01,
        noise_pct=1.2,
        noise_abs=3.4,
        delta=5.5,
        n=8,
    )

    assert verdict.method == "permutation"
    assert verdict.verdict == "improved"
    assert verdict.p == 0.01
    assert verdict.noise_pct == 1.2
    assert verdict.noise_abs == 3.4
    assert verdict.delta == 5.5
    assert verdict.n == 8


def test_band_verdict_when_constructed_does_store_fields():
    verdict = BandVerdict(
        method="band",
        verdict="unstable",
        usable_n=4,
        noise_pct=2.0,
        noise_abs=5.0,
        delta=-7.0,
        n=6,
    )

    assert verdict.method == "band"
    assert verdict.verdict == "unstable"
    assert verdict.usable_n == 4
    assert verdict.noise_pct == 2.0
    assert verdict.noise_abs == 5.0
    assert verdict.delta == -7.0
    assert verdict.n == 6


def test_exact_verdict_when_constructed_does_store_fields():
    verdict = ExactVerdict(
        method="exact",
        verdict="regressed",
        delta=-1.5,
        n=10,
    )

    assert verdict.method == "exact"
    assert verdict.verdict == "regressed"
    assert verdict.delta == -1.5
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
        delta=0.0,
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
                delta=1.0,
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
                delta=2.0,
                n=4,
            ),
            "band",
        ),
        (
            ExactVerdict(
                method="exact",
                verdict="improved",
                delta=1.0,
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
# Exclusion taxonomy
# ---------------------------------------------------------------------------


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
