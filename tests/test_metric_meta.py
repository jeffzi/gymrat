"""Behavioral tests for metric-meta resolution.

Adapter defaults, then kind, then metric, including the warning for time
readings that look quantized, and the direction of the configured primary.
"""

from collections.abc import Callable
from dataclasses import replace

import pytest

from gymrat.adapters.contract import TIME_SUFFIX, Adapter, MetricDefaults
from gymrat.adapters.metric_lines import metric_lines_adapter
from gymrat.config import KindEntry, MetricEntry
from gymrat.errors import GymratError
from gymrat.metric_meta import (
    primary_direction,
    resolve_metric_meta,
    resolve_metric_meta_from_samples,
)
from gymrat.model import GEOMEAN_DIRECTION, ResolvedMetricMeta
from gymrat.utils import WarnSink, warn_to_stderr
from tests._config import resolved_config
from tests._pipeline import is_quantized_warning
from tests.report._verdicts import metric_meta


def make_adapter(
    defaults_fn: Callable[[str], MetricDefaults] = lambda _name: MetricDefaults(direction="lower"),
) -> Adapter:
    """Build a mock adapter whose per-metric defaults come from ``defaults_fn``."""

    class MockAdapter:
        name = "test-adapter"

        def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
            return {}

        def defaults(self, metric_name: str) -> MetricDefaults:
            return defaults_fn(metric_name)

    return MockAdapter()


# ---------------------------------------------------------------------------
# resolve_metric_meta
# ---------------------------------------------------------------------------

_NAME = "bench-a/heap"
_HEAP = MetricDefaults(direction="higher", kind="memory", short_name="heap", unit="bytes")
_MEMORY_NOT_GATING = {"memory": KindEntry(gating=False)}
_HEAP_STEP = replace(_HEAP, resolution=1.0)
_MEMORY_STEP = {"memory": KindEntry(resolution=1024.0)}


def _heap_step_meta(
    resolution: float, *, gating: bool = True, exact: bool = False
) -> ResolvedMetricMeta:
    return metric_meta(
        "heap",
        direction="higher",
        kind="memory",
        unit="bytes",
        gating=gating,
        exact=exact,
        resolution=resolution,
    )


@pytest.mark.parametrize(
    ("defaults", "entry", "config_kinds", "expected"),
    [
        pytest.param(
            MetricDefaults(direction="higher", unit="ns"),
            None,
            None,
            metric_meta(_NAME, direction="higher", unit="ns"),
            id="adapter-direction-and-unit",
        ),
        pytest.param(
            _HEAP,
            MetricEntry(direction="lower"),
            None,
            metric_meta("heap", direction="lower", kind="memory", unit="bytes"),
            id="entry-direction",
        ),
        pytest.param(
            _HEAP,
            MetricEntry(exact=True),
            _MEMORY_NOT_GATING,
            metric_meta(
                "heap", direction="higher", kind="memory", unit="bytes", gating=False, exact=True
            ),
            id="kind-gating-fills-an-entry-without-gating",
        ),
        pytest.param(
            _HEAP,
            None,
            {"time": KindEntry(gating=False)},
            metric_meta("heap", direction="higher", kind="memory", unit="bytes"),
            id="other-kind-gating-ignored",
        ),
        pytest.param(
            _HEAP,
            MetricEntry(gating=True),
            _MEMORY_NOT_GATING,
            metric_meta("heap", direction="higher", kind="memory", unit="bytes"),
            id="entry-gating-beats-kind-gating",
        ),
        pytest.param(
            MetricDefaults(direction="lower", short_name="heap"),
            None,
            {"other": KindEntry(gating=False)},
            metric_meta("heap", kind="other", gating=False),
            id="adapter-reports-no-kind",
        ),
        pytest.param(_HEAP_STEP, None, None, _heap_step_meta(1.0), id="adapter-resolution"),
        pytest.param(
            _HEAP_STEP,
            MetricEntry(resolution=8.0),
            _MEMORY_STEP,
            _heap_step_meta(8.0),
            id="entry-resolution-beats-kind-resolution",
        ),
        pytest.param(
            _HEAP_STEP,
            MetricEntry(exact=True),
            _MEMORY_STEP,
            _heap_step_meta(1024.0, exact=True),
            id="kind-resolution-fills-an-entry-without-resolution",
        ),
    ],
)
def test_resolve_metric_meta_when_layers_given_does_layer_entry_over_kind_over_adapter(
    defaults: MetricDefaults,
    entry: MetricEntry | None,
    config_kinds: dict[str, KindEntry] | None,
    expected: ResolvedMetricMeta,
):
    adapter = make_adapter(lambda _name: defaults)

    result = resolve_metric_meta(_NAME, entry, adapter, config_kinds)

    assert result == expected


# ---------------------------------------------------------------------------
# resolve_metric_meta_from_samples
# ---------------------------------------------------------------------------


def _by_metric_name(name: str) -> MetricDefaults:
    if name == "response-time":
        return MetricDefaults(direction="lower", unit="ns")
    if name == "throughput":
        return MetricDefaults(direction="higher")
    return MetricDefaults(direction="lower")


def _by_metric_suffix(name: str) -> MetricDefaults:
    if name.endswith("/heap"):
        return MetricDefaults(direction="lower", kind="memory", short_name="heap")
    return MetricDefaults(direction="lower", kind="time", short_name="time")


def test_resolve_metric_meta_from_samples_when_no_set_reports_a_metric_does_raise():
    sample_sets = [[{}, {}], [{}, {}]]

    with pytest.raises(GymratError):
        resolve_metric_meta_from_samples(sample_sets, None, make_adapter(), None)


@pytest.mark.parametrize(
    ("sample_sets", "defaults_fn", "config_metrics", "config_kinds", "expected"),
    [
        pytest.param(
            [[{"throughput": 1.0}], [{"response-time": 1.0}]],
            _by_metric_name,
            {
                "response-time": MetricEntry(gating=False),
                "throughput": MetricEntry(exact=True),
            },
            None,
            {
                "throughput": metric_meta("throughput", direction="higher", exact=True),
                "response-time": metric_meta("response-time", unit="ns", gating=False),
            },
            id="per-metric-entries-first-seen-in-a-later-set",
        ),
        pytest.param(
            [[{"bench-a/time": 1.0}, {"bench-a/time": 1.0, "bench-a/heap": 1.0}]],
            _by_metric_suffix,
            None,
            _MEMORY_NOT_GATING,
            {
                "bench-a/time": metric_meta("time", kind="time"),
                "bench-a/heap": metric_meta("heap", gating=False, kind="memory"),
            },
            id="kind-gating-first-seen-in-a-later-round",
        ),
    ],
)
def test_resolve_metric_meta_from_samples_when_several_names_reported_does_resolve_each_in_first_seen_order(
    sample_sets: list[list[dict[str, float]]],
    defaults_fn: Callable[[str], MetricDefaults],
    config_metrics: dict[str, MetricEntry] | None,
    config_kinds: dict[str, KindEntry] | None,
    expected: dict[str, ResolvedMetricMeta],
):
    result = resolve_metric_meta_from_samples(
        sample_sets, config_metrics, make_adapter(defaults_fn), config_kinds
    )

    assert list(result) == list(expected)
    assert result == expected


_TIME = f"bench{TIME_SUFFIX}"


def _readings(*per_target: list[float], name: str = _TIME) -> list[list[dict[str, float]]]:
    return [[{name: value} for value in values] for values in per_target]


_ONE_STEP_UP_VALUES = ([333.0] * 5 + [375.0], [375.0] * 5 + [417.0])
_ONE_STEP_UP = _readings(*_ONE_STEP_UP_VALUES)


@pytest.mark.parametrize(
    ("sample_sets", "gap"),
    [
        pytest.param(_ONE_STEP_UP, "42ns", id="one-step-up-on-every-pair"),
        pytest.param(_readings([333.0] * 10, [417.0] * 10), "84ns", id="two-values-ten-times"),
        pytest.param(
            _readings([332.9999999550637, 333.0], [375.0, 417.0]),
            "42ns",
            id="float-noise-counts-as-one-value",
        ),
        pytest.param(_readings([333.0] * 3, [333.0] * 3), None, id="every-reading-one-value"),
        pytest.param(_readings([0.0, 0.0], [0.0, 5.0]), "5ns", id="zero-median"),
        pytest.param(
            _readings(*([-value for value in values] for values in _ONE_STEP_UP_VALUES)),
            "42ns",
            id="negative-median",
        ),
        pytest.param(
            _readings([1000.0, 1000.0, 1084.0], [1210.0, 1294.0]),
            "42ns",
            id="coarse-lattice-names-the-fitted-step-not-the-smallest-gap",
        ),
        pytest.param(
            _readings([1000.0, 1042.0], [1126.0, 1210.0]),
            "42ns",
            id="coarse-lattice-without-a-repeat",
        ),
        pytest.param(
            _readings([1000.0, 1042.00001], [1084.00001, 1126.0]),
            "42ns",
            id="coarse-lattice-off-by-float-noise",
        ),
        pytest.param(
            _readings([1000.0, 1000.0000001], [1010.0, 1027.0]),
            "10ns",
            id="float-noise-keeps-three-distinct-readings-in-the-upper-bound-rule",
        ),
        pytest.param(
            _readings([0.0, 0.0, 0.0, 0.0], [-7.0, 0.0, 5.0, 12.0]),
            "5ns",
            id="zero-median-with-four-distinct-readings",
        ),
    ],
)
def test_resolve_metric_meta_from_samples_when_time_readings_look_quantized_does_warn_once(
    sample_sets: list[list[dict[str, float]]], gap: str | None
):
    warnings: list[str] = []

    resolve_metric_meta_from_samples(
        sample_sets, None, metric_lines_adapter, None, warn=warnings.append
    )

    assert [is_quantized_warning(warning, _TIME, gap) for warning in warnings] == [True]


@pytest.mark.parametrize(
    ("sample_sets", "config_metrics"),
    [
        pytest.param(_ONE_STEP_UP, {_TIME: MetricEntry(resolution=42.0)}, id="metric-resolution"),
        pytest.param(_ONE_STEP_UP, {_TIME: MetricEntry(exact=True)}, id="exact"),
        pytest.param(_readings(*_ONE_STEP_UP_VALUES, name="bench"), None, id="unit-not-ns"),
        pytest.param(_readings([333.0, 375.0], [333.0, 375.0]), None, id="no-target-repeats"),
        pytest.param(
            _readings([1000.0, 1000.0], [1000.0, 1005.0]), None, id="gap-at-half-a-percent"
        ),
        pytest.param(
            _readings([-1000.0, -1000.0], [-1000.0, -1005.0]),
            None,
            id="gap-at-half-a-percent-of-a-negative-median",
        ),
        pytest.param(
            _readings([1000.0, 1000.0, 1007.0], [1013.0, 1020.0]),
            None,
            id="fine-lattice-at-four-distinct-readings",
        ),
        pytest.param(
            _readings([1000.0, 1007.0, 1013.0], [1020.0, 1020.0, 1027.0]),
            None,
            id="fine-lattice-with-one-repeated-pair",
        ),
        pytest.param(
            _readings([990.0, 1000.0, 1000.0], [995.0, 1005.0, 1010.0]),
            None,
            id="lattice-at-half-a-percent-with-four-distinct-readings",
        ),
    ],
)
def test_resolve_metric_meta_from_samples_when_readings_not_suspect_does_not_warn(
    sample_sets: list[list[dict[str, float]]],
    config_metrics: dict[str, MetricEntry] | None,
):
    warnings: list[str] = []

    resolve_metric_meta_from_samples(
        sample_sets, config_metrics, metric_lines_adapter, None, warn=warnings.append
    )

    assert warnings == []


# ---------------------------------------------------------------------------
# primary_direction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        pytest.param("geomean", GEOMEAN_DIRECTION, id="geomean-ignores-a-geomean-metric-entry"),
        pytest.param("throughput", "higher", id="named-metric-resolves-its-entry"),
    ],
)
def test_primary_direction_when_primary_named_does_return_the_direction_it_resolves_to(
    primary: str, expected: str
):
    config = resolved_config(
        primary=primary,
        metrics={
            "throughput": MetricEntry(direction="higher"),
            "geomean": MetricEntry(direction="higher"),
        },
    )

    direction = primary_direction(config)

    assert direction == expected


def test_primary_direction_when_no_metric_entries_does_take_the_adapters_default():
    config = resolved_config(primary="throughput", metrics=None)

    direction = primary_direction(config)

    assert direction == "lower"
