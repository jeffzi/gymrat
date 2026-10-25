"""Behavioral tests for metric-meta resolution.

Adapter defaults, then kind, then metric, including the warning for time
readings that look quantized.
"""

from collections.abc import Callable
from dataclasses import replace

import pytest

from gymrat.adapters.contract import TIME_SUFFIX, MetricDefaults
from gymrat.adapters.metric_lines import metric_lines_adapter
from gymrat.config import KindEntry, MetricEntry
from gymrat.errors import GymratError
from gymrat.model import ResolvedMetricMeta
from gymrat.sampling import (
    resolve_metric_meta,
    resolve_metric_meta_from_samples,
)
from tests._pipeline import quantized_warning
from tests.report._verdicts import metric_meta
from tests.sampling._adapters import make_adapter

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
            MetricDefaults(direction="lower", kind="memory", short_name="heap"),
            None,
            None,
            metric_meta("heap", kind="memory"),
            id="adapter-kind-and-short-name",
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
            MetricEntry(gating=False),
            None,
            metric_meta("heap", direction="higher", kind="memory", unit="bytes", gating=False),
            id="entry-gating",
        ),
        pytest.param(
            _HEAP,
            MetricEntry(exact=True),
            None,
            metric_meta("heap", direction="higher", kind="memory", unit="bytes", exact=True),
            id="entry-exact",
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
            _MEMORY_NOT_GATING,
            metric_meta("heap", direction="higher", kind="memory", unit="bytes", gating=False),
            id="kind-gating-without-an-entry",
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
            _HEAP,
            None,
            {"memory": KindEntry(gating=None)},
            metric_meta("heap", direction="higher", kind="memory", unit="bytes"),
            id="kind-entry-without-gating-keeps-the-default",
        ),
        pytest.param(
            MetricDefaults(direction="lower", short_name="heap"),
            None,
            {"other": KindEntry(gating=False)},
            metric_meta("heap", kind="other", gating=False),
            id="adapter-reports-no-kind",
        ),
        pytest.param(_HEAP_STEP, None, None, _heap_step_meta(1.0), id="adapter-resolution"),
        pytest.param(_HEAP_STEP, None, _MEMORY_STEP, _heap_step_meta(1024.0), id="kind-resolution"),
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
        pytest.param(
            _HEAP_STEP,
            None,
            {"time": KindEntry(resolution=1024.0)},
            _heap_step_meta(1.0),
            id="other-kind-resolution-ignored",
        ),
        pytest.param(
            _HEAP_STEP,
            None,
            _MEMORY_NOT_GATING,
            _heap_step_meta(1.0, gating=False),
            id="kind-entry-without-resolution-keeps-the-default",
        ),
        pytest.param(
            _HEAP_STEP,
            MetricEntry(resolution=8.0),
            None,
            _heap_step_meta(8.0),
            id="entry-resolution-without-kinds",
        ),
        pytest.param(
            _HEAP,
            None,
            _MEMORY_STEP,
            _heap_step_meta(1024.0),
            id="kind-resolution-without-an-adapter-resolution",
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

    with pytest.raises(GymratError, match="^No metrics found in benchmark output$"):
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
    ],
)
def test_resolve_metric_meta_from_samples_when_time_readings_look_quantized_does_warn_once(
    sample_sets: list[list[dict[str, float]]], gap: str | None
):
    warnings: list[str] = []

    result = resolve_metric_meta_from_samples(
        sample_sets, None, metric_lines_adapter, None, warn=warnings.append
    )

    assert (result, warnings) == (
        {_TIME: metric_meta("bench", unit="ns", kind="time")},
        [quantized_warning(_TIME, gap)],
    )


def test_resolve_metric_meta_from_samples_when_no_sink_given_does_warn_on_stderr(
    capsys: pytest.CaptureFixture[str],
):
    resolve_metric_meta_from_samples(_ONE_STEP_UP, None, metric_lines_adapter, None)

    assert capsys.readouterr().err == f"{quantized_warning(_TIME, '42ns')}\n"


@pytest.mark.parametrize(
    ("sample_sets", "config_metrics", "config_kinds"),
    [
        pytest.param(
            _ONE_STEP_UP, {_TIME: MetricEntry(resolution=42.0)}, None, id="metric-resolution"
        ),
        pytest.param(_ONE_STEP_UP, {_TIME: MetricEntry(exact=True)}, None, id="exact"),
        pytest.param(
            _readings(*_ONE_STEP_UP_VALUES, name="bench"),
            None,
            None,
            id="unit-not-ns",
        ),
        pytest.param(_readings([333.0, 375.0], [333.0, 375.0]), None, None, id="no-target-repeats"),
        pytest.param(
            _readings([1000.0, 1000.0], [1000.0, 1005.0]), None, None, id="gap-at-half-a-percent"
        ),
        pytest.param(
            _readings([10_000.0, 10_001.0, 10_003.0, 10_000.0], [10_002.0, 10_005.0, 10_007.0]),
            None,
            None,
            id="ten-microseconds-on-a-one-nanosecond-timer",
        ),
    ],
)
def test_resolve_metric_meta_from_samples_when_readings_not_suspect_does_not_warn(
    sample_sets: list[list[dict[str, float]]],
    config_metrics: dict[str, MetricEntry] | None,
    config_kinds: dict[str, KindEntry] | None,
):
    warnings: list[str] = []

    resolve_metric_meta_from_samples(
        sample_sets, config_metrics, metric_lines_adapter, config_kinds, warn=warnings.append
    )

    assert warnings == []
