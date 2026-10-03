"""Tests for metric-meta resolution: adapter defaults, then kind, then metric."""

from collections.abc import Callable, Sequence

import pytest

from gymrat.adapters import Adapter, MetricDefaults
from gymrat.config import KindEntry, MetricEntry
from gymrat.model import Direction, MetricUnit, ResolvedMetricMeta
from gymrat.sampling import resolve_metric_meta_from_samples
from gymrat.warn import WarnSink, warn_to_stderr


def resolve(
    names: Sequence[str],
    config_metrics: dict[str, MetricEntry] | None,
    adapter: Adapter,
    config_kinds: dict[str, KindEntry] | None = None,
) -> dict[str, ResolvedMetricMeta]:
    """Resolve the metadata of one round that reported every name in ``names``, in order."""
    samples = [[dict.fromkeys(names, 1.0)]]
    return resolve_metric_meta_from_samples(samples, config_metrics, adapter, config_kinds)


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


def metric_meta(
    short_name: str,
    *,
    direction: Direction = "lower",
    gating: bool = True,
    exact: bool = False,
    unit: MetricUnit | None = None,
    kind: str = "other",
) -> ResolvedMetricMeta:
    """A resolved meta defaulting to a lower-is-better, gating, non-exact "other" metric."""
    return ResolvedMetricMeta(
        direction=direction,
        gating=gating,
        exact=exact,
        unit=unit,
        kind=kind,
        short_name=short_name,
    )


# ---------------------------------------------------------------------------
# resolve_metric_meta — adapter defaults
# ---------------------------------------------------------------------------


def test_resolve_metric_meta_when_config_metrics_none_does_default_gating_exact_kind_short_name():
    adapter = make_adapter()

    result = resolve(["response-time"], None, adapter)

    assert result == {"response-time": metric_meta("response-time")}


def test_resolve_metric_meta_when_adapter_returns_unit_does_carry_unit():
    adapter = make_adapter(lambda _name: MetricDefaults(direction="lower", unit="ns"))

    result = resolve(["response-time"], None, adapter)

    assert result == {"response-time": metric_meta("response-time", unit="ns")}


def test_resolve_metric_meta_when_adapter_reports_kind_and_short_name_does_carry_both():
    adapter = make_adapter(
        lambda _name: MetricDefaults(direction="lower", kind="memory", short_name="heap")
    )

    result = resolve(["bench-a/heap"], None, adapter)

    assert result == {"bench-a/heap": metric_meta("heap", kind="memory")}


# ---------------------------------------------------------------------------
# resolve_metric_meta — per-metric config overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "entry", "expected"),
    [
        pytest.param(
            "throughput",
            MetricEntry(direction="higher"),
            metric_meta("throughput", direction="higher"),
            id="direction",
        ),
        pytest.param(
            "response-time",
            MetricEntry(gating=False),
            metric_meta("response-time", gating=False),
            id="gating",
        ),
        pytest.param(
            "response-time",
            MetricEntry(exact=True),
            metric_meta("response-time", exact=True),
            id="exact",
        ),
    ],
)
def test_resolve_metric_meta_when_config_sets_single_field_does_override(
    metric_name: str, entry: MetricEntry, expected: ResolvedMetricMeta
):
    adapter = make_adapter()
    config_metrics = {metric_name: entry}

    result = resolve([metric_name], config_metrics, adapter)

    assert result == {metric_name: expected}


def test_resolve_metric_meta_when_config_sets_only_gating_does_keep_direction_and_default_exact():
    adapter = make_adapter(lambda _name: MetricDefaults(direction="lower", unit="bytes"))
    config_metrics = {"memory-usage": MetricEntry(gating=False)}

    result = resolve(["memory-usage"], config_metrics, adapter)

    assert result == {"memory-usage": metric_meta("memory-usage", unit="bytes", gating=False)}


# ---------------------------------------------------------------------------
# resolve_metric_meta — multiple metrics and unused entries
# ---------------------------------------------------------------------------


def test_resolve_metric_meta_when_multiple_names_does_resolve_each_in_order():
    def defaults_fn(name: str) -> MetricDefaults:
        if name == "response-time":
            return MetricDefaults(direction="lower", unit="ns")
        if name == "throughput":
            return MetricDefaults(direction="higher")
        return MetricDefaults(direction="lower")

    adapter = make_adapter(defaults_fn)
    config_metrics = {
        "response-time": MetricEntry(gating=False),
        "throughput": MetricEntry(exact=True),
    }

    result = resolve(["response-time", "throughput"], config_metrics, adapter)

    assert list(result) == ["response-time", "throughput"]
    assert result == {
        "response-time": metric_meta("response-time", unit="ns", gating=False),
        "throughput": metric_meta("throughput", direction="higher", exact=True),
    }


def test_resolve_metric_meta_when_config_has_unused_entries_does_ignore_them():
    adapter = make_adapter()
    config_metrics = {
        "response-time": MetricEntry(gating=False),
        "unused": MetricEntry(gating=True, exact=True),
    }

    result = resolve(["response-time"], config_metrics, adapter)

    assert result == {"response-time": metric_meta("response-time", gating=False)}


# ---------------------------------------------------------------------------
# resolve_metric_meta — kind-level gating
# ---------------------------------------------------------------------------


def test_resolve_metric_meta_when_kind_sets_gating_does_apply_only_to_matching_kind():
    def defaults_fn(name: str) -> MetricDefaults:
        if name.endswith("/heap"):
            return MetricDefaults(direction="lower", kind="memory", short_name="heap")
        return MetricDefaults(direction="lower", kind="time", short_name="time")

    adapter = make_adapter(defaults_fn)
    config_kinds = {"memory": KindEntry(gating=False)}

    result = resolve(["bench-a/heap", "bench-a/time"], None, adapter, config_kinds)

    assert result == {
        "bench-a/heap": metric_meta("heap", gating=False, kind="memory"),
        "bench-a/time": metric_meta("time", kind="time"),
    }


def test_resolve_metric_meta_when_kind_entry_matches_no_metric_does_ignore_it():
    adapter = make_adapter(
        lambda _name: MetricDefaults(direction="lower", kind="memory", short_name="heap")
    )
    config_kinds = {"io": KindEntry(gating=False)}

    result = resolve(["bench-a/heap"], None, adapter, config_kinds)

    assert result == {"bench-a/heap": metric_meta("heap", kind="memory")}


def test_resolve_metric_meta_when_metric_and_kind_disagree_does_let_metric_win():
    adapter = make_adapter(
        lambda name: MetricDefaults(
            direction="lower", kind="memory", short_name=name.split("/")[-1]
        )
    )
    config_metrics = {"bench-a/heap": MetricEntry(gating=True)}
    config_kinds = {"memory": KindEntry(gating=False)}

    result = resolve(["bench-a/heap", "bench-a/rss"], config_metrics, adapter, config_kinds)

    assert result == {
        "bench-a/heap": metric_meta("heap", kind="memory"),
        "bench-a/rss": metric_meta("rss", kind="memory", gating=False),
    }


@pytest.mark.parametrize(
    ("adapter_kind", "config_metrics", "config_kinds", "expected"),
    [
        pytest.param(
            "memory",
            {"bench-a/heap": MetricEntry(exact=True)},
            {"memory": KindEntry(gating=False)},
            metric_meta("heap", kind="memory", gating=False, exact=True),
            id="metric-entry-without-gating",
        ),
        pytest.param(
            "memory",
            None,
            {"memory": KindEntry(gating=None)},
            metric_meta("heap", kind="memory"),
            id="kind-entry-without-gating",
        ),
        pytest.param(
            None,
            None,
            {"other": KindEntry(gating=False)},
            metric_meta("heap", kind="other", gating=False),
            id="adapter-reports-no-kind",
        ),
    ],
)
def test_resolve_metric_meta_when_metric_leaves_gating_unset_does_take_it_from_kind_or_default(
    adapter_kind: str | None,
    config_metrics: dict[str, MetricEntry] | None,
    config_kinds: dict[str, KindEntry],
    expected: ResolvedMetricMeta,
):
    adapter = make_adapter(
        lambda _name: MetricDefaults(direction="lower", kind=adapter_kind, short_name="heap")
    )

    result = resolve(["bench-a/heap"], config_metrics, adapter, config_kinds)

    assert result == {"bench-a/heap": expected}
