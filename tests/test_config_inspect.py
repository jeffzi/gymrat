from collections.abc import Callable

from gymrat_py.adapters.types import Adapter, MetricDefaults
from gymrat_py.config import KindEntry, MetricEntry, resolve_metric_meta
from gymrat_py.model import Direction, MetricUnit, ResolvedMetricMeta
from gymrat_py.warn import WarnSink, warn_to_stderr


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


def metric_meta(  # noqa: PLR0913 -- one keyword per resolved field, so a test pins only what it cares about
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

    result = resolve_metric_meta(["response-time"], None, adapter)

    assert result == {"response-time": metric_meta("response-time")}


def test_resolve_metric_meta_when_adapter_returns_unit_does_carry_unit():
    adapter = make_adapter(lambda _name: MetricDefaults(direction="lower", unit="ns"))

    result = resolve_metric_meta(["response-time"], None, adapter)

    assert result == {"response-time": metric_meta("response-time", unit="ns")}


def test_resolve_metric_meta_when_adapter_reports_kind_and_short_name_does_carry_both():
    adapter = make_adapter(
        lambda _name: MetricDefaults(direction="lower", kind="memory", short_name="heap")
    )

    result = resolve_metric_meta(["bench-a/heap"], None, adapter)

    assert result == {"bench-a/heap": metric_meta("heap", kind="memory")}


# ---------------------------------------------------------------------------
# resolve_metric_meta — per-metric config overrides
# ---------------------------------------------------------------------------


def test_resolve_metric_meta_when_config_sets_direction_does_override_adapter():
    adapter = make_adapter()
    config_metrics = {"throughput": MetricEntry(direction="higher")}

    result = resolve_metric_meta(["throughput"], config_metrics, adapter)

    assert result == {"throughput": metric_meta("throughput", direction="higher")}


def test_resolve_metric_meta_when_config_sets_gating_does_override_default():
    adapter = make_adapter()
    config_metrics = {"response-time": MetricEntry(gating=False)}

    result = resolve_metric_meta(["response-time"], config_metrics, adapter)

    assert result == {"response-time": metric_meta("response-time", gating=False)}


def test_resolve_metric_meta_when_config_sets_exact_does_override_default():
    adapter = make_adapter()
    config_metrics = {"response-time": MetricEntry(exact=True)}

    result = resolve_metric_meta(["response-time"], config_metrics, adapter)

    assert result == {"response-time": metric_meta("response-time", exact=True)}


def test_resolve_metric_meta_when_config_sets_only_gating_does_keep_direction_and_default_exact():
    adapter = make_adapter(lambda _name: MetricDefaults(direction="lower", unit="bytes"))
    config_metrics = {"memory-usage": MetricEntry(gating=False)}

    result = resolve_metric_meta(["memory-usage"], config_metrics, adapter)

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

    result = resolve_metric_meta(["response-time", "throughput"], config_metrics, adapter)

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

    result = resolve_metric_meta(["response-time"], config_metrics, adapter)

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

    result = resolve_metric_meta(["bench-a/heap", "bench-a/time"], None, adapter, config_kinds)

    assert result == {
        "bench-a/heap": metric_meta("heap", gating=False, kind="memory"),
        "bench-a/time": metric_meta("time", kind="time"),
    }


def test_resolve_metric_meta_when_kind_entry_matches_no_metric_does_ignore_it():
    adapter = make_adapter(
        lambda _name: MetricDefaults(direction="lower", kind="memory", short_name="heap")
    )
    config_kinds = {"io": KindEntry(gating=False)}

    result = resolve_metric_meta(["bench-a/heap"], None, adapter, config_kinds)

    assert result == {"bench-a/heap": metric_meta("heap", kind="memory")}


def test_resolve_metric_meta_when_metric_and_kind_disagree_does_let_metric_win():
    adapter = make_adapter(
        lambda name: MetricDefaults(
            direction="lower", kind="memory", short_name=name.split("/")[-1]
        )
    )
    config_metrics = {"bench-a/heap": MetricEntry(gating=True)}
    config_kinds = {"memory": KindEntry(gating=False)}

    result = resolve_metric_meta(
        ["bench-a/heap", "bench-a/rss"], config_metrics, adapter, config_kinds
    )

    assert result == {
        "bench-a/heap": metric_meta("heap", kind="memory"),
        "bench-a/rss": metric_meta("rss", kind="memory", gating=False),
    }
