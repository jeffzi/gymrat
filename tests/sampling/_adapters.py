"""Adapter doubles for tests that resolve metric metadata."""

from collections.abc import Callable

from gymrat.adapters import Adapter, MetricDefaults
from gymrat.utils import WarnSink, warn_to_stderr


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
