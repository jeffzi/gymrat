"""Shared settled-config builder for tests that need a benchless config."""

from dataclasses import replace
from typing import Any

from gymrat.config import BenchlessConfig


def benchless_config(**overrides: Any) -> BenchlessConfig:
    """The fully defaulted metric-lines config, with any field overridable.

    The defaults are what ``inspect_config`` settles on when neither flags nor
    a config file supply a value.

    Args:
        **overrides: ``BenchlessConfig`` fields to set in place of the defaults.

    Returns:
        The config with ``overrides`` applied.
    """
    default = BenchlessConfig(
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200,
        primary="geomean",
    )
    return replace(default, **overrides)
