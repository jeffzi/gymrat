"""Shared settled-config builders for tests that need a benchless or a run config."""

from dataclasses import replace
from typing import Any

from gymrat.config import BenchlessConfig, ResolvedConfig


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


def resolved_config(**overrides: Any) -> ResolvedConfig:
    """A settled run configuration, geomean-led unless a test names its own primary.

    Args:
        **overrides: ``ResolvedConfig`` fields to set in place of the defaults.

    Returns:
        The config with ``overrides`` applied.
    """
    default = ResolvedConfig(
        bench="npm run bench",
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200.0,
        primary="geomean",
    )
    return replace(default, **overrides)
