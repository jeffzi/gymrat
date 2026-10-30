"""Shared settled-config builders for tests that need a benchless or a run config."""

from dataclasses import replace
from typing import Any

from gymrat.config import CONFIG_DEFAULTS, BenchlessConfig, ResolvedConfig


def benchless_config(**overrides: Any) -> BenchlessConfig:
    """The project's built-in config defaults, with any field overridable.

    Starts from ``CONFIG_DEFAULTS``, which ``inspect_config`` settles on when
    neither flags nor a config file supply a value.

    Args:
        **overrides: ``BenchlessConfig`` fields to set in place of the defaults.

    Returns:
        The config with ``overrides`` applied.
    """
    return replace(CONFIG_DEFAULTS, **overrides)


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
