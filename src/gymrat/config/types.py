"""Public frozen dataclasses and constants for the gymrat config surface."""

from dataclasses import dataclass
from typing import Literal

from gymrat.model import DEFAULT_UNSTABLE_NOISE_PCT, Direction

#: The effort dial the CLI and config file both accept for a supervised session.
Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True, slots=True)
class MetricEntry:
    """Per-metric overrides declared under the ``metrics`` section."""

    direction: Direction | None = None
    gating: bool | None = None
    exact: bool | None = None


@dataclass(frozen=True, slots=True)
class KindEntry:
    """Per-kind overrides declared under the ``kinds`` section."""

    gating: bool | None = None


@dataclass(frozen=True, slots=True)
class StopConfig:
    """Loop stopping criteria declared under the ``stop`` section."""

    target_value: float | None = None
    max_iterations: int | None = None


@dataclass(frozen=True, slots=True)
class HooksConfig:
    """Loop lifecycle commands declared under the ``hooks`` section."""

    before: str | None = None
    after: str | None = None


@dataclass(frozen=True, slots=True)
class SuperviseConfig:
    """Agent supervision settings declared under the ``supervise`` section."""

    model: str | None = None
    effort: Effort | None = None


@dataclass(frozen=True, slots=True)
class ConfigFile:
    """Parsed ``gymrat.toml`` contents; every key is optional."""

    bench: str | None = None
    prepare: str | None = None
    adapter: str | None = None
    samples: int | None = None
    timeout_seconds: int | None = None
    unstable_noise_pct: float | None = None
    metrics: dict[str, MetricEntry] | None = None
    kinds: dict[str, KindEntry] | None = None
    checks: str | None = None
    runbook: str | None = None
    filter: str | None = None
    primary: str | None = None
    stop: StopConfig | None = None
    hooks: HooksConfig | None = None
    supervise: SuperviseConfig | None = None


@dataclass(frozen=True, slots=True)
class ConfigFileResult:
    """Outcome of a collecting load.

    Carries the parsed config (when valid), whether the file existed, and every
    validation problem found.
    """

    config_file: ConfigFile | None
    exists: bool
    problems: list[str]


@dataclass(frozen=True, slots=True)
class CliFlags:
    """Command-line overrides, named after the flags rather than the config keys."""

    bench: str | None = None
    prepare: str | None = None
    adapter: str | None = None
    samples: int | None = None
    timeout: int | None = None
    config: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchlessConfig:
    """A settled configuration for a command that runs no benchmark.

    Every value a non-benchmarking command (``status``, ``keep``) reads is present
    with defaults already applied; ``bench`` is absent because such commands never
    run one. Keyword-only so the required fields can precede the optional ones.
    """

    adapter: str
    samples: int
    timeout_seconds: int
    unstable_noise_pct: float
    primary: str
    prepare: str | None = None
    metrics: dict[str, MetricEntry] | None = None
    kinds: dict[str, KindEntry] | None = None
    checks: str | None = None
    runbook: str | None = None
    filter: str | None = None
    stop: StopConfig | None = None
    hooks: HooksConfig | None = None
    supervise: SuperviseConfig | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedConfig(BenchlessConfig):
    """A settled run configuration: every value a run needs, including ``bench``."""

    bench: str


@dataclass(frozen=True, slots=True)
class _ConfigDefaults:
    adapter: str
    samples: int
    timeout_seconds: int
    unstable_noise_pct: float
    primary: str


#: The config file basename the CLI writes, loads, and probes for.
CONFIG_FILENAME = "gymrat.toml"

#: The primary that aggregates every gating metric rather than naming one.
GEOMEAN_PRIMARY = "geomean"

#: The token a ``filter`` command must carry, where the loop substitutes benchmark names.
FILTER_PLACEHOLDER = "{names}"

#: Built-in fallbacks for the fields no flag, env var, or config file sets.
CONFIG_DEFAULTS = _ConfigDefaults(
    adapter="metric-lines",
    samples=10,
    timeout_seconds=1800,
    unstable_noise_pct=DEFAULT_UNSTABLE_NOISE_PCT,
    primary=GEOMEAN_PRIMARY,
)
