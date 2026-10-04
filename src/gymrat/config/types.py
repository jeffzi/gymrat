"""Public frozen dataclasses and constants for the gymrat config surface.

The dataclasses a ``gymrat.toml`` settles into carry their own validation
annotations, so ``TypeAdapter(ConfigFile)`` validates a parsed file directly.
Constructing one in code runs no validation.
"""

import json
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pydantic import BeforeValidator, ConfigDict, Field, Strict

from gymrat.config.env import MAX_SAFE_INTEGER, MAX_TIMEOUT_SECONDS
from gymrat.metric_name import LINE_TERMINATORS
from gymrat.model import DEFAULT_UNSTABLE_NOISE_PCT, NOISE_FLOOR_PCT, Direction
from gymrat.pydantic_errors import NON_BLANK_PATTERN, coerce_integer

#: The effort dial the CLI and config file both accept for a supervised session.
Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Unknown keys must fail validation so ``config/resolve.py`` reports an "Unknown config key"
# problem; every nested dataclass sets this, not only ``ConfigFile``.
_FORBID_EXTRA = ConfigDict(extra="forbid")


def _reject_line_break_keys(value: object) -> object:
    """Reject any mapping whose key embeds a line terminator.

    A non-mapping falls through to the field's own ``dict``-type error. The
    offending key is JSON-escaped so naming it cannot itself split the reported
    problem across lines.

    Args:
        value: The raw value being validated.

    Returns:
        The value unchanged when it is not a mapping or all keys are clean.

    Raises:
        ValueError: When a mapping key contains a line terminator.
    """
    if not isinstance(value, dict):
        return value
    for key in value:
        if isinstance(key, str) and LINE_TERMINATORS.search(key):
            msg = f"key {json.dumps(key)} must not embed a line break"
            raise ValueError(msg)
    return value


_NoLineBreakKeys = BeforeValidator(_reject_line_break_keys)
_Str = Annotated[str, Strict()]
_NonEmptyStr = Annotated[str, Strict(), Field(min_length=1, pattern=NON_BLANK_PATTERN)]
_Bool = Annotated[bool, Strict()]
_FiniteFloat = Annotated[float, Strict(), Field(allow_inf_nan=False)]
# TOML writes ``5.0`` for a whole number as readily as ``5``; fold it before the strict check.
_PositiveInt = Annotated[int, BeforeValidator(coerce_integer), Strict(), Field(ge=1)]


@dataclass(frozen=True, slots=True)
class MetricEntry:
    """Per-metric overrides declared under the ``metrics`` section."""

    __pydantic_config__: ClassVar[ConfigDict] = _FORBID_EXTRA

    direction: Direction | None = None
    gating: _Bool | None = None
    exact: _Bool | None = None


@dataclass(frozen=True, slots=True)
class KindEntry:
    """Per-kind overrides declared under the ``kinds`` section."""

    __pydantic_config__: ClassVar[ConfigDict] = _FORBID_EXTRA

    gating: _Bool | None = None


@dataclass(frozen=True, slots=True)
class StopConfig:
    """Loop stopping criteria declared under the ``stop`` section."""

    __pydantic_config__: ClassVar[ConfigDict] = _FORBID_EXTRA

    target_value: _FiniteFloat | None = None
    max_iterations: _PositiveInt | None = None


@dataclass(frozen=True, slots=True)
class HooksConfig:
    """Loop lifecycle commands declared under the ``hooks`` section."""

    __pydantic_config__: ClassVar[ConfigDict] = _FORBID_EXTRA

    before: _NonEmptyStr | None = None
    after: _NonEmptyStr | None = None


@dataclass(frozen=True, slots=True)
class SuperviseConfig:
    """Agent supervision settings declared under the ``supervise`` section."""

    __pydantic_config__: ClassVar[ConfigDict] = _FORBID_EXTRA

    model: _NonEmptyStr | None = None
    effort: Effort | None = None


@dataclass(frozen=True, slots=True)
class ConfigFile:
    """Parsed ``gymrat.toml`` contents; every key is optional."""

    __pydantic_config__: ClassVar[ConfigDict] = _FORBID_EXTRA

    bench: _NonEmptyStr | None = None
    prepare: _NonEmptyStr | None = None
    adapter: _NonEmptyStr | None = None
    samples: Annotated[_PositiveInt, Field(le=MAX_SAFE_INTEGER)] | None = None
    timeout_seconds: Annotated[_PositiveInt, Field(le=MAX_TIMEOUT_SECONDS)] | None = None
    unstable_noise_pct: Annotated[_FiniteFloat, Field(ge=NOISE_FLOOR_PCT)] | None = None
    metrics: Annotated[dict[str, MetricEntry], _NoLineBreakKeys] | None = None
    kinds: Annotated[dict[str, KindEntry], _NoLineBreakKeys] | None = None
    checks: _NonEmptyStr | None = None
    runbook: _NonEmptyStr | None = None
    filter: _Str | None = None
    primary: _NonEmptyStr | None = None
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
