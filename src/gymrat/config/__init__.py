"""Config-file schema and loading for gymrat.

The on-disk ``gymrat.toml`` file is validated by pydantic against the frozen
dataclasses themselves. Consumers receive plain frozen dataclasses and values;
the pydantic metadata lives only in their class definitions, in field
annotations and a ``__pydantic_config__`` class attribute. Two entry points
share one read/parse/validate pipeline:

- :func:`load_config_file` raises a :class:`GymratError` on the first problem.
- :func:`load_config_file_collecting` returns every problem alongside an
  ``exists`` flag, for callers that want to report all issues at once.

Config keys are snake_case (``timeout_seconds``, ``unstable_noise_pct``,
``stop.max_iterations``), matching the frozen dataclass attributes. Validation
error paths always name the snake_case key the user wrote.
"""

from gymrat.config.env import MAX_SAFE_INTEGER, MAX_TIMEOUT_SECONDS
from gymrat.config.load import load_config_file, load_config_file_collecting
from gymrat.config.meta import resolve_metric_meta
from gymrat.config.resolve import (
    ConfigInspection,
    find_implicit_base,
    flag_problem,
    inspect_config,
    loop_key_problems,
    merge_config,
    resolve_benchless_config,
    resolve_config,
    runbook_problem,
    validate_config_dict,
)
from gymrat.config.types import (
    CONFIG_DEFAULTS,
    CONFIG_FILENAME,
    FILTER_PLACEHOLDER,
    GEOMEAN_PRIMARY,
    BenchlessConfig,
    CliFlags,
    ConfigFile,
    ConfigFileResult,
    Effort,
    HooksConfig,
    KindEntry,
    MetricEntry,
    ResolvedConfig,
    StopConfig,
    SuperviseConfig,
)

__all__ = [
    "CONFIG_DEFAULTS",
    "CONFIG_FILENAME",
    "FILTER_PLACEHOLDER",
    "GEOMEAN_PRIMARY",
    "MAX_SAFE_INTEGER",
    "MAX_TIMEOUT_SECONDS",
    "BenchlessConfig",
    "CliFlags",
    "ConfigFile",
    "ConfigFileResult",
    "ConfigInspection",
    "Effort",
    "HooksConfig",
    "KindEntry",
    "MetricEntry",
    "ResolvedConfig",
    "StopConfig",
    "SuperviseConfig",
    "find_implicit_base",
    "flag_problem",
    "inspect_config",
    "load_config_file",
    "load_config_file_collecting",
    "loop_key_problems",
    "merge_config",
    "resolve_benchless_config",
    "resolve_config",
    "resolve_metric_meta",
    "runbook_problem",
    "validate_config_dict",
]
