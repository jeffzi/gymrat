"""Config-file schema and loading for gymrat.

The on-disk ``gymrat.json`` file is validated against pydantic models internally,
but the public surface is plain frozen dataclasses -- no pydantic type ever leaks
to consumers. Two entry points share one read/parse/validate pipeline:

- :func:`load_config_file` raises a :class:`GymratError` on the first problem.
- :func:`load_config_file_collecting` returns every problem alongside an
  ``exists`` flag, for callers that want to report all issues at once.

Config keys are written in camelCase (``timeoutSeconds``, ``unstableNoisePct``,
``stop.maxIterations``); the frozen dataclasses expose them as snake_case
attributes. Validation error paths always name the camelCase key the user wrote.
"""

import json
import os
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError
from pydantic_core import ErrorDetails

from gymrat_py.adapters.defaults import DEFAULT_GATING, DEFAULT_METRIC_KIND
from gymrat_py.adapters.types import Adapter
from gymrat_py.errors import GymratError
from gymrat_py.model import (
    DEFAULT_UNSTABLE_NOISE_PCT,
    NOISE_FLOOR_PCT,
    Direction,
    ResolvedMetricMeta,
)

MAX_TIMEOUT_SECONDS = 2_147_483
"""Largest ``timeoutSeconds`` a 32-bit millisecond timer can represent."""

# config_env reads MAX_TIMEOUT_SECONDS from this module, so its import must follow
# the constant's definition -- importing it at the top would hit a half-initialized
# module and fail. Keep this line below MAX_TIMEOUT_SECONDS.
from gymrat_py.config_env import (  # noqa: E402
    NUMBER_ENV_FIELDS,
    STRING_ENV_FIELDS,
    env_string_result,
)

_NULL_MESSAGE = "value must not be null"

# Characters a `^…$` key pattern would treat as line terminators; embedding one
# in a config key can smuggle the rest of the key past validation, so any key
# containing one is rejected outright.
_LINE_BREAKS = ("\n", "\r", "\u2028", "\u2029")

# Top-level string keys that must be non-empty (reject empty/whitespace-only).
_NON_EMPTY_STRING_FIELDS = frozenset(
    {"bench", "prepare", "adapter", "checks", "runbook", "primary"}
)


# ---------------------------------------------------------------------------
# Public frozen dataclasses
# ---------------------------------------------------------------------------


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
class ConfigFile:
    """Parsed ``gymrat.json`` contents; every key is optional."""

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
CONFIG_FILENAME = "gymrat.json"

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


# ---------------------------------------------------------------------------
# Value coercion / rejection for the internal pydantic models
# ---------------------------------------------------------------------------


def _reject_none(value: object) -> object:
    """Reject an explicitly-provided ``null``.

    An absent key falls back to the ``None`` default without invoking this
    validator; only a present ``null`` reaches here, and we reject it so the
    error path names the field with its real type expectation.
    """
    if value is None:
        raise ValueError(_NULL_MESSAGE)
    return value


def _coerce_integer(value: object) -> object:
    """Reject ``null`` and fold an integral float into ``int``.

    Folding ``5.0`` to ``5`` lets it satisfy strict integer validation; every
    other value is passed through for the model to accept or reject.
    """
    if value is None:
        raise ValueError(_NULL_MESSAGE)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _coerce_number(value: object) -> object:
    """Reject ``null`` and widen a plain ``int`` to ``float``.

    Widening lets an integer satisfy strict float validation; ``bool`` is left
    untouched so it is rejected as a non-number.
    """
    if value is None:
        raise ValueError(_NULL_MESSAGE)
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _reject_bad_dict(value: object) -> object:
    """Reject ``null`` and any mapping whose key embeds a line terminator.

    A non-mapping falls through to the model's own ``dict``-type error.
    """
    if value is None:
        raise ValueError(_NULL_MESSAGE)
    if isinstance(value, dict):
        for key in value:
            if isinstance(key, str) and any(char in key for char in _LINE_BREAKS):
                msg = "config key must not embed a line break"
                raise ValueError(msg)
    return value


# ---------------------------------------------------------------------------
# Internal pydantic models (never exposed)
# ---------------------------------------------------------------------------

_STRICT = ConfigDict(strict=True, extra="forbid")

# A field's key is optional, but a present `null` must be rejected: annotate the
# union so the BeforeValidator wraps it (running before None matches the None
# arm), and give a None default so an absent key skips validation entirely.
_NonEmptyStr = Annotated[
    str | None, BeforeValidator(_reject_none), Field(default=None, min_length=1, pattern=r"\S")
]
_AnyStr = Annotated[str | None, BeforeValidator(_reject_none), Field(default=None)]
_Bool = Annotated[bool | None, BeforeValidator(_reject_none), Field(default=None)]


class _MetricModel(BaseModel):
    model_config = _STRICT

    direction: Annotated[Direction | None, BeforeValidator(_reject_none), Field(default=None)]
    gating: _Bool
    exact: _Bool


class _KindModel(BaseModel):
    model_config = _STRICT

    gating: _Bool


class _StopModel(BaseModel):
    model_config = _STRICT

    target_value: Annotated[
        float | None, BeforeValidator(_coerce_number), Field(default=None, alias="targetValue")
    ]
    max_iterations: Annotated[
        int | None,
        BeforeValidator(_coerce_integer),
        Field(default=None, ge=1, alias="maxIterations"),
    ]


class _HooksModel(BaseModel):
    model_config = _STRICT

    before: _NonEmptyStr
    after: _NonEmptyStr


class _ConfigModel(BaseModel):
    model_config = _STRICT

    bench: _NonEmptyStr
    prepare: _NonEmptyStr
    adapter: _NonEmptyStr
    samples: Annotated[int | None, BeforeValidator(_coerce_integer), Field(default=None, ge=1)]
    timeout_seconds: Annotated[
        int | None,
        BeforeValidator(_coerce_integer),
        Field(default=None, ge=1, le=MAX_TIMEOUT_SECONDS, alias="timeoutSeconds"),
    ]
    unstable_noise_pct: Annotated[
        float | None,
        BeforeValidator(_coerce_number),
        Field(default=None, ge=NOISE_FLOOR_PCT, alias="unstableNoisePct"),
    ]
    metrics: Annotated[
        dict[str, _MetricModel] | None, BeforeValidator(_reject_bad_dict), Field(default=None)
    ]
    kinds: Annotated[
        dict[str, _KindModel] | None, BeforeValidator(_reject_bad_dict), Field(default=None)
    ]
    checks: _NonEmptyStr
    runbook: _NonEmptyStr
    filter: _AnyStr
    primary: _NonEmptyStr
    stop: Annotated[_StopModel | None, BeforeValidator(_reject_none), Field(default=None)]
    hooks: Annotated[_HooksModel | None, BeforeValidator(_reject_none), Field(default=None)]


# ---------------------------------------------------------------------------
# Validation-error translation
# ---------------------------------------------------------------------------

# Phrase describing the value each location shape expects. A dynamic metrics or
# kinds entry key is collapsed to "*" by `_phrase_for_loc` before lookup.
_PHRASES: dict[tuple[str, ...], str] = {
    **{(field,): "a non-empty string" for field in _NON_EMPTY_STRING_FIELDS},
    ("filter",): "a string",
    ("samples",): "a positive integer",
    ("timeoutSeconds",): f"a positive integer no greater than {MAX_TIMEOUT_SECONDS}",
    ("unstableNoisePct",): f"a number at or above the {NOISE_FLOOR_PCT}% noise floor",
    ("metrics",): "an object",
    ("kinds",): "an object",
    ("stop",): "an object",
    ("hooks",): "an object",
    ("stop", "targetValue"): "a number",
    ("stop", "maxIterations"): "a positive integer",
    ("hooks", "before"): "a non-empty string",
    ("hooks", "after"): "a non-empty string",
    ("metrics", "*"): "an object",
    ("kinds", "*"): "an object",
    ("metrics", "*", "direction"): '"lower" or "higher"',
    ("metrics", "*", "gating"): "a boolean",
    ("metrics", "*", "exact"): "a boolean",
    ("kinds", "*", "gating"): "a boolean",
}


def _describe_key(loc: tuple[str, ...]) -> str:
    """Join an error location into a dotted config key.

    An empty part renders as a quoted empty string so the path never ends in a
    bare dot.
    """
    return ".".join('""' if part == "" else part for part in loc)


def _phrase_for_loc(loc: tuple[str, ...]) -> str:
    """Human-facing description of the value a location expects."""
    nested = len(loc) > 1 and loc[0] in {"metrics", "kinds"}
    shape = (loc[0], "*", *loc[2:]) if nested else loc
    return _PHRASES.get(shape, "a valid value")


def _message_for_error(error: ErrorDetails) -> str:
    """Translate one pydantic error into a gymrat-worded problem string."""
    loc = tuple(str(part) for part in error["loc"])
    if error["type"] == "extra_forbidden":
        return f"Unknown config key: {_describe_key(loc)}"
    phrase = _phrase_for_loc(loc)
    got = json.dumps(error["input"])
    return f"Invalid config value for {_describe_key(loc)}: expected {phrase}, got {got}"


def _drop_prefix_errors(errors: list[ErrorDetails]) -> list[ErrorDetails]:
    """Drop any error whose location is a strict prefix of another's.

    When a parent and its child both fail, only the more specific child error is
    worth reporting; the parent prefix is redundant noise.
    """
    locs = [error["loc"] for error in errors]
    kept: list[ErrorDetails] = []
    for index, error in enumerate(errors):
        loc = error["loc"]
        is_prefix = any(
            other != index and len(candidate) > len(loc) and candidate[: len(loc)] == loc
            for other, candidate in enumerate(locs)
        )
        if not is_prefix:
            kept.append(error)
    return kept


# ---------------------------------------------------------------------------
# File I/O and parsing
# ---------------------------------------------------------------------------


def _reject_constant(literal: str) -> float:
    """Turn a non-finite JSON literal (``NaN``/``Infinity``) into a parse error."""
    msg = f"unsupported non-finite literal: {literal}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class _ReadOk:
    """The config file was read; ``text`` is its content with any BOM stripped."""

    text: str


@dataclass(frozen=True, slots=True)
class _ReadAbsent:
    """The config file does not exist."""


@dataclass(frozen=True, slots=True)
class _ReadError:
    """The config file could not be read; ``message`` describes why."""

    message: str


_ReadResult = _ReadOk | _ReadAbsent | _ReadError


def _read_source(path: Path) -> _ReadResult:
    """Read the config file, returning the outcome as data rather than raising.

    Strips a leading UTF-8 BOM. Returns :class:`_ReadAbsent` when the file does
    not exist and :class:`_ReadError` for any other read failure (e.g. the path
    is a directory), leaving each caller free to raise or collect the problem.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _ReadAbsent()
    except OSError as exc:
        reason = exc.strerror or str(exc)
        return _ReadError(f"Cannot read config file at {path}: {reason}")
    return _ReadOk(text.removeprefix("﻿"))


def _to_metric_entry(model: _MetricModel) -> MetricEntry:
    return MetricEntry(direction=model.direction, gating=model.gating, exact=model.exact)


def _to_config_file(model: _ConfigModel) -> ConfigFile:
    metrics = (
        {name: _to_metric_entry(entry) for name, entry in model.metrics.items()}
        if model.metrics is not None
        else None
    )
    kinds = (
        {name: KindEntry(gating=entry.gating) for name, entry in model.kinds.items()}
        if model.kinds is not None
        else None
    )
    stop = (
        StopConfig(target_value=model.stop.target_value, max_iterations=model.stop.max_iterations)
        if model.stop is not None
        else None
    )
    hooks = (
        HooksConfig(before=model.hooks.before, after=model.hooks.after)
        if model.hooks is not None
        else None
    )
    return ConfigFile(
        bench=model.bench,
        prepare=model.prepare,
        adapter=model.adapter,
        samples=model.samples,
        timeout_seconds=model.timeout_seconds,
        unstable_noise_pct=model.unstable_noise_pct,
        metrics=metrics,
        kinds=kinds,
        checks=model.checks,
        runbook=model.runbook,
        filter=model.filter,
        primary=model.primary,
        stop=stop,
        hooks=hooks,
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_config_file_collecting(path: str | Path, *, required: bool) -> ConfigFileResult:
    """Load and validate a config file, collecting every problem.

    Args:
        path: Path to the ``gymrat.json`` file.
        required: When ``True``, an absent file is itself a reported problem.

    Returns:
        A :class:`ConfigFileResult` carrying the parsed config (when valid),
        whether the file existed, and every validation problem found.
    """
    config_path = Path(path)
    source = _read_source(config_path)
    if isinstance(source, _ReadAbsent):
        if required:
            return ConfigFileResult(
                config_file=None,
                exists=False,
                problems=[f"Config file not found at {config_path}"],
            )
        return ConfigFileResult(config_file=ConfigFile(), exists=False, problems=[])

    config_file, problems = _validate_read(source, config_path)
    return ConfigFileResult(config_file=config_file, exists=True, problems=problems)


def _validate_read(
    source: _ReadOk | _ReadError,
    config_path: Path,
) -> tuple[ConfigFile | None, list[str]]:
    """Turn a successful read (or read error) into a config or a problem list.

    Shared by the collecting loader and the throwing loader: it never raises, so
    each caller decides whether a non-empty problem list becomes an exception or
    a returned result.
    """
    if isinstance(source, _ReadError):
        return None, [source.message]

    try:
        data = json.loads(source.text, parse_constant=_reject_constant)
    except ValueError as exc:
        return None, [f"Failed to parse config file at {config_path}: {exc}"]

    if not isinstance(data, dict):
        rendered = json.dumps(data)
        return None, [
            f"Invalid config file at {config_path}: expected a JSON object, got {rendered}"
        ]

    try:
        model = _ConfigModel.model_validate(data)
    except ValidationError as exc:
        return None, [_message_for_error(error) for error in _drop_prefix_errors(exc.errors())]

    return _to_config_file(model), []


def load_config_file(path: str | Path, *, required: bool = False) -> ConfigFile:
    """Load and validate a config file, raising on the first problem.

    Args:
        path: Path to the ``gymrat.json`` file.
        required: When ``True``, an absent file raises rather than returning an
            empty config.

    Returns:
        The parsed :class:`ConfigFile`; an empty one when the file is absent and
        not required.

    Raises:
        GymratError: On any read, parse, or validation problem.
    """
    result = load_config_file_collecting(path, required=required)
    if result.problems:
        raise GymratError(result.problems[0])
    return result.config_file if result.config_file is not None else ConfigFile()


# ---------------------------------------------------------------------------
# Settlement: merge flags, env vars, config file, and defaults
# ---------------------------------------------------------------------------


def _invalid_value_message(field_name: str, expected_phrase: str, value: object) -> str:
    """Word an invalid-value problem the same way the schema translator does."""
    got = json.dumps(value)
    return f"Invalid config value for {field_name}: expected {expected_phrase}, got {got}"


def flag_problem(field_name: str, value: str | None) -> str | None:
    """Return a problem string when a flag holds the empty string, else ``None``.

    Flags bypass the file schema, so ``--bench ""`` is the one way an empty string
    reaches a settled field. The message names the flag, not the config key,
    because the flag is what the user typed.
    """
    if value == "":
        return _invalid_value_message(f"--{field_name}", "a non-empty string", value)
    return None


def _assert_flag_not_empty(field_name: str, value: str | None) -> None:
    problem = flag_problem(field_name, value)
    if problem is not None:
        raise GymratError(problem)


def loop_key_problems(config: BenchlessConfig) -> list[str]:
    """Return the cross-field violations the schema alone cannot express.

    ``filter`` must carry its placeholder, and ``stop.targetValue`` only makes
    sense when ``primary`` names a metric -- the geomean is a ratio, not a value.
    """
    problems: list[str] = []
    if config.filter is not None and FILTER_PLACEHOLDER not in config.filter:
        problems.append(
            _invalid_value_message(
                "filter",
                f"a string containing the {FILTER_PLACEHOLDER} placeholder",
                config.filter,
            )
        )
    if (
        config.stop is not None
        and config.stop.target_value is not None
        and config.primary == GEOMEAN_PRIMARY
    ):
        problems.append(
            "Invalid config value for stop.targetValue: it needs primary to name a metric, "
            f"not {json.dumps(GEOMEAN_PRIMARY)}"
        )
    return problems


def _validate_loop_keys(config: BenchlessConfig) -> None:
    problems = loop_key_problems(config)
    if problems:
        raise GymratError(problems[0])


def runbook_problem(runbook: str, base_dir: str | Path | None) -> str | None:
    """Return a problem string when ``runbook`` does not name an existing file.

    Resolved against ``base_dir`` (or the cwd when ``None``), matching how the
    implicit ``gymrat.json`` lookup is anchored -- a runbook path is authored
    relative to the repository the config lives in.
    """
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    resolved = Path(os.path.normpath(base / runbook))
    try:
        info = resolved.stat()
    except FileNotFoundError:
        info = None
    except OSError as exc:
        reason = exc.strerror or str(exc)
        return f"Cannot read runbook path {runbook}: {reason}"
    if info is None or not stat.S_ISREG(info.st_mode):
        return _invalid_value_message("runbook", "a path to an existing file", runbook)
    return None


def _assert_runbook_exists(runbook: str, base_dir: str | Path | None) -> None:
    problem = runbook_problem(runbook, base_dir)
    if problem is not None:
        raise GymratError(problem)


def _read_env_string(env_var: str) -> str | None:
    """Throwing wrapper around :func:`env_string_result`."""
    result = env_string_result(env_var)
    if result.problem is not None:
        raise GymratError(result.problem)
    return str(result.value) if result.value is not None else None


def _read_env_flags(flags: CliFlags) -> CliFlags:
    """Overlay ``GYMRAT_*`` values onto every flag left unset: flag > env > (later) file.

    An env var is consulted only when its flag is ``None``, so a flag always wins
    without the env var's validation ever firing. ``GYMRAT_CONFIG`` is handled in
    :func:`_settle_config` because it selects the file to load, not a field.
    """
    strings: dict[str, str] = {}
    for field_name, env_var in STRING_ENV_FIELDS:
        if getattr(flags, field_name) is None:
            value = _read_env_string(env_var)
            if value is not None:
                strings[field_name] = value
    numbers: dict[str, int] = {}
    for field_name, env_var, reader in NUMBER_ENV_FIELDS:
        if getattr(flags, field_name) is None:
            result = reader(env_var)
            if result.problem is not None:
                raise GymratError(result.problem)
            if isinstance(result.value, int):
                numbers[field_name] = result.value
    return CliFlags(
        bench=strings.get("bench", flags.bench),
        prepare=strings.get("prepare", flags.prepare),
        adapter=strings.get("adapter", flags.adapter),
        samples=numbers.get("samples", flags.samples),
        timeout=numbers.get("timeout", flags.timeout),
        config=flags.config,
    )


def find_implicit_base() -> str:
    """Return the anchor directory for the implicit ``gymrat.json`` lookup.

    Inside a git repository the config lives at the repo root, so moving the cwd
    into a subdirectory must not lose it. Outside a repository -- or when git is
    unavailable -- the lookup falls back to the process cwd.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return str(Path.cwd())
    return completed.stdout.strip()


def merge_config(flags: CliFlags, config_file: ConfigFile) -> BenchlessConfig:
    """Merge flags, config file, and defaults into every setting but ``bench``.

    ``bench`` is settled separately: benchmarking commands spread it into the
    result, and the rest never ask for it.
    """
    return BenchlessConfig(
        adapter=flags.adapter or config_file.adapter or CONFIG_DEFAULTS.adapter,
        samples=flags.samples or config_file.samples or CONFIG_DEFAULTS.samples,
        timeout_seconds=(
            flags.timeout or config_file.timeout_seconds or CONFIG_DEFAULTS.timeout_seconds
        ),
        unstable_noise_pct=config_file.unstable_noise_pct or CONFIG_DEFAULTS.unstable_noise_pct,
        primary=config_file.primary or CONFIG_DEFAULTS.primary,
        prepare=flags.prepare if flags.prepare is not None else config_file.prepare,
        metrics=dict(config_file.metrics) if config_file.metrics is not None else None,
        kinds=dict(config_file.kinds) if config_file.kinds is not None else None,
        checks=config_file.checks,
        runbook=config_file.runbook,
        filter=config_file.filter,
        stop=config_file.stop,
        hooks=config_file.hooks,
    )


def _settle_config(
    flags: CliFlags, base_dir: str | Path | None
) -> tuple[BenchlessConfig, str | None]:
    _assert_flag_not_empty("bench", flags.bench)
    _assert_flag_not_empty("prepare", flags.prepare)
    _assert_flag_not_empty("adapter", flags.adapter)
    _assert_flag_not_empty("config", flags.config)

    # Env vars fill in for absent flags: flag > env > file > default.
    effective = _read_env_flags(flags)

    # --config > GYMRAT_CONFIG > implicit gymrat.json
    env_config_path = _read_env_string("GYMRAT_CONFIG") if flags.config is None else None
    explicit_config = flags.config if flags.config is not None else env_config_path
    if explicit_config is not None:
        config_path = Path(explicit_config)
    else:
        anchor = Path(base_dir) if base_dir is not None else Path(find_implicit_base())
        config_path = anchor / CONFIG_FILENAME
    config_file = load_config_file(config_path, required=explicit_config is not None)

    config = merge_config(effective, config_file)
    _validate_loop_keys(config)

    if config.runbook is not None:
        config_dir = config_path.parent
        _assert_runbook_exists(config.runbook, config_dir)
        resolved_runbook = os.path.normpath(config_dir / config.runbook)
        config = replace(config, runbook=resolved_runbook)

    bench = effective.bench if effective.bench is not None else config_file.bench
    return config, bench


def resolve_benchless_config(
    flags: CliFlags, base_dir: str | Path | None = None
) -> BenchlessConfig:
    """Settle everything :func:`resolve_config` does except ``bench``.

    Use for a command that runs no benchmark: it settles the same values without
    demanding a bench command none of them would run.
    """
    config, _ = _settle_config(flags, base_dir)
    return config


def resolve_config(flags: CliFlags, base_dir: str | Path | None = None) -> ResolvedConfig:
    """Settle a run configuration from flags, env vars, config file, and defaults.

    ``bench`` has no default and must come from a flag or the config file -- a run
    without it raises.
    """
    config, bench = _settle_config(flags, base_dir)
    if bench is None:
        message = "bench is required. Provide it via --bench flag or in config file."
        raise GymratError(message)
    return ResolvedConfig(
        bench=bench,
        adapter=config.adapter,
        samples=config.samples,
        timeout_seconds=config.timeout_seconds,
        unstable_noise_pct=config.unstable_noise_pct,
        primary=config.primary,
        prepare=config.prepare,
        metrics=config.metrics,
        kinds=config.kinds,
        checks=config.checks,
        runbook=config.runbook,
        filter=config.filter,
        stop=config.stop,
        hooks=config.hooks,
    )


# ---------------------------------------------------------------------------
# Metric metadata resolution
# ---------------------------------------------------------------------------


def _resolve_one_metric(
    name: str,
    entry: MetricEntry | None,
    adapter: Adapter,
    config_kinds: dict[str, KindEntry] | None,
) -> ResolvedMetricMeta:
    defaults = adapter.defaults(name)
    kind = defaults.kind if defaults.kind is not None else DEFAULT_METRIC_KIND
    direction = (
        entry.direction if entry is not None and entry.direction is not None else defaults.direction
    )

    # Precedence: the metric's own entry, then its kind's entry, then the default.
    gating = DEFAULT_GATING
    if entry is not None and entry.gating is not None:
        gating = entry.gating
    elif config_kinds is not None:
        kind_entry = config_kinds.get(kind)
        if kind_entry is not None and kind_entry.gating is not None:
            gating = kind_entry.gating

    exact = entry.exact if entry is not None and entry.exact is not None else False
    short_name = defaults.short_name if defaults.short_name is not None else name

    return ResolvedMetricMeta(
        direction=direction,
        gating=gating,
        exact=exact,
        unit=defaults.unit,
        kind=kind,
        short_name=short_name,
    )


def resolve_metric_meta(
    metric_names: Sequence[str],
    config_metrics: dict[str, MetricEntry] | None,
    adapter: Adapter,
    config_kinds: dict[str, KindEntry] | None = None,
) -> dict[str, ResolvedMetricMeta]:
    """Resolve each metric's display metadata from adapter defaults and config overrides.

    For every name in ``metric_names`` (preserving input order), the adapter's
    per-metric defaults are the base; a matching ``config_metrics`` entry overrides
    direction, gating, and exact, and a ``config_kinds`` entry for the resolved kind
    supplies gating when the metric entry does not. A per-metric gating override wins
    over its kind's gating.
    """
    return {
        name: _resolve_one_metric(
            name,
            config_metrics.get(name) if config_metrics is not None else None,
            adapter,
            config_kinds,
        )
        for name in metric_names
    }
