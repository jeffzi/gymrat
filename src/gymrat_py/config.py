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
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError
from pydantic_core import ErrorDetails

from gymrat_py.errors import GymratError
from gymrat_py.model import NOISE_FLOOR_PCT, Direction

MAX_TIMEOUT_SECONDS = 2_147_483
"""Largest ``timeoutSeconds`` a 32-bit millisecond timer can represent."""

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
