"""Internal pydantic models and validation-error translation for gymrat config.

These models are never exposed to consumers — they validate the on-disk TOML and
produce the frozen dataclasses from :mod:`gymrat.config.types`.
"""

import json
import math
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    ValidationError,
    model_validator,
)
from pydantic_core import ErrorDetails

from gymrat.config.env import MAX_SAFE_INTEGER, MAX_TIMEOUT_SECONDS
from gymrat.config.types import (
    ConfigFile,
    HooksConfig,
    KindEntry,
    MetricEntry,
    StopConfig,
)
from gymrat.model import NOISE_FLOOR_PCT, Direction
from gymrat.pydantic_errors import STRICT_FORBID, coerce_integer, describe_key, drop_prefix_errors

_LINE_BREAKS = ("\n", "\r", "\u2028", "\u2029")

_NON_EMPTY_STRING_FIELDS = frozenset(
    {"bench", "prepare", "adapter", "checks", "runbook", "primary"}
)


# ---------------------------------------------------------------------------
# Value coercion / rejection for the internal pydantic models
# ---------------------------------------------------------------------------


def _coerce_number(value: object) -> object:
    """Widen a plain ``int`` to ``float`` so it satisfies strict float validation.

    ``bool`` is left untouched so it is rejected as a non-number.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _reject_non_finite(value: float | None) -> float | None:
    """Reject a non-finite float that TOML accepts as a literal.

    TOML parses ``nan``/``inf``/``-inf`` into real floats, so a numeric key can
    carry one past parsing. This surfaces it as an invalid value naming the key
    rather than letting it settle as a silent NaN or infinity.
    """
    if value is not None and not math.isfinite(value):
        msg = "value must be finite"
        raise ValueError(msg)
    return value


def _reject_bad_dict(value: object) -> object:
    """Reject any mapping whose key embeds a line terminator.

    A non-mapping falls through to the model's own ``dict``-type error. The
    offending key is JSON-escaped so naming it cannot itself split the reported
    problem across lines.
    """
    if isinstance(value, dict):
        for key in value:
            if isinstance(key, str) and any(char in key for char in _LINE_BREAKS):
                msg = f"key {json.dumps(key)} must not embed a line break"
                raise ValueError(msg)
    return value


# ---------------------------------------------------------------------------
# Internal pydantic models (never exposed)
# ---------------------------------------------------------------------------

_NonEmptyStr = Annotated[str | None, Field(default=None, min_length=1, pattern=r"\S")]
_AnyStr = Annotated[str | None, Field(default=None)]
_Bool = Annotated[bool | None, Field(default=None)]


class _MetricModel(BaseModel):
    model_config = STRICT_FORBID

    direction: Annotated[Direction | None, Field(default=None)]
    gating: _Bool
    exact: _Bool


class _KindModel(BaseModel):
    model_config = STRICT_FORBID

    gating: _Bool


class _StopModel(BaseModel):
    model_config = STRICT_FORBID

    target_value: Annotated[
        float | None,
        BeforeValidator(_coerce_number),
        AfterValidator(_reject_non_finite),
        Field(default=None),
    ]
    max_iterations: Annotated[
        int | None,
        BeforeValidator(coerce_integer),
        Field(default=None, ge=1),
    ]


class _HooksModel(BaseModel):
    model_config = STRICT_FORBID

    before: _NonEmptyStr
    after: _NonEmptyStr


class _ConfigModel(BaseModel):
    model_config = STRICT_FORBID

    @model_validator(mode="before")
    @classmethod
    def _reject_line_break_keys(cls, value: object) -> object:
        """Apply the line-break key guard to the document root, not just its tables."""
        return _reject_bad_dict(value)

    bench: _NonEmptyStr
    prepare: _NonEmptyStr
    adapter: _NonEmptyStr
    samples: Annotated[
        int | None,
        BeforeValidator(coerce_integer),
        Field(default=None, ge=1, le=MAX_SAFE_INTEGER),
    ]
    timeout_seconds: Annotated[
        int | None,
        BeforeValidator(coerce_integer),
        Field(default=None, ge=1, le=MAX_TIMEOUT_SECONDS),
    ]
    unstable_noise_pct: Annotated[
        float | None,
        BeforeValidator(_coerce_number),
        AfterValidator(_reject_non_finite),
        Field(default=None, ge=NOISE_FLOOR_PCT),
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
    stop: Annotated[_StopModel | None, Field(default=None)]
    hooks: Annotated[_HooksModel | None, Field(default=None)]


# ---------------------------------------------------------------------------
# Validation-error translation
# ---------------------------------------------------------------------------

_PHRASES: dict[tuple[str, ...], str] = {
    **{(field,): "a non-empty string" for field in _NON_EMPTY_STRING_FIELDS},
    ("filter",): "a string",
    ("samples",): "a positive integer",
    ("timeout_seconds",): "a positive integer",
    ("unstable_noise_pct",): f"a number at or above the {NOISE_FLOOR_PCT}% noise floor",
    ("metrics",): "an object",
    ("kinds",): "an object",
    ("stop",): "an object",
    ("hooks",): "an object",
    ("stop", "target_value"): "a number",
    ("stop", "max_iterations"): "a positive integer",
    ("hooks", "before"): "a non-empty string",
    ("hooks", "after"): "a non-empty string",
    ("metrics", "*"): "an object",
    ("kinds", "*"): "an object",
    ("metrics", "*", "direction"): '"lower" or "higher"',
    ("metrics", "*", "gating"): "a boolean",
    ("metrics", "*", "exact"): "a boolean",
    ("kinds", "*", "gating"): "a boolean",
}

_PYDANTIC_VALUE_ERROR_PREFIX = "Value error, "


def _phrase_for_loc(loc: tuple[str, ...]) -> str:
    """Human-facing description of the value a location expects."""
    nested = len(loc) > 1 and loc[0] in {"metrics", "kinds"}
    shape = (loc[0], "*", *loc[2:]) if nested else loc
    return _PHRASES.get(shape, "a valid value")


def invalid_value_message(field_name: str, expected_phrase: str, value: object) -> str:
    """Word an invalid-value problem.

    The single shape both the schema translator and the cross-field settlement
    checks report.
    """
    try:
        got = json.dumps(value)
    except TypeError:
        got = repr(value)
    return f"Invalid config value for {field_name}: expected {expected_phrase}, got {got}"


def _message_for_error(error: ErrorDetails) -> str:
    """Translate one pydantic error into a gymrat-worded problem string.

    A custom validator (``_reject_non_finite``, ``_reject_bad_dict``) already
    knows why it refused the value, so its own message is reported: the shape
    phrase for the location would describe a fault the value does not have.
    """
    loc = tuple(str(part) for part in error["loc"])
    if error["type"] == "extra_forbidden":
        return f"Unknown config key: {describe_key(loc)}"
    if error["type"] == "value_error":
        detail = error["msg"].removeprefix(_PYDANTIC_VALUE_ERROR_PREFIX)
        if not loc:
            return f"Invalid config file: {detail}"
        return f"Invalid config value for {describe_key(loc)}: {detail}"
    return invalid_value_message(describe_key(loc), _phrase_for_loc(loc), error["input"])


# ---------------------------------------------------------------------------
# Model-to-dataclass conversion
# ---------------------------------------------------------------------------


def _to_metric_entry(model: _MetricModel) -> MetricEntry:
    return MetricEntry(direction=model.direction, gating=model.gating, exact=model.exact)


def _to_config_file(model: _ConfigModel) -> ConfigFile:
    """Convert a validated pydantic model to its frozen dataclass equivalent."""
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


def validate_and_convert(data: dict[str, object]) -> tuple[ConfigFile | None, list[str]]:
    """Validate a parsed TOML dict and convert to a ConfigFile, or return problems."""
    try:
        model = _ConfigModel.model_validate(data)
    except ValidationError as exc:
        return None, [_message_for_error(error) for error in drop_prefix_errors(exc.errors())]
    return _to_config_file(model), []
