"""Validation of a parsed ``gymrat.toml`` and translation of its pydantic errors.

The frozen dataclasses from :mod:`gymrat.config.types` carry the validation
annotations; this module runs them through a pydantic ``TypeAdapter`` and words
each failure as a gymrat problem string.
"""

import json

from pydantic import TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

from gymrat.config.types import ConfigFile
from gymrat.pydantic_errors import (
    UNKNOWN_SHAPE_PHRASE,
    VALUE_ERROR_PREFIX,
    describe_key,
    drop_prefix_errors,
    phrase_for_error,
)

_CONFIG_ADAPTER = TypeAdapter(ConfigFile)


def invalid_value_message(field_name: str, expected_phrase: str, value: object) -> str:
    """Word an invalid-value problem.

    The single shape both the schema translator and the cross-field settlement
    checks report.

    Args:
        field_name: Dotted name of the offending config field.
        expected_phrase: Human-readable description of the expected shape.
        value: The actual value that failed validation.

    Returns:
        A human-readable problem string naming the field, its expected shape,
        and the actual value.
    """
    try:
        got = json.dumps(value)
    except TypeError:
        got = repr(value)
    return f"Invalid config value for {field_name}: expected {expected_phrase}, got {got}"


def _message_for_error(error: ErrorDetails) -> str:
    """Translate one pydantic error into a gymrat-worded problem string.

    A custom validator (the line-break key guard) already knows why it refused
    the value, so its own message is reported: a shape phrase would describe a
    fault the value does not have.

    Args:
        error: The pydantic error detail to translate.

    Returns:
        The problem string describing the validation failure.
    """
    key = describe_key(tuple(str(part) for part in error["loc"]))
    if error["type"] == "unexpected_keyword_argument":
        return f"Unknown config key: {key}"
    if error["type"] == "value_error":
        detail = error["msg"].removeprefix(VALUE_ERROR_PREFIX)
        return f"Invalid config value for {key}: {detail}"
    phrase = phrase_for_error(error) or UNKNOWN_SHAPE_PHRASE
    return invalid_value_message(key, phrase, error["input"])


def validate_config_file(data: dict[str, object]) -> tuple[ConfigFile | None, list[str]]:
    """Validate parsed config data into a :class:`ConfigFile`.

    Never raises: validation failures are returned as a problem list, not
    exceptions, so callers can collect and display all errors at once.

    Args:
        data: Raw config data, shaped like a parsed ``gymrat.toml``.

    Returns:
        A ``(config_file, problems)`` pair: the validated :class:`ConfigFile`
        (``None`` on failure) and any validation problems.
    """
    try:
        config_file = _CONFIG_ADAPTER.validate_python(data)
    except ValidationError as exc:
        return None, [_message_for_error(error) for error in drop_prefix_errors(exc.errors())]
    return config_file, []
