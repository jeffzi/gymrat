"""Shared helpers for translating pydantic ``ErrorDetails`` into gymrat-worded problems.

Both the config-file schema (``config.py``) and the session-log schema
(``session/records.py``) validate with pydantic and need to render the
same things from a pydantic ``ValidationError``: a dotted location string, a
list pruned of parent errors whose only fault is that a child under them also
failed, and the expected-shape phrase a pydantic error's ``type`` and ``ctx``
imply.

It also holds :func:`coerce_integer`, the boundary coercion both schemas apply
before strict integer validation.
"""

import json

from pydantic_core import ErrorDetails

NON_BLANK_PATTERN = r"\S"
"""String ``pattern`` constraint that rejects a whitespace-only value."""

VALUE_ERROR_PREFIX = "Value error, "
"""Prefix pydantic prepends to a model validator's ``ValueError`` message."""

_TYPE_PHRASES: dict[str, str] = {
    "int_type": "an integer",
    "float_type": "a number",
    "finite_number": "a number",
    "string_type": "a string",
    "bool_type": "a boolean",
    "dict_type": "an object",
    "dataclass_type": "an object",
    "model_type": "an object",
    "list_type": "an array",
    "tuple_type": "an array",
}

# Templates filled from the error ``ctx``, which carries the constraint's own value.
_CONSTRAINT_PHRASES: dict[str, str] = {
    "greater_than_equal": "a number at or above {ge}",
    "less_than_equal": "a number at or below {le}",
    "literal_error": "{expected}",
}


def _rejects_blank(error_type: str, ctx: dict[str, object]) -> bool:
    if error_type == "string_too_short":
        return ctx.get("min_length") == 1
    return error_type == "string_pattern_mismatch" and ctx.get("pattern") == NON_BLANK_PATTERN


def coerce_integer(value: object) -> object:
    """Fold an integral float into ``int`` so it satisfies strict integer validation.

    Only the fold happens here; accepting or rejecting the value stays the
    model's job.

    Args:
        value: The value to coerce.

    Returns:
        The coerced ``int`` when *value* is an integral float, otherwise *value*
        unchanged.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _needs_quoting(part: str) -> bool:
    """Whether a location part would not read back as itself unquoted.

    An empty part would vanish into a bare dot; a part carrying a dot, a quote,
    or whitespace would read as a deeper path (or a truncated one) than the
    writer actually named.

    Args:
        part: One segment of an error location.

    Returns:
        Whether the part needs quoting to survive a round-trip.
    """
    return part == "" or any(char in '."' or char.isspace() for char in part)


def describe_key(loc: tuple[str | int, ...]) -> str:
    """Join an error location into a dotted key path.

    Parts that would be misread bare are quoted with :func:`json.dumps`, which
    also escapes the quotes, backslashes, and line terminators a part may carry
    so the rendered path stays on one line.

    Args:
        loc: The error location parts to join; a list index is written as its
            digits.

    Returns:
        The dot-joined key path.
    """
    parts = (str(part) for part in loc)
    return ".".join(json.dumps(part) if _needs_quoting(part) else part for part in parts)


def drop_prefix_errors(errors: list[ErrorDetails]) -> list[ErrorDetails]:
    """Drop any error whose location is a strict prefix of another's.

    When a parent and its child both fail, only the more specific child error is
    worth reporting; the parent prefix is redundant noise.

    Args:
        errors: The pydantic error details to filter.

    Returns:
        The errors with prefix-only entries removed.
    """
    locs = [error["loc"] for error in errors]
    return [
        error
        for error in errors
        if not any(
            len(candidate) > len(error["loc"]) and candidate[: len(error["loc"])] == error["loc"]
            for candidate in locs
        )
    ]


def phrase_for_error(error: ErrorDetails) -> str:
    """Describe the value shape a pydantic error says was expected.

    The phrase is derived from the error ``type`` and its ``ctx`` constraint
    values alone, so every field sharing a constraint shares its wording.

    Args:
        error: The pydantic error detail to describe.

    Returns:
        The phrase that completes ``expected ...``, such as ``"an integer"`` or
        ``"a number at or above 1"``, or ``"a valid value"`` when the error
        type implies no shape (``missing``, or any type not mapped here).
    """
    error_type = error["type"]
    ctx = error.get("ctx", {})
    if _rejects_blank(error_type, ctx):
        return "a non-empty string"
    if error_type in _CONSTRAINT_PHRASES:
        return _CONSTRAINT_PHRASES[error_type].format_map(ctx)
    return _TYPE_PHRASES.get(error_type, "a valid value")
