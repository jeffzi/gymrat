"""Shared helpers for translating pydantic ``ErrorDetails`` into gymrat-worded problems.

Both the config-file schema (``config.py``) and the session-log schema
(``session/records.py``) validate against pydantic models and need to render the
same things from a pydantic ``ValidationError``: a dotted location string, a
list pruned of parent errors whose only fault is that a child under them also
failed, and the ``"a", "b" or "c"`` phrase naming a ``Literal``'s accepted
values.
"""

import json
from typing import get_args

from pydantic import ConfigDict
from pydantic_core import ErrorDetails

STRICT_FORBID = ConfigDict(strict=True, extra="forbid")
"""Shared ``model_config`` for all internal pydantic models."""


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


def describe_key(loc: tuple[str, ...]) -> str:
    """Join an error location into a dotted key path.

    Parts that would be misread bare are quoted with :func:`json.dumps`, which
    also escapes the quotes, backslashes, and line terminators a part may carry
    so the rendered path stays on one line.

    Args:
        loc: The error location parts to join.

    Returns:
        The dot-joined key path.
    """
    return ".".join(json.dumps(part) if _needs_quoting(part) else part for part in loc)


def alternatives(literal: object) -> str:
    """Render a ``Literal``'s values as ``"a", "b" or "c"`` for a problem message.

    Each value is JSON-encoded, so strings are quoted and integers stay bare. A
    single-value ``Literal`` renders as that value alone.

    Args:
        literal: The ``Literal`` type whose values to list.

    Returns:
        The values joined by commas, with ``or`` before the last.
    """
    *head, last = (json.dumps(value) for value in get_args(literal))
    return f"{', '.join(head)} or {last}" if head else last


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
