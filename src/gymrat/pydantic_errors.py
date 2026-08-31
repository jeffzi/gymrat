"""Shared helpers for translating pydantic ``ErrorDetails`` into gymrat-worded problems.

Both the config-file schema (``config.py``) and the session-log schema
(``session/records.py``) validate against pydantic models and need to render the
same two things from a pydantic ``ValidationError``: a dotted location string,
and a list pruned of parent errors whose only fault is that a child under them
also failed.
"""

import json

from pydantic import ConfigDict
from pydantic_core import ErrorDetails

STRICT_FORBID = ConfigDict(strict=True, extra="forbid")
"""Shared ``model_config`` for all internal pydantic models."""


def coerce_integer(value: object) -> object:
    """Fold an integral float into ``int`` so it satisfies strict integer validation.

    Folding ``5.0`` to ``5`` lets it pass; every other value is passed through for
    the model to accept or reject.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _needs_quoting(part: str) -> bool:
    """Whether a location part would not read back as itself unquoted.

    An empty part would vanish into a bare dot; a part carrying a dot, a quote,
    or whitespace would read as a deeper path (or a truncated one) than the
    writer actually named.
    """
    return part == "" or any(char in '."' or char.isspace() for char in part)


def describe_key(loc: tuple[str, ...]) -> str:
    """Join an error location into a dotted key path.

    Parts that would be misread bare are quoted with :func:`json.dumps`, which
    also escapes the quotes, backslashes, and line terminators a part may carry
    so the rendered path stays on one line.
    """
    return ".".join(json.dumps(part) if _needs_quoting(part) else part for part in loc)


def drop_prefix_errors(errors: list[ErrorDetails]) -> list[ErrorDetails]:
    """Drop any error whose location is a strict prefix of another's.

    When a parent and its child both fail, only the more specific child error is
    worth reporting; the parent prefix is redundant noise.
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
