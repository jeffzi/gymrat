"""Shared helpers for translating pydantic ``ErrorDetails`` into gymrat-worded problems.

Both the config-file schema (``config.py``) and the session-log schema
(``session/records.py``) validate against pydantic models and need to render the
same two things from a pydantic ``ValidationError``: a dotted location string,
and a list pruned of parent errors whose only fault is that a child under them
also failed.
"""

from pydantic_core import ErrorDetails


def describe_key(loc: tuple[str, ...]) -> str:
    """Join an error location into a dotted key.

    An empty part renders as a quoted empty string so the path never ends in a
    bare dot.
    """
    return ".".join('""' if part == "" else part for part in loc)


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
