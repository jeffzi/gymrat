"""Reading one entry of a harness's JSON payload.

The ``mitata`` and ``tinybench`` extractors validate each entry, word the warning
that skips a malformed one, and store the metrics that survive through these
helpers, so both harnesses phrase a skipped entry the same way.
"""

import json
import math
import re

from pydantic import ValidationError

from gymrat.adapters.json_payload import warn_skip
from gymrat.utils import (
    LINE_TERMINATORS,
    WarnSink,
    describe_key,
    drop_prefix_errors,
    expected_got,
    phrase_for_error,
)


def record_metric(
    metrics: dict[str, float], name: str, value: float, warn: WarnSink, advice: str
) -> None:
    """Store ``value`` under ``name``, warning when it displaces an earlier reading.

    A collision means two readings resolved to one metric name, so the report
    would otherwise silently show only the last one's value.

    Args:
        metrics: The readings collected so far, updated in place.
        name: The resolved metric name.
        value: The reading to store.
        warn: The sink that receives the collision warning.
        advice: How the user separates the colliding readings in their harness,
            closing the warning.
    """
    if name in metrics:
        warn(f"Duplicate metric name: {name} (keeping the last value; {advice})")
    metrics[name] = value


def _escape_line_terminator(match: re.Match[str]) -> str:
    return json.dumps(match.group())[1:-1]


def describe_error(error: object) -> str:
    r"""Render a harness's error value (mitata's ``run.error``, tinybench's ``result.error``).

    ``error`` is read from parsed JSON, so it can be any JSON value, not just a
    string — ``str()`` on a plain dict would print an unhelpful Python repr.
    ``json.dumps`` renders that case usefully instead, and cannot fail since the
    value round-trips from ``json.loads``.

    A string is written as-is so a plain message reads naturally, except that
    each character of ``LINE_TERMINATORS`` becomes its JSON escape (``\n``,
    ``\u000b``, ``\f``, ``\r``, ``\u001c`` to ``\u001e``, ``\u0085``,
    ``\u2028``, ``\u2029``): the warning must stay on one line.
    ``json.dumps`` keeps ``ensure_ascii`` on, so a non-string value is already
    escaped the same way.

    Args:
        error: The error value as parsed from JSON.

    Returns:
        A single-line, human-readable rendering of the error value.
    """
    if isinstance(error, str):
        return LINE_TERMINATORS.sub(_escape_line_terminator, error)
    return json.dumps(error)


def _invalid_value_tail(key: str, phrase: str, value: object) -> str:
    """Phrase a rejected value as the tail of a skip warning.

    Args:
        key: The dotted path of the rejected value; empty for the entry itself.
        phrase: What was expected, such as ``"a number"``.
        value: The rejected value.

    Returns:
        The text that follows the skipped entry's name in the warning.
    """
    detail = expected_got(phrase, value)
    return f" with invalid {key}: {detail}" if key else f": {detail}"


def first_problem(exc: ValidationError, prefix: tuple[str, ...] = ()) -> str:
    """Phrase the first problem pydantic found as the tail of a skip warning.

    Args:
        exc: The validation failure.
        prefix: The path of the validated value inside the entry (a mitata run,
            a tinybench task), prepended to each error location.

    Returns:
        The text that follows the skipped entry's name in the warning.
    """
    error = drop_prefix_errors(exc.errors())[0]
    key = describe_key((*prefix, *error["loc"]))
    if error["type"] == "missing":
        return f" with missing {key}"
    return _invalid_value_tail(key, phrase_for_error(error), error["input"])


def is_finite_or_warn(value: float, warn: WarnSink, subject: str, key: str) -> bool:
    """Tell whether ``value`` is finite, sending a skip warning for ``subject`` when not.

    Args:
        value: The reading to check.
        warn: The sink that receives the warning.
        subject: What is being skipped, as for :func:`warn_skip`.
        key: The dotted path of ``value``, named in the warning.

    Returns:
        ``False`` when ``value`` is not finite, or is an integer too large to
        convert to a float, and the entry is to be skipped.
    """
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if finite:
        return True
    warn_skip(warn, subject, _invalid_value_tail(key, "a finite number", value))
    return False


def named_subject(kind: str, entry: object, key: str) -> str:
    """Name a skipped entry by its ``key`` field when that field is a string.

    Args:
        kind: What the entry is, e.g. ``"benchmark"``.
        entry: The raw entry as parsed from JSON.
        key: The field naming the entry, e.g. ``"alias"``.

    Returns:
        ``kind`` followed by the JSON-escaped name, or ``kind`` alone.
    """
    name = entry.get(key) if isinstance(entry, dict) else None
    return f"{kind} {json.dumps(name)}" if isinstance(name, str) else kind
