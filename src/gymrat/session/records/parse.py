"""Inbound codec: wire value to typed session-log model.

Includes validation-error translation from pydantic errors to problem strings.
"""

import json
from typing import Annotated

from pydantic import Field, TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

from gymrat.errors import GymratError
from gymrat.pydantic_errors import describe_key, drop_prefix_errors
from gymrat.session.records.models import (
    SessionLogRecord,
    _wire_validation,
)
from gymrat.session.schema import SCHEMA_VERSION

_SessionLogUnion = TypeAdapter(Annotated[SessionLogRecord, Field(discriminator="type")])


def parse_record(value: object) -> SessionLogRecord:
    """Validate one decoded session-log line against the schema for its ``type``.

    Args:
        value: A decoded-JSON value -- expected to be a mapping carrying a
            ``type`` discriminator.

    Returns:
        The typed model for the matching record type.

    Raises:
        GymratError: When ``value`` is not an object, carries no recognized
            ``type``, or violates that type's schema.
    """
    if not isinstance(value, dict):
        message = f"Invalid session record: expected a JSON object, got {json.dumps(value)}"
        raise GymratError(message)
    token = _wire_validation.set(True)
    try:
        return _SessionLogUnion.validate_python(value)
    except ValidationError as exc:
        errors = exc.errors()
        _raise_discriminator_error(errors, value)
        error = drop_prefix_errors(errors)[0]
        record_type = value.get("type", "")
        raise GymratError(message_for_error(error, str(record_type))) from exc
    finally:
        _wire_validation.reset(token)


_KNOWN_TYPES = ("session", "baseline", "iteration", "keep", "discard", "hook", "finalize")


def _raise_discriminator_error(errors: list[ErrorDetails], value: dict[str, object]) -> None:
    """Raise with the canonical "Unknown session record type" message.

    Callers match on this exact wording and the ``_KNOWN_TYPES`` hint.
    """
    if not errors:
        return
    error_type = errors[0]["type"]
    if error_type not in ("union_tag_not_found", "union_tag_invalid"):
        return
    type_value = value.get("type")
    tag_missing = error_type == "union_tag_not_found" and type_value is None and "type" not in value
    rendered = "undefined" if tag_missing else json.dumps(type_value)
    hint = "Expected one of: " + ", ".join(_KNOWN_TYPES) + "."
    message = f"Unknown session record type: {rendered}"
    raise GymratError(message, hint=hint)


_STRING = "a string"
_NUMBER = "a number"
_BOOL = "a boolean"
_OBJECT = "an object"
_POSITIVE_INT = "a positive integer"
_NON_NEGATIVE_INT = "a non-negative integer"
_INT = "an integer"
_SAMPLE_ROUNDS = "an array of objects mapping metric names to numbers"
_STRING_ARRAY = "an array of strings"
_DELTA = "a number or null"
_VERDICT = '"improved", "regressed", "no-signal" or "unstable"'
_METHOD = '"permutation", "band" or "exact"'
_KIND = '"geomean" or "metric"'
_OUTCOME = '"improved", "regressed" or "no-signal"'
_STATUS = '"committed" or "blocked"'
_REASON = '"checks-failed", "gating-regression", "nothing-measured" or "nothing-to-commit"'
_STAGE = '"before" or "after"'

_PHRASES: dict[tuple[str, ...], str] = {
    # session
    ("session", "schemaVersion"): str(SCHEMA_VERSION),
    ("session", "sessionId"): _STRING,
    ("session", "createdAt"): _STRING,
    ("session", "baseline"): _OBJECT,
    ("session", "baseline", "ref"): _STRING,
    ("session", "baseline", "sha"): _STRING,
    ("session", "branch"): _STRING,
    ("session", "worktrees"): _OBJECT,
    ("session", "worktrees", "experiment"): _STRING,
    ("session", "worktrees", "baseline"): _STRING,
    ("session", "config"): _OBJECT,
    ("session", "config", "bench"): _STRING,
    ("session", "config", "prepare"): _STRING,
    ("session", "config", "adapter"): _STRING,
    ("session", "config", "samples"): _POSITIVE_INT,
    ("session", "config", "timeoutSeconds"): _POSITIVE_INT,
    ("session", "config", "primary"): _STRING,
    ("session", "config", "filter"): _STRING,
    ("session", "config", "hooks"): _OBJECT,
    ("session", "config", "hooks", "before"): "a non-empty string",
    ("session", "config", "hooks", "after"): "a non-empty string",
    # baseline
    ("baseline", "at"): _STRING,
    ("baseline", "label"): _STRING,
    ("baseline", "samples"): _SAMPLE_ROUNDS,
    ("baseline", "samples", "*"): _OBJECT,
    ("baseline", "samples", "*", "*"): _NUMBER,
    # iteration
    ("iteration", "seq"): _POSITIVE_INT,
    ("iteration", "at"): _STRING,
    ("iteration", "samples"): _OBJECT,
    ("iteration", "samples", "experiment"): _SAMPLE_ROUNDS,
    ("iteration", "samples", "experiment", "*"): _OBJECT,
    ("iteration", "samples", "experiment", "*", "*"): _NUMBER,
    ("iteration", "samples", "baseline"): _SAMPLE_ROUNDS,
    ("iteration", "samples", "baseline", "*"): _OBJECT,
    ("iteration", "samples", "baseline", "*", "*"): _NUMBER,
    ("iteration", "metrics"): _OBJECT,
    ("iteration", "metrics", "*"): _OBJECT,
    ("iteration", "metrics", "*", "deltaPct"): _DELTA,
    ("iteration", "metrics", "*", "verdict"): _VERDICT,
    ("iteration", "metrics", "*", "method"): _METHOD,
    ("iteration", "metrics", "*", "p"): _NUMBER,
    ("iteration", "metrics", "*", "noisePct"): _NUMBER,
    ("iteration", "metrics", "*", "gating"): _BOOL,
    ("iteration", "metrics", "*", "confirmed"): _BOOL,
    ("iteration", "confirm"): _OBJECT,
    ("iteration", "confirm", "ran"): _BOOL,
    ("iteration", "confirm", "filtered"): _STRING_ARRAY,
    ("iteration", "confirm", "filtered", "*"): _STRING,
    ("iteration", "confirm", "absent"): _STRING_ARRAY,
    ("iteration", "confirm", "absent", "*"): _STRING,
    ("iteration", "confirm", "samples"): _OBJECT,
    ("iteration", "confirm", "samples", "experiment"): _SAMPLE_ROUNDS,
    ("iteration", "confirm", "samples", "experiment", "*"): _OBJECT,
    ("iteration", "confirm", "samples", "experiment", "*", "*"): _NUMBER,
    ("iteration", "confirm", "samples", "baseline"): _SAMPLE_ROUNDS,
    ("iteration", "confirm", "samples", "baseline", "*"): _OBJECT,
    ("iteration", "confirm", "samples", "baseline", "*", "*"): _NUMBER,
    ("iteration", "primary"): _OBJECT,
    ("iteration", "primary", "kind"): _KIND,
    ("iteration", "primary", "name"): _STRING,
    ("iteration", "primary", "deltaPct"): _DELTA,
    ("iteration", "outcome"): _OUTCOME,
    ("iteration", "targetReached"): _BOOL,
    # keep
    ("keep", "seq"): _NON_NEGATIVE_INT,
    ("keep", "at"): _STRING,
    ("keep", "status"): _STATUS,
    ("keep", "commit"): _STRING,
    ("keep", "message"): _STRING,
    ("keep", "reason"): _REASON,
    ("keep", "checks"): _OBJECT,
    ("keep", "checks", "configured"): _BOOL,
    ("keep", "checks", "passed"): _BOOL,
    ("keep", "checks", "stdoutBytes"): _NON_NEGATIVE_INT,
    ("keep", "checks", "stderrBytes"): _NON_NEGATIVE_INT,
    # discard
    ("discard", "seq"): _NON_NEGATIVE_INT,
    ("discard", "at"): _STRING,
    # hook
    ("hook", "stage"): _STAGE,
    ("hook", "seq"): _NON_NEGATIVE_INT,
    ("hook", "exitCode"): _INT,
    ("hook", "durationMs"): _NUMBER,
    ("hook", "stdoutBytes"): _INT,
    ("hook", "stderrBytes"): _INT,
    ("hook", "timedOut"): _BOOL,
    # finalize
    ("finalize", "at"): _STRING,
    ("finalize", "branch"): _STRING,
    ("finalize", "commit"): _STRING,
    ("finalize", "message"): _STRING,
}


def _normalize_loc(loc: tuple[int | str, ...]) -> tuple[str, ...]:
    """Collapse array indices and dynamic map keys to ``"*"`` for phrase lookup.

    An array index is an ``int`` segment (pydantic preserves the type). A dynamic
    map key is either a metric name directly under ``metrics``, or a metric name
    inside a sample round -- the segment following an index. Both are collapsed so
    one phrase entry covers every concrete name.

    Using ``isinstance(segment, int)`` instead of ``str.isdigit`` keeps all-digit
    dict keys (e.g. a metric named ``"123"``) from being conflated with array
    indices.
    """
    out: list[str] = []
    prev_index = False
    for i, segment in enumerate(loc):
        if isinstance(segment, int):
            out.append("*")
            prev_index = True
            continue
        if prev_index or (i > 0 and loc[i - 1] == "metrics"):
            out.append("*")
            prev_index = False
            continue
        out.append(segment)
        prev_index = False
    return tuple(out)


def _strip_type_prefix(loc: tuple[int | str, ...], record_type: str) -> tuple[int | str, ...]:
    """Strip the discriminated-union type prefix from an error location.

    The ``TypeAdapter`` on the tagged union prepends the record type (e.g.
    ``"iteration"``) to every field-level error location.  The ``_PHRASES``
    table and the display path both expect the location without that prefix.
    """
    if loc and loc[0] == record_type:
        return loc[1:]
    return loc


def message_for_error(error: ErrorDetails, record_type: str) -> str:
    """Translate one pydantic error into a session-record problem string."""
    raw_loc = _strip_type_prefix(error["loc"], record_type)
    display_loc = tuple(str(part) for part in raw_loc)
    if error["type"] == "extra_forbidden":
        return f"Unknown session record key: {describe_key(display_loc)}"
    phrase = _PHRASES.get((record_type, *_normalize_loc(raw_loc)), "a valid value")
    got = "undefined" if error["type"] == "missing" else json.dumps(error["input"])
    key = describe_key(display_loc)
    return f"Invalid session record value for {key}: expected {phrase}, got {got}"
