"""Inbound codec: wire value to typed session-log model.

Includes validation-error translation from pydantic errors to problem strings.
"""

import json
from typing import Annotated

from pydantic import Field, TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

from gymrat.errors import GymratError
from gymrat.pydantic_errors import alternatives, describe_key, drop_prefix_errors
from gymrat.session.records.models import (
    SESSION_LOG_MODELS,
    CommandRecord,
    SessionLogRecord,
    _wire_validation,
    wire_type,
)
from gymrat.session.schema import (
    SCHEMA_VERSION,
    CommandOrigin,
    CommandReason,
    HookStage,
    KeepReason,
    KeepStatus,
    Method,
    Outcome,
    PrimaryKind,
    Verdict,
)

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
        raise GymratError(message_for_error(error, value)) from exc
    finally:
        _wire_validation.reset(token)


def _known_types() -> list[str]:
    return [wire_type(member) for member in SESSION_LOG_MODELS]


def _raise_discriminator_error(errors: list[ErrorDetails], value: dict[str, object]) -> None:
    """Raise with the canonical "Unknown session record type" message.

    Callers match on this exact wording and on the hint, which lists every
    ``SessionLogRecord`` member's ``type`` literal in union order.

    Args:
        errors: The validation errors pydantic reported.
        value: The raw record that failed validation.

    Raises:
        GymratError: When ``errors`` contains a discriminator-mismatch error.
    """
    if not errors:
        return
    error_type = errors[0]["type"]
    if error_type not in ("union_tag_not_found", "union_tag_invalid"):
        return
    type_value = value.get("type")
    tag_missing = error_type == "union_tag_not_found" and type_value is None and "type" not in value
    rendered = "undefined" if tag_missing else json.dumps(type_value)
    hint = "Expected one of: " + ", ".join(_known_types()) + "."
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


_VERDICT = alternatives(Verdict)
_METHOD = alternatives(Method)
_KIND = alternatives(PrimaryKind)
_OUTCOME = alternatives(Outcome)
_STATUS = alternatives(KeepStatus)
_REASON = alternatives(KeepReason)
_STAGE = alternatives(HookStage)
_EXIT_CODE = alternatives(CommandRecord.model_fields["exit_code"].annotation)
_COMMAND_REASON = f"one of {alternatives(CommandReason)}"
_ORIGIN = alternatives(CommandOrigin)

_PHRASES: dict[tuple[str, ...], str] = {
    # session
    ("session", "schema"): str(SCHEMA_VERSION),
    ("session", "session_id"): _STRING,
    ("session", "at"): _INT,
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
    ("session", "config", "timeout_seconds"): _POSITIVE_INT,
    ("session", "config", "primary"): _STRING,
    ("session", "config", "filter"): _STRING,
    ("session", "config", "hooks"): _OBJECT,
    ("session", "config", "hooks", "before"): "a non-empty string",
    ("session", "config", "hooks", "after"): "a non-empty string",
    # baseline
    ("baseline", "at"): _INT,
    ("baseline", "label"): _STRING,
    ("baseline", "duration_ms"): _NUMBER,
    ("baseline", "samples"): _SAMPLE_ROUNDS,
    ("baseline", "samples", "*"): _OBJECT,
    ("baseline", "samples", "*", "*"): _NUMBER,
    # iteration
    ("iteration", "seq"): _POSITIVE_INT,
    ("iteration", "at"): _INT,
    ("iteration", "samples"): _OBJECT,
    ("iteration", "samples", "experiment"): _SAMPLE_ROUNDS,
    ("iteration", "samples", "experiment", "*"): _OBJECT,
    ("iteration", "samples", "experiment", "*", "*"): _NUMBER,
    ("iteration", "samples", "baseline"): _SAMPLE_ROUNDS,
    ("iteration", "samples", "baseline", "*"): _OBJECT,
    ("iteration", "samples", "baseline", "*", "*"): _NUMBER,
    ("iteration", "metrics"): _OBJECT,
    ("iteration", "metrics", "*"): _OBJECT,
    ("iteration", "metrics", "*", "delta_pct"): _DELTA,
    ("iteration", "metrics", "*", "verdict"): _VERDICT,
    ("iteration", "metrics", "*", "method"): _METHOD,
    ("iteration", "metrics", "*", "p"): _NUMBER,
    ("iteration", "metrics", "*", "noise_pct"): _NUMBER,
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
    ("iteration", "primary", "delta_pct"): _DELTA,
    ("iteration", "outcome"): _OUTCOME,
    ("iteration", "target_reached"): _BOOL,
    ("iteration", "duration_ms"): _NUMBER,
    ("iteration", "measured_tree"): _STRING,
    # keep
    ("keep", "seq"): _NON_NEGATIVE_INT,
    ("keep", "at"): _INT,
    ("keep", "status"): _STATUS,
    ("keep", "commit"): _STRING,
    ("keep", "message"): _STRING,
    ("keep", "reason"): _REASON,
    ("keep", "checks"): _OBJECT,
    ("keep", "checks", "configured"): _BOOL,
    ("keep", "checks", "passed"): _BOOL,
    ("keep", "checks", "stdout_bytes"): _NON_NEGATIVE_INT,
    ("keep", "checks", "stderr_bytes"): _NON_NEGATIVE_INT,
    # discard
    ("discard", "seq"): _NON_NEGATIVE_INT,
    ("discard", "at"): _INT,
    # hook
    ("hook", "at"): _INT,
    ("hook", "stage"): _STAGE,
    ("hook", "seq"): _NON_NEGATIVE_INT,
    ("hook", "exit_code"): _INT,
    ("hook", "duration_ms"): _NUMBER,
    ("hook", "stdout_bytes"): _INT,
    ("hook", "stderr_bytes"): _INT,
    ("hook", "timed_out"): _BOOL,
    # finalize
    ("finalize", "at"): _INT,
    ("finalize", "branch"): _STRING,
    ("finalize", "commit"): _STRING,
    ("finalize", "message"): _STRING,
    # stop
    ("stop", "at"): _INT,
    ("stop", "message"): "a non-empty string",
    # command
    ("command", "at"): _INT,
    ("command", "name"): "a non-empty string",
    ("command", "args"): _OBJECT,
    ("command", "exit_code"): _EXIT_CODE,
    ("command", "reason"): _COMMAND_REASON,
    ("command", "origin"): _ORIGIN,
    ("command", "seq"): _INT,
    ("command", "duration_ms"): _NON_NEGATIVE_INT,
    ("command", "traceparent"): _STRING,
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

    Args:
        loc: A pydantic error location.

    Returns:
        The location tuple with indices and dynamic keys replaced by ``"*"``.
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

    Args:
        loc: A pydantic error location.
        record_type: The record's ``type`` value.

    Returns:
        The location tuple without the leading type segment.
    """
    if loc and loc[0] == record_type:
        return loc[1:]
    return loc


def _walk_to_target(
    loc: tuple[int | str, ...], node: object, target: object
) -> tuple[int | str, ...] | None:
    """Find the location prefix that walks ``node`` down to ``target``.

    Nodes are matched against ``target`` by identity, so ``target`` must be a
    container: a scalar may be one object shared by every equal value.

    A segment that also names a key or index of ``node`` is tried first, since
    a coincidental match should win when it leads to ``target``. If it
    doesn't, the segment is retried as a tag naming nothing in the input,
    leaving ``node`` unchanged for the next segment.

    Args:
        loc: The remaining location segments to consume.
        node: The record node the walk is currently on.
        target: The node the walk must end on.

    Returns:
        The consumed segments on the path to ``target``, or ``None`` when no
        path reaches it.
    """
    if not loc:
        return () if node is target else None
    segment, rest = loc[0], loc[1:]
    match node:
        case dict() if segment in node:
            taken = _walk_to_target(rest, node[segment], target)
            if taken is not None:
                return (segment, *taken)
        case list() if isinstance(segment, int) and 0 <= segment < len(node):
            taken = _walk_to_target(rest, node[segment], target)
            if taken is not None:
                return (segment, *taken)
    return _walk_to_target(rest, node, target)


def _greedy_walk(loc: tuple[int | str, ...], node: object) -> tuple[int | str, ...]:
    """Consume every location segment that names a key or index of ``node``.

    Used when the error's ``input`` cannot be located by identity: it is a
    scalar, whose object CPython may share with equal values elsewhere in the
    record, or a container a ``BeforeValidator`` replaced with a new object no
    walk can land on. Without a target to confirm against, a segment that also
    names a key or index of ``node`` is always followed; a tag segment naming
    nothing in the input is skipped, leaving ``node`` unchanged for the next
    segment.

    Args:
        loc: The remaining location segments to consume.
        node: The record node the walk is currently on.

    Returns:
        The consumed segments.
    """
    if not loc:
        return ()
    segment, rest = loc[0], loc[1:]
    match node:
        case dict() if segment in node:
            return (segment, *_greedy_walk(rest, node[segment]))
        case list() if isinstance(segment, int) and 0 <= segment < len(node):
            return (segment, *_greedy_walk(rest, node[segment]))
    return _greedy_walk(rest, node)


def _data_path(
    loc: tuple[int | str, ...], record: dict[str, object], target: object, *, missing: bool
) -> tuple[int | str, ...]:
    """Keep only the location segments that address a key or index of ``record``.

    For every union member it tries, pydantic inserts a tag segment into the
    location -- a model's class name, or a type such as ``int`` or
    ``tuple[str, ...]``. A tag names nothing in the input, so a segment that
    also happens to be a present key is disambiguated by whether continuing
    the walk under it still reaches ``target`` -- the error's ``input``
    (for a ``missing`` error, the parent object). A ``missing`` error's final
    segment names a key absent by definition, so it is kept without being
    looked up.

    The identity walk runs only when ``target`` is a ``dict`` or ``list``:
    each decoded JSON container is a distinct object, so identity pins down
    exactly one node. A scalar ``target`` proves nothing -- CPython shares one
    object for equal small ints, ``True``, ``False`` and ``None``, and a
    ``BeforeValidator`` that coerces ``0.0`` to ``0`` hands back that shared
    object -- so identity would land on whichever equal value the walk meets
    first, anywhere in the record. For a scalar, and whenever no path reaches
    a container ``target`` because a ``BeforeValidator`` replaced it,
    ``_greedy_walk`` follows every segment that matches a key or index
    instead.

    Args:
        loc: A pydantic error location, relative to ``record``.
        record: The raw record that failed validation.
        target: The error's ``input``; a ``dict`` or ``list`` is the node the
            walk must end on, any other value is ignored.
        missing: Whether the error reports a missing key.

    Returns:
        The location without union member tags.
    """
    walked = loc[:-1] if missing else loc
    match target:
        case dict() | list():
            path = _walk_to_target(walked, record, target)
        case _:
            path = None
    if path is None:
        path = _greedy_walk(walked, record)
    if missing and loc:
        path = (*path, loc[-1])
    return path


_VALUE_ERROR_PREFIX = "Value error, "


def message_for_error(error: ErrorDetails, record: dict[str, object]) -> str:
    """Translate one pydantic error into a session-record problem string.

    Model-level validators (``type="value_error"``, empty ``loc``) carry their
    own sentence in ``msg``; field-level errors use the phrase table. The
    reported path is the one the user wrote in ``record``, free of the union
    member tags pydantic adds to the error location.

    Args:
        error: The pydantic ``ErrorDetails`` being translated.
        record: The raw record that failed validation; its ``type`` selects the
            phrase table entries.

    Returns:
        A human-readable problem string for the error.
    """
    record_type = str(record.get("type", ""))
    missing = error["type"] == "missing"
    path = _data_path(
        _strip_type_prefix(error["loc"], record_type), record, error["input"], missing=missing
    )
    display_loc = tuple(str(part) for part in path)
    key = describe_key(display_loc)
    if error["type"] == "extra_forbidden":
        return f"Unknown session record key: {key}"
    if error["type"] == "value_error" and not path:
        # Pydantic renders a model validator's ValueError as
        # "Value error, <text>"; surface <text> directly.
        msg = error["msg"].removeprefix(_VALUE_ERROR_PREFIX)
        separator = ": " if key else ""
        return f"Invalid session record: {key}{separator}{msg}"
    normalized = _normalize_loc(path)
    phrase = _PHRASES.get((record_type, *normalized), "a valid value")
    got = "undefined" if missing else json.dumps(error["input"])
    return f"Invalid session record value for {key}: expected {phrase}, got {got}"
