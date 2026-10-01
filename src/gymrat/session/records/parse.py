"""Inbound codec: log line to wire value to typed session-log model.

Includes validation-error translation from pydantic errors to problem strings.
"""

import json
import math
from typing import Annotated, NoReturn

from pydantic import Field, TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

from gymrat.errors import GymratError
from gymrat.pydantic_errors import (
    UNKNOWN_SHAPE_PHRASE,
    VALUE_ERROR_PREFIX,
    describe_key,
    drop_prefix_errors,
    phrase_for_error,
)
from gymrat.session.records.models import (
    SESSION_LOG_MODELS,
    SessionLogRecord,
    _wire_validation,
    wire_type,
)

_SessionLogUnion = TypeAdapter(Annotated[SessionLogRecord, Field(discriminator="type")])


class NonFiniteNumberError(ValueError):
    """A log line holds a number that does not load as a finite float."""


def _reject_non_finite(literal: str) -> NoReturn:
    message = f"{literal} is a non-finite number, which is not valid JSON"
    raise NonFiniteNumberError(message)


def _parse_finite_float(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):
        message = f"{literal} overflows a float"
        raise NonFiniteNumberError(message)
    return value


def decode_log_line(line: str) -> object:
    """Decode one JSONL log line as strict JSON, refusing every non-finite number.

    Python's decoder accepts the non-standard ``NaN``, ``Infinity`` and
    ``-Infinity`` literals, and decodes a number literal too large for a float,
    such as ``1e999``, to infinity. A log gymrat writes never holds either, so a
    line that does was hand-edited or written by another tool and is refused like
    any other malformed line rather than loading a non-finite measurement.

    Args:
        line: The line to decode.

    Returns:
        The decoded JSON value.

    Raises:
        NonFiniteNumberError: When the line holds a non-finite number, at any
            depth.
        ValueError: When the line is not JSON.
    """
    return json.loads(line, parse_constant=_reject_non_finite, parse_float=_parse_finite_float)


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


def _strip_type_prefix(loc: tuple[int | str, ...], record_type: str) -> tuple[int | str, ...]:
    """Strip the discriminated-union type prefix from an error location.

    The ``TypeAdapter`` on the tagged union prepends the record type (e.g.
    ``"iteration"``) to every field-level error location, which the display
    path expects without that prefix.

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


def message_for_error(error: ErrorDetails, record: dict[str, object]) -> str:
    """Translate one pydantic error into a session-record problem string.

    Model-level validators (``type="value_error"``, empty ``loc``) carry their
    own sentence in ``msg``; a field-level error names the shape its pydantic
    error ``type`` and ``ctx`` imply, and a missing key names only the key. The
    reported path is the one the user wrote in ``record``, free of the union
    member tags pydantic adds to the error location.

    Args:
        error: The pydantic ``ErrorDetails`` being translated.
        record: The raw record that failed validation; its ``type`` is the
            union member tag stripped from the error location.

    Returns:
        A human-readable problem string for the error.
    """
    record_type = str(record.get("type", ""))
    missing = error["type"] == "missing"
    path = _data_path(
        _strip_type_prefix(error["loc"], record_type), record, error["input"], missing=missing
    )
    key = describe_key(tuple(str(part) for part in path))
    if missing:
        return f"Missing session record key: {key}"
    if error["type"] == "extra_forbidden":
        return f"Unknown session record key: {key}"
    if error["type"] == "value_error" and not path:
        # Pydantic renders a model validator's ValueError as
        # "Value error, <text>"; surface <text> directly.
        msg = error["msg"].removeprefix(VALUE_ERROR_PREFIX)
        separator = ": " if key else ""
        return f"Invalid session record: {key}{separator}{msg}"
    phrase = phrase_for_error(error) or UNKNOWN_SHAPE_PHRASE
    got = json.dumps(error["input"])
    return f"Invalid session record value for {key}: expected {phrase}, got {got}"
