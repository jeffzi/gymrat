"""Tests for the telemetry attribute namespace.

Enumerates every attribute name the telemetry layer can emit, checks that the
README's attribute table documents exactly that set, and that every attribute a
record or command span emits is in it and follows the naming rule.
"""

from __future__ import annotations

import functools
import re
import types
import typing
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gymrat.session.records import (
    SESSION_LOG_MODELS,
    CommandRecord,
    Confirm,
    IterationRecord,
    PairedSamples,
    SessionLogRecord,
    SessionRecord,
    wire_type,
)
from gymrat.telemetry.provider import (
    CAP_NAME,
    COMMAND_ARGS_PREFIX,
    COMMAND_DURATION_MS,
    COMMAND_EXIT_CODE,
    COMMAND_NAME,
    COMMAND_REASON,
    FOLLOW_UP_ACTION,
    FOLLOW_UP_REASON,
    GEN_AI_MODEL,
    GEN_AI_PROVIDER,
    ITERATION_DELTA_PCT,
    ITERATION_OUTCOME,
    ITERATION_SEQ,
    RUN_COST_USD,
    RUN_DURATION_MS,
    RUN_EFFORT,
    RUN_END_REASON,
    RUN_ENDED_BY,
    RUN_HEAD_SHA,
    RUN_MAX_MINUTES,
    RUN_MAX_USD,
    SESSION_BRANCH,
    TURN_BUDGET_EXHAUSTED,
    TURN_ORIGIN,
    TURN_SESSION_COST_USD,
    command_attributes,
    record_event,
)
from gymrat.telemetry.provider import SESSION_ID as SESSION_ID_ATTR
from tests.session.records._fixtures import (
    SESSION_ID,
    baseline_record,
    blocked_keep,
    command_record,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    iteration_record,
    stop_record,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


#: The annotations a record field may carry to be emitted as an attribute.
_SCALAR_TYPES = (str, int, float, bool)

#: The envelope fields every record carries, never emitted as their own attribute.
_SKIPPED_FIELD_NAMES = frozenset({"at", "seq", "type"})

# The attribute names that are not derived from a record model's fields.
_FIXED_ATTRS = frozenset({
    SESSION_ID_ATTR,
    SESSION_BRANCH,
    COMMAND_NAME,
    COMMAND_EXIT_CODE,
    COMMAND_DURATION_MS,
    COMMAND_REASON,
    COMMAND_ARGS_PREFIX,
    RUN_HEAD_SHA,
    RUN_MAX_MINUTES,
    RUN_MAX_USD,
    RUN_EFFORT,
    RUN_COST_USD,
    RUN_ENDED_BY,
    RUN_END_REASON,
    RUN_DURATION_MS,
    TURN_SESSION_COST_USD,
    TURN_ORIGIN,
    TURN_BUDGET_EXHAUSTED,
    FOLLOW_UP_ACTION,
    FOLLOW_UP_REASON,
    CAP_NAME,
    GEN_AI_MODEL,
    GEN_AI_PROVIDER,
    ITERATION_SEQ,
    ITERATION_OUTCOME,
    ITERATION_DELTA_PCT,
})


def _is_scalar_or_literal(annotation: object) -> bool:
    return annotation in _SCALAR_TYPES or typing.get_origin(annotation) is typing.Literal


def _is_scalar_or_none(annotation: object) -> bool:
    if typing.get_origin(annotation) is typing.Annotated:
        inner = typing.get_args(annotation)
        if inner:
            annotation = inner[0]
    return _is_scalar_or_literal(annotation) or annotation is type(None)


def _is_scalar_type(annotation: object) -> bool:
    if _is_scalar_or_literal(annotation):
        return True
    if typing.get_origin(annotation) in {typing.Union, types.UnionType}:
        return all(_is_scalar_or_none(arg) for arg in typing.get_args(annotation))
    return False


@functools.cache
def all_attribute_names() -> frozenset[str]:
    """Return every attribute name the telemetry layer can emit.

    The set includes fixed constants for session, run, turn, follow-up, cap,
    and command spans, derived ``gymrat.<type>.<field>`` names from record
    models other than session, iteration, and command — those are covered
    by the fixed constants above and are excluded here — iteration names,
    ``gen_ai.*`` names, and the ``gymrat.command.args`` pattern placeholder.

    Returns:
        A frozenset of dotted attribute name strings.
    """
    record_derived: set[str] = set()
    for record_cls in SESSION_LOG_MODELS:
        if record_cls in (SessionRecord, IterationRecord, CommandRecord):
            continue
        record_type = wire_type(record_cls)
        for field_name, field_info in record_cls.model_fields.items():
            if field_name in _SKIPPED_FIELD_NAMES:
                continue
            if _is_scalar_type(field_info.annotation):
                record_derived.add(f"gymrat.{record_type}.{field_name}")

    return _FIXED_ATTRS | frozenset(record_derived)


_README = Path(__file__).resolve().parents[2] / "README.md"
_HEADING = "### Attribute reference"
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_PLACEHOLDER_SUFFIX_RE = re.compile(r"\.<[^>]+>$")


def _parse_readme_attribute_names() -> frozenset[str]:
    """Extract attribute names from the first column of the README table.

    Returns:
        Attribute names found under the ``### Attribute reference`` heading.
    """
    lines = _README.read_text().splitlines()

    in_section = False
    header_seen = False
    names: set[str] = set()

    for line in lines:
        if line.strip() == _HEADING:
            in_section = True
            continue

        if not in_section:
            continue

        # Stop at the next heading of equal or higher level.
        if line.startswith("#") and not line.startswith("####"):
            break

        stripped = line.strip()

        # Skip the separator row (e.g. "| --- | --- |").
        if stripped.startswith("| -"):
            header_seen = True
            continue

        # Skip the header row itself (first pipe-delimited row before separator).
        if not header_seen:
            continue

        if not stripped.startswith("|"):
            continue

        first_col = stripped.split("|")[1]
        match = _BACKTICK_RE.search(first_col)
        if match:
            name = match.group(1)
            # Normalize pattern placeholder: `gymrat.command.args.<key>` -> `gymrat.command.args`
            name = _PLACEHOLDER_SUFFIX_RE.sub("", name)
            names.add(name)

    return frozenset(names)


def test_readme_attribute_reference_when_parsed_does_list_exactly_the_emitted_attribute_names() -> (
    None
):
    readme_names = _parse_readme_attribute_names()
    code_names = all_attribute_names()

    assert readme_names == code_names


_ATTR_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*)*$")


def _assert_valid_attribute_names(keys: Iterable[str]) -> None:
    for key in keys:
        assert _ATTR_NAME_RE.match(key), f"bad attribute name: {key!r}"


# ---------------------------------------------------------------------------
# record_event and command_attributes — every emitted key is documented and well formed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(baseline_record(duration_ms=42), id="baseline"),
        pytest.param(
            iteration_record(
                duration_ms=500,
                measured_tree="experiment",
                confirm=Confirm(
                    ran=True,
                    filtered=("total_ms",),
                    samples=PairedSamples(
                        experiment=({"total_ms": 14050},),
                        baseline=({"total_ms": 15200},),
                    ),
                ),
            ),
            id="iteration",
        ),
        pytest.param(
            committed_keep(1, commit="a" * 40, message="cache the regex", reason=None),
            id="committed_keep",
        ),
        pytest.param(blocked_keep(1, reason="checks-failed"), id="blocked_keep"),
        pytest.param(discard_record(2), id="discard"),
        pytest.param(hook_record(stderr_bytes=120), id="hook"),
        pytest.param(finalize_record(branch="gymrat/test-final"), id="finalize"),
        pytest.param(stop_record(), id="stop"),
    ],
)
def test_record_event_when_every_optional_field_set_does_emit_only_documented_well_formed_scalar_attributes(
    record: SessionLogRecord,
):
    # All optional fields are set so a new field with an illegal name fails this test.
    namespace = all_attribute_names()

    _name, attrs = record_event(record)

    assert set(attrs) <= namespace
    _assert_valid_attribute_names(attrs)
    assert all(isinstance(value, (str, int, float, bool)) for value in attrs.values())


def test_command_attributes_when_called_does_emit_only_documented_well_formed_attributes():
    record = command_record(
        name="iterate",
        args={"session_id": SESSION_ID, "ref": "main", "samples": 5},
        reason="budget-exceeded",
        seq=2,
    )

    result = command_attributes(record, session_id=SESSION_ID)

    _assert_valid_attribute_names(result)
    assert {key for key in result if not key.startswith(COMMAND_ARGS_PREFIX + ".")} <= (
        all_attribute_names()
    )
