"""Schemas and parsing for the lines of a session JSONL log.

Each line of a session log is one record, discriminated on its ``type`` field.
A record is validated against a pydantic model internally, but the public
surface is plain frozen dataclasses -- no pydantic type ever leaks to consumers.
The wire form is camelCase (``schemaVersion``, ``timeoutSeconds``, ``deltaPct``);
the dataclasses expose snake_case attributes, and validation error paths always
name the camelCase key the writer wrote.

Two entry points bridge the two forms:

- :func:`parse_record` validates a decoded-JSON value into the typed dataclass
  for its ``type``, raising a :class:`GymratError` worded for a session log.
- :func:`record_to_wire` renders a dataclass back to its camelCase wire dict,
  the form the store serializes. Optional fields whose value is ``None`` are
  omitted, except ``deltaPct`` (on a metric verdict and on an iteration's
  primary), which is always present and serializes ``None`` as JSON ``null``.
"""

from gymrat.session.records.parse import parse_record
from gymrat.session.records.types import (
    BaselineRecord,
    Confirm,
    DiscardRecord,
    FinalizeRecord,
    HookRecord,
    IterationPrimary,
    IterationRecord,
    KeepChecks,
    KeepRecord,
    MetricVerdict,
    PairedSamples,
    SampleRound,
    SessionConfig,
    SessionHooks,
    SessionLogRecord,
    SessionRecord,
)
from gymrat.session.records.wire import record_to_wire
from gymrat.session.schema import SCHEMA_VERSION
from gymrat.session.workspace import BaselineRef, Worktrees

__all__ = [
    "SCHEMA_VERSION",
    "BaselineRecord",
    "BaselineRef",
    "Confirm",
    "DiscardRecord",
    "FinalizeRecord",
    "HookRecord",
    "IterationPrimary",
    "IterationRecord",
    "KeepChecks",
    "KeepRecord",
    "MetricVerdict",
    "PairedSamples",
    "SampleRound",
    "SessionConfig",
    "SessionHooks",
    "SessionLogRecord",
    "SessionRecord",
    "Worktrees",
    "parse_record",
    "record_to_wire",
]
