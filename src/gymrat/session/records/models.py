"""Pydantic models for session log records.

Each model is frozen, uses ``alias_generator=to_camel`` for camelCase wire keys,
and ``populate_by_name=True`` so consumers construct with snake_case kwargs.

Two entry points bridge the two forms:

- :func:`parse_record` (in ``parse.py``) validates a decoded-JSON value into the
  typed model for its ``type``, raising a :class:`GymratError` worded for a
  session log.
- :func:`record_to_wire` renders a model back to its camelCase wire dict,
  the form the store serializes. Optional fields whose value is ``None`` are
  omitted, except ``deltaPct`` (on a metric verdict and on an iteration's
  primary), which is always present and serializes ``None`` as JSON ``null``.
"""

from collections.abc import Callable
from contextvars import ContextVar
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
)
from pydantic.alias_generators import to_camel

from gymrat.pydantic_errors import coerce_integer
from gymrat.session.schema import (
    HookStage,
    KeepReason,
    KeepStatus,
    Method,
    Outcome,
    PrimaryKind,
    Verdict,
)
from gymrat.session.workspace import BaselineRef, Worktrees

type SampleRound = dict[str, float | int]


def _to_tuple(value: object) -> object:
    """Coerce a list into a tuple before strict-mode validation."""
    if isinstance(value, list):
        return tuple(value)
    return value


_BaselineRefAdapter = TypeAdapter(BaselineRef)
_WorktreesAdapter = TypeAdapter(Worktrees)


def _coerce[T](cls: type[T], adapter: TypeAdapter[T]) -> Callable[[object], T]:
    """Build a validator that coerces a dict into ``cls`` via lax-mode validation.

    Strict mode rejects dict-to-dataclass coercion, so the returned validator
    uses a ``TypeAdapter`` in lax mode to validate and construct the dataclass,
    preserving field-level error locations.
    """

    def coerce(value: object) -> T:
        if isinstance(value, cls):
            return value
        return adapter.validate_python(value)

    return coerce


_coerce_baseline_ref = _coerce(BaselineRef, _BaselineRefAdapter)
_coerce_worktrees = _coerce(Worktrees, _WorktreesAdapter)


_RECORD_CONFIG = ConfigDict(
    strict=True,
    extra="forbid",
    frozen=True,
    populate_by_name=True,
    alias_generator=to_camel,
)

_NULL_MESSAGE = "value must not be null"

_wire_validation: ContextVar[bool] = ContextVar("_wire_validation", default=False)


def _reject_none(value: object) -> object:
    """Reject an explicitly-provided ``null`` during wire validation.

    An absent key falls back to the ``None`` default without invoking this
    validator; only a present ``null`` reaches here, so an optional-but-not-
    nullable field rejects it while an absent key stays ``None``.

    The check is skipped when the ``_wire_validation`` context variable is not
    set, so Python-side construction with ``field=None`` passes through.
    """
    if value is None and _wire_validation.get():
        raise ValueError(_NULL_MESSAGE)
    return value


def _coerce_integer(value: object) -> object:
    """Reject ``null``, then fold an integral float into ``int``.

    Composes :func:`_reject_none` with the shared :func:`coerce_integer` so
    JSON ``null`` is caught before the coercion pass.
    """
    _reject_none(value)
    return coerce_integer(value)


_Number = int | float

_OptStr = Annotated[str | None, BeforeValidator(_reject_none)]
_OptNonEmptyStr = Annotated[str | None, BeforeValidator(_reject_none), Field(min_length=1)]
_OptBool = Annotated[bool | None, BeforeValidator(_reject_none)]
_OptNumber = Annotated[_Number | None, BeforeValidator(_reject_none)]

_DeltaPct = _Number | None

_PositiveInt = Annotated[int, BeforeValidator(_coerce_integer), Field(ge=1)]
_NonNegativeInt = Annotated[int, BeforeValidator(_coerce_integer), Field(ge=0)]

_SampleRounds = Annotated[tuple[dict[str, _Number], ...], BeforeValidator(_to_tuple)]


class SessionHooks(BaseModel):
    """The hook commands a session was started with; an absent stage ran nothing."""

    model_config = _RECORD_CONFIG

    before: _OptNonEmptyStr = None
    after: _OptNonEmptyStr = None


class SessionConfig(BaseModel):
    """The settled configuration a session was opened with, kept for provenance."""

    model_config = _RECORD_CONFIG

    bench: str
    prepare: _OptStr = None
    adapter: str
    samples: _PositiveInt
    timeout_seconds: Annotated[int, BeforeValidator(_coerce_integer), Field(ge=1)]
    primary: str
    filter: _OptStr = None
    hooks: Annotated[SessionHooks | None, BeforeValidator(_reject_none)] = None


class SessionRecord(BaseModel):
    """Opens a session log: identity, worktrees, and a config snapshot."""

    model_config = _RECORD_CONFIG

    type: Literal["session"]
    schema_version: Literal[1]
    session_id: str
    created_at: str
    baseline: Annotated[BaselineRef, BeforeValidator(_coerce_baseline_ref)]
    branch: str
    worktrees: Annotated[Worktrees, BeforeValidator(_coerce_worktrees)]
    config: SessionConfig


class BaselineRecord(BaseModel):
    """A labelled set of baseline sample rounds."""

    model_config = _RECORD_CONFIG

    type: Literal["baseline"]
    at: str
    label: str
    samples: _SampleRounds
    duration_ms: _OptNumber = None


class _DeltaPctSerializer(BaseModel):
    """Shared base for models whose ``delta_pct`` is always emitted, even as ``None``."""

    model_config = _RECORD_CONFIG

    delta_pct: _DeltaPct

    @model_serializer(mode="wrap")
    def _always_emit_delta_pct(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        data["deltaPct"] = self.delta_pct
        return data


class MetricVerdict(_DeltaPctSerializer):
    """How one metric moved, with the statistics behind the judgement.

    ``delta_pct`` is ``None`` when a zero baseline median left the ratio
    undefined; the field is always present so a dropped delta reads as a broken
    writer, not a degenerate measurement.
    """

    model_config = _RECORD_CONFIG

    verdict: Verdict
    method: Method
    p: _OptNumber = None
    noise_pct: Annotated[_Number | None, BeforeValidator(_reject_none)] = None
    gating: bool
    confirmed: bool


class PairedSamples(BaseModel):
    """The experiment and baseline sample rounds measured in one iteration."""

    model_config = _RECORD_CONFIG

    experiment: _SampleRounds
    baseline: _SampleRounds


class Confirm(BaseModel):
    """A confirmation rerun: which metrics it re-measured, and the samples it took.

    ``absent`` names metrics the rerun was asked about but skipped; it is
    ``None`` on a log written before the field existed.
    """

    model_config = _RECORD_CONFIG

    ran: bool
    filtered: Annotated[tuple[str, ...], BeforeValidator(_to_tuple)]
    absent: Annotated[
        tuple[str, ...] | None,
        BeforeValidator(_reject_none),
        BeforeValidator(_to_tuple),
    ] = None
    samples: PairedSamples


class IterationPrimary(_DeltaPctSerializer):
    """The primary an iteration was judged on -- the geomean, or a named metric."""

    model_config = _RECORD_CONFIG

    kind: PrimaryKind
    name: _OptStr = None


class IterationRecord(BaseModel):
    """One measured edit: raw samples, per-metric verdicts, and the outcome.

    ``duration_ms`` is the wall-clock milliseconds from the readiness guard
    passing to the record being appended -- the before hook, both bench sides,
    the confirmation rerun, and judging.  It excludes the after hook.
    """

    model_config = _RECORD_CONFIG

    type: Literal["iteration"]
    seq: _PositiveInt
    at: str
    samples: PairedSamples
    metrics: dict[str, MetricVerdict]
    confirm: Annotated[Confirm | None, BeforeValidator(_reject_none)] = None
    primary: IterationPrimary
    outcome: Outcome
    target_reached: bool
    duration_ms: _OptNumber = None
    measured_tree: _OptStr = None


class KeepChecks(BaseModel):
    """The outcome of a keep's configured checks, with relayed output sizes."""

    model_config = _RECORD_CONFIG

    configured: bool
    passed: _OptBool = None
    stdout_bytes: Annotated[int | None, BeforeValidator(_coerce_integer), Field(ge=0)] = None
    stderr_bytes: Annotated[int | None, BeforeValidator(_coerce_integer), Field(ge=0)] = None


class KeepRecord(BaseModel):
    """The settlement of an iteration: committed, or blocked with a reason."""

    model_config = _RECORD_CONFIG

    type: Literal["keep"]
    seq: _NonNegativeInt
    at: str
    status: KeepStatus
    commit: _OptStr = None
    message: _OptStr = None
    reason: Annotated[KeepReason | None, BeforeValidator(_reject_none)] = None
    checks: KeepChecks


class DiscardRecord(BaseModel):
    """The reverted settlement of an iteration."""

    model_config = _RECORD_CONFIG

    type: Literal["discard"]
    seq: _NonNegativeInt
    at: str


class HookRecord(BaseModel):
    """One hook invocation around an iteration."""

    model_config = _RECORD_CONFIG

    type: Literal["hook"]
    stage: HookStage
    seq: _NonNegativeInt
    exit_code: Annotated[int, BeforeValidator(_coerce_integer)]
    duration_ms: _Number
    stdout_bytes: Annotated[_NonNegativeInt, BeforeValidator(_coerce_integer)]
    stderr_bytes: Annotated[
        _NonNegativeInt | None,
        BeforeValidator(_coerce_integer),
    ] = None
    timed_out: bool


class FinalizeRecord(BaseModel):
    """Closes a session: the branch and squash commit its kept work collapsed onto."""

    model_config = _RECORD_CONFIG

    type: Literal["finalize"]
    at: str
    branch: str
    commit: str
    message: str


class StopRecord(BaseModel):
    """A user-requested stop: halts the loop without finalizing the session."""

    model_config = _RECORD_CONFIG

    type: Literal["stop"]
    at: str
    message: Annotated[str, Field(min_length=1)]


type SessionLogRecord = (
    SessionRecord
    | BaselineRecord
    | IterationRecord
    | KeepRecord
    | DiscardRecord
    | HookRecord
    | FinalizeRecord
    | StopRecord
)


def record_to_wire(record: SessionLogRecord) -> dict[str, object]:
    """Render a session-log model back to its camelCase wire dict.

    Optional fields whose value is ``None`` are omitted, so a parsed record and
    its serialization round-trip. The exception is ``deltaPct`` on a metric
    verdict and on an iteration's primary: it is always emitted, carrying JSON
    ``null`` when the delta is undefined.
    """
    return record.model_dump(mode="json", by_alias=True, exclude_none=True)
