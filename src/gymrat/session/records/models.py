"""Pydantic models for session log records.

Each model is frozen with ``strict=True`` and ``extra="forbid"``. Wire keys
are the Python attribute names (snake_case) -- no alias generator, except
:attr:`SessionRecord.schema_version`, which aliases to ``schema`` because
``schema`` collides with :meth:`BaseModel.schema`.

Two entry points bridge the two forms:

- :func:`parse_record` (in ``parse.py``) validates a decoded-JSON value into the
  typed model for its ``type``, raising a :class:`GymratError` worded for a
  session log.
- :func:`record_to_wire` renders a model back to its snake_case wire dict,
  the form the store serializes. Optional fields whose value is ``None`` are
  omitted, except ``delta_pct`` (on a metric verdict and on an iteration's
  primary), which is always present and serializes ``None`` as JSON ``null``.
"""

from collections.abc import Callable
from contextvars import ContextVar
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema, WithJsonSchema

from gymrat.pydantic_errors import coerce_integer
from gymrat.session.schema import (
    CommandReason,
    HookStage,
    KeepReason,
    KeepStatus,
    Method,
    Outcome,
    PrimaryKind,
    Verdict,
)
from gymrat.session.workspace import BaselineRef, Worktrees

# ---------------------------------------------------------------------------
# Validation and coercion helpers
# ---------------------------------------------------------------------------

#: Metric name to measured value for one sample round.
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

    Args:
        cls: The dataclass type to coerce values into.
        adapter: The ``TypeAdapter`` used to validate and construct ``cls``.

    Returns:
        A before-validator callable that coerces dicts into ``cls``.
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
    validate_by_name=True,
    serialize_by_alias=True,
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

    Args:
        value: The value under validation.

    Returns:
        The original value, unchanged.

    Raises:
        ValueError: When ``value`` is ``None`` and wire validation is active.
    """
    if value is None and _wire_validation.get():
        raise ValueError(_NULL_MESSAGE)
    return value


def _coerce_integer(value: object) -> object:
    """Reject ``null``, then fold an integral float into ``int``."""
    _reject_none(value)
    return coerce_integer(value)


_Number = int | float

_OptStr = Annotated[
    str | SkipJsonSchema[None],
    BeforeValidator(_reject_none),
]
_OptNonEmptyStr = Annotated[
    str | SkipJsonSchema[None],
    BeforeValidator(_reject_none),
    Field(min_length=1),
]
_OptBool = Annotated[
    bool | SkipJsonSchema[None],
    BeforeValidator(_reject_none),
]
_OptNumber = Annotated[
    _Number | SkipJsonSchema[None],
    BeforeValidator(_reject_none),
    WithJsonSchema({"type": "number"}),
]

_DeltaPct = _Number | None

_PositiveInt = Annotated[int, Field(ge=1), BeforeValidator(_coerce_integer)]
_NonNegativeInt = Annotated[int, Field(ge=0), BeforeValidator(_coerce_integer)]
_OptNonNegativeInt = _NonNegativeInt | SkipJsonSchema[None]

_SampleRounds = Annotated[tuple[dict[str, _Number], ...], BeforeValidator(_to_tuple)]


# ---------------------------------------------------------------------------
# Envelope bases
# ---------------------------------------------------------------------------


class _RecordEnvelope(BaseModel):
    """Base for every session-log record: a nanosecond timestamp.

    Each subclass declares its own ``type: Literal[...]`` discriminator.
    """

    model_config = _RECORD_CONFIG

    at: int = Field(description="Nanoseconds since the Unix epoch when the record was created.")


class _SequencedEnvelope(_RecordEnvelope):
    """Records tied to an iteration carry a sequence number.

    Iteration, Keep, Discard, and Hook override ``seq`` as required;
    Command inherits the optional default.
    """

    seq: Annotated[
        int | SkipJsonSchema[None],
        BeforeValidator(_coerce_integer),
    ] = Field(
        default=None, description="Iteration sequence number, present on per-iteration records."
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionHooks(BaseModel):
    """The hook commands a session was started with; an absent stage ran nothing."""

    model_config = _RECORD_CONFIG

    before: _OptNonEmptyStr = Field(
        default=None, description="Command to run before each iteration."
    )
    after: _OptNonEmptyStr = Field(default=None, description="Command to run after each iteration.")


class SessionConfig(BaseModel):
    """The settled configuration a session was opened with, kept for provenance."""

    model_config = _RECORD_CONFIG

    bench: str = Field(description="Benchmark command to run.")
    prepare: _OptStr = Field(
        default=None, description="Preparation command to run before benchmarking."
    )
    adapter: str = Field(description="Output adapter for parsing benchmark results.")
    samples: _PositiveInt = Field(description="Number of sample rounds per side.")
    timeout_seconds: Annotated[
        int,
        Field(ge=1, description="Maximum seconds per bench invocation."),
        BeforeValidator(_coerce_integer),
    ]
    primary: str = Field(description="Primary metric or aggregation method for judging iterations.")
    filter: _OptStr = Field(default=None, description="Filter expression for selecting benchmarks.")
    hooks: Annotated[
        SessionHooks | SkipJsonSchema[None],
        BeforeValidator(_reject_none),
    ] = Field(default=None, description="Hook commands to run around iterations.")


class SessionRecord(_RecordEnvelope):
    """Opens a session log: identity, worktrees, and a config snapshot."""

    type: Literal["session"] = Field(description="Record type discriminator.")
    schema_version: Literal[1] = Field(alias="schema", description="Session log format version.")
    session_id: str = Field(description="Unique identifier for this session.")
    baseline: Annotated[BaselineRef, BeforeValidator(_coerce_baseline_ref)] = Field(
        description="Git ref and SHA the baseline was taken from."
    )
    branch: str = Field(description="Git branch created for this session.")
    worktrees: Annotated[Worktrees, BeforeValidator(_coerce_worktrees)] = Field(
        description="Paths to the experiment and baseline worktrees."
    )
    config: SessionConfig = Field(
        description="Configuration snapshot the session was started with."
    )


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


class BaselineRecord(_RecordEnvelope):
    """A labelled set of baseline sample rounds."""

    type: Literal["baseline"] = Field(description="Record type discriminator.")
    label: str = Field(description="Human-readable label for this baseline measurement.")
    samples: _SampleRounds = Field(
        description="Baseline sample rounds mapping metric names to numbers."
    )
    duration_ms: _OptNumber = Field(
        default=None, description="Wall-clock milliseconds the baseline measurement took."
    )


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


class _DeltaPctSerializer(BaseModel):
    """Shared base for models whose ``delta_pct`` is always emitted, even as ``None``."""

    model_config = _RECORD_CONFIG

    delta_pct: _DeltaPct = Field(
        description="Percentage change from baseline, or null when undefined."
    )

    @model_serializer(mode="wrap")
    def _always_emit_delta_pct(
        self,
        handler: SerializerFunctionWrapHandler,
    ) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        data["delta_pct"] = self.delta_pct
        return data


class MetricVerdict(_DeltaPctSerializer):
    """How one metric moved, with the statistics behind the judgement.

    ``delta_pct`` is ``None`` when a zero baseline median left the ratio
    undefined; the field is always present so a dropped delta reads as a broken
    writer, not a degenerate measurement.
    """

    verdict: Verdict = Field(
        description="Whether the metric improved, regressed, or showed no signal."
    )
    method: Method = Field(description="Statistical test used to judge the metric.")
    p: _OptNumber = Field(default=None, description="P-value from the statistical test.")
    noise_pct: _OptNumber = Field(
        default=None, description="Estimated noise as a percentage of the baseline median."
    )
    gating: bool = Field(description="Whether this metric gates the iteration outcome.")
    confirmed: bool = Field(description="Whether a confirmation rerun validated this verdict.")


class PairedSamples(BaseModel):
    """The experiment and baseline sample rounds measured in one iteration."""

    model_config = _RECORD_CONFIG

    experiment: _SampleRounds = Field(description="Sample rounds from the experiment worktree.")
    baseline: _SampleRounds = Field(description="Sample rounds from the baseline worktree.")


class Confirm(BaseModel):
    """A confirmation rerun: which metrics it re-measured, and the samples it took.

    ``absent`` names metrics the rerun was asked about but skipped; it is
    ``None`` on a log written before the field existed.
    """

    model_config = _RECORD_CONFIG

    ran: bool = Field(description="Whether the confirmation rerun actually executed.")
    filtered: Annotated[tuple[str, ...], BeforeValidator(_to_tuple)] = Field(
        description="Metric names the rerun was filtered to."
    )
    absent: Annotated[
        tuple[str, ...] | SkipJsonSchema[None],
        BeforeValidator(_reject_none),
        BeforeValidator(_to_tuple),
    ] = Field(default=None, description="Metric names the rerun was asked about but skipped.")
    samples: PairedSamples = Field(
        description="Sample rounds collected during the confirmation rerun."
    )


class IterationPrimary(_DeltaPctSerializer):
    """The primary an iteration was judged on -- the geomean, or a named metric."""

    kind: PrimaryKind = Field(
        description="Whether the primary is a geomean aggregate or a named metric."
    )
    name: _OptStr = Field(default=None, description="Metric name when kind is 'metric'.")


class IterationRecord(_SequencedEnvelope):
    """One measured edit: raw samples, per-metric verdicts, and the outcome.

    ``duration_ms`` is the wall-clock milliseconds from the readiness guard
    passing to the record being appended -- the before hook, both bench sides,
    the confirmation rerun, and judging.  It excludes the after hook.
    """

    type: Literal["iteration"] = Field(description="Record type discriminator.")
    # pyrefly: ignore[bad-override-mutable-attribute] -- pydantic narrows optional to required
    seq: _PositiveInt = Field(description="Iteration sequence number, starting at 1.")
    samples: PairedSamples = Field(description="Raw sample rounds from both worktrees.")
    metrics: dict[str, MetricVerdict] = Field(
        description="Per-metric verdicts keyed by metric name."
    )
    confirm: Annotated[
        Confirm | SkipJsonSchema[None],
        BeforeValidator(_reject_none),
    ] = Field(default=None, description="Confirmation rerun results, if one was triggered.")
    primary: IterationPrimary = Field(
        description="The primary metric or aggregate the outcome was judged on."
    )
    outcome: Outcome = Field(
        description="Overall iteration outcome: improved, regressed, or no-signal."
    )
    target_reached: bool = Field(description="Whether the iteration met the target threshold.")
    duration_ms: _OptNumber = Field(
        default=None, description="Wall-clock milliseconds the iteration measurement took."
    )
    measured_tree: _OptStr = Field(
        default=None, description="Git tree hash of the experiment worktree at measurement time."
    )


# ---------------------------------------------------------------------------
# Keep / discard / hook
# ---------------------------------------------------------------------------


class KeepChecks(BaseModel):
    """The outcome of a keep's configured checks, with relayed output sizes."""

    model_config = _RECORD_CONFIG

    configured: bool = Field(description="Whether checks were configured for this session.")
    passed: _OptBool = Field(default=None, description="Whether all configured checks passed.")
    stdout_bytes: _OptNonNegativeInt = Field(
        default=None, description="Bytes the check command wrote to stdout."
    )
    stderr_bytes: _OptNonNegativeInt = Field(
        default=None, description="Bytes the check command wrote to stderr."
    )


class KeepRecord(_SequencedEnvelope):
    """The settlement of an iteration: committed, or blocked with a reason."""

    type: Literal["keep"] = Field(description="Record type discriminator.")
    # pyrefly: ignore[bad-override-mutable-attribute] -- pydantic narrows optional to required
    seq: _NonNegativeInt = Field(description="Iteration sequence number this keep settles.")
    status: KeepStatus = Field(description="Whether the iteration was committed or blocked.")
    commit: _OptStr = Field(default=None, description="Git commit SHA when status is committed.")
    message: _OptStr = Field(default=None, description="Commit message when status is committed.")
    reason: Annotated[
        KeepReason | SkipJsonSchema[None],
        BeforeValidator(_reject_none),
    ] = Field(default=None, description="Why the keep was blocked, when status is blocked.")
    checks: KeepChecks = Field(description="Outcome of the configured checks.")


class DiscardRecord(_SequencedEnvelope):
    """The reverted settlement of an iteration."""

    type: Literal["discard"] = Field(description="Record type discriminator.")
    # pyrefly: ignore[bad-override-mutable-attribute] -- pydantic narrows optional to required
    seq: _NonNegativeInt = Field(description="Iteration sequence number this discard settles.")


class HookRecord(_SequencedEnvelope):
    """One hook invocation around an iteration."""

    type: Literal["hook"] = Field(description="Record type discriminator.")
    stage: HookStage = Field(description="Whether the hook ran before or after the iteration.")
    # pyrefly: ignore[bad-override-mutable-attribute] -- pydantic narrows optional to required
    seq: _NonNegativeInt = Field(
        description="Iteration sequence number this hook is associated with."
    )
    exit_code: Annotated[int, BeforeValidator(_coerce_integer)] = Field(
        description="Process exit code of the hook command."
    )
    duration_ms: _Number = Field(description="Wall-clock milliseconds the hook command ran.")
    stdout_bytes: _NonNegativeInt = Field(description="Bytes the hook command wrote to stdout.")
    stderr_bytes: _OptNonNegativeInt = Field(
        default=None, description="Bytes the hook command wrote to stderr."
    )
    timed_out: bool = Field(description="Whether the hook command exceeded its timeout.")


# ---------------------------------------------------------------------------
# Terminal records
# ---------------------------------------------------------------------------


class FinalizeRecord(_RecordEnvelope):
    """Closes a session: the branch and squash commit its kept work collapsed onto."""

    type: Literal["finalize"] = Field(description="Record type discriminator.")
    branch: str = Field(description="Git branch the finalize squash landed on.")
    commit: str = Field(description="Git commit SHA of the squash commit.")
    message: str = Field(description="Commit message of the squash commit.")


class StopRecord(_RecordEnvelope):
    """A user-requested stop: halts the loop without finalizing the session."""

    type: Literal["stop"] = Field(description="Record type discriminator.")
    message: Annotated[str, Field(min_length=1, description="Reason the stop was requested.")]


class CommandRecord(_SequencedEnvelope):
    """A CLI command invocation: name, arguments, exit code, and optional reason.

    The ``exit_code`` / ``reason`` pairing is constrained: a zero exit has no
    reason, and a non-zero exit always carries one.
    """

    type: Literal["command"] = Field(description="Record type discriminator.")
    name: Annotated[str, Field(min_length=1, description="CLI command name.")]
    args: dict[str, object] = Field(description="Arguments the command was invoked with.")
    exit_code: Literal[0, 1, 2] = Field(
        description="Process-style exit code: 0 success, 1 or 2 failure."
    )
    reason: Annotated[
        CommandReason | SkipJsonSchema[None],
        BeforeValidator(_reject_none),
    ] = Field(default=None, description="Why the command exited non-zero, when it did.")
    duration_ms: _NonNegativeInt = Field(description="Wall-clock milliseconds the command took.")
    traceparent: _OptStr = Field(
        default=None, description="W3C Trace Context traceparent header for distributed tracing."
    )

    @model_validator(mode="after")
    def _validate_exit_reason_consistency(self) -> Self:
        if self.exit_code == 0 and self.reason is not None:
            msg = (
                f"command.exit_code is 0 but reason is set to {self.reason!r}; "
                "a successful command must not carry a reason"
            )
            raise ValueError(msg)
        if self.exit_code != 0 and self.reason is None:
            msg = (
                f"command.exit_code is {self.exit_code} but reason is missing; "
                "a failed command must carry a reason"
            )
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Wire codec
# ---------------------------------------------------------------------------

#: The discriminated union of every record type a session JSONL line can decode to.
type SessionLogRecord = (
    SessionRecord
    | BaselineRecord
    | IterationRecord
    | KeepRecord
    | DiscardRecord
    | HookRecord
    | FinalizeRecord
    | StopRecord
    | CommandRecord
)


def record_to_wire(record: SessionLogRecord) -> dict[str, object]:
    """Render a session-log model back to its snake_case wire dict.

    Optional fields whose value is ``None`` are omitted, so a parsed record and
    its serialization round-trip. The exception is ``delta_pct`` on a metric
    verdict and on an iteration's primary: it is always emitted, carrying JSON
    ``null`` when the delta is undefined.

    Args:
        record: The parsed session-log model to render.

    Returns:
        The wire-format dict, ready for JSON serialization.
    """
    return record.model_dump(mode="json", exclude_none=True)
