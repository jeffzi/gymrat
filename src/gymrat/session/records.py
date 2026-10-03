"""Schemas and parsing for the lines of a session JSONL log.

Each line of a session log is one record, discriminated on its ``type`` field.
Records are frozen pydantic models with ``strict=True`` and ``extra="forbid"``.
Wire keys are the Python attribute names (snake_case) -- no alias generator,
except :attr:`SessionRecord.schema_version`, which aliases to ``schema`` because
``schema`` collides with :meth:`BaseModel.schema`.

Two entry points bridge the two forms:

- :func:`parse_record` validates a decoded-JSON value into the typed model
  for its ``type``, raising a :class:`GymratError` worded for a session log; it
  translates pydantic validation errors into problem strings.
- :func:`record_to_wire` renders a model back to its snake_case wire dict,
  the same shape the store writes. Optional fields whose value is ``None`` are
  omitted, except ``delta_pct`` (on a metric verdict and on an iteration's
  primary), which is always present and serializes ``None`` as JSON ``null``.

Inbound, a log line goes to a wire value through :func:`decode_log_line`, which
refuses non-finite numbers, and then to a typed model through
:func:`parse_record`. Outbound, the store renders a model as a compact JSON line
with ``model_dump_json(exclude_none=True)``.
"""

import json
import math
from collections.abc import Callable
from contextvars import ContextVar
from typing import Annotated, Literal, NoReturn, Self, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    Strict,
    TypeAdapter,
    ValidationError,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema, WithJsonSchema
from pydantic_core import ErrorDetails

from gymrat.errors import GymratError
from gymrat.pydantic_errors import (
    UNKNOWN_SHAPE_PHRASE,
    VALUE_ERROR_PREFIX,
    coerce_integer,
    describe_key,
    drop_prefix_errors,
    phrase_for_error,
)
from gymrat.session.schema import (
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
from gymrat.session.workspace import BaselineRef, Worktrees

# ---------------------------------------------------------------------------
# Validation and coercion helpers
# ---------------------------------------------------------------------------


def _coerce[T](cls: type[T]) -> Callable[[object], T]:
    """Build a validator that coerces a dict into ``cls`` via lax-mode validation.

    Strict mode rejects dict-to-dataclass coercion, so the returned validator
    uses a ``TypeAdapter`` in lax mode to validate and construct the dataclass,
    preserving field-level error locations.

    Args:
        cls: The dataclass type to coerce values into.

    Returns:
        A before-validator callable that coerces dicts into ``cls``.
    """
    adapter = TypeAdapter(cls)

    def coerce(value: object) -> T:
        if isinstance(value, cls):
            return value
        return adapter.validate_python(value)

    return coerce


_coerce_baseline_ref = _coerce(BaselineRef)
_coerce_worktrees = _coerce(Worktrees)


_RECORD_CONFIG = ConfigDict(
    strict=True,
    extra="forbid",
    frozen=True,
    validate_by_name=True,
    serialize_by_alias=True,
)

_wire_validation: ContextVar[bool] = ContextVar("_wire_validation", default=False)


def _not_null(member: object) -> BeforeValidator:
    """Build a validator that holds an explicit wire ``null`` to ``member``'s type check.

    An absent key falls back to the ``None`` default without invoking the
    validator; only a present ``null`` reaches it, so an optional-but-not-
    nullable field rejects it while an absent key stays ``None``. The ``null``
    fails with the error pydantic reports for ``member`` itself -- a
    ``string_type`` for ``str``, a ``literal_error`` for a ``Literal`` -- so
    the problem message names the field's own type.

    The check is skipped when the ``_wire_validation`` context variable is not
    set, so Python-side construction with ``field=None`` passes through.

    Args:
        member: The non-null type the optional field holds.

    Returns:
        A before-validator for an ``Annotated`` optional of ``member``.
    """
    adapter = TypeAdapter(member)

    def check(value: object) -> object:
        if value is None and _wire_validation.get():
            # pydantic-core re-reports a ValidationError raised in a validator
            # as its own line errors, keeping their type, ctx and input.
            adapter.validate_python(value)
        return value

    return BeforeValidator(check)


# ``float`` leads so a rejected value's first union error is ``float_type``, which
# the problem message words as "a number". Smart-mode union validation still keeps
# an ``int`` input as ``int``, since that branch matches its type exactly.
_Number = float | int

_OptStr = Annotated[str | SkipJsonSchema[None], _not_null(str)]
_OptNonEmptyStr = Annotated[str | SkipJsonSchema[None], _not_null(str), Field(min_length=1)]
_OptBool = Annotated[bool | SkipJsonSchema[None], _not_null(bool)]
_OptNumber = Annotated[
    _Number | SkipJsonSchema[None],
    _not_null(_Number),
    WithJsonSchema({"type": "number"}),
]

_PositiveInt = Annotated[int, Field(ge=1), BeforeValidator(coerce_integer)]
_NonNegativeInt = Annotated[int, Field(ge=0), BeforeValidator(coerce_integer)]
_OptNonNegativeInt = Annotated[_NonNegativeInt | SkipJsonSchema[None], _not_null(int)]

# Lax mode on the tuple itself lets a JSON array fill it; the items keep the
# model's strict mode, so a bool or numeric string inside is still rejected.
_NameList = Annotated[tuple[str, ...], Strict(strict=False)]
_SampleRounds = Annotated[tuple[dict[str, _Number], ...], Strict(strict=False)]


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
        _not_null(int),
        BeforeValidator(coerce_integer),
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
    timeout_seconds: _PositiveInt = Field(description="Maximum seconds per bench invocation.")
    primary: str = Field(description="Primary metric or aggregation method for judging iterations.")
    filter: _OptStr = Field(default=None, description="Filter expression for selecting benchmarks.")
    hooks: Annotated[
        SessionHooks | SkipJsonSchema[None],
        _not_null(SessionHooks),
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

    delta_pct: _Number | None = Field(
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
    filtered: _NameList = Field(description="Metric names the rerun was filtered to.")
    absent: Annotated[
        _NameList | SkipJsonSchema[None],
        _not_null(_NameList),
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
        _not_null(Confirm),
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
        _not_null(KeepReason),
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
    exit_code: Annotated[int, BeforeValidator(coerce_integer)] = Field(
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
        _not_null(CommandReason),
    ] = Field(default=None, description="Why the command exited non-zero, when it did.")
    duration_ms: _NonNegativeInt = Field(description="Wall-clock milliseconds the command took.")
    origin: CommandOrigin = Field(
        default="cli",
        description=(
            "What invoked the command: 'tool' when the supervised agent ran it through the "
            "in-process tool host, 'cli' when it was run directly."
        ),
    )
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

#: Session-log record models, in ``SessionLogRecord`` union order.
SESSION_LOG_MODELS: tuple[type[BaseModel], ...] = get_args(SessionLogRecord.__value__)


def wire_type(model: type[BaseModel]) -> str:
    """Read the wire ``type`` literal a session-log model discriminates on.

    Args:
        model: A model with a ``type: Literal[...]`` discriminator field --
            typically a ``SessionLogRecord`` union member.

    Returns:
        The model's ``type`` literal value.
    """
    return get_args(model.model_fields["type"].annotation)[0]


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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


SESSION_LOG_ADAPTER: TypeAdapter[SessionLogRecord] = TypeAdapter(
    Annotated[SessionLogRecord, Field(discriminator="type")]
)
"""Validates a wire object into its session-log record and renders the union's JSON Schema."""


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
        return SESSION_LOG_ADAPTER.validate_python(value)
    except ValidationError as exc:
        errors = exc.errors()
        _raise_discriminator_error(errors, value)
        error = drop_prefix_errors(errors)[0]
        raise GymratError(message_for_error(error, value)) from exc
    finally:
        _wire_validation.reset(token)


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
    hint = "Expected one of: " + ", ".join(wire_type(member) for member in SESSION_LOG_MODELS) + "."
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
