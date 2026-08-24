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

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError
from pydantic_core import ErrorDetails

from gymrat_py.errors import GymratError
from gymrat_py.session.schema import (
    SCHEMA_VERSION,
    HookStage,
    KeepReason,
    KeepStatus,
    Method,
    Outcome,
    PrimaryKind,
    Verdict,
)
from gymrat_py.session.workspace import BaselineRef, Worktrees

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
    "SessionConfig",
    "SessionHooks",
    "SessionLogRecord",
    "SessionRecord",
    "Worktrees",
    "parse_record",
    "record_to_wire",
]

# A single round of raw samples: one flat metric-name -> value map. A value is a
# JSON number (int or float); the int/float distinction is preserved so a
# round-trip through the wire returns the same literal it was given.
SampleRound = dict[str, float | int]


# ---------------------------------------------------------------------------
# Public frozen dataclasses (never pydantic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionHooks:
    """The hook commands a session was started with; an absent stage ran nothing."""

    before: str | None = None
    after: str | None = None


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """The settled configuration a session was opened with, kept for provenance."""

    bench: str
    adapter: str
    samples: int
    timeout_seconds: int
    primary: str
    prepare: str | None = None
    filter: str | None = None
    hooks: SessionHooks | None = None


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """Opens a session log: identity, worktrees, and a config snapshot."""

    type: Literal["session"]
    schema_version: int
    session_id: str
    created_at: str
    baseline: BaselineRef
    branch: str
    worktrees: Worktrees
    config: SessionConfig


@dataclass(frozen=True, slots=True)
class BaselineRecord:
    """A labelled set of baseline sample rounds."""

    type: Literal["baseline"]
    at: str
    label: str
    samples: tuple[SampleRound, ...]


@dataclass(frozen=True, slots=True)
class MetricVerdict:
    """How one metric moved, with the statistics behind the judgement.

    ``delta_pct`` is ``None`` when a zero baseline median left the ratio
    undefined; the field is always present so a dropped delta reads as a broken
    writer, not a degenerate measurement.
    """

    delta_pct: float | int | None
    verdict: Verdict
    method: Method
    gating: bool
    confirmed: bool
    p: float | int | None = None
    noise_pct: float | int | None = None


@dataclass(frozen=True, slots=True)
class PairedSamples:
    """The experiment and baseline sample rounds measured in one iteration."""

    experiment: tuple[SampleRound, ...]
    baseline: tuple[SampleRound, ...]


@dataclass(frozen=True, slots=True)
class Confirm:
    """A confirmation rerun: which metrics it re-measured, and the samples it took.

    ``absent`` names metrics the rerun was asked about but skipped; it is
    ``None`` on a log written before the field existed.
    """

    ran: bool
    filtered: tuple[str, ...]
    samples: PairedSamples
    absent: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class IterationPrimary:
    """The primary an iteration was judged on -- the geomean, or a named metric."""

    kind: PrimaryKind
    delta_pct: float | int | None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """One measured edit: raw samples, per-metric verdicts, and the outcome."""

    type: Literal["iteration"]
    seq: int
    at: str
    samples: PairedSamples
    metrics: dict[str, MetricVerdict]
    primary: IterationPrimary
    outcome: Outcome
    target_reached: bool
    confirm: Confirm | None = None


@dataclass(frozen=True, slots=True)
class KeepChecks:
    """The outcome of a keep's configured checks, with relayed output sizes."""

    configured: bool
    passed: bool | None = None
    stdout_bytes: int | None = None
    stderr_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class KeepRecord:
    """The settlement of an iteration: committed, or blocked with a reason."""

    type: Literal["keep"]
    seq: int
    at: str
    status: KeepStatus
    checks: KeepChecks
    commit: str | None = None
    message: str | None = None
    reason: KeepReason | None = None


@dataclass(frozen=True, slots=True)
class DiscardRecord:
    """The reverted settlement of an iteration."""

    type: Literal["discard"]
    seq: int
    at: str


@dataclass(frozen=True, slots=True)
class HookRecord:
    """One hook invocation around an iteration."""

    type: Literal["hook"]
    stage: HookStage
    seq: int
    exit_code: int
    duration_ms: float | int
    stdout_bytes: int
    timed_out: bool
    stderr_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class FinalizeRecord:
    """Closes a session: the branch and squash commit its kept work collapsed onto."""

    type: Literal["finalize"]
    at: str
    branch: str
    commit: str
    message: str


#: Any line of a session log, discriminated on ``type``.
SessionLogRecord = (
    SessionRecord
    | BaselineRecord
    | IterationRecord
    | KeepRecord
    | DiscardRecord
    | HookRecord
    | FinalizeRecord
)


# ---------------------------------------------------------------------------
# Value coercion / rejection for the internal pydantic models
# ---------------------------------------------------------------------------

_NULL_MESSAGE = "value must not be null"


def _reject_none(value: object) -> object:
    """Reject an explicitly-provided ``null``.

    An absent key falls back to the ``None`` default without invoking this
    validator; only a present ``null`` reaches here, so an optional-but-not-
    nullable field rejects it while an absent key stays ``None``.
    """
    if value is None:
        raise ValueError(_NULL_MESSAGE)
    return value


def _coerce_integer(value: object) -> object:
    """Reject ``null`` and fold an integral float into ``int``.

    Folding ``5.0`` to ``5`` lets it satisfy strict integer validation; every
    other value is passed through for the model to accept or reject.
    """
    if value is None:
        raise ValueError(_NULL_MESSAGE)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


# ---------------------------------------------------------------------------
# Internal pydantic models (never exposed)
# ---------------------------------------------------------------------------

_STRICT = ConfigDict(strict=True, extra="forbid")

# A JSON number that keeps its int-vs-float identity through validation.
_Number = int | float

# Optional fields: the key may be absent (default None), but a present `null` is
# rejected so the error names the field with its real type expectation.
_OptStr = Annotated[str | None, BeforeValidator(_reject_none), Field(default=None)]
_OptNonEmptyStr = Annotated[
    str | None, BeforeValidator(_reject_none), Field(default=None, min_length=1)
]
_OptBool = Annotated[bool | None, BeforeValidator(_reject_none), Field(default=None)]
_OptNumber = Annotated[_Number | None, BeforeValidator(_reject_none), Field(default=None)]

# Delta percentages are required but genuinely nullable: a zero baseline median
# leaves the ratio undefined, which reaches the log as JSON null.
_DeltaPct = Annotated[_Number | None, Field(alias="deltaPct")]

_PositiveInt = Annotated[int, BeforeValidator(_coerce_integer), Field(ge=1)]
_NonNegativeInt = Annotated[int, BeforeValidator(_coerce_integer), Field(ge=0)]


class _BaselineRefModel(BaseModel):
    model_config = _STRICT

    ref: str
    sha: str


class _WorktreesModel(BaseModel):
    model_config = _STRICT

    experiment: str
    baseline: str


class _SessionHooksModel(BaseModel):
    model_config = _STRICT

    before: _OptNonEmptyStr
    after: _OptNonEmptyStr


class _SessionConfigModel(BaseModel):
    model_config = _STRICT

    bench: str
    prepare: _OptStr
    adapter: str
    samples: _PositiveInt
    timeout_seconds: Annotated[
        int, BeforeValidator(_coerce_integer), Field(ge=1, alias="timeoutSeconds")
    ]
    primary: str
    filter: _OptStr
    hooks: Annotated[_SessionHooksModel | None, BeforeValidator(_reject_none), Field(default=None)]


class _SessionModel(BaseModel):
    model_config = _STRICT

    type: Literal["session"]
    schema_version: Annotated[Literal[1], Field(alias="schemaVersion")]
    session_id: Annotated[str, Field(alias="sessionId")]
    created_at: Annotated[str, Field(alias="createdAt")]
    baseline: _BaselineRefModel
    branch: str
    worktrees: _WorktreesModel
    config: _SessionConfigModel


class _BaselineModel(BaseModel):
    model_config = _STRICT

    type: Literal["baseline"]
    at: str
    label: str
    samples: list[dict[str, _Number]]


class _MetricVerdictModel(BaseModel):
    model_config = _STRICT

    delta_pct: _DeltaPct
    verdict: Verdict
    method: Method
    p: _OptNumber
    noise_pct: Annotated[
        _Number | None, BeforeValidator(_reject_none), Field(default=None, alias="noisePct")
    ]
    gating: bool
    confirmed: bool


class _PairedSamplesModel(BaseModel):
    model_config = _STRICT

    experiment: list[dict[str, _Number]]
    baseline: list[dict[str, _Number]]


class _ConfirmModel(BaseModel):
    model_config = _STRICT

    ran: bool
    filtered: list[str]
    absent: Annotated[list[str] | None, BeforeValidator(_reject_none), Field(default=None)]
    samples: _PairedSamplesModel


class _IterationPrimaryModel(BaseModel):
    model_config = _STRICT

    kind: PrimaryKind
    name: _OptStr
    delta_pct: _DeltaPct


class _IterationModel(BaseModel):
    model_config = _STRICT

    type: Literal["iteration"]
    seq: _PositiveInt
    at: str
    samples: _PairedSamplesModel
    metrics: dict[str, _MetricVerdictModel]
    confirm: Annotated[_ConfirmModel | None, BeforeValidator(_reject_none), Field(default=None)]
    primary: _IterationPrimaryModel
    outcome: Outcome
    target_reached: Annotated[bool, Field(alias="targetReached")]


class _KeepChecksModel(BaseModel):
    model_config = _STRICT

    configured: bool
    passed: _OptBool
    stdout_bytes: Annotated[
        int | None, BeforeValidator(_coerce_integer), Field(default=None, ge=0, alias="stdoutBytes")
    ]
    stderr_bytes: Annotated[
        int | None, BeforeValidator(_coerce_integer), Field(default=None, ge=0, alias="stderrBytes")
    ]


class _KeepModel(BaseModel):
    model_config = _STRICT

    type: Literal["keep"]
    seq: _NonNegativeInt
    at: str
    status: KeepStatus
    commit: _OptStr
    message: _OptStr
    reason: Annotated[KeepReason | None, BeforeValidator(_reject_none), Field(default=None)]
    checks: _KeepChecksModel


class _DiscardModel(BaseModel):
    model_config = _STRICT

    type: Literal["discard"]
    seq: _NonNegativeInt
    at: str


class _HookModel(BaseModel):
    model_config = _STRICT

    type: Literal["hook"]
    stage: HookStage
    seq: _NonNegativeInt
    exit_code: Annotated[int, BeforeValidator(_coerce_integer), Field(alias="exitCode")]
    duration_ms: Annotated[_Number, Field(alias="durationMs")]
    stdout_bytes: Annotated[int, BeforeValidator(_coerce_integer), Field(alias="stdoutBytes")]
    stderr_bytes: Annotated[
        int | None, BeforeValidator(_coerce_integer), Field(default=None, alias="stderrBytes")
    ]
    timed_out: Annotated[bool, Field(alias="timedOut")]


class _FinalizeModel(BaseModel):
    model_config = _STRICT

    type: Literal["finalize"]
    at: str
    branch: str
    commit: str
    message: str


# ---------------------------------------------------------------------------
# Validation-error translation
# ---------------------------------------------------------------------------

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
_METHOD = '"signed-rank", "band" or "exact"'
_KIND = '"geomean" or "metric"'
_OUTCOME = '"improved", "regressed" or "no-signal"'
_STATUS = '"committed" or "blocked"'
_REASON = '"checks-failed", "gating-regression", "nothing-measured" or "nothing-to-commit"'
_STAGE = '"before" or "after"'

# Phrase describing the value each schema location expects, keyed by
# ``(record_type, *normalized_loc)``. The record type disambiguates fields that
# share a name across records (``seq`` is positive on an iteration but
# non-negative on a keep). ``_normalize_loc`` collapses array indices and
# dynamic map keys (metric names) to ``"*"`` before lookup.
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


def _normalize_loc(loc: tuple[str, ...]) -> tuple[str, ...]:
    """Collapse array indices and dynamic map keys to ``"*"`` for phrase lookup.

    An array index is a purely-numeric segment. A dynamic map key is either a
    metric name directly under ``metrics``, or a metric name inside a sample
    round -- the segment following an index. Both are collapsed so one phrase
    entry covers every concrete name.
    """
    out: list[str] = []
    prev_index = False
    for i, segment in enumerate(loc):
        if segment.isdigit():
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


def _describe_key(loc: tuple[str, ...]) -> str:
    """Join an error location into a dotted key, quoting any empty segment."""
    return ".".join('""' if part == "" else part for part in loc)


def _message_for_error(error: ErrorDetails, record_type: str) -> str:
    """Translate one pydantic error into a session-record problem string."""
    loc = tuple(str(part) for part in error["loc"])
    if error["type"] == "extra_forbidden":
        return f"Unknown session record key: {_describe_key(loc)}"
    phrase = _PHRASES.get((record_type, *_normalize_loc(loc)), "a valid value")
    got = "undefined" if error["type"] == "missing" else json.dumps(error["input"])
    return f"Invalid session record value for {_describe_key(loc)}: expected {phrase}, got {got}"


def _drop_prefix_errors(errors: list[ErrorDetails]) -> list[ErrorDetails]:
    """Drop any error whose location is a strict prefix of another's.

    When a parent and its child both fail, only the more specific child error is
    worth reporting; the parent prefix is redundant noise.
    """
    locs = [error["loc"] for error in errors]
    kept: list[ErrorDetails] = []
    for index, error in enumerate(errors):
        loc = error["loc"]
        is_prefix = any(
            other != index and len(candidate) > len(loc) and candidate[: len(loc)] == loc
            for other, candidate in enumerate(locs)
        )
        if not is_prefix:
            kept.append(error)
    return kept


# ---------------------------------------------------------------------------
# Model -> public dataclass converters
# ---------------------------------------------------------------------------


def _rounds(rounds: list[dict[str, _Number]]) -> tuple[SampleRound, ...]:
    return tuple(dict(round_) for round_ in rounds)


def _to_paired(model: _PairedSamplesModel) -> PairedSamples:
    return PairedSamples(experiment=_rounds(model.experiment), baseline=_rounds(model.baseline))


def _to_verdict(model: _MetricVerdictModel) -> MetricVerdict:
    return MetricVerdict(
        delta_pct=model.delta_pct,
        verdict=model.verdict,
        method=model.method,
        gating=model.gating,
        confirmed=model.confirmed,
        p=model.p,
        noise_pct=model.noise_pct,
    )


def _to_session(model: _SessionModel) -> SessionRecord:
    hooks = (
        SessionHooks(before=model.config.hooks.before, after=model.config.hooks.after)
        if model.config.hooks is not None
        else None
    )
    config = SessionConfig(
        bench=model.config.bench,
        adapter=model.config.adapter,
        samples=model.config.samples,
        timeout_seconds=model.config.timeout_seconds,
        primary=model.config.primary,
        prepare=model.config.prepare,
        filter=model.config.filter,
        hooks=hooks,
    )
    return SessionRecord(
        type=model.type,
        schema_version=model.schema_version,
        session_id=model.session_id,
        created_at=model.created_at,
        baseline=BaselineRef(ref=model.baseline.ref, sha=model.baseline.sha),
        branch=model.branch,
        worktrees=Worktrees(
            experiment=model.worktrees.experiment, baseline=model.worktrees.baseline
        ),
        config=config,
    )


def _to_baseline(model: _BaselineModel) -> BaselineRecord:
    return BaselineRecord(
        type=model.type, at=model.at, label=model.label, samples=_rounds(model.samples)
    )


def _to_iteration(model: _IterationModel) -> IterationRecord:
    confirm = (
        Confirm(
            ran=model.confirm.ran,
            filtered=tuple(model.confirm.filtered),
            samples=_to_paired(model.confirm.samples),
            absent=tuple(model.confirm.absent) if model.confirm.absent is not None else None,
        )
        if model.confirm is not None
        else None
    )
    return IterationRecord(
        type=model.type,
        seq=model.seq,
        at=model.at,
        samples=_to_paired(model.samples),
        metrics={name: _to_verdict(entry) for name, entry in model.metrics.items()},
        primary=IterationPrimary(
            kind=model.primary.kind, delta_pct=model.primary.delta_pct, name=model.primary.name
        ),
        outcome=model.outcome,
        target_reached=model.target_reached,
        confirm=confirm,
    )


def _to_keep(model: _KeepModel) -> KeepRecord:
    return KeepRecord(
        type=model.type,
        seq=model.seq,
        at=model.at,
        status=model.status,
        checks=KeepChecks(
            configured=model.checks.configured,
            passed=model.checks.passed,
            stdout_bytes=model.checks.stdout_bytes,
            stderr_bytes=model.checks.stderr_bytes,
        ),
        commit=model.commit,
        message=model.message,
        reason=model.reason,
    )


def _to_discard(model: _DiscardModel) -> DiscardRecord:
    return DiscardRecord(type=model.type, seq=model.seq, at=model.at)


def _to_hook(model: _HookModel) -> HookRecord:
    return HookRecord(
        type=model.type,
        stage=model.stage,
        seq=model.seq,
        exit_code=model.exit_code,
        duration_ms=model.duration_ms,
        stdout_bytes=model.stdout_bytes,
        timed_out=model.timed_out,
        stderr_bytes=model.stderr_bytes,
    )


def _to_finalize(model: _FinalizeModel) -> FinalizeRecord:
    return FinalizeRecord(
        type=model.type,
        at=model.at,
        branch=model.branch,
        commit=model.commit,
        message=model.message,
    )


# ---------------------------------------------------------------------------
# Parsing: wire value -> typed dataclass
# ---------------------------------------------------------------------------


def _parser_for(
    record_type: str,
    model_cls: type[BaseModel],
    to_dataclass: Callable[..., SessionLogRecord],
) -> Callable[[dict[str, object]], SessionLogRecord]:
    """Build a parser that validates against ``model_cls`` and words its failures."""

    def parse(value: dict[str, object]) -> SessionLogRecord:
        try:
            model = model_cls.model_validate(value)
        except ValidationError as exc:
            error = _drop_prefix_errors(exc.errors())[0]
            raise GymratError(_message_for_error(error, record_type)) from exc
        return to_dataclass(model)

    return parse


# One parser per `type` a session log line can carry. The hint in the
# unknown-type error is read off these keys so it cannot name a set the parser
# no longer accepts.
_PARSERS: dict[str, Callable[[dict[str, object]], SessionLogRecord]] = {
    "session": _parser_for("session", _SessionModel, _to_session),
    "baseline": _parser_for("baseline", _BaselineModel, _to_baseline),
    "iteration": _parser_for("iteration", _IterationModel, _to_iteration),
    "keep": _parser_for("keep", _KeepModel, _to_keep),
    "discard": _parser_for("discard", _DiscardModel, _to_discard),
    "hook": _parser_for("hook", _HookModel, _to_hook),
    "finalize": _parser_for("finalize", _FinalizeModel, _to_finalize),
}

_MISSING = object()


def parse_record(value: object) -> SessionLogRecord:
    """Validate one decoded session-log line against the schema for its ``type``.

    Args:
        value: A decoded-JSON value -- expected to be a mapping carrying a
            ``type`` discriminator.

    Returns:
        The typed dataclass for the matching record type.

    Raises:
        GymratError: When ``value`` is not an object, carries no recognized
            ``type``, or violates that type's schema.
    """
    if not isinstance(value, dict):
        message = f"Invalid session record: expected a JSON object, got {json.dumps(value)}"
        raise GymratError(message)
    type_value = value.get("type", _MISSING)
    if isinstance(type_value, str) and type_value in _PARSERS:
        return _PARSERS[type_value](value)
    rendered = "undefined" if type_value is _MISSING else json.dumps(type_value)
    hint = "Expected one of: " + ", ".join(_PARSERS) + "."
    message = f"Unknown session record type: {rendered}"
    raise GymratError(message, hint=hint)


# ---------------------------------------------------------------------------
# Serialization: typed dataclass -> wire dict
# ---------------------------------------------------------------------------


def _rounds_to_wire(rounds: tuple[SampleRound, ...]) -> list[SampleRound]:
    return [dict(round_) for round_ in rounds]


def _paired_to_wire(samples: PairedSamples) -> dict[str, object]:
    return {
        "experiment": _rounds_to_wire(samples.experiment),
        "baseline": _rounds_to_wire(samples.baseline),
    }


def _verdict_to_wire(verdict: MetricVerdict) -> dict[str, object]:
    out: dict[str, object] = {
        "deltaPct": verdict.delta_pct,
        "verdict": verdict.verdict,
        "method": verdict.method,
    }
    if verdict.p is not None:
        out["p"] = verdict.p
    if verdict.noise_pct is not None:
        out["noisePct"] = verdict.noise_pct
    out["gating"] = verdict.gating
    out["confirmed"] = verdict.confirmed
    return out


def _session_to_wire(record: SessionRecord) -> dict[str, object]:
    config = record.config
    config_wire: dict[str, object] = {"bench": config.bench}
    if config.prepare is not None:
        config_wire["prepare"] = config.prepare
    config_wire["adapter"] = config.adapter
    config_wire["samples"] = config.samples
    config_wire["timeoutSeconds"] = config.timeout_seconds
    config_wire["primary"] = config.primary
    if config.filter is not None:
        config_wire["filter"] = config.filter
    if config.hooks is not None:
        hooks_wire: dict[str, object] = {}
        if config.hooks.before is not None:
            hooks_wire["before"] = config.hooks.before
        if config.hooks.after is not None:
            hooks_wire["after"] = config.hooks.after
        config_wire["hooks"] = hooks_wire
    return {
        "type": record.type,
        "schemaVersion": record.schema_version,
        "sessionId": record.session_id,
        "createdAt": record.created_at,
        "baseline": {"ref": record.baseline.ref, "sha": record.baseline.sha},
        "branch": record.branch,
        "worktrees": {
            "experiment": record.worktrees.experiment,
            "baseline": record.worktrees.baseline,
        },
        "config": config_wire,
    }


def _baseline_to_wire(record: BaselineRecord) -> dict[str, object]:
    return {
        "type": record.type,
        "at": record.at,
        "label": record.label,
        "samples": _rounds_to_wire(record.samples),
    }


def _iteration_to_wire(record: IterationRecord) -> dict[str, object]:
    out: dict[str, object] = {
        "type": record.type,
        "seq": record.seq,
        "at": record.at,
        "samples": _paired_to_wire(record.samples),
        "metrics": {name: _verdict_to_wire(entry) for name, entry in record.metrics.items()},
    }
    if record.confirm is not None:
        confirm_wire: dict[str, object] = {
            "ran": record.confirm.ran,
            "filtered": list(record.confirm.filtered),
        }
        if record.confirm.absent is not None:
            confirm_wire["absent"] = list(record.confirm.absent)
        confirm_wire["samples"] = _paired_to_wire(record.confirm.samples)
        out["confirm"] = confirm_wire
    primary_wire: dict[str, object] = {"kind": record.primary.kind}
    if record.primary.name is not None:
        primary_wire["name"] = record.primary.name
    primary_wire["deltaPct"] = record.primary.delta_pct
    out["primary"] = primary_wire
    out["outcome"] = record.outcome
    out["targetReached"] = record.target_reached
    return out


def _keep_to_wire(record: KeepRecord) -> dict[str, object]:
    checks_wire: dict[str, object] = {"configured": record.checks.configured}
    if record.checks.passed is not None:
        checks_wire["passed"] = record.checks.passed
    if record.checks.stdout_bytes is not None:
        checks_wire["stdoutBytes"] = record.checks.stdout_bytes
    if record.checks.stderr_bytes is not None:
        checks_wire["stderrBytes"] = record.checks.stderr_bytes
    out: dict[str, object] = {
        "type": record.type,
        "seq": record.seq,
        "at": record.at,
        "status": record.status,
    }
    if record.commit is not None:
        out["commit"] = record.commit
    if record.message is not None:
        out["message"] = record.message
    if record.reason is not None:
        out["reason"] = record.reason
    out["checks"] = checks_wire
    return out


def _discard_to_wire(record: DiscardRecord) -> dict[str, object]:
    return {"type": record.type, "seq": record.seq, "at": record.at}


def _hook_to_wire(record: HookRecord) -> dict[str, object]:
    out: dict[str, object] = {
        "type": record.type,
        "stage": record.stage,
        "seq": record.seq,
        "exitCode": record.exit_code,
        "durationMs": record.duration_ms,
        "stdoutBytes": record.stdout_bytes,
    }
    if record.stderr_bytes is not None:
        out["stderrBytes"] = record.stderr_bytes
    out["timedOut"] = record.timed_out
    return out


def _finalize_to_wire(record: FinalizeRecord) -> dict[str, object]:
    return {
        "type": record.type,
        "at": record.at,
        "branch": record.branch,
        "commit": record.commit,
        "message": record.message,
    }


def record_to_wire(record: SessionLogRecord) -> dict[str, object]:
    """Render a session-log dataclass back to its camelCase wire dict.

    Optional fields whose value is ``None`` are omitted, so a parsed record and
    its serialization round-trip. The exception is ``deltaPct`` on a metric
    verdict and on an iteration's primary: it is always emitted, carrying JSON
    ``null`` when the delta is undefined.
    """
    match record:
        case SessionRecord():
            wire = _session_to_wire(record)
        case BaselineRecord():
            wire = _baseline_to_wire(record)
        case IterationRecord():
            wire = _iteration_to_wire(record)
        case KeepRecord():
            wire = _keep_to_wire(record)
        case DiscardRecord():
            wire = _discard_to_wire(record)
        case HookRecord():
            wire = _hook_to_wire(record)
        case FinalizeRecord():
            wire = _finalize_to_wire(record)
    return wire
