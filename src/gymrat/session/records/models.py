"""Internal pydantic models for session log validation (never exposed)."""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

from gymrat.pydantic_errors import STRICT_FORBID, coerce_integer
from gymrat.session.schema import (
    HookStage,
    KeepReason,
    KeepStatus,
    Method,
    Outcome,
    PrimaryKind,
    Verdict,
)

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
    """Reject ``null``, then fold an integral float into ``int``.

    Composes :func:`_reject_none` with the shared :func:`coerce_integer` so
    JSON ``null`` is caught before the coercion pass.
    """
    _reject_none(value)
    return coerce_integer(value)


_Number = int | float

_OptStr = Annotated[str | None, BeforeValidator(_reject_none), Field(default=None)]
_OptNonEmptyStr = Annotated[
    str | None, BeforeValidator(_reject_none), Field(default=None, min_length=1)
]
_OptBool = Annotated[bool | None, BeforeValidator(_reject_none), Field(default=None)]
_OptNumber = Annotated[_Number | None, BeforeValidator(_reject_none), Field(default=None)]

_DeltaPct = Annotated[_Number | None, Field(alias="deltaPct")]

_PositiveInt = Annotated[int, BeforeValidator(_coerce_integer), Field(ge=1)]
_NonNegativeInt = Annotated[int, BeforeValidator(_coerce_integer), Field(ge=0)]


class BaselineRefModel(BaseModel):
    """Wire format for a session's baseline git reference."""

    model_config = STRICT_FORBID

    ref: str
    sha: str


class WorktreesModel(BaseModel):
    """Wire format for the session's worktree directory pair."""

    model_config = STRICT_FORBID

    experiment: str
    baseline: str


class SessionHooksModel(BaseModel):
    """Wire format for the session's before/after hook commands."""

    model_config = STRICT_FORBID

    before: _OptNonEmptyStr
    after: _OptNonEmptyStr


class SessionConfigModel(BaseModel):
    """Wire format for the session's resolved configuration snapshot."""

    model_config = STRICT_FORBID

    bench: str
    prepare: _OptStr
    adapter: str
    samples: _PositiveInt
    timeout_seconds: Annotated[
        int, BeforeValidator(_coerce_integer), Field(ge=1, alias="timeoutSeconds")
    ]
    primary: str
    filter: _OptStr
    hooks: Annotated[SessionHooksModel | None, BeforeValidator(_reject_none), Field(default=None)]


class SessionModel(BaseModel):
    """Wire format for the session-open log record."""

    model_config = STRICT_FORBID

    type: Literal["session"]
    schema_version: Annotated[Literal[1], Field(alias="schemaVersion")]
    session_id: Annotated[str, Field(alias="sessionId")]
    created_at: Annotated[str, Field(alias="createdAt")]
    baseline: BaselineRefModel
    branch: str
    worktrees: WorktreesModel
    config: SessionConfigModel


class BaselineModel(BaseModel):
    """Wire format for the baseline-measurement log record."""

    model_config = STRICT_FORBID

    type: Literal["baseline"]
    at: str
    label: str
    samples: list[dict[str, _Number]]


class MetricVerdictModel(BaseModel):
    """Wire format for a single metric's verdict within an iteration."""

    model_config = STRICT_FORBID

    delta_pct: _DeltaPct
    verdict: Verdict
    method: Method
    p: _OptNumber
    noise_pct: Annotated[
        _Number | None, BeforeValidator(_reject_none), Field(default=None, alias="noisePct")
    ]
    gating: bool
    confirmed: bool


class PairedSamplesModel(BaseModel):
    """Wire format for the experiment/baseline sample pair."""

    model_config = STRICT_FORBID

    experiment: list[dict[str, _Number]]
    baseline: list[dict[str, _Number]]


class ConfirmModel(BaseModel):
    """Wire format for the confirmation rerun within an iteration."""

    model_config = STRICT_FORBID

    ran: bool
    filtered: list[str]
    absent: Annotated[list[str] | None, BeforeValidator(_reject_none), Field(default=None)]
    samples: PairedSamplesModel


class IterationPrimaryModel(BaseModel):
    """Wire format for the iteration's primary-figure summary."""

    model_config = STRICT_FORBID

    kind: PrimaryKind
    name: _OptStr
    delta_pct: _DeltaPct


class IterationModel(BaseModel):
    """Wire format for the iteration log record."""

    model_config = STRICT_FORBID

    type: Literal["iteration"]
    seq: _PositiveInt
    at: str
    samples: PairedSamplesModel
    metrics: dict[str, MetricVerdictModel]
    confirm: Annotated[ConfirmModel | None, BeforeValidator(_reject_none), Field(default=None)]
    primary: IterationPrimaryModel
    outcome: Outcome
    target_reached: Annotated[bool, Field(alias="targetReached")]


class KeepChecksModel(BaseModel):
    """Wire format for the checks block within a keep record."""

    model_config = STRICT_FORBID

    configured: bool
    passed: _OptBool
    stdout_bytes: Annotated[
        int | None, BeforeValidator(_coerce_integer), Field(default=None, ge=0, alias="stdoutBytes")
    ]
    stderr_bytes: Annotated[
        int | None, BeforeValidator(_coerce_integer), Field(default=None, ge=0, alias="stderrBytes")
    ]


class KeepModel(BaseModel):
    """Wire format for the keep log record."""

    model_config = STRICT_FORBID

    type: Literal["keep"]
    seq: _NonNegativeInt
    at: str
    status: KeepStatus
    commit: _OptStr
    message: _OptStr
    reason: Annotated[KeepReason | None, BeforeValidator(_reject_none), Field(default=None)]
    checks: KeepChecksModel


class DiscardModel(BaseModel):
    """Wire format for the discard log record."""

    model_config = STRICT_FORBID

    type: Literal["discard"]
    seq: _NonNegativeInt
    at: str


class HookModel(BaseModel):
    """Wire format for the hook log record."""

    model_config = STRICT_FORBID

    type: Literal["hook"]
    stage: HookStage
    seq: _NonNegativeInt
    exit_code: Annotated[int, BeforeValidator(_coerce_integer), Field(alias="exitCode")]
    duration_ms: Annotated[_Number, Field(alias="durationMs")]
    stdout_bytes: Annotated[
        _NonNegativeInt, BeforeValidator(_coerce_integer), Field(alias="stdoutBytes")
    ]
    stderr_bytes: Annotated[
        _NonNegativeInt | None,
        BeforeValidator(_coerce_integer),
        Field(default=None, alias="stderrBytes"),
    ]
    timed_out: Annotated[bool, Field(alias="timedOut")]


class FinalizeModel(BaseModel):
    """Wire format for the finalize log record."""

    model_config = STRICT_FORBID

    type: Literal["finalize"]
    at: str
    branch: str
    commit: str
    message: str
