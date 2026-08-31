"""Public frozen dataclasses for session log records."""

from dataclasses import dataclass
from typing import Literal

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


type SessionLogRecord = (
    SessionRecord
    | BaselineRecord
    | IterationRecord
    | KeepRecord
    | DiscardRecord
    | HookRecord
    | FinalizeRecord
)
