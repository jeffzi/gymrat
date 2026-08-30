"""Inbound codec: wire value to typed session-log dataclass.

Includes validation-error translation from pydantic errors to problem strings.
"""

import json
from collections.abc import Callable

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from gymrat.errors import GymratError
from gymrat.pydantic_errors import describe_key, drop_prefix_errors
from gymrat.session.records.models import (
    BaselineModel,
    DiscardModel,
    FinalizeModel,
    HookModel,
    IterationModel,
    KeepModel,
    MetricVerdictModel,
    PairedSamplesModel,
    SessionModel,
)
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
from gymrat.session.schema import SCHEMA_VERSION
from gymrat.session.workspace import BaselineRef, Worktrees

_Number = int | float


def _rounds(rounds: list[dict[str, _Number]]) -> tuple[SampleRound, ...]:
    return tuple(dict(round_) for round_ in rounds)


def _to_paired(model: PairedSamplesModel) -> PairedSamples:
    return PairedSamples(experiment=_rounds(model.experiment), baseline=_rounds(model.baseline))


def _to_verdict(model: MetricVerdictModel) -> MetricVerdict:
    return MetricVerdict(
        delta_pct=model.delta_pct,
        verdict=model.verdict,
        method=model.method,
        gating=model.gating,
        confirmed=model.confirmed,
        p=model.p,
        noise_pct=model.noise_pct,
    )


def _to_session(model: SessionModel) -> SessionRecord:
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


def _to_baseline(model: BaselineModel) -> BaselineRecord:
    return BaselineRecord(
        type=model.type, at=model.at, label=model.label, samples=_rounds(model.samples)
    )


def _to_iteration(model: IterationModel) -> IterationRecord:
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


def _to_keep(model: KeepModel) -> KeepRecord:
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


def _to_discard(model: DiscardModel) -> DiscardRecord:
    return DiscardRecord(type=model.type, seq=model.seq, at=model.at)


def _to_hook(model: HookModel) -> HookRecord:
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


def _to_finalize(model: FinalizeModel) -> FinalizeRecord:
    return FinalizeRecord(
        type=model.type,
        at=model.at,
        branch=model.branch,
        commit=model.commit,
        message=model.message,
    )


def _parser_for[M: BaseModel](
    record_type: str,
    model_cls: type[M],
    to_dataclass: Callable[[M], SessionLogRecord],
) -> Callable[[dict[str, object]], SessionLogRecord]:
    """Build a parser that validates against ``model_cls`` and words its failures."""

    def parse(value: dict[str, object]) -> SessionLogRecord:
        try:
            model = model_cls.model_validate(value)
        except ValidationError as exc:
            error = drop_prefix_errors(exc.errors())[0]
            raise GymratError(message_for_error(error, record_type)) from exc
        return to_dataclass(model)

    return parse


_PARSERS: dict[str, Callable[[dict[str, object]], SessionLogRecord]] = {
    "session": _parser_for("session", SessionModel, _to_session),
    "baseline": _parser_for("baseline", BaselineModel, _to_baseline),
    "iteration": _parser_for("iteration", IterationModel, _to_iteration),
    "keep": _parser_for("keep", KeepModel, _to_keep),
    "discard": _parser_for("discard", DiscardModel, _to_discard),
    "hook": _parser_for("hook", HookModel, _to_hook),
    "finalize": _parser_for("finalize", FinalizeModel, _to_finalize),
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
_METHOD = '"permutation", "band" or "exact"'
_KIND = '"geomean" or "metric"'
_OUTCOME = '"improved", "regressed" or "no-signal"'
_STATUS = '"committed" or "blocked"'
_REASON = '"checks-failed", "gating-regression", "nothing-measured" or "nothing-to-commit"'
_STAGE = '"before" or "after"'

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


def _normalize_loc(loc: tuple[int | str, ...]) -> tuple[str, ...]:
    """Collapse array indices and dynamic map keys to ``"*"`` for phrase lookup.

    An array index is an ``int`` segment (pydantic preserves the type). A dynamic
    map key is either a metric name directly under ``metrics``, or a metric name
    inside a sample round — the segment following an index. Both are collapsed so
    one phrase entry covers every concrete name.

    Using ``isinstance(segment, int)`` instead of ``str.isdigit`` keeps all-digit
    dict keys (e.g. a metric named ``"123"``) from being conflated with array
    indices.
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


def message_for_error(error: ErrorDetails, record_type: str) -> str:
    """Translate one pydantic error into a session-record problem string."""
    raw_loc = error["loc"]
    display_loc = tuple(str(part) for part in raw_loc)
    if error["type"] == "extra_forbidden":
        return f"Unknown session record key: {describe_key(display_loc)}"
    phrase = _PHRASES.get((record_type, *_normalize_loc(raw_loc)), "a valid value")
    got = "undefined" if error["type"] == "missing" else json.dumps(error["input"])
    key = describe_key(display_loc)
    return f"Invalid session record value for {key}: expected {phrase}, got {got}"
