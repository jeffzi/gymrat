"""Outbound codec: typed session-log dataclass to camelCase wire dict."""

from typing import assert_never

from gymrat.session.records.types import (
    BaselineRecord,
    DiscardRecord,
    FinalizeRecord,
    HookRecord,
    IterationRecord,
    KeepRecord,
    MetricVerdict,
    PairedSamples,
    SampleRound,
    SessionLogRecord,
    SessionRecord,
)


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
        case _ as unreachable:
            assert_never(unreachable)
    return wire
