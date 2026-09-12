import re
import time
from typing import Any, cast

import pytest
from pydantic import TypeAdapter

from gymrat.errors import GymratError
from gymrat.session import Worktrees, parse_record, record_to_wire
from gymrat.session.records import (
    BaselineRecord,
    FinalizeRecord,
    HookRecord,
    IterationRecord,
    SessionLogRecord,
    SessionRecord,
    StopRecord,
)
from tests.session.records._fixtures import session_record

AT = 1_786_198_530_000_000_000
SHA = "a" * 40
COMMIT = "b" * 40

SESSION_RECORD: dict[str, object] = {
    "type": "session",
    "schema": 1,
    "session_id": "20260808-141530-a3f2",
    "at": AT,
    "baseline": {"ref": "main", "sha": SHA},
    "branch": "gymrat/20260808-141530-a3f2",
    "worktrees": {
        "experiment": "/repo/.gymrat/experiment",
        "baseline": "/repo/.gymrat/baseline",
    },
    "config": {
        "bench": "npm run bench",
        "adapter": "metric-lines",
        "samples": 10,
        "timeout_seconds": 1800,
        "primary": "geomean",
    },
}

BASELINE_RECORD: dict[str, object] = {
    "type": "baseline",
    "at": AT,
    "label": "main",
    "samples": [{"total_ms": 15200}, {"total_ms": 15184}],
}

METRIC_VERDICT: dict[str, object] = {
    "delta_pct": -7.2,
    "verdict": "improved",
    "method": "permutation",
    "p": 0.002,
    "noise_pct": 1.4,
    "gating": True,
    "confirmed": False,
}

ITERATION_RECORD: dict[str, object] = {
    "type": "iteration",
    "seq": 1,
    "at": AT,
    "samples": {
        "experiment": [{"total_ms": 14100}, {"total_ms": 14088}],
        "baseline": [{"total_ms": 15200}, {"total_ms": 15190}],
    },
    "metrics": {"total_ms": METRIC_VERDICT},
    "primary": {"kind": "geomean", "delta_pct": -7.2},
    "outcome": "improved",
    "target_reached": False,
}

COMMITTED_KEEP_RECORD: dict[str, object] = {
    "type": "keep",
    "seq": 1,
    "at": AT,
    "status": "committed",
    "commit": COMMIT,
    "message": "cache the regex",
    "checks": {"configured": True, "passed": True},
}

BLOCKED_KEEP_RECORD: dict[str, object] = {
    "type": "keep",
    "seq": 2,
    "at": AT,
    "status": "blocked",
    "reason": "checks-failed",
    "checks": {"configured": True, "passed": False},
}

DISCARD_RECORD: dict[str, object] = {"type": "discard", "seq": 3, "at": AT}

HOOK_RECORD: dict[str, object] = {
    "type": "hook",
    "at": AT,
    "stage": "before",
    "seq": 4,
    "exit_code": 0,
    "duration_ms": 120,
    "stdout_bytes": 80,
    "timed_out": False,
}

FINALIZE_RECORD: dict[str, object] = {
    "type": "finalize",
    "at": AT,
    "branch": "gymrat/20260808-141530-a3f2-final",
    "commit": COMMIT,
    "message": "squash 3 kept iterations",
}

STOP_RECORD: dict[str, object] = {
    "type": "stop",
    "at": AT,
    "message": "user requested stop",
}

COMMAND_RECORD: dict[str, object] = {
    "type": "command",
    "at": AT,
    "name": "iterate",
    "args": {},
    "exit_code": 1,
    "reason": "budget-exceeded",
    "duration_ms": 1840,
    "seq": 3,
}

COMMAND_RECORD_SUCCESS: dict[str, object] = {
    "type": "command",
    "at": AT,
    "name": "keep",
    "args": {"message": "cache the regex"},
    "exit_code": 0,
    "duration_ms": 520,
    "seq": 1,
}

COMMAND_RECORD_WITH_TRACEPARENT: dict[str, object] = {
    "type": "command",
    "at": AT,
    "name": "iterate",
    "args": {},
    "exit_code": 2,
    "reason": "error",
    "duration_ms": 100,
    "seq": 5,
    "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
}


def omitting(record: dict[str, object], key: str) -> dict[str, object]:
    """Copy of ``record`` without ``key``."""
    clone = dict(record)
    del clone[key]
    return clone


def patching(record: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    """Copy of ``record`` with ``patch`` merged over it."""
    return {**record, **patch}


def mentions(field: str) -> re.Pattern[str]:
    """Matches an error message that names ``field`` as the failing location."""
    return re.compile(rf"\b{re.escape(field)}\b")


def _field(record: dict[str, object], key: str) -> dict[str, object]:
    return cast("dict[str, object]", record[key])


def _config_with(**overrides: object) -> dict[str, object]:
    base = dict(_field(SESSION_RECORD, "config"))
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# now_ns
# ---------------------------------------------------------------------------


def test_now_ns_when_called_does_return_nanosecond_epoch_integer():
    from gymrat.session.clock import now_ns

    before = time.time_ns()

    result = now_ns()

    after = time.time_ns()
    assert isinstance(result, int)
    assert before <= result <= after


# ---------------------------------------------------------------------------
# parse_record — valid records round-trip through the wire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(SESSION_RECORD, id="session"),
        pytest.param(
            patching(
                SESSION_RECORD,
                {
                    "config": _config_with(
                        prepare="npm run build",
                        filter="npm run bench -- --filter {names}",
                    )
                },
            ),
            id="session-with-prepare-and-filter",
        ),
        pytest.param(
            patching(
                SESSION_RECORD,
                {
                    "config": _config_with(
                        hooks={"before": "npm run warm-cache", "after": "npm run cool-down"}
                    )
                },
            ),
            id="session-with-hooks",
        ),
        pytest.param(BASELINE_RECORD, id="baseline"),
        pytest.param(
            patching(BASELINE_RECORD, {"duration_ms": 15192.3}),
            id="baseline-with-duration",
        ),
        pytest.param(ITERATION_RECORD, id="iteration"),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "metrics": {
                        "total_ms": {
                            "delta_pct": -7.2,
                            "verdict": "improved",
                            "method": "band",
                            "gating": True,
                            "confirmed": False,
                        }
                    }
                },
            ),
            id="iteration-metric-omits-optional-stats",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "metrics": {
                        "total_ms": {**METRIC_VERDICT, "delta_pct": None, "verdict": "no-signal"}
                    },
                    "primary": {"kind": "geomean", "delta_pct": None},
                    "outcome": "no-signal",
                },
            ),
            id="iteration-nulled-deltas",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "confirm": {
                        "ran": True,
                        "filtered": ["total_ms"],
                        "samples": {
                            "experiment": [{"total_ms": 14120}],
                            "baseline": [{"total_ms": 15170}],
                        },
                    }
                },
            ),
            id="iteration-reran-to-confirm",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {"metrics": {"__proto__": {**METRIC_VERDICT, "delta_pct": -1.0}}},
            ),
            id="iteration-metric-name-is-proto",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "samples": {
                        "experiment": [{"__proto__": 100, "total_ms": 200}],
                        "baseline": _field(ITERATION_RECORD, "samples")["baseline"],
                    }
                },
            ),
            id="iteration-sample-round-key-is-proto",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"duration_ms": 4200.5, "measured_tree": "abc123def456"}),
            id="iteration-with-duration-and-tree",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"duration_ms": 4200}),
            id="iteration-with-duration-only",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"measured_tree": "abc123def456"}),
            id="iteration-with-tree-only",
        ),
        pytest.param(COMMITTED_KEEP_RECORD, id="committed-keep"),
        pytest.param(
            omitting(COMMITTED_KEEP_RECORD, "message"),
            id="committed-keep-without-message",
        ),
        pytest.param(BLOCKED_KEEP_RECORD, id="blocked-keep"),
        pytest.param(
            patching(
                BLOCKED_KEEP_RECORD,
                {"seq": 0, "reason": "nothing-measured", "checks": {"configured": False}},
            ),
            id="keep-blocked-nothing-measured",
        ),
        pytest.param(
            patching(
                BLOCKED_KEEP_RECORD,
                {"seq": 1, "reason": "nothing-to-commit", "checks": {"configured": True}},
            ),
            id="keep-blocked-nothing-to-commit",
        ),
        pytest.param(
            patching(
                BLOCKED_KEEP_RECORD,
                {"seq": 1, "reason": "not-improved", "checks": {"configured": True}},
            ),
            id="keep-blocked-not-improved",
        ),
        pytest.param(DISCARD_RECORD, id="discard"),
        pytest.param(HOOK_RECORD, id="hook"),
        pytest.param(patching(HOOK_RECORD, {"stderr_bytes": 42}), id="hook-with-stderr-bytes"),
        pytest.param(patching(HOOK_RECORD, {"exit_code": -9}), id="hook-exit-code-negative-signal"),
        pytest.param(FINALIZE_RECORD, id="finalize"),
        pytest.param(STOP_RECORD, id="stop"),
        pytest.param(COMMAND_RECORD, id="command"),
        pytest.param(COMMAND_RECORD_SUCCESS, id="command-success-no-reason"),
        pytest.param(COMMAND_RECORD_WITH_TRACEPARENT, id="command-with-traceparent"),
        pytest.param(
            patching(COMMAND_RECORD, {"name": "probe", "reason": "no-filter"}),
            id="command-probe-no-filter",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"name": "probe", "reason": "no-baseline"}),
            id="command-probe-no-baseline",
        ),
    ],
)
def test_parse_record_when_record_satisfies_schema_does_round_trip(record: dict[str, object]):
    assert record_to_wire(parse_record(record)) == record


# ---------------------------------------------------------------------------
# parse_record — backward compatibility with older logs
# ---------------------------------------------------------------------------


def test_parse_record_when_iteration_lacks_duration_and_tree_does_default_to_none():
    parsed = parse_record(ITERATION_RECORD)

    assert isinstance(parsed, IterationRecord)
    assert parsed.duration_ms is None
    assert parsed.measured_tree is None


def test_parse_record_when_baseline_lacks_duration_does_default_to_none():
    parsed = parse_record(BASELINE_RECORD)

    assert isinstance(parsed, BaselineRecord)
    assert parsed.duration_ms is None


# ---------------------------------------------------------------------------
# parse_record — duration / measured-tree type rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "field", "phrase"),
    [
        pytest.param(
            patching(ITERATION_RECORD, {"duration_ms": "fast"}),
            "duration_ms",
            "a number",
            id="iteration-duration-not-number",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"measured_tree": 42}),
            "measured_tree",
            "a string",
            id="iteration-tree-not-string",
        ),
        pytest.param(
            patching(BASELINE_RECORD, {"duration_ms": "slow"}),
            "duration_ms",
            "a number",
            id="baseline-duration-not-number",
        ),
    ],
)
def test_parse_record_when_duration_or_tree_wrong_type_does_reject_with_phrase(
    record: dict[str, object], field: str, phrase: str
):
    with pytest.raises(GymratError) as exc:
        parse_record(record)

    msg = str(exc.value)
    assert mentions(field).search(msg)
    assert phrase in msg


# ---------------------------------------------------------------------------
# parse_record — at field rejects non-integer types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "bad_at"),
    [
        pytest.param(ITERATION_RECORD, "2026-08-08T14:15:30.000Z", id="iteration-at-string"),
        pytest.param(ITERATION_RECORD, 1_786_198_530.0, id="iteration-at-float"),
        pytest.param(SESSION_RECORD, "2026-08-08T14:15:30.000Z", id="session-at-string"),
        pytest.param(SESSION_RECORD, 1_786_198_530.0, id="session-at-float"),
        pytest.param(HOOK_RECORD, "2026-08-08T14:15:30.000Z", id="hook-at-string"),
        pytest.param(HOOK_RECORD, 1_786_198_530.0, id="hook-at-float"),
    ],
)
def test_parse_record_when_at_not_integer_does_reject(record: dict[str, object], bad_at: object):
    with pytest.raises(GymratError) as exc:
        parse_record(patching(record, {"at": bad_at}))

    msg = str(exc.value)
    assert mentions("at").search(msg)
    assert "an integer" in msg


# ---------------------------------------------------------------------------
# parse_record — camelCase keys rejected as unknown fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("record", "old_key"),
    [
        pytest.param(SESSION_RECORD, "schemaVersion", id="session-schemaVersion"),
        pytest.param(SESSION_RECORD, "sessionId", id="session-sessionId"),
        pytest.param(SESSION_RECORD, "createdAt", id="session-createdAt"),
        pytest.param(
            patching(SESSION_RECORD, {"config": _config_with(timeoutSeconds=1800)}),
            "config.timeoutSeconds",
            id="config-timeoutSeconds",
        ),
        pytest.param(ITERATION_RECORD, "targetReached", id="iteration-targetReached"),
        pytest.param(ITERATION_RECORD, "durationMs", id="iteration-durationMs"),
        pytest.param(ITERATION_RECORD, "measuredTree", id="iteration-measuredTree"),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {"metrics": {"total_ms": {**METRIC_VERDICT, "deltaPct": -7.2}}},
            ),
            "metrics.total_ms.deltaPct",
            id="verdict-deltaPct",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {"metrics": {"total_ms": {**METRIC_VERDICT, "noisePct": 1.4}}},
            ),
            "metrics.total_ms.noisePct",
            id="verdict-noisePct",
        ),
        pytest.param(HOOK_RECORD, "exitCode", id="hook-exitCode"),
        pytest.param(HOOK_RECORD, "durationMs", id="hook-durationMs"),
        pytest.param(HOOK_RECORD, "stdoutBytes", id="hook-stdoutBytes"),
        pytest.param(HOOK_RECORD, "timedOut", id="hook-timedOut"),
    ],
)
def test_parse_record_when_camel_case_key_sent_does_reject_as_unknown(
    record: dict[str, object], old_key: str
):
    leaf = old_key.rsplit(".", maxsplit=1)[-1]
    with pytest.raises(GymratError) as exc:
        parse_record(patching(record, {leaf: record.get(leaf, 0)}))

    assert mentions(leaf).search(str(exc.value))


# ---------------------------------------------------------------------------
# parse_record — schema key on non-session records is rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(patching(ITERATION_RECORD, {"schema": 1}), id="iteration"),
        pytest.param(patching(HOOK_RECORD, {"schema": 1}), id="hook"),
        pytest.param(patching(DISCARD_RECORD, {"schema": 1}), id="discard"),
        pytest.param(patching(FINALIZE_RECORD, {"schema": 1}), id="finalize"),
        pytest.param(patching(STOP_RECORD, {"schema": 1}), id="stop"),
    ],
)
def test_parse_record_when_schema_on_non_session_record_does_reject(record: dict[str, object]):
    with pytest.raises(GymratError) as exc:
        parse_record(record)

    assert mentions("schema").search(str(exc.value))


# ---------------------------------------------------------------------------
# parse_record — rejections that name the offending field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "field"),
    [
        # a required field is missing
        pytest.param(omitting(SESSION_RECORD, "schema"), "schema", id="session-no-schema"),
        pytest.param(omitting(SESSION_RECORD, "session_id"), "session_id", id="session-no-id"),
        pytest.param(
            patching(SESSION_RECORD, {"config": omitting(_config_with(), "bench")}),
            "config.bench",
            id="config-no-bench",
        ),
        pytest.param(omitting(BASELINE_RECORD, "samples"), "samples", id="baseline-no-samples"),
        pytest.param(omitting(ITERATION_RECORD, "metrics"), "metrics", id="iteration-no-metrics"),
        pytest.param(
            patching(
                ITERATION_RECORD, {"metrics": {"total_ms": omitting(METRIC_VERDICT, "delta_pct")}}
            ),
            "metrics.total_ms.delta_pct",
            id="verdict-drops-delta",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {"primary": omitting(_field(ITERATION_RECORD, "primary"), "delta_pct")},
            ),
            "primary.delta_pct",
            id="primary-drops-delta",
        ),
        pytest.param(omitting(COMMITTED_KEEP_RECORD, "status"), "status", id="keep-no-status"),
        pytest.param(omitting(DISCARD_RECORD, "at"), "at", id="discard-no-at"),
        pytest.param(omitting(HOOK_RECORD, "exit_code"), "exit_code", id="hook-no-exit-code"),
        pytest.param(omitting(FINALIZE_RECORD, "commit"), "commit", id="finalize-no-commit"),
        # a field violates its schema
        pytest.param(
            patching(SESSION_RECORD, {"baseline": {"ref": 42, "sha": SHA}}),
            "baseline.ref",
            id="baseline-ref-not-string",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"config": _config_with(samples=10.5)}),
            "config.samples",
            id="samples-fractional",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"config": _config_with(hooks="gymrat.hooks")}),
            "config.hooks",
            id="config-hooks-string",
        ),
        pytest.param(
            patching(BASELINE_RECORD, {"samples": [{"total_ms": "15200"}]}),
            "samples.0.total_ms",
            id="baseline-sample-not-number",
        ),
        pytest.param(patching(ITERATION_RECORD, {"seq": 0}), "seq", id="iteration-seq-below-one"),
        pytest.param(
            patching(ITERATION_RECORD, {"outcome": "unknown"}),
            "outcome",
            id="iteration-bad-outcome",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD, {"metrics": {"total_ms": omitting(METRIC_VERDICT, "gating")}}
            ),
            "metrics.total_ms.gating",
            id="verdict-no-gating",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"target_reached": "false"}),
            "target_reached",
            id="target-reached-not-boolean",
        ),
        pytest.param(
            patching(COMMITTED_KEEP_RECORD, {"status": "pending"}), "status", id="keep-bad-status"
        ),
        pytest.param(
            patching(BLOCKED_KEEP_RECORD, {"reason": "bored"}), "reason", id="keep-bad-reason"
        ),
        pytest.param(patching(COMMITTED_KEEP_RECORD, {"seq": -1}), "seq", id="keep-seq-below-zero"),
        pytest.param(patching(DISCARD_RECORD, {"seq": 1.5}), "seq", id="discard-seq-fractional"),
        pytest.param(patching(HOOK_RECORD, {"stage": "during"}), "stage", id="hook-bad-stage"),
        pytest.param(
            patching(HOOK_RECORD, {"stdout_bytes": -1}),
            "stdout_bytes",
            id="hook-stdout-bytes-negative",
        ),
        pytest.param(
            patching(HOOK_RECORD, {"stderr_bytes": -1}),
            "stderr_bytes",
            id="hook-stderr-bytes-negative",
        ),
        pytest.param(
            patching(HOOK_RECORD, {"timed_out": 0}), "timed_out", id="hook-timed-out-not-boolean"
        ),
        pytest.param(
            patching(FINALIZE_RECORD, {"branch": 42}), "branch", id="finalize-branch-not-string"
        ),
        pytest.param(omitting(STOP_RECORD, "at"), "at", id="stop-no-at"),
        pytest.param(omitting(STOP_RECORD, "message"), "message", id="stop-no-message"),
        pytest.param(patching(STOP_RECORD, {"message": ""}), "message", id="stop-message-empty"),
        # an undeclared key
        pytest.param(patching(DISCARD_RECORD, {"note": "why not"}), "note", id="unknown-top-level"),
        pytest.param(
            patching(SESSION_RECORD, {"config": _config_with(retries=3)}),
            "config.retries",
            id="unknown-nested",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"metrics": {"total_ms": {**METRIC_VERDICT, "band": 1.4}}}),
            "metrics.total_ms.band",
            id="unknown-in-verdict",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {"metrics": {"total_ms": {**METRIC_VERDICT, "method": "signed-rank"}}},
            ),
            "metrics.total_ms.method",
            id="verdict-method-signed-rank-retired",
        ),
    ],
)
def test_parse_record_when_field_invalid_does_name_field(value: object, field: str):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    assert mentions(field).search(str(exc.value))


# ---------------------------------------------------------------------------
# parse_record — values that match no record type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="null"),
        pytest.param(42, id="number"),
        pytest.param("session", id="string"),
        pytest.param([SESSION_RECORD], id="array"),
        pytest.param(omitting(DISCARD_RECORD, "type"), id="no-type-discriminator"),
        pytest.param({"type": 42}, id="type-not-a-string"),
    ],
)
def test_parse_record_when_value_has_no_recognized_type_does_raise(value: object):
    with pytest.raises(GymratError):
        parse_record(value)


def test_parse_record_when_keep_reason_unknown_does_list_every_accepted_reason():
    with pytest.raises(GymratError) as exc:
        parse_record(patching(BLOCKED_KEEP_RECORD, {"reason": "bored"}))

    msg = str(exc.value)
    for reason in (
        "checks-failed",
        "gating-regression",
        "nothing-measured",
        "nothing-to-commit",
        "not-improved",
    ):
        assert reason in msg


def test_parse_record_when_type_unknown_does_name_it_and_list_known_types():
    with pytest.raises(GymratError) as exc:
        parse_record({"type": "banana", "seq": 1})

    assert mentions("banana").search(str(exc.value))
    hint = exc.value.hint or ""
    assert "finalize" in hint
    assert "stop" in hint


# ---------------------------------------------------------------------------
# parse_record — all-digit metric name phrasing
# ---------------------------------------------------------------------------


def test_parse_record_when_metric_name_all_digits_does_phrase_expected_value_correctly():
    record = patching(
        ITERATION_RECORD,
        {"metrics": {"123": omitting(METRIC_VERDICT, "delta_pct")}},
    )

    with pytest.raises(GymratError) as exc:
        parse_record(record)

    msg = str(exc.value)
    assert mentions("123").search(msg)
    assert mentions("delta_pct").search(msg)
    # Must produce the specific type description, not the generic "a valid value"
    # fallback that the expected-type lookup misses when the metric name is all
    # digits and _normalize_loc conflates it with an array index.
    assert "a number or null" in msg


# ---------------------------------------------------------------------------
# parse_record — error messages name the snake_case path
# ---------------------------------------------------------------------------


def test_parse_record_when_iteration_field_invalid_does_name_snake_case_path():
    record = patching(
        ITERATION_RECORD,
        {"metrics": {"total_ms": omitting(METRIC_VERDICT, "delta_pct")}},
    )

    with pytest.raises(GymratError) as exc:
        parse_record(record)

    msg = str(exc.value)
    assert "iteration.metrics.total_ms.delta_pct" in msg or mentions("delta_pct").search(msg)


# ---------------------------------------------------------------------------
# Record model behavior
# ---------------------------------------------------------------------------


def test_record_when_model_copy_called_does_return_updated_copy():
    record = HookRecord(
        type="hook",
        at=AT,
        stage="before",
        seq=4,
        exit_code=0,
        duration_ms=120,
        stdout_bytes=80,
        timed_out=False,
    )

    updated = record.model_copy(update={"exit_code": 42})

    assert updated.exit_code == 42
    assert record.exit_code == 0


def test_session_record_when_dumped_does_use_schema_key_not_schema_version():
    record = session_record(
        worktrees=Worktrees(
            experiment="/repo/.gymrat/experiment",
            baseline="/repo/.gymrat/baseline",
        ),
    )

    wire = record_to_wire(record)

    assert "schema" in wire
    assert "schema_version" not in wire
    assert wire["schema"] == 1


def test_session_record_when_parsed_from_wire_schema_does_populate_schema_version():
    parsed = parse_record(SESSION_RECORD)

    assert isinstance(parsed, SessionRecord)
    assert parsed.schema_version == 1


def test_record_to_wire_when_called_does_produce_snake_case_wire():
    record = session_record(
        worktrees=Worktrees(
            experiment="/repo/.gymrat/experiment",
            baseline="/repo/.gymrat/baseline",
        ),
    )

    wire = record_to_wire(record)

    assert wire == SESSION_RECORD


# ---------------------------------------------------------------------------
# RecordEnvelope — seq omitted from wire when None
# ---------------------------------------------------------------------------


def test_record_to_wire_when_seq_none_does_omit_seq_from_wire():
    record = BaselineRecord(
        type="baseline",
        at=AT,
        label="main",
        samples=({"total_ms": 15200},),
    )

    wire = record_to_wire(record)

    assert "seq" not in wire


# ---------------------------------------------------------------------------
# JSON schema — WithJsonSchema overrides on optional fields
# ---------------------------------------------------------------------------


def _session_log_schema() -> dict[str, Any]:
    """The JSON schema ``TypeAdapter`` generates directly from ``SessionLogRecord``."""
    return TypeAdapter(SessionLogRecord).json_schema()


def test_json_schema_when_optional_never_null_fields_present_does_emit_non_null_types():
    schema = _session_log_schema()
    defs = schema["$defs"]

    # KeepChecks.stdout_bytes is the sentinel: if it gains a null alternative
    # the assertion fails unconditionally.
    stdout_bytes_schema = defs["KeepChecks"]["properties"]["stdout_bytes"]
    if "anyOf" in stdout_bytes_schema:
        types = [alt.get("type") for alt in stdout_bytes_schema["anyOf"]]
        assert "null" not in types, (
            f"stdout_bytes should not include null in anyOf: {stdout_bytes_schema}"
        )
    else:
        assert stdout_bytes_schema.get("type") == "integer"

    delta_pct_schema = defs["MetricVerdict"]["properties"]["delta_pct"]
    assert delta_pct_schema, "delta_pct must appear in MetricVerdict properties"


# ---------------------------------------------------------------------------
# JSON schema — Field(description=...) on every field
# ---------------------------------------------------------------------------


def test_json_schema_when_generated_does_carry_descriptions_on_every_field():
    schema = _session_log_schema()
    defs = schema.get("$defs", {})

    missing: list[str] = []
    for model_name, model_schema in defs.items():
        props = model_schema.get("properties", {})
        for field_name, field_schema in props.items():
            if "description" not in field_schema:
                missing.append(f"{model_name}.{field_name}")

    assert not missing, f"Fields without description: {missing}"


# ---------------------------------------------------------------------------
# JSON schema — SessionHooks and Confirm lifted to $defs as $ref
# ---------------------------------------------------------------------------


def test_json_schema_when_generated_does_define_session_hooks_and_confirm_under_defs():
    schema = _session_log_schema()
    defs = schema.get("$defs", {})

    assert "SessionHooks" in defs
    hooks_props = defs["SessionHooks"]["properties"]
    assert "before" in hooks_props
    assert "after" in hooks_props

    assert "Confirm" in defs
    confirm_props = defs["Confirm"]["properties"]
    assert "ran" in confirm_props
    assert "filtered" in confirm_props
    assert "samples" in confirm_props


def test_json_schema_when_generated_does_ref_hooks_from_session_config_without_null():
    schema = _session_log_schema()
    config_props = schema["$defs"]["SessionConfig"]["properties"]
    hooks_prop = config_props["hooks"]

    assert "$ref" in hooks_prop
    assert hooks_prop["$ref"] == "#/$defs/SessionHooks"
    assert "anyOf" not in hooks_prop


def test_json_schema_when_generated_does_ref_confirm_from_iteration_record_without_null():
    schema = _session_log_schema()
    iteration_props = schema["$defs"]["IterationRecord"]["properties"]
    confirm_prop = iteration_props["confirm"]

    assert "$ref" in confirm_prop
    assert confirm_prop["$ref"] == "#/$defs/Confirm"
    assert "anyOf" not in confirm_prop


# ---------------------------------------------------------------------------
# JSON schema — seq distribution across record types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name",
    [
        pytest.param("SessionRecord", id="session"),
        pytest.param("BaselineRecord", id="baseline"),
        pytest.param("FinalizeRecord", id="finalize"),
        pytest.param("StopRecord", id="stop"),
    ],
)
def test_json_schema_when_generated_does_not_include_seq_on_non_sequenced_record(
    model_name: str,
):
    schema = _session_log_schema()
    model_schema = schema["$defs"][model_name]

    assert "seq" not in model_schema["properties"]


# ---------------------------------------------------------------------------
# Model — seq field absent on non-sequenced records
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(SessionRecord, id="session"),
        pytest.param(BaselineRecord, id="baseline"),
        pytest.param(FinalizeRecord, id="finalize"),
        pytest.param(StopRecord, id="stop"),
    ],
)
def test_model_when_non_sequenced_record_does_not_have_seq_field(
    model_cls: type,
):
    assert "seq" not in model_cls.model_fields
