"""Field validation for ``parse_record``: what each field accepts and how it rejects the rest.

These cover the problem message for each rejected value — wrong types, bounds,
missing and unknown keys, unknown record types and literal values — and that
tuple fields take a JSON array but reject strings and non-numeric or boolean
items.
"""

from typing import get_args

import pytest

from gymrat.errors import GymratError
from gymrat.session.records import (
    BaselineRecord,
    IterationRecord,
    KeepChecks,
    KeepRecord,
    SessionLogRecord,
    parse_record,
)
from tests.session.records._fixtures import AT
from tests.session.records._wire import (
    BASELINE_RECORD,
    BLOCKED_KEEP_RECORD,
    COMMAND_RECORD,
    COMMAND_RECORD_WITH_TRACEPARENT,
    COMMITTED_KEEP_RECORD,
    DISCARD_RECORD,
    FINALIZE_RECORD,
    HOOK_RECORD,
    ITERATION_RECORD,
    METRIC_VERDICT,
    SESSION_RECORD,
    SHA,
    STOP_RECORD,
    config_with,
    field_of,
    omitting,
    patching,
)


def _verdict_with(**overrides: object) -> dict[str, object]:
    return patching(ITERATION_RECORD, {"metrics": {"total_ms": {**METRIC_VERDICT, **overrides}}})


def _verdict_without(key: str) -> dict[str, object]:
    return patching(ITERATION_RECORD, {"metrics": {"total_ms": omitting(METRIC_VERDICT, key)}})


def _nested_with(record: dict[str, object], field: str, **overrides: object) -> dict[str, object]:
    return patching(record, {field: {**field_of(record, field), **overrides}})


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


def test_parse_record_when_type_unknown_does_raise_with_known_types_hint():
    known_types = ", ".join(
        get_args(member.model_fields["type"].annotation)[0]
        for member in get_args(SessionLogRecord.__value__)
    )

    with pytest.raises(GymratError) as exc:
        parse_record({"type": "banana", "seq": 1})

    assert (str(exc.value), exc.value.hint) == (
        'Unknown session record type: "banana"',
        f"Expected one of: {known_types}.",
    )


# ---------------------------------------------------------------------------
# parse_record — missing keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "key"),
    [
        pytest.param(omitting(STOP_RECORD, "message"), "message", id="stop-no-message"),
        pytest.param(omitting(SESSION_RECORD, "schema"), "schema", id="session-no-schema"),
        pytest.param(omitting(SESSION_RECORD, "session_id"), "session_id", id="session-no-id"),
        pytest.param(
            patching(SESSION_RECORD, {"config": omitting(config_with(), "bench")}),
            "config.bench",
            id="config-no-bench",
        ),
        pytest.param(omitting(BASELINE_RECORD, "samples"), "samples", id="baseline-no-samples"),
        pytest.param(omitting(ITERATION_RECORD, "metrics"), "metrics", id="iteration-no-metrics"),
        pytest.param(
            _verdict_without("delta_pct"), "metrics.total_ms.delta_pct", id="verdict-drops-delta"
        ),
        pytest.param(_verdict_without("gating"), "metrics.total_ms.gating", id="verdict-no-gating"),
        pytest.param(
            patching(ITERATION_RECORD, {"metrics": {"123": omitting(METRIC_VERDICT, "delta_pct")}}),
            "metrics.123.delta_pct",
            id="under-all-digit-metric-name",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {"primary": omitting(field_of(ITERATION_RECORD, "primary"), "delta_pct")},
            ),
            "primary.delta_pct",
            id="primary-drops-delta",
        ),
        pytest.param(omitting(COMMITTED_KEEP_RECORD, "status"), "status", id="keep-no-status"),
        pytest.param(omitting(DISCARD_RECORD, "at"), "at", id="discard-no-at"),
        pytest.param(omitting(HOOK_RECORD, "exit_code"), "exit_code", id="hook-no-exit-code"),
        pytest.param(omitting(FINALIZE_RECORD, "commit"), "commit", id="finalize-no-commit"),
        pytest.param(omitting(COMMAND_RECORD, "name"), "name", id="command-no-name"),
        pytest.param(omitting(COMMAND_RECORD, "args"), "args", id="command-no-args"),
        pytest.param(omitting(COMMAND_RECORD, "exit_code"), "exit_code", id="command-no-exit-code"),
        pytest.param(
            omitting(COMMAND_RECORD, "duration_ms"), "duration_ms", id="command-no-duration-ms"
        ),
    ],
)
def test_parse_record_when_key_missing_does_reject_naming_key(value: object, key: str):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    assert str(exc.value) == f"Missing session record key: {key}"


# ---------------------------------------------------------------------------
# parse_record — unknown keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "key"),
    [
        pytest.param(patching(DISCARD_RECORD, {"note": "why not"}), "note", id="unknown-top-level"),
        pytest.param(
            patching(SESSION_RECORD, {"config": config_with(retries=3)}),
            "config.retries",
            id="unknown-nested",
        ),
        pytest.param(_verdict_with(band=1.4), "metrics.total_ms.band", id="unknown-in-verdict"),
        pytest.param(
            patching(ITERATION_RECORD, {"schema": 1}), "schema", id="schema-on-non-session"
        ),
    ],
)
def test_parse_record_when_key_unknown_does_reject_naming_key(value: object, key: str):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    assert str(exc.value) == f"Unknown session record key: {key}"


# ---------------------------------------------------------------------------
# Tuple fields — a JSON array is accepted, anything else is rejected
# ---------------------------------------------------------------------------


CONFIRM: dict[str, object] = {
    "ran": True,
    "filtered": ["total_ms"],
    "absent": ["rss_kb"],
    "samples": {
        "experiment": [{"total_ms": 14120}],
        "baseline": [{"total_ms": 15170}],
    },
}


def _confirm_with(**overrides: object) -> dict[str, object]:
    return {**CONFIRM, **overrides}


def test_parse_record_when_tuple_fields_sent_as_arrays_does_hold_tuples():
    record = patching(ITERATION_RECORD, {"confirm": CONFIRM})

    parsed = parse_record(record)

    assert isinstance(parsed, IterationRecord)
    assert parsed.confirm is not None
    assert (
        parsed.samples.experiment,
        parsed.samples.baseline,
        parsed.confirm.filtered,
        parsed.confirm.absent,
        parsed.confirm.samples.experiment,
    ) == (
        ({"total_ms": 14100}, {"total_ms": 14088}),
        ({"total_ms": 15200}, {"total_ms": 15190}),
        ("total_ms",),
        ("rss_kb",),
        ({"total_ms": 14120},),
    )


def test_parse_record_when_baseline_samples_sent_as_array_does_hold_tuple():
    parsed = parse_record(BASELINE_RECORD)

    assert isinstance(parsed, BaselineRecord)
    assert parsed.samples == ({"total_ms": 15200}, {"total_ms": 15184})


# ---------------------------------------------------------------------------
# parse_record — exact "Invalid session record value for ..." messages
# ---------------------------------------------------------------------------


_INVALID_VALUE_PREFIX = "Invalid session record value for "

_COMMAND_REASONS = (
    "'stop-condition', 'budget-exceeded', 'unsettled', 'gating-block', 'already-stopped', "
    "'no-session', 'finalized', 'nothing-measured', 'gating-regression', 'nothing-to-commit', "
    "'checks-failed', 'not-improved', 'nothing-to-discard', 'stale-session', 'nothing-kept', "
    "'dirty-worktree', 'unkept-commits', 'bad-branch', 'branch-exists', 'fail-on', 'no-filter', "
    "'no-baseline', 'supervised-use-tool' or 'error'"
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # literal fields
        pytest.param(
            patching(ITERATION_RECORD, {"outcome": "banana"}),
            "outcome: expected 'improved', 'regressed' or 'no-signal', got \"banana\"",
            id="outcome",
        ),
        pytest.param(
            patching(COMMITTED_KEEP_RECORD, {"status": "banana"}),
            "status: expected 'committed' or 'blocked', got \"banana\"",
            id="keep-status",
        ),
        pytest.param(
            patching(BLOCKED_KEEP_RECORD, {"reason": "banana"}),
            "reason: expected 'checks-failed', 'gating-regression', 'nothing-measured', "
            "'nothing-to-commit' or 'not-improved', got \"banana\"",
            id="keep-reason",
        ),
        pytest.param(
            patching(HOOK_RECORD, {"stage": "banana"}),
            "stage: expected 'before' or 'after', got \"banana\"",
            id="hook-stage",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"exit_code": "banana"}),
            'exit_code: expected 0, 1 or 2, got "banana"',
            id="command-exit-code",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"reason": "banana"}),
            f'reason: expected {_COMMAND_REASONS}, got "banana"',
            id="command-reason",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"origin": "banana"}),
            "origin: expected 'cli' or 'tool', got \"banana\"",
            id="command-origin",
        ),
        pytest.param(
            _verdict_with(verdict="banana"),
            "metrics.total_ms.verdict: expected 'improved', 'regressed', 'no-signal' or "
            "'unstable', got \"banana\"",
            id="verdict",
        ),
        pytest.param(
            _verdict_with(method="banana"),
            "metrics.total_ms.method: expected 'permutation', 'band' or 'exact', got \"banana\"",
            id="method",
        ),
        pytest.param(
            _nested_with(ITERATION_RECORD, "primary", kind="banana"),
            "primary.kind: expected 'geomean' or 'metric', got \"banana\"",
            id="primary-kind",
        ),
        # scalar type mismatches
        pytest.param(
            patching(ITERATION_RECORD, {"at": "2026-08-08T14:15:30.000Z"}),
            'at: expected an integer, got "2026-08-08T14:15:30.000Z"',
            id="at-string",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"at": 1_786_198_530.0}),
            "at: expected an integer, got 1786198530.0",
            id="at-float",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"duration_ms": "fast"}),
            'duration_ms: expected a number, got "fast"',
            id="duration-string",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"measured_tree": 42}),
            "measured_tree: expected a string, got 42",
            id="tree-number",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"target_reached": "false"}),
            'target_reached: expected a boolean, got "false"',
            id="target-reached-string",
        ),
        pytest.param(
            patching(COMMAND_RECORD_WITH_TRACEPARENT, {"traceparent": 42}),
            "traceparent: expected a string, got 42",
            id="command-traceparent-not-string",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"args": "banana"}),
            'args: expected an object, got "banana"',
            id="command-args-string",
        ),
        # nullable fields take the phrase of their non-null type
        pytest.param(
            _verdict_with(delta_pct="banana"),
            'metrics.total_ms.delta_pct: expected a number, got "banana"',
            id="verdict-delta-string",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {"metrics": {"decode.time": {**METRIC_VERDICT, "delta_pct": "banana"}}},
            ),
            'metrics."decode.time".delta_pct: expected a number, got "banana"',
            id="verdict-delta-string-quoted-metric",
        ),
        # optional fields that are not nullable reject an explicit null with their type
        pytest.param(
            patching(SESSION_RECORD, {"config": config_with(filter=None)}),
            "config.filter: expected a string, got null",
            id="config-filter-null",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"config": config_with(hooks={"before": None})}),
            "config.hooks.before: expected a string, got null",
            id="config-hooks-before-null",
        ),
        pytest.param(
            _verdict_with(p=None),
            "metrics.total_ms.p: expected a number, got null",
            id="verdict-p-null",
        ),
        pytest.param(
            _nested_with(COMMITTED_KEEP_RECORD, "checks", passed=None),
            "checks.passed: expected a boolean, got null",
            id="keep-checks-passed-null",
        ),
        pytest.param(
            _nested_with(COMMITTED_KEEP_RECORD, "checks", stdout_bytes=None),
            "checks.stdout_bytes: expected an integer, got null",
            id="keep-checks-stdout-bytes-null",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"config": config_with(hooks=None)}),
            "config.hooks: expected an object, got null",
            id="config-hooks-null",
        ),
        # bounds
        pytest.param(
            patching(HOOK_RECORD, {"stdout_bytes": -1}),
            "stdout_bytes: expected a number at or above 0, got -1",
            id="hook-stdout-bytes-negative",
        ),
        pytest.param(
            patching(STOP_RECORD, {"message": ""}),
            'message: expected a non-empty string, got ""',
            id="stop-message-empty",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"duration_ms": -1}),
            "duration_ms: expected a number at or above 0, got -1",
            id="command-duration-ms-negative",
        ),
        # sample rounds
        pytest.param(
            patching(BASELINE_RECORD, {"samples": "banana"}),
            'samples: expected an array, got "banana"',
            id="baseline-samples-string",
        ),
        pytest.param(
            _nested_with(ITERATION_RECORD, "samples", experiment="banana"),
            'samples.experiment: expected an array, got "banana"',
            id="iteration-experiment-string",
        ),
        pytest.param(
            patching(BASELINE_RECORD, {"samples": [{"total_ms": True}]}),
            "samples.0.total_ms: expected a number, got true",
            id="baseline-sample-bool",
        ),
        pytest.param(
            patching(BASELINE_RECORD, {"samples": [{"total_ms": "15200"}]}),
            'samples.0.total_ms: expected a number, got "15200"',
            id="baseline-sample-string",
        ),
        pytest.param(
            _nested_with(ITERATION_RECORD, "samples", baseline=[{"total_ms": "banana"}]),
            'samples.baseline.0.total_ms: expected a number, got "banana"',
            id="iteration-baseline-sample-string",
        ),
        # integers coerced from a whole float
        pytest.param(
            patching(ITERATION_RECORD, {"seq": 0.0}),
            "seq: expected a number at or above 1, got 0",
            id="iteration-seq-zero-float",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"config": config_with(samples=0.0), "samples": 0}),
            "config.samples: expected a number at or above 1, got 0",
            id="config-samples-zero-float-beside-equal-stray-key",
        ),
        # confirm field
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": _confirm_with(filtered="total_ms")}),
            'confirm.filtered: expected an array, got "total_ms"',
            id="filtered-string",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": _confirm_with(absent="total_ms")}),
            'confirm.absent: expected an array, got "total_ms"',
            id="absent-string",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": _confirm_with(filtered=[True])}),
            "confirm.filtered.0: expected a string, got true",
            id="filtered-bool",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": _confirm_with(absent=[False])}),
            "confirm.absent.0: expected a string, got false",
            id="absent-bool",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "confirm": _confirm_with(
                        samples={"experiment": [{"total_ms": True}], "baseline": []}
                    )
                },
            ),
            "confirm.samples.experiment.0.total_ms: expected a number, got true",
            id="sample-bool",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "confirm": _confirm_with(
                        samples={"experiment": [], "baseline": [{"total_ms": "banana"}]}
                    )
                },
            ),
            'confirm.samples.baseline.0.total_ms: expected a number, got "banana"',
            id="sample-string",
        ),
        # fields that violate their schema
        pytest.param(
            patching(SESSION_RECORD, {"baseline": {"ref": 42, "sha": SHA}}),
            "baseline.ref: expected a string, got 42",
            id="baseline-ref-not-string",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"config": config_with(samples=10.5)}),
            "config.samples: expected an integer, got 10.5",
            id="samples-fractional",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"config": config_with(hooks="gymrat.hooks")}),
            'config.hooks: expected an object, got "gymrat.hooks"',
            id="config-hooks-string",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"seq": 0}),
            "seq: expected a number at or above 1, got 0",
            id="iteration-seq-below-one",
        ),
        pytest.param(
            patching(COMMITTED_KEEP_RECORD, {"seq": -1}),
            "seq: expected a number at or above 0, got -1",
            id="keep-seq-below-zero",
        ),
        pytest.param(
            patching(DISCARD_RECORD, {"seq": 1.5}),
            "seq: expected an integer, got 1.5",
            id="discard-seq-fractional",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"seq": 3.5}),
            "seq: expected an integer, got 3.5",
            id="command-seq-fractional",
        ),
        pytest.param(
            patching(HOOK_RECORD, {"stderr_bytes": -1}),
            "stderr_bytes: expected a number at or above 0, got -1",
            id="hook-stderr-bytes-negative",
        ),
        pytest.param(
            patching(HOOK_RECORD, {"timed_out": 0}),
            "timed_out: expected a boolean, got 0",
            id="hook-timed-out-not-boolean",
        ),
        pytest.param(
            patching(FINALIZE_RECORD, {"branch": 42}),
            "branch: expected a string, got 42",
            id="finalize-branch-not-string",
        ),
    ],
)
def test_parse_record_when_field_invalid_does_reject_with_message(value: object, expected: str):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    assert str(exc.value) == _INVALID_VALUE_PREFIX + expected


# ---------------------------------------------------------------------------
# Python-side construction — an optional field takes None
# ---------------------------------------------------------------------------


def test_keep_record_when_constructed_with_none_optionals_does_hold_none():
    record = KeepRecord(
        type="keep",
        seq=1,
        at=AT,
        status="committed",
        checks=KeepChecks(configured=True, passed=None, stdout_bytes=None),
        commit=None,
        message=None,
    )

    assert (
        record.commit,
        record.message,
        record.checks.passed,
        record.checks.stdout_bytes,
    ) == (None, None, None, None)
