"""Field validation for ``parse_record``: what each field accepts and how it rejects the rest.

These cover the problem message for each rejected value — wrong types, bounds,
missing and unknown keys, unknown record types and literal values — and that
tuple fields take a JSON array but reject strings and non-numeric or boolean
items.
"""

import pytest

from gymrat.errors import GymratError
from gymrat.session.records import (
    IterationRecord,
    parse_record,
)
from tests.session.records._fixtures import (
    BASELINE_SHA,
)
from tests.session.records._wire import (
    BASELINE_RECORD,
    BLOCKED_KEEP_RECORD,
    COMMAND_RECORD,
    COMMAND_RECORD_SUCCESS,
    COMMITTED_KEEP_RECORD,
    CONFIRM,
    DISCARD_RECORD,
    FINALIZE_RECORD,
    HOOK_RECORD,
    ITERATION_RECORD,
    METRIC_VERDICT,
    SESSION_RECORD,
    STOP_RECORD,
    confirm_with,
    field_of,
    nested_with,
    omitting,
    patching,
    verdict_with,
    verdict_without,
)

# ---------------------------------------------------------------------------
# parse_record — values that match no record type
# ---------------------------------------------------------------------------


_KNOWN_TYPES_HINT = (
    "Expected one of: session, baseline, iteration, keep, discard, hook, finalize, stop, command."
)


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        pytest.param(None, "null", id="null"),
        pytest.param(42, "42", id="number"),
        pytest.param("session", '"session"', id="string"),
        pytest.param(["session"], '["session"]', id="array"),
    ],
)
def test_parse_record_when_value_not_an_object_does_raise_naming_it(value: object, shown: str):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    assert str(exc.value) == f"Invalid session record: expected a JSON object, got {shown}"


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        pytest.param({"type": "banana", "seq": 1}, '"banana"', id="unknown-name"),
        pytest.param(omitting(DISCARD_RECORD, "type"), "undefined", id="no-type-discriminator"),
        pytest.param({"type": 42}, "42", id="type-not-a-string"),
    ],
)
def test_parse_record_when_type_unknown_does_raise_with_known_types_hint(value: object, shown: str):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    assert (str(exc.value), exc.value.hint) == (
        f"Unknown session record type: {shown}",
        _KNOWN_TYPES_HINT,
    )


# ---------------------------------------------------------------------------
# parse_record — missing keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "key"),
    [
        pytest.param(omitting(SESSION_RECORD, "schema"), "schema", id="session-no-schema"),
        pytest.param(
            patching(
                SESSION_RECORD, {"config": omitting(field_of(SESSION_RECORD, "config"), "bench")}
            ),
            "config.bench",
            id="config-no-bench",
        ),
        pytest.param(
            verdict_without("delta_pct"), "metrics.total_ms.delta_pct", id="verdict-drops-delta"
        ),
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
        pytest.param(omitting(DISCARD_RECORD, "at"), "at", id="discard-no-at"),
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
            nested_with(SESSION_RECORD, "config", retries=3),
            "config.retries",
            id="unknown-nested",
        ),
        pytest.param(verdict_with(band=1.4), "metrics.total_ms.band", id="unknown-in-verdict"),
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


def test_parse_record_when_iteration_tuple_fields_sent_as_arrays_does_hold_tuples():
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
            verdict_with(verdict="banana"),
            "metrics.total_ms.verdict: expected 'improved', 'regressed', 'no-signal' or "
            "'unstable', got \"banana\"",
            id="verdict",
        ),
        pytest.param(
            verdict_with(method="banana"),
            "metrics.total_ms.method: expected 'permutation', 'band' or 'exact', got \"banana\"",
            id="method",
        ),
        pytest.param(
            nested_with(ITERATION_RECORD, "primary", kind="banana"),
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
            patching(COMMAND_RECORD, {"args": "banana"}),
            'args: expected an object, got "banana"',
            id="command-args-string",
        ),
        # nullable fields take the phrase of their non-null type
        pytest.param(
            verdict_with(delta_pct="banana"),
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
            nested_with(SESSION_RECORD, "config", filter=None),
            "config.filter: expected a string, got null",
            id="config-filter-null",
        ),
        pytest.param(
            nested_with(SESSION_RECORD, "config", hooks={"before": None}),
            "config.hooks.before: expected a string, got null",
            id="config-hooks-before-null",
        ),
        pytest.param(
            verdict_with(p=None),
            "metrics.total_ms.p: expected a number, got null",
            id="verdict-p-null",
        ),
        pytest.param(
            nested_with(COMMITTED_KEEP_RECORD, "checks", passed=None),
            "checks.passed: expected a boolean, got null",
            id="keep-checks-passed-null",
        ),
        pytest.param(
            nested_with(COMMITTED_KEEP_RECORD, "checks", stdout_bytes=None),
            "checks.stdout_bytes: expected an integer, got null",
            id="keep-checks-stdout-bytes-null",
        ),
        pytest.param(
            nested_with(SESSION_RECORD, "config", hooks=None),
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
        # sample rounds
        pytest.param(
            patching(BASELINE_RECORD, {"samples": "banana"}),
            'samples: expected an array, got "banana"',
            id="baseline-samples-string",
        ),
        pytest.param(
            nested_with(ITERATION_RECORD, "samples", experiment="banana"),
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
            nested_with(ITERATION_RECORD, "samples", baseline=[{"total_ms": "banana"}]),
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
            patching(nested_with(SESSION_RECORD, "config", samples=0.0), {"samples": 0}),
            "config.samples: expected a number at or above 1, got 0",
            id="config-samples-zero-float-beside-equal-stray-key",
        ),
        pytest.param(
            nested_with(COMMITTED_KEEP_RECORD, "checks", stdout_bytes=-1.0),
            "checks.stdout_bytes: expected a number at or above 0, got -1",
            id="keep-checks-stdout-bytes-negative-float",
        ),
        # confirm field
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": confirm_with(filtered="total_ms")}),
            'confirm.filtered: expected an array, got "total_ms"',
            id="filtered-string",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": confirm_with(absent="total_ms")}),
            'confirm.absent: expected an array, got "total_ms"',
            id="absent-string",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": confirm_with(filtered=[True])}),
            "confirm.filtered.0: expected a string, got true",
            id="filtered-bool",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": confirm_with(absent=[False])}),
            "confirm.absent.0: expected a string, got false",
            id="absent-bool",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "confirm": confirm_with(
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
                    "confirm": confirm_with(
                        samples={"experiment": [], "baseline": [{"total_ms": "banana"}]}
                    )
                },
            ),
            'confirm.samples.baseline.0.total_ms: expected a number, got "banana"',
            id="sample-string",
        ),
        # fields that violate their schema
        pytest.param(
            patching(SESSION_RECORD, {"baseline": {"ref": 42, "sha": BASELINE_SHA}}),
            "baseline.ref: expected a string, got 42",
            id="baseline-ref-not-string",
        ),
        pytest.param(
            nested_with(SESSION_RECORD, "config", samples=10.5),
            "config.samples: expected an integer, got 10.5",
            id="samples-fractional",
        ),
        pytest.param(
            nested_with(SESSION_RECORD, "config", hooks="gymrat.hooks"),
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
# CommandRecord — exit_code / reason consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            patching(COMMAND_RECORD_SUCCESS, {"reason": "budget-exceeded"}),
            "Invalid session record: command.exit_code is 0 but reason is set to "
            "'budget-exceeded'; a successful command must not carry a reason",
            id="success-with-reason",
        ),
        pytest.param(
            omitting(COMMAND_RECORD, "reason"),
            "Invalid session record: command.exit_code is 1 but reason is missing; "
            "a failed command must carry a reason",
            id="failure-without-reason",
        ),
    ],
)
def test_parse_record_when_command_exit_code_and_reason_disagree_does_reject_with_the_rule(
    value: object, expected: str
):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    assert str(exc.value) == expected
