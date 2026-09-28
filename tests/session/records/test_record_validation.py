"""Field validation for ``parse_record``: what each field accepts and how it rejects the rest.

These cover the problem message for each rejected value — wrong types, unknown
keys, unknown record types and literal values, all-digit metric names — and
that tuple fields take a JSON array but reject strings and non-numeric or
boolean items.
"""

from typing import get_args

import pytest

from gymrat.errors import GymratError
from gymrat.session import parse_record
from gymrat.session.records import (
    BaselineRecord,
    IterationRecord,
    SessionLogRecord,
)
from tests.session.records._wire import (
    BASELINE_RECORD,
    BLOCKED_KEEP_RECORD,
    COMMAND_RECORD,
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
    mentions,
    omitting,
    patching,
)

# ---------------------------------------------------------------------------
# parse_record — field wrong-type rejections
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
        pytest.param(
            patching(ITERATION_RECORD, {"at": "2026-08-08T14:15:30.000Z"}),
            "at",
            "an integer",
            id="iteration-at-string",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"at": 1_786_198_530.0}),
            "at",
            "an integer",
            id="iteration-at-float",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"at": "2026-08-08T14:15:30.000Z"}),
            "at",
            "an integer",
            id="session-at-string",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"at": 1_786_198_530.0}),
            "at",
            "an integer",
            id="session-at-float",
        ),
        pytest.param(
            patching(HOOK_RECORD, {"at": "2026-08-08T14:15:30.000Z"}),
            "at",
            "an integer",
            id="hook-at-string",
        ),
        pytest.param(
            patching(HOOK_RECORD, {"at": 1_786_198_530.0}),
            "at",
            "an integer",
            id="hook-at-float",
        ),
    ],
)
def test_parse_record_when_field_wrong_type_does_reject_with_phrase(
    record: dict[str, object], field: str, phrase: str
):
    with pytest.raises(GymratError) as exc:
        parse_record(record)

    msg = str(exc.value)
    assert mentions(field).search(msg)
    assert phrase in msg


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
            patching(SESSION_RECORD, {"config": omitting(config_with(), "bench")}),
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
                {"primary": omitting(field_of(ITERATION_RECORD, "primary"), "delta_pct")},
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
            patching(SESSION_RECORD, {"config": config_with(samples=10.5)}),
            "config.samples",
            id="samples-fractional",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"config": config_with(hooks="gymrat.hooks")}),
            "config.hooks",
            id="config-hooks-string",
        ),
        pytest.param(patching(ITERATION_RECORD, {"seq": 0}), "seq", id="iteration-seq-below-one"),
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
        pytest.param(patching(COMMITTED_KEEP_RECORD, {"seq": -1}), "seq", id="keep-seq-below-zero"),
        pytest.param(patching(DISCARD_RECORD, {"seq": 1.5}), "seq", id="discard-seq-fractional"),
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
            patching(SESSION_RECORD, {"config": config_with(retries=3)}),
            "config.retries",
            id="unknown-nested",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"metrics": {"total_ms": {**METRIC_VERDICT, "band": 1.4}}}),
            "metrics.total_ms.band",
            id="unknown-in-verdict",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"schema": 1}), "schema", id="schema-on-non-session"
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
    '"stop-condition", "budget-exceeded", "unsettled", "gating-block", "already-stopped", '
    '"no-session", "finalized", "nothing-measured", "gating-regression", "nothing-to-commit", '
    '"checks-failed", "not-improved", "nothing-to-discard", "stale-session", "nothing-kept", '
    '"dirty-worktree", "unkept-commits", "bad-branch", "branch-exists", "fail-on", "no-filter", '
    '"no-baseline", "supervised-use-tool" or "error"'
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # literal fields
        pytest.param(
            patching(ITERATION_RECORD, {"outcome": "banana"}),
            'outcome: expected "improved", "regressed" or "no-signal", got "banana"',
            id="outcome",
        ),
        pytest.param(
            patching(COMMITTED_KEEP_RECORD, {"status": "banana"}),
            'status: expected "committed" or "blocked", got "banana"',
            id="keep-status",
        ),
        pytest.param(
            patching(BLOCKED_KEEP_RECORD, {"reason": "banana"}),
            'reason: expected "checks-failed", "gating-regression", "nothing-measured", '
            '"nothing-to-commit" or "not-improved", got "banana"',
            id="keep-reason",
        ),
        pytest.param(
            patching(HOOK_RECORD, {"stage": "banana"}),
            'stage: expected "before" or "after", got "banana"',
            id="hook-stage",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"exit_code": "banana"}),
            'exit_code: expected 0, 1 or 2, got "banana"',
            id="command-exit-code",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"reason": "banana"}),
            f'reason: expected one of {_COMMAND_REASONS}, got "banana"',
            id="command-reason",
        ),
        pytest.param(
            patching(COMMAND_RECORD, {"origin": "banana"}),
            'origin: expected "cli" or "tool", got "banana"',
            id="command-origin",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD, {"metrics": {"total_ms": {**METRIC_VERDICT, "verdict": "banana"}}}
            ),
            'metrics.total_ms.verdict: expected "improved", "regressed", "no-signal" or '
            '"unstable", got "banana"',
            id="verdict",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD, {"metrics": {"total_ms": {**METRIC_VERDICT, "method": "banana"}}}
            ),
            'metrics.total_ms.method: expected "permutation", "band" or "exact", got "banana"',
            id="method",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {"primary": {**field_of(ITERATION_RECORD, "primary"), "kind": "banana"}},
            ),
            'primary.kind: expected "geomean" or "metric", got "banana"',
            id="primary-kind",
        ),
        # sample rounds
        pytest.param(
            patching(BASELINE_RECORD, {"samples": "banana"}),
            'samples: expected an array of objects mapping metric names to numbers, got "banana"',
            id="baseline-samples-string",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {"samples": {**field_of(ITERATION_RECORD, "samples"), "experiment": "banana"}},
            ),
            "samples.experiment: expected an array of objects mapping metric names to numbers, "
            'got "banana"',
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
            patching(
                ITERATION_RECORD,
                {
                    "samples": {
                        **field_of(ITERATION_RECORD, "samples"),
                        "baseline": [{"total_ms": "banana"}],
                    }
                },
            ),
            'samples.baseline.0.total_ms: expected a number, got "banana"',
            id="iteration-baseline-sample-string",
        ),
        # confirm field
        # integers coerced from a whole float
        pytest.param(
            patching(ITERATION_RECORD, {"seq": 0.0}),
            "seq: expected a positive integer, got 0",
            id="iteration-seq-zero-float",
        ),
        pytest.param(
            patching(SESSION_RECORD, {"config": config_with(samples=0.0), "samples": 0}),
            "config.samples: expected a positive integer, got 0",
            id="config-samples-zero-float-beside-equal-stray-key",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": _confirm_with(filtered="total_ms")}),
            'confirm.filtered: expected an array of strings, got "total_ms"',
            id="filtered-string",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": _confirm_with(absent="total_ms")}),
            'confirm.absent: expected an array of strings, got "total_ms"',
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
    ],
)
def test_parse_record_when_field_invalid_does_reject_with_message(value: object, expected: str):
    with pytest.raises(GymratError) as exc:
        parse_record(value)

    assert str(exc.value) == _INVALID_VALUE_PREFIX + expected
