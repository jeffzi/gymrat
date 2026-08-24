import re
from typing import cast

import pytest

from gymrat_py.errors import GymratError
from gymrat_py.session import parse_record, record_to_wire

AT = "2026-08-08T14:15:30.000Z"
SHA = "a" * 40
COMMIT = "b" * 40

SESSION_RECORD: dict[str, object] = {
    "type": "session",
    "schemaVersion": 1,
    "sessionId": "20260808-141530-a3f2",
    "createdAt": AT,
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
        "timeoutSeconds": 1800,
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
    "deltaPct": -7.2,
    "verdict": "improved",
    "method": "signed-rank",
    "p": 0.002,
    "noisePct": 1.4,
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
    "primary": {"kind": "geomean", "deltaPct": -7.2},
    "outcome": "improved",
    "targetReached": False,
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
    "stage": "before",
    "seq": 4,
    "exitCode": 0,
    "durationMs": 120,
    "stdoutBytes": 80,
    "timedOut": False,
}

FINALIZE_RECORD: dict[str, object] = {
    "type": "finalize",
    "at": AT,
    "branch": "gymrat/20260808-141530-a3f2-final",
    "commit": COMMIT,
    "message": "squash 3 kept iterations",
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


def _config_with(**overrides: object) -> dict[str, object]:
    base = dict(cast("dict[str, object]", SESSION_RECORD["config"]))
    base.update(overrides)
    return base


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
        pytest.param(ITERATION_RECORD, id="iteration"),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "metrics": {
                        "total_ms": {
                            "deltaPct": -7.2,
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
                        "total_ms": {**METRIC_VERDICT, "deltaPct": None, "verdict": "no-signal"}
                    },
                    "primary": {"kind": "geomean", "deltaPct": None},
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
                {"metrics": {"__proto__": {**METRIC_VERDICT, "deltaPct": -1.0}}},
            ),
            id="iteration-metric-name-is-proto",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "samples": {
                        "experiment": [{"__proto__": 100, "total_ms": 200}],
                        "baseline": cast("dict[str, object]", ITERATION_RECORD["samples"])[
                            "baseline"
                        ],
                    }
                },
            ),
            id="iteration-sample-round-key-is-proto",
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
        pytest.param(DISCARD_RECORD, id="discard"),
        pytest.param(HOOK_RECORD, id="hook"),
        pytest.param(patching(HOOK_RECORD, {"stderrBytes": 42}), id="hook-with-stderr-bytes"),
        pytest.param(FINALIZE_RECORD, id="finalize"),
    ],
)
def test_parse_record_when_record_satisfies_schema_does_round_trip(record: dict[str, object]):
    assert record_to_wire(parse_record(record)) == record


# ---------------------------------------------------------------------------
# parse_record — rejections that name the offending field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "field"),
    [
        # a required field is missing
        pytest.param(omitting(SESSION_RECORD, "sessionId"), "sessionId", id="session-no-id"),
        pytest.param(
            patching(SESSION_RECORD, {"config": omitting(_config_with(), "bench")}),
            "config.bench",
            id="config-no-bench",
        ),
        pytest.param(omitting(BASELINE_RECORD, "samples"), "samples", id="baseline-no-samples"),
        pytest.param(omitting(ITERATION_RECORD, "metrics"), "metrics", id="iteration-no-metrics"),
        pytest.param(
            patching(
                ITERATION_RECORD, {"metrics": {"total_ms": omitting(METRIC_VERDICT, "deltaPct")}}
            ),
            "metrics.total_ms.deltaPct",
            id="verdict-drops-delta",
        ),
        pytest.param(
            patching(
                ITERATION_RECORD,
                {
                    "primary": omitting(
                        cast("dict[str, object]", ITERATION_RECORD["primary"]), "deltaPct"
                    )
                },
            ),
            "primary.deltaPct",
            id="primary-drops-delta",
        ),
        pytest.param(omitting(COMMITTED_KEEP_RECORD, "status"), "status", id="keep-no-status"),
        pytest.param(omitting(DISCARD_RECORD, "at"), "at", id="discard-no-at"),
        pytest.param(omitting(HOOK_RECORD, "exitCode"), "exitCode", id="hook-no-exit-code"),
        pytest.param(omitting(FINALIZE_RECORD, "commit"), "commit", id="finalize-no-commit"),
        # a field violates its schema
        pytest.param(
            patching(SESSION_RECORD, {"schemaVersion": 2}),
            "schemaVersion",
            id="wrong-schema-version",
        ),
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
            patching(ITERATION_RECORD, {"targetReached": "false"}),
            "targetReached",
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
            patching(HOOK_RECORD, {"timedOut": 0}), "timedOut", id="hook-timed-out-not-boolean"
        ),
        pytest.param(
            patching(FINALIZE_RECORD, {"branch": 42}), "branch", id="finalize-branch-not-string"
        ),
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
        pytest.param({"type": "banana"}, id="unknown-type"),
    ],
)
def test_parse_record_when_value_has_no_recognized_type_does_raise(value: object):
    with pytest.raises(GymratError):
        parse_record(value)


def test_parse_record_when_type_unknown_does_name_it_and_list_known_types():
    with pytest.raises(GymratError) as exc:
        parse_record({"type": "banana", "seq": 1})

    assert mentions("banana").search(str(exc.value))
    assert "finalize" in (exc.value.hint or "")
