import time

import pytest

from gymrat.session import Worktrees, parse_record, record_to_wire
from gymrat.session.records import (
    BaselineRecord,
    FinalizeRecord,
    HookRecord,
    IterationRecord,
    SessionRecord,
    StopRecord,
)
from tests.session.records._fixtures import session_record
from tests.session.records._wire import (
    AT,
    BASELINE_RECORD,
    BLOCKED_KEEP_RECORD,
    COMMAND_RECORD,
    COMMAND_RECORD_SUCCESS,
    COMMAND_RECORD_WITH_TRACEPARENT,
    COMMITTED_KEEP_RECORD,
    DISCARD_RECORD,
    FINALIZE_RECORD,
    HOOK_RECORD,
    ITERATION_RECORD,
    METRIC_VERDICT,
    SESSION_RECORD,
    STOP_RECORD,
    config_with,
    field_of,
    omitting,
    patching,
)

# ---------------------------------------------------------------------------
# now_ns
# ---------------------------------------------------------------------------


def test_now_ns_when_called_does_return_nanosecond_epoch_integer():
    from gymrat.clock import now_ns

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
                    "config": config_with(
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
                    "config": config_with(
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
                        "baseline": field_of(ITERATION_RECORD, "samples")["baseline"],
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
        pytest.param(
            patching(COMMAND_RECORD, {"exit_code": 2, "reason": "supervised-use-tool"}),
            id="command-supervised-use-tool",
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
