import pytest

from gymrat.session.records import parse_record, record_to_wire
from tests.session.records._wire import (
    BASELINE_RECORD,
    BLOCKED_KEEP_RECORD,
    COMMAND_RECORD,
    COMMAND_RECORD_SUCCESS,
    COMMAND_RECORD_WITH_TRACEPARENT,
    COMMITTED_KEEP_RECORD,
    CONFIRM,
    DISCARD_RECORD,
    FINALIZE_RECORD,
    HOOK_RECORD,
    ITERATION_RECORD,
    METRIC_VERDICT,
    SESSION_RECORD,
    STOP_RECORD,
    nested_with,
    omitting,
    patching,
    verdict_with,
)


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(SESSION_RECORD, id="session"),
        pytest.param(
            nested_with(
                SESSION_RECORD,
                "config",
                prepare="npm run build",
                filter="npm run bench -- --filter {names}",
            ),
            id="session-with-prepare-and-filter",
        ),
        pytest.param(
            nested_with(
                SESSION_RECORD,
                "config",
                hooks={"before": "npm run warm-cache", "after": "npm run cool-down"},
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
                            **omitting(omitting(METRIC_VERDICT, "p"), "noise_pct"),
                            "method": "band",
                        }
                    }
                },
            ),
            id="iteration-metric-omits-optional-stats",
        ),
        pytest.param(
            patching(
                verdict_with(delta_pct=None, verdict="no-signal"),
                {"primary": {"kind": "geomean", "delta_pct": None}, "outcome": "no-signal"},
            ),
            id="iteration-nulled-deltas",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"confirm": CONFIRM}),
            id="iteration-reran-to-confirm",
        ),
        pytest.param(
            patching(ITERATION_RECORD, {"duration_ms": 4200.5, "measured_tree": "abc123def456"}),
            id="iteration-with-duration-and-tree",
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
        pytest.param(DISCARD_RECORD, id="discard"),
        pytest.param(HOOK_RECORD, id="hook"),
        pytest.param(patching(HOOK_RECORD, {"stderr_bytes": 42}), id="hook-with-stderr-bytes"),
        pytest.param(patching(HOOK_RECORD, {"exit_code": -9}), id="hook-exit-code-negative-signal"),
        pytest.param(FINALIZE_RECORD, id="finalize"),
        pytest.param(STOP_RECORD, id="stop"),
        pytest.param(COMMAND_RECORD, id="command"),
        pytest.param(COMMAND_RECORD_SUCCESS, id="command-success-no-reason"),
        pytest.param(COMMAND_RECORD_WITH_TRACEPARENT, id="command-with-traceparent"),
    ],
)
def test_parse_record_when_record_satisfies_schema_does_round_trip(record: dict[str, object]):
    wire = record_to_wire(parse_record(record))

    assert wire == record
