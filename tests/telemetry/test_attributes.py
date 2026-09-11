"""Tests for record-to-attribute mapping (pure dict, no OTel imports)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

from gymrat.session.records import (
    BaselineRecord,
    Confirm,
    FinalizeRecord,
    IterationPrimary,
)
from gymrat.session.records.models import PairedSamples, SessionLogRecord
from gymrat.telemetry.attributes import (
    all_attribute_names,
    command_attributes,
    command_span_inputs,
    record_event,
)
from tests.session.records._fixtures import (
    AT,
    SESSION_ID,
    blocked_keep,
    command_record,
    committed_keep,
    discard_record,
    finalize_record,
    hook_record,
    iteration_record,
    stop_record,
)

_ATTR_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*)*$")


def _assert_valid_attribute_names(keys: Iterable[str]) -> None:
    for key in keys:
        assert _ATTR_NAME_RE.match(key), f"bad attribute name: {key!r}"


def _baseline_record(**overrides: object) -> BaselineRecord:
    """A baseline record labeled ``"initial"`` unless ``label`` is overridden."""
    default = BaselineRecord(type="baseline", at=AT, label="initial", samples=({"total_ms": 100},))
    return default.model_copy(update=overrides) if overrides else default


# ---------------------------------------------------------------------------
# command_attributes — fixed fields
# ---------------------------------------------------------------------------


def test_command_attributes_when_called_does_map_session_id():
    record = command_record(name="measure", args={"samples": 5})

    result = command_attributes(record, session_id=SESSION_ID)

    assert result["gymrat.session.id"] == SESSION_ID


def test_command_attributes_when_called_does_map_name():
    record = command_record(name="measure")

    result = command_attributes(record, session_id=SESSION_ID)

    assert result["gymrat.command.name"] == "measure"


def test_command_attributes_when_called_does_map_exit_code():
    record = command_record(exit_code=0)

    result = command_attributes(record, session_id=SESSION_ID)

    assert result["gymrat.command.exit_code"] == 0


def test_command_attributes_when_called_does_map_duration_ms():
    record = command_record(duration_ms=1234)

    result = command_attributes(record, session_id=SESSION_ID)

    assert result["gymrat.command.duration_ms"] == 1234


def test_command_attributes_when_reason_present_does_map_reason():
    record = command_record(reason="budget-exceeded")

    result = command_attributes(record, session_id=SESSION_ID)

    assert result["gymrat.command.reason"] == "budget-exceeded"


def test_command_attributes_when_reason_none_does_omit_key():
    record = command_record(reason=None)

    result = command_attributes(record, session_id=SESSION_ID)

    assert "gymrat.command.reason" not in result


# ---------------------------------------------------------------------------
# command_attributes — seq
# ---------------------------------------------------------------------------


def test_command_attributes_when_seq_present_does_map_iteration_seq():
    record = command_record(seq=3)

    result = command_attributes(record, session_id=SESSION_ID)

    assert result["gymrat.iteration.seq"] == 3


def test_command_attributes_when_seq_none_does_omit_key():
    record = command_record(seq=None)

    result = command_attributes(record, session_id=SESSION_ID)

    assert "gymrat.iteration.seq" not in result


# ---------------------------------------------------------------------------
# command_attributes — args expansion
# ---------------------------------------------------------------------------


def test_command_attributes_when_args_scalar_does_map_each_key():
    record = command_record(
        args={"ref": "main", "samples": 5, "verbose": True, "threshold": 0.05},
    )

    result = command_attributes(record, session_id=SESSION_ID)

    assert result["gymrat.command.args.ref"] == "main"
    assert result["gymrat.command.args.samples"] == 5
    assert result["gymrat.command.args.verbose"] is True
    assert result["gymrat.command.args.threshold"] == 0.05


def test_command_attributes_when_args_non_scalar_does_skip():
    record = command_record(
        args={"paths": ["/a", "/b"], "config": {"key": "val"}, "empty": None, "ok": "yes"},
    )

    result = command_attributes(record, session_id=SESSION_ID)

    assert "gymrat.command.args.paths" not in result
    assert "gymrat.command.args.config" not in result
    assert "gymrat.command.args.empty" not in result
    assert result["gymrat.command.args.ok"] == "yes"


# ---------------------------------------------------------------------------
# command_attributes — attribute naming convention
# ---------------------------------------------------------------------------


def test_command_attributes_when_called_does_produce_valid_attribute_names():
    record = command_record(
        name="iterate",
        args={"session_id": SESSION_ID, "ref": "main"},
        reason="budget-exceeded",
        seq=2,
    )

    result = command_attributes(record, session_id=SESSION_ID)

    _assert_valid_attribute_names(result)


# ---------------------------------------------------------------------------
# record_event — IterationRecord
# ---------------------------------------------------------------------------


def test_record_event_when_iteration_does_return_event_name():
    record = iteration_record()

    name, _attrs = record_event(record)

    assert name == "gymrat.iteration"


def test_record_event_when_iteration_does_map_outcome():
    record = iteration_record(outcome="improved")

    _name, attrs = record_event(record)

    assert attrs["gymrat.iteration.outcome"] == "improved"


def test_record_event_when_iteration_does_map_delta_pct():
    record = iteration_record(primary=IterationPrimary(kind="geomean", delta_pct=-7.2))

    _name, attrs = record_event(record)

    assert attrs["gymrat.iteration.delta_pct"] == pytest.approx(-7.2)


def test_record_event_when_iteration_delta_pct_none_does_omit_key():
    record = iteration_record(primary=IterationPrimary(kind="geomean", delta_pct=None))

    _name, attrs = record_event(record)

    assert "gymrat.iteration.delta_pct" not in attrs


def test_record_event_when_iteration_does_map_seq():
    record = iteration_record(seq=3)

    _name, attrs = record_event(record)

    assert attrs["gymrat.iteration.seq"] == 3


# ---------------------------------------------------------------------------
# record_event — HookRecord
# ---------------------------------------------------------------------------


def test_record_event_when_hook_does_return_event_name():
    record = hook_record()

    name, _attrs = record_event(record)

    assert name == "gymrat.hook"


def test_record_event_when_hook_does_map_scalar_fields():
    record = hook_record(stage="before", exit_code=0)

    _name, attrs = record_event(record)

    assert attrs["gymrat.hook.stage"] == "before"
    assert attrs["gymrat.hook.exit_code"] == 0


# ---------------------------------------------------------------------------
# record_event — KeepRecord
# ---------------------------------------------------------------------------


def test_record_event_when_keep_does_return_event_name():
    record = committed_keep(1)

    name, _attrs = record_event(record)

    assert name == "gymrat.keep"


def test_record_event_when_keep_does_map_status():
    record = committed_keep(1)

    _name, attrs = record_event(record)

    assert attrs["gymrat.keep.status"] == "committed"


def test_record_event_when_keep_reason_present_does_map_reason():
    record = blocked_keep(1)

    _name, attrs = record_event(record)

    assert attrs["gymrat.keep.reason"] == "checks-failed"


def test_record_event_when_keep_reason_none_does_omit_key():
    record = committed_keep(1)

    _name, attrs = record_event(record)

    assert "gymrat.keep.reason" not in attrs


# ---------------------------------------------------------------------------
# record_event — BaselineRecord
# ---------------------------------------------------------------------------


def test_record_event_when_baseline_does_map_label():
    record = _baseline_record()

    _name, attrs = record_event(record)

    assert attrs["gymrat.baseline.label"] == "initial"


# ---------------------------------------------------------------------------
# record_event — StopRecord, DiscardRecord, FinalizeRecord
# ---------------------------------------------------------------------------


def test_record_event_when_stop_does_return_event_name():
    record = stop_record()

    name, _attrs = record_event(record)

    assert name == "gymrat.stop"


def test_record_event_when_discard_does_return_event_name():
    record = discard_record(2)

    name, _attrs = record_event(record)

    assert name == "gymrat.discard"


def test_record_event_when_finalize_does_return_event_name():
    record = finalize_record(branch="gymrat/test-final")

    name, _attrs = record_event(record)

    assert name == "gymrat.finalize"


# ---------------------------------------------------------------------------
# record_event — seq on non-iteration records
# ---------------------------------------------------------------------------


def test_record_event_when_hook_has_seq_does_map_iteration_seq():
    record = hook_record(seq=5)

    _name, attrs = record_event(record)

    assert attrs["gymrat.iteration.seq"] == 5


def test_record_event_when_stop_has_no_seq_does_omit_iteration_seq():
    record = stop_record()

    _name, attrs = record_event(record)

    assert "gymrat.iteration.seq" not in attrs


def test_record_event_when_baseline_does_not_carry_seq_field_does_omit_iteration_seq():
    assert "seq" not in BaselineRecord.model_fields

    record = _baseline_record(label="main")

    _name, attrs = record_event(record)

    assert "gymrat.iteration.seq" not in attrs


def test_record_event_when_finalize_does_not_carry_seq_field_does_omit_iteration_seq():
    assert "seq" not in FinalizeRecord.model_fields

    record = finalize_record(branch="gymrat/test-final")

    _name, attrs = record_event(record)

    assert "gymrat.iteration.seq" not in attrs


# ---------------------------------------------------------------------------
# record_event — no list/dict/None values
# ---------------------------------------------------------------------------


def test_record_event_when_called_does_never_produce_complex_values():
    records = [
        iteration_record(),
        hook_record(),
        stop_record(),
        committed_keep(1),
        _baseline_record(),
        discard_record(2),
        finalize_record(branch="gymrat/test-final"),
    ]

    for record in records:
        _name, attrs = record_event(record)
        for key, val in attrs.items():
            assert isinstance(val, (str, int, float, bool)), (
                f"{key!r} has disallowed type {type(val).__name__}"
            )


# ---------------------------------------------------------------------------
# record_event — attribute naming convention
# ---------------------------------------------------------------------------


def test_record_event_when_called_does_produce_valid_attribute_names():
    records = [
        iteration_record(),
        hook_record(),
        stop_record(),
        committed_keep(1),
    ]

    for record in records:
        _name, attrs = record_event(record)
        _assert_valid_attribute_names(attrs)


# ---------------------------------------------------------------------------
# command_span_inputs — shared span-building helper
# ---------------------------------------------------------------------------


def test_command_span_inputs_when_called_does_return_span_name():
    record = command_record(name="measure")

    result = command_span_inputs(record, session_id=SESSION_ID, line_number=3)

    assert result.name == "gymrat.command.measure"


def test_command_span_inputs_when_called_does_return_span_key():
    record = command_record(name="measure")

    result = command_span_inputs(record, session_id=SESSION_ID, line_number=5)

    assert result.key == "command:5"


def test_command_span_inputs_when_called_does_return_attributes_from_command_attributes():
    record = command_record(name="iterate", args={"samples": 10}, exit_code=0)

    result = command_span_inputs(record, session_id=SESSION_ID, line_number=2)

    assert result.attributes["gymrat.session.id"] == SESSION_ID
    assert result.attributes["gymrat.command.name"] == "iterate"
    assert result.attributes["gymrat.command.exit_code"] == 0
    assert result.attributes["gymrat.command.args.samples"] == 10


def test_command_span_inputs_when_traceparent_present_does_return_link():
    traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    record = command_record(name="measure", traceparent=traceparent)

    result = command_span_inputs(record, session_id=SESSION_ID, line_number=2)

    assert result.link is not None
    assert result.link.span_id == 0xB7AD6B7169203331


@pytest.mark.parametrize(
    "traceparent",
    [None, "not-valid"],
    ids=["absent", "malformed"],
)
def test_command_span_inputs_when_traceparent_missing_or_malformed_does_return_none_link(
    traceparent: str | None,
):
    record = command_record(name="measure", traceparent=traceparent)

    result = command_span_inputs(record, session_id=SESSION_ID, line_number=2)

    assert result.link is None


# ---------------------------------------------------------------------------
# all_attribute_names — enumerable namespace
# ---------------------------------------------------------------------------


def test_all_attribute_names_when_called_does_return_nonempty_frozenset():
    result = all_attribute_names()

    assert isinstance(result, frozenset)
    assert len(result) > 0


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("gymrat.session.id", id="session"),
        pytest.param("gymrat.command.name", id="command-name"),
        pytest.param("gymrat.command.exit_code", id="command-exit_code"),
        pytest.param("gymrat.command.duration_ms", id="command-duration_ms"),
        pytest.param("gymrat.command.reason", id="command-reason"),
        pytest.param("gymrat.run.head_sha", id="run-head_sha"),
        pytest.param("gymrat.run.max_minutes", id="run-max_minutes"),
        pytest.param("gymrat.run.cost_usd", id="run-cost_usd"),
        pytest.param("gymrat.turn.cost_usd", id="turn_and_follow_up-turn_cost_usd"),
        pytest.param("gymrat.follow_up.action", id="turn_and_follow_up-follow_up_action"),
        pytest.param("gymrat.cap.name", id="turn_and_follow_up-cap_name"),
        pytest.param("gymrat.iteration.seq", id="iteration-seq"),
        pytest.param("gymrat.iteration.outcome", id="iteration-outcome"),
        pytest.param("gymrat.iteration.delta_pct", id="iteration-delta_pct"),
        pytest.param("gymrat.baseline.label", id="record_derived_scalar_fields-baseline_label"),
        pytest.param("gymrat.hook.stage", id="record_derived_scalar_fields-hook_stage"),
        pytest.param("gymrat.hook.exit_code", id="record_derived_scalar_fields-hook_exit_code"),
        pytest.param("gymrat.hook.duration_ms", id="record_derived_scalar_fields-hook_duration_ms"),
        pytest.param(
            "gymrat.hook.stdout_bytes", id="record_derived_scalar_fields-hook_stdout_bytes"
        ),
        pytest.param("gymrat.hook.timed_out", id="record_derived_scalar_fields-hook_timed_out"),
        pytest.param("gymrat.keep.status", id="record_derived_scalar_fields-keep_status"),
        pytest.param("gymrat.stop.message", id="record_derived_scalar_fields-stop_message"),
        pytest.param("gymrat.finalize.branch", id="record_derived_scalar_fields-finalize_branch"),
        pytest.param("gymrat.finalize.commit", id="record_derived_scalar_fields-finalize_commit"),
        pytest.param("gymrat.finalize.message", id="record_derived_scalar_fields-finalize_message"),
        # discard has no scalar fields beyond seq, so no gymrat.discard.* names
        pytest.param("gymrat.command.args", id="args_pattern"),
        pytest.param("gen_ai.request.model", id="gen_ai-request_model"),
        pytest.param("gen_ai.provider.name", id="gen_ai-provider_name"),
    ],
)
def test_all_attribute_names_when_called_does_include_constant(name: str):
    result = all_attribute_names()

    assert name in result


def test_all_attribute_names_when_called_does_produce_valid_attribute_names():
    result = all_attribute_names()

    for name in result:
        if name.startswith("gen_ai."):
            continue
        assert _ATTR_NAME_RE.match(name), f"bad attribute name in namespace: {name!r}"


# ---------------------------------------------------------------------------
# all_attribute_names — naming rule covers every record type with all optional fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(_baseline_record(duration_ms=42), id="baseline"),
        pytest.param(
            iteration_record(
                duration_ms=500,
                measured_tree="experiment",
                confirm=Confirm(
                    ran=True,
                    filtered=("total_ms",),
                    samples=PairedSamples(
                        experiment=({"total_ms": 14050},),
                        baseline=({"total_ms": 15200},),
                    ),
                ),
            ),
            id="iteration",
        ),
        pytest.param(
            committed_keep(1, commit="a" * 40, message="cache the regex", reason=None),
            id="committed_keep",
        ),
        pytest.param(blocked_keep(1, reason="checks-failed"), id="blocked_keep"),
        pytest.param(discard_record(2), id="discard"),
        pytest.param(hook_record(stderr_bytes=120), id="hook"),
        pytest.param(finalize_record(branch="gymrat/test-final"), id="finalize"),
        pytest.param(stop_record(), id="stop"),
    ],
)
def test_all_attribute_names_when_record_event_called_does_contain_every_emitted_key(
    record: SessionLogRecord,
):
    # All optional fields are set so a new field with an illegal name fails this test.
    namespace = all_attribute_names()

    _name, attrs = record_event(record)

    assert set(attrs) <= namespace


def test_all_attribute_names_when_command_attributes_called_does_contain_every_emitted_key():
    namespace = all_attribute_names()

    record = command_record(
        name="iterate",
        args={"ref": "main", "samples": 5},
        reason="budget-exceeded",
        seq=2,
    )

    result = command_attributes(record, session_id=SESSION_ID)

    for key in result:
        if key.startswith("gymrat.command.args."):
            continue
        assert key in namespace, (
            f"attribute {key!r} from command_attributes missing in all_attribute_names()"
        )
