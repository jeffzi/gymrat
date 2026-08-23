import dataclasses

import pytest

from gymrat_py.adapters.defaults import (
    DEFAULT_GATING,
    DEFAULT_METRIC_KIND,
    defaults_from_suffixes,
)
from gymrat_py.adapters.types import (
    Adapter,
    AdapterError,
    MetricDefaults,
    WarnSink,
    warn_to_stderr,
)
from gymrat_py.errors import GymratError

# ---------------------------------------------------------------------------
# AdapterError
# ---------------------------------------------------------------------------


def test_adapter_error_when_raised_does_subclass_gymrat_error():
    error = AdapterError("unparseable output")

    with pytest.raises(GymratError):
        raise error


def test_adapter_error_when_stringified_does_return_message():
    assert str(AdapterError("unparseable output")) == "unparseable output"


def test_adapter_error_when_given_hint_does_store_hint():
    error = AdapterError("unparseable output", hint="emit newline-delimited JSON")

    assert error.hint == "emit newline-delimited JSON"


def test_adapter_error_when_no_hint_does_default_hint_to_none():
    assert AdapterError("unparseable output").hint is None


# ---------------------------------------------------------------------------
# MetricDefaults
# ---------------------------------------------------------------------------


def test_metric_defaults_when_constructed_does_store_all_fields():
    defaults = MetricDefaults(direction="lower", unit="ns", kind="time", short_name="bench")

    assert defaults.direction == "lower"
    assert defaults.unit == "ns"
    assert defaults.kind == "time"
    assert defaults.short_name == "bench"


def test_metric_defaults_when_only_direction_given_does_default_rest_to_none():
    defaults = MetricDefaults(direction="lower")

    assert defaults.unit is None
    assert defaults.kind is None
    assert defaults.short_name is None


def test_metric_defaults_when_field_assigned_does_raise_frozen_instance_error():
    defaults = MetricDefaults(direction="lower")

    with pytest.raises(dataclasses.FrozenInstanceError):
        # The write is rejected at runtime; the type checker flags it statically, so the
        # suppression documents the intentional frozen-field violation under test.
        defaults.unit = "bytes"  # pyrefly: ignore


def test_metric_defaults_when_fields_equal_does_compare_equal():
    assert MetricDefaults(direction="lower", unit="ns") == MetricDefaults(
        direction="lower", unit="ns"
    )


def test_metric_defaults_when_fields_differ_does_compare_unequal():
    assert MetricDefaults(direction="lower", unit="ns") != MetricDefaults(
        direction="lower", unit="bytes"
    )


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------


class _StructuralAdapter:
    """Satisfies ``Adapter`` structurally without inheriting from it."""

    name = "structural"

    def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
        return {}

    def defaults(self, metric_name: str) -> MetricDefaults:
        return MetricDefaults(direction="lower")


class _MissingParse:
    """Lacks ``parse``, so it does not satisfy the ``Adapter`` protocol."""

    name = "incomplete"

    def defaults(self, metric_name: str) -> MetricDefaults:
        return MetricDefaults(direction="lower")


def test_adapter_when_object_has_all_members_does_satisfy_protocol_without_inheriting():
    assert isinstance(_StructuralAdapter(), Adapter)


def test_adapter_when_object_missing_parse_does_not_satisfy_protocol():
    assert not isinstance(_MissingParse(), Adapter)


# ---------------------------------------------------------------------------
# defaults_from_suffixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        (
            "bench/time",
            MetricDefaults(direction="lower", unit="ns", kind="time", short_name="bench"),
        ),
        (
            "a/b/time",
            MetricDefaults(direction="lower", unit="ns", kind="time", short_name="a/b"),
        ),
        (
            "bench/heap",
            MetricDefaults(direction="lower", unit="bytes", kind="memory", short_name="bench"),
        ),
        (
            "/time",
            MetricDefaults(direction="lower", unit="ns", kind="time", short_name="/time"),
        ),
        (
            "/heap",
            MetricDefaults(direction="lower", unit="bytes", kind="memory", short_name="/heap"),
        ),
        ("foo", MetricDefaults(direction="lower")),
        ("latency", MetricDefaults(direction="lower")),
        ("memory", MetricDefaults(direction="lower")),
        ("custom_metric", MetricDefaults(direction="lower")),
        ("test/throughput", MetricDefaults(direction="lower")),
        ("test/ops", MetricDefaults(direction="lower")),
    ],
)
def test_defaults_from_suffixes_when_given_metric_name_does_return_expected_defaults(
    metric_name: str,
    expected: MetricDefaults,
):
    assert defaults_from_suffixes(metric_name) == expected


# ---------------------------------------------------------------------------
# Seam-7 constants
# ---------------------------------------------------------------------------


def test_default_metric_kind_when_referenced_does_equal_other():
    assert DEFAULT_METRIC_KIND == "other"


def test_default_gating_when_referenced_does_equal_true():
    assert DEFAULT_GATING is True
