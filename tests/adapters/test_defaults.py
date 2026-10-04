import dataclasses

import pytest

from gymrat.adapters import (
    AdapterError,
    MetricDefaults,
    defaults_from_suffixes,
)
from gymrat.errors import GymratError
from gymrat.sampling import DEFAULT_METRIC_KIND

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
# defaults_from_suffixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        (
            "bench#time",
            MetricDefaults(direction="lower", unit="ns", kind="time", short_name="bench"),
        ),
        (
            "a/b#time",
            MetricDefaults(direction="lower", unit="ns", kind="time", short_name="a/b"),
        ),
        (
            "bench#heap",
            MetricDefaults(direction="lower", unit="bytes", kind="memory", short_name="bench"),
        ),
        (
            "#time",
            MetricDefaults(direction="lower", unit="ns", kind="time", short_name="#time"),
        ),
        (
            "#heap",
            MetricDefaults(direction="lower", unit="bytes", kind="memory", short_name="#heap"),
        ),
        ("foo", MetricDefaults(direction="lower")),
        ("test/throughput", MetricDefaults(direction="lower")),
    ],
)
def test_defaults_from_suffixes_when_given_metric_name_does_return_expected_defaults(
    metric_name: str,
    expected: MetricDefaults,
):
    assert defaults_from_suffixes(metric_name) == expected


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------


def test_default_metric_kind_when_referenced_does_equal_other():
    assert DEFAULT_METRIC_KIND == "other"
