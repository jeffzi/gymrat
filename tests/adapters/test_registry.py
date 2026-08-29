import json

import pytest

from gymrat.adapters import (
    ADAPTER_NAMES,
    Adapter,
    MetricDefaults,
    get_adapter,
    metric_lines_adapter,
    mitata_adapter,
)
from gymrat.errors import GymratError

_MITATA_STDOUT = json.dumps(
    {
        "benchmarks": [
            {"alias": "test", "runs": [{"name": "test", "args": {}, "stats": {"p50": 42}}]}
        ]
    }
)


# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------


def test_adapters_package_when_imported_does_export_public_surface():
    from gymrat import adapters

    assert set(adapters.__all__) == {
        "AdapterError",
        "Adapter",
        "MetricDefaults",
        "WarnSink",
        "get_adapter",
        "ADAPTER_NAMES",
        "metric_lines_adapter",
        "mitata_adapter",
    }


def test_adapter_names_when_referenced_does_list_builtins_sorted():
    assert ADAPTER_NAMES == ("metric-lines", "mitata")


# ---------------------------------------------------------------------------
# get_adapter — registered names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "singleton"),
    [
        pytest.param("metric-lines", metric_lines_adapter, id="metric-lines"),
        pytest.param("mitata", mitata_adapter, id="mitata"),
    ],
)
def test_get_adapter_when_name_registered_does_return_matching_singleton(
    name: str, singleton: Adapter
):
    adapter = get_adapter(name)

    assert adapter is singleton
    assert adapter.name == name


@pytest.mark.parametrize(
    "name",
    [pytest.param("metric-lines", id="metric-lines"), pytest.param("mitata", id="mitata")],
)
def test_get_adapter_when_name_registered_does_return_object_satisfying_protocol(name: str):
    assert isinstance(get_adapter(name), Adapter)


@pytest.mark.parametrize(
    ("name", "stdout", "expected"),
    [
        pytest.param("metric-lines", "METRIC foo=42", {"foo": 42.0}, id="metric-lines"),
        pytest.param("mitata", _MITATA_STDOUT, {"test/time": 42.0}, id="mitata"),
    ],
)
def test_get_adapter_when_registered_adapter_parses_does_return_expected_metrics(
    name: str, stdout: str, expected: dict[str, float]
):
    assert get_adapter(name).parse(stdout) == expected


@pytest.mark.parametrize(
    ("name", "metric_name", "expected"),
    [
        pytest.param(
            "metric-lines", "test_metric", MetricDefaults(direction="lower"), id="metric-lines"
        ),
        pytest.param(
            "mitata",
            "test/time",
            MetricDefaults(direction="lower", unit="ns", kind="time", short_name="test"),
            id="mitata",
        ),
    ],
)
def test_get_adapter_when_registered_adapter_computes_defaults_does_return_expected(
    name: str, metric_name: str, expected: MetricDefaults
):
    assert get_adapter(name).defaults(metric_name) == expected


# ---------------------------------------------------------------------------
# get_adapter — unknown name
# ---------------------------------------------------------------------------


def test_get_adapter_when_name_unknown_does_raise_gymrat_error_describing_valid_adapters():
    with pytest.raises(GymratError, match=r"Unknown adapter") as excinfo:
        get_adapter("unknown")

    error = excinfo.value
    assert type(error) is GymratError
    assert str(error) == 'Unknown adapter: "unknown".'
    assert error.hint == "valid adapters are: metric-lines, mitata"
