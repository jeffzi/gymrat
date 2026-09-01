import pytest

from gymrat.adapters import (
    ADAPTER_NAMES,
    Adapter,
    get_adapter,
    metric_lines_adapter,
    mitata_adapter,
)
from gymrat.errors import GymratError

# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------


def test_adapters_package_when_imported_does_export_public_surface():
    from gymrat import adapters

    assert set(adapters.__all__) == {
        "AdapterError",
        "Adapter",
        "MetricDefaults",
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
