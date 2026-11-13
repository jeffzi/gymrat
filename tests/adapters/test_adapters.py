import pytest

from gymrat.adapters.contract import (
    MEMORY_SUFFIX,
    AdapterError,
    defaults_from_suffixes,
)
from gymrat.adapters.registry import get_adapter
from gymrat.errors import GymratError

# ---------------------------------------------------------------------------
# get_adapter — unknown name
# ---------------------------------------------------------------------------


def test_get_adapter_when_name_unknown_does_raise_gymrat_error_describing_valid_adapters():
    with pytest.raises(GymratError) as excinfo:
        get_adapter("unknown")

    error = excinfo.value
    # An adapter error would gain a class-name prefix on the CLI; this one names no adapter fault.
    assert not isinstance(error, AdapterError)
    assert '"unknown"' in str(error)
    assert all(name in (error.hint or "") for name in ("metric-lines", "mitata", "tinybench"))


# ---------------------------------------------------------------------------
# defaults_from_suffixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        pytest.param(f"bench{MEMORY_SUFFIX}", ("bench", "memory"), id="memory-suffix"),
        pytest.param("foo", (None, None), id="no-suffix"),
    ],
)
def test_defaults_from_suffixes_when_kind_suffix_varies_does_derive_defaults_from_it(
    metric_name: str, expected: tuple[str | None, str | None]
):
    defaults = defaults_from_suffixes(metric_name)

    assert (defaults.short_name, defaults.kind) == expected
