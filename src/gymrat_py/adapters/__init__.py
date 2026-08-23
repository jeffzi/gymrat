"""Closed registry of the built-in bench-harness adapters.

The set of adapters gymrat ships is fixed at import time: nothing registers an
adapter at runtime, so :func:`get_adapter` is a lookup in a closed mapping rather
than a plugin dispatch. A name that misses the mapping is a user typo, not a
missing plugin, so the raised :class:`~gymrat_py.errors.GymratError` carries a
``hint`` listing every valid name — the same inventory the CLI shows when it
reports what adapters are available.

This module is also the package's public surface: the adapter contract types, the
two built-in singletons, and the registry accessors are re-exported here so
callers import from ``gymrat_py.adapters`` rather than reaching into submodules.
"""

from gymrat_py.adapters.metric_lines import metric_lines_adapter
from gymrat_py.adapters.mitata import mitata_adapter
from gymrat_py.adapters.types import Adapter, AdapterError, MetricDefaults, WarnSink
from gymrat_py.errors import GymratError

__all__ = [
    "ADAPTER_NAMES",
    "Adapter",
    "AdapterError",
    "MetricDefaults",
    "WarnSink",
    "get_adapter",
    "metric_lines_adapter",
    "mitata_adapter",
]

_ADAPTERS: dict[str, Adapter] = {
    metric_lines_adapter.name: metric_lines_adapter,
    mitata_adapter.name: mitata_adapter,
}

ADAPTER_NAMES: tuple[str, ...] = tuple(sorted(_ADAPTERS))
"""The built-in adapter names, sorted, for display and error hints."""


def get_adapter(name: str) -> Adapter:
    """Return the built-in adapter registered under ``name``.

    Args:
        name: The adapter name to look up, e.g. ``"metric-lines"`` or ``"mitata"``.

    Returns:
        The singleton :class:`~gymrat_py.adapters.types.Adapter` for ``name``.

    Raises:
        GymratError: When ``name`` is not a built-in adapter. The message names
            the offending value and the ``hint`` lists every valid name.
    """
    try:
        return _ADAPTERS[name]
    except KeyError:
        msg = f'Unknown adapter: "{name}".'
        hint = f"valid adapters are: {', '.join(ADAPTER_NAMES)}"
        raise GymratError(msg, hint=hint) from None
