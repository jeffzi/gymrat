"""The closed registry of built-in bench-harness adapters.

The set of adapters is fixed at import time: nothing registers an adapter at
runtime, so :func:`get_adapter` is a lookup in a closed mapping rather than a
plugin dispatch. A name that misses the mapping is a user typo, not a missing
plugin, so the raised :class:`~gymrat.errors.GymratError` carries a ``hint``
listing every valid name.
"""

from gymrat.adapters.contract import Adapter
from gymrat.adapters.metric_lines import metric_lines_adapter
from gymrat.adapters.mitata import mitata_adapter
from gymrat.adapters.tinybench import tinybench_adapter
from gymrat.errors import GymratError

_ADAPTERS: dict[str, Adapter] = {
    metric_lines_adapter.name: metric_lines_adapter,
    mitata_adapter.name: mitata_adapter,
    tinybench_adapter.name: tinybench_adapter,
}


def get_adapter(name: str) -> Adapter:
    """Return the built-in adapter registered under ``name``.

    Args:
        name: The adapter name to look up, e.g. ``"metric-lines"`` or ``"mitata"``.

    Returns:
        The singleton :class:`~gymrat.adapters.contract.Adapter` for ``name``.

    Raises:
        GymratError: When ``name`` is not a built-in adapter. The message names
            the offending value and the ``hint`` lists every valid name.
    """
    try:
        return _ADAPTERS[name]
    except KeyError:
        msg = f'Unknown adapter: "{name}".'
        hint = f"valid adapters are: {', '.join(sorted(_ADAPTERS))}"
        raise GymratError(msg, hint=hint) from None
