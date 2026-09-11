"""Shared helper for nulling non-finite floats before a JSON dump.

``json.dumps`` writes invalid ``NaN``/``Infinity`` literals by default. Both the
session-log store (``session/store.py``) and the report JSON builders
(``report/json_doc.py``) need the JavaScript-compatible contract instead: a
non-finite float serializes as JSON ``null``.
"""

import math


def null_non_finite(value: object) -> object:
    """Recursively replace non-finite floats with ``None``, matching ``JSON.stringify``.

    Args:
        value: The JSON-bound value to sanitize. Dicts and lists are walked
            recursively; non-finite floats (``NaN``, ``±Infinity``) are replaced
            with ``None``.

    Returns:
        A structure identical to *value* with every non-finite float replaced by
        ``None``.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: null_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [null_non_finite(item) for item in value]
    return value
