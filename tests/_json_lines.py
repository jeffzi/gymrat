"""Rewrite JSON lines to hold a bare ``NaN`` number literal.

This is test-support code, not a test module: it carries no test functions of
its own. :func:`with_raw_number` is shared by the tests that feed a reader a
line holding a bare ``NaN``.
"""

import json


def with_raw_number(line: str, keys: tuple[str, ...]) -> str:
    """Rewrite a JSON ``line`` so the value at ``keys`` is the bare number ``NaN``.

    ``json.dumps`` writes ``float("nan")`` as the bare ``NaN`` literal, which a
    strict JSON reader must reject.

    Args:
        line: A JSON object line.
        keys: The key path from the top-level object to the value to replace.

    Returns:
        The rewritten JSON line.
    """
    data = json.loads(line)
    node = data
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = float("nan")
    return json.dumps(data)
