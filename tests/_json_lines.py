"""Rewrite JSON lines to hold number literals ``json.dumps`` refuses to write.

This is test-support code, not a test module: it carries no test functions of
its own. :func:`with_raw_number` is shared by the tests that feed a reader a
line holding a bare ``NaN``.
"""

import json

_RAW_NUMBER = "raw-number-placeholder"


def with_raw_number(line: str, keys: tuple[str, ...], literal: str) -> str:
    """Rewrite a JSON ``line`` so the value at ``keys`` is the bare number text ``literal``.

    ``json.dumps`` cannot emit ``NaN``-style or overflowing literals as bare
    numbers, so a placeholder string stands in for the value and is replaced,
    quotes included, once the line is dumped.

    Args:
        line: A JSON object line.
        keys: The key path from the top-level object to the value to replace.
        literal: The number text to splice in unquoted, such as ``"NaN"`` or ``"1e999"``.

    Returns:
        The rewritten JSON line.
    """
    data = json.loads(line)
    node = data
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = _RAW_NUMBER
    return json.dumps(data).replace(f'"{_RAW_NUMBER}"', literal)
