"""Ensures every telemetry attribute has a row in README.md."""

from __future__ import annotations

import re
from pathlib import Path

from gymrat.telemetry.attributes import all_attribute_names

_README = Path(__file__).resolve().parents[2] / "README.md"
_HEADING = "### Attribute reference"
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_PLACEHOLDER_SUFFIX_RE = re.compile(r"\.<[^>]+>$")


def _parse_readme_attribute_names() -> frozenset[str]:
    """Extract attribute names from the first column of the README table.

    Returns:
        Attribute names found under the ``### Attribute reference`` heading.
    """
    lines = _README.read_text().splitlines()

    in_section = False
    header_seen = False
    names: set[str] = set()

    for line in lines:
        if line.strip() == _HEADING:
            in_section = True
            continue

        if not in_section:
            continue

        # Stop at the next heading of equal or higher level.
        if line.startswith("#") and not line.startswith("####"):
            break

        stripped = line.strip()

        # Skip the separator row (e.g. "| --- | --- |").
        if stripped.startswith("| -"):
            header_seen = True
            continue

        # Skip the header row itself (first pipe-delimited row before separator).
        if not header_seen:
            continue

        if not stripped.startswith("|"):
            continue

        first_col = stripped.split("|")[1]
        match = _BACKTICK_RE.search(first_col)
        if match:
            name = match.group(1)
            # Normalize pattern placeholder: `gymrat.command.args.<key>` -> `gymrat.command.args`
            name = _PLACEHOLDER_SUFFIX_RE.sub("", name)
            names.add(name)

    return frozenset(names)


def test_all_attribute_names_when_compared_to_readme_table_does_match_exactly() -> None:
    readme_names = _parse_readme_attribute_names()
    code_names = all_attribute_names()

    assert readme_names == code_names
