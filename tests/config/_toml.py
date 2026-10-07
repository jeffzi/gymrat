"""Writers for the ``gymrat.toml`` the config resolution tests read back."""

from pathlib import Path

import tomli_w

#: Bit length of the hexadecimal literal in ``HUGE_HEX_LITERAL``.
HUGE_HEX_BITS = 20_000

#: A hexadecimal integer literal TOML parses, whose decimal form runs past the
#: interpreter's limit on integer-to-text conversion.
HUGE_HEX_LITERAL = "0x" + "F" * (HUGE_HEX_BITS // 4)

#: A document whose decimal integer literal has more digits than the
#: interpreter converts, so the TOML parser raises a plain ``ValueError``.
DIGIT_LIMIT_DOCUMENT = "samples = 1" + "0" * 5000

#: A document nested deeper than the interpreter's recursion limit, so the TOML
#: parser raises ``RecursionError``.
DEEP_NESTING_DOCUMENT = "bench = " + "[" * 100_000 + "]" * 100_000


def write_raw(directory: Path, text: str) -> Path:
    """Write ``text`` verbatim as ``directory``'s ``gymrat.toml`` and return its path."""
    config_path = directory / "gymrat.toml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def write_config(directory: Path, content: dict[str, object]) -> Path:
    """Write ``content`` as ``directory``'s ``gymrat.toml`` and return its path."""
    return write_raw(directory, tomli_w.dumps(content))
