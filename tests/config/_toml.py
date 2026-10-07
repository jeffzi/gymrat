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


#: The loop keys a fully configured ``gymrat.toml`` carries beside its bench.
LOOP_CONFIG: dict[str, object] = {
    "checks": "npm test",
    "filter": "npm run bench -- {names}",
    "primary": "decode/time",
    "stop": {"target_value": 1.5, "max_iterations": 20},
    "hooks": {"before": "npm run warm-cache", "after": "npm run cool-down"},
}


def write_raw(directory: Path, text: str, *, name: str = "gymrat.toml") -> Path:
    """Write ``text`` verbatim as the config file ``name`` in ``directory``.

    Args:
        directory: Where the file is written.
        text: The file's exact content.
        name: The file name; the implicit ``gymrat.toml`` unless a test names
            an explicit config path.

    Returns:
        The written file's path.
    """
    config_path = directory / name
    config_path.write_text(text, encoding="utf-8")
    return config_path


def write_config(directory: Path, content: dict[str, object], *, name: str = "gymrat.toml") -> Path:
    """Write ``content`` as TOML to the config file ``name`` in ``directory``.

    Args:
        directory: Where the file is written.
        content: The table the file holds.
        name: The file name; the implicit ``gymrat.toml`` unless a test names
            an explicit config path.

    Returns:
        The written file's path.
    """
    return write_raw(directory, tomli_w.dumps(content), name=name)
