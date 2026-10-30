"""Writers for the ``gymrat.toml`` files tests put on disk, and a pre-existing one.

Shared across the suite: the config, scaffold, and command tests all write their
``gymrat.toml`` through these.
"""

from pathlib import Path

import tomli_w

#: A ``gymrat.toml`` already on disk before a scaffold, so a test can see it left untouched.
EXISTING_CONFIG = 'bench = "old"\n'


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
