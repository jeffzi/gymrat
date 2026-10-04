"""Writers for the ``gymrat.toml`` the config resolution tests read back."""

from pathlib import Path

import tomli_w


def write_raw(directory: Path, text: str) -> Path:
    """Write ``text`` verbatim as ``directory``'s ``gymrat.toml`` and return its path."""
    config_path = directory / "gymrat.toml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def write_config(directory: Path, content: dict[str, object]) -> Path:
    """Write ``content`` as ``directory``'s ``gymrat.toml`` and return its path."""
    return write_raw(directory, tomli_w.dumps(content))
