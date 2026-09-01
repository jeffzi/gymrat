"""Config file I/O and the two public loader entry points."""

import tomllib
from pathlib import Path

from gymrat.config.schema import validate_and_convert
from gymrat.config.types import ConfigFile, ConfigFileResult
from gymrat.errors import GymratError

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _read_source(path: Path) -> tuple[str | None, str | None]:
    """Read the config file, returning ``(text, problem)`` rather than raising.

    On success the first element is the file content (BOM-stripped) and the
    second is ``None``.  When the file does not exist both elements are
    ``None``.  On any other read failure the first element is ``None`` and
    the second describes the error.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
        return None, f"Cannot read config file at {path}: {reason}"
    return text.removeprefix("﻿"), None


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------


def _validate_read(
    text: str,
    config_path: Path,
) -> tuple[ConfigFile | None, list[str]]:
    """Turn file content into a config or a problem list.

    Shared by the collecting loader and the throwing loader: it never raises, so
    each caller decides whether a non-empty problem list becomes an exception or
    a returned result.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return None, [f"Failed to parse config file at {config_path}: {exc}"]

    return validate_and_convert(data)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_config_file_collecting(path: str | Path, *, required: bool) -> ConfigFileResult:
    """Load and validate a config file, collecting every problem.

    Args:
        path: Path to the ``gymrat.toml`` file.
        required: When ``True``, an absent file is itself a reported problem.

    Returns:
        A :class:`ConfigFileResult` carrying the parsed config (when valid),
        whether the file existed, and every validation problem found.
    """
    config_path = Path(path)
    text, read_problem = _read_source(config_path)
    if read_problem is not None:
        return ConfigFileResult(config_file=None, exists=True, problems=[read_problem])
    if text is None:
        if required:
            return ConfigFileResult(
                config_file=None,
                exists=False,
                problems=[f"Config file not found at {config_path}"],
            )
        return ConfigFileResult(config_file=ConfigFile(), exists=False, problems=[])

    config_file, problems = _validate_read(text, config_path)
    return ConfigFileResult(config_file=config_file, exists=True, problems=problems)


def load_config_file(path: str | Path, *, required: bool = False) -> ConfigFile:
    """Load and validate a config file, raising on the first problem.

    Args:
        path: Path to the ``gymrat.toml`` file.
        required: When ``True``, an absent file raises rather than returning an
            empty config.

    Returns:
        The parsed :class:`ConfigFile`; an empty one when the file is absent and
        not required.

    Raises:
        GymratError: On any read, parse, or validation problem.
    """
    result = load_config_file_collecting(path, required=required)
    if result.problems:
        raise GymratError(result.problems[0])
    return result.config_file if result.config_file is not None else ConfigFile()
