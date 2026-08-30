"""Config file I/O and the two public loader entry points."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from gymrat.config.schema import validate_and_convert
from gymrat.config.types import ConfigFile, ConfigFileResult
from gymrat.errors import GymratError

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ReadOk:
    """The config file was read; ``text`` is its content with any BOM stripped."""

    text: str


@dataclass(frozen=True, slots=True)
class _ReadAbsent:
    """The config file does not exist."""


@dataclass(frozen=True, slots=True)
class _ReadError:
    """The config file could not be read; ``message`` describes why."""

    message: str


_ReadResult = _ReadOk | _ReadAbsent | _ReadError


def _read_source(path: Path) -> _ReadResult:
    """Read the config file, returning the outcome as data rather than raising.

    Strips a leading UTF-8 BOM. Returns :class:`_ReadAbsent` when the file does
    not exist and :class:`_ReadError` for any other read failure (e.g. the path
    is a directory), leaving each caller free to raise or collect the problem.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _ReadAbsent()
    except (OSError, ValueError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
        return _ReadError(f"Cannot read config file at {path}: {reason}")
    return _ReadOk(text.removeprefix("﻿"))


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------


def _validate_read(
    source: _ReadOk | _ReadError,
    config_path: Path,
) -> tuple[ConfigFile | None, list[str]]:
    """Turn a successful read (or read error) into a config or a problem list.

    Shared by the collecting loader and the throwing loader: it never raises, so
    each caller decides whether a non-empty problem list becomes an exception or
    a returned result.
    """
    if isinstance(source, _ReadError):
        return None, [source.message]

    try:
        data = tomllib.loads(source.text)
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
    source = _read_source(config_path)
    if isinstance(source, _ReadAbsent):
        if required:
            return ConfigFileResult(
                config_file=None,
                exists=False,
                problems=[f"Config file not found at {config_path}"],
            )
        return ConfigFileResult(config_file=ConfigFile(), exists=False, problems=[])

    config_file, problems = _validate_read(source, config_path)
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
