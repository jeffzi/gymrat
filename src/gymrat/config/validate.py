"""Standalone config validators — cross-field checks and flag/runbook guards."""

import json
import os
import stat
from pathlib import Path

from gymrat.config.schema import invalid_value_message, validate_and_convert
from gymrat.config.types import (
    FILTER_PLACEHOLDER,
    GEOMEAN_PRIMARY,
    BenchlessConfig,
    CliFlags,
)
from gymrat.errors import GymratError


def flag_problem(field_name: str, value: str | None) -> str | None:
    """Return a problem string when a flag is blank (empty or whitespace-only).

    Flags bypass the file schema, so ``--bench ""`` or ``--bench "   "`` is the
    one way a blank string reaches a settled field. The message names the flag,
    not the config key, because the flag is what the user typed.
    """
    if value is not None and not value.strip():
        return invalid_value_message(f"--{field_name}", "a non-empty string", value)
    return None


def assert_flag_not_empty(field_name: str, value: str | None) -> None:
    """Raise on a blank flag value."""
    problem = flag_problem(field_name, value)
    if problem is not None:
        raise GymratError(problem)


def loop_key_problems(config: BenchlessConfig) -> list[str]:
    """Return the cross-field violations the schema alone cannot express.

    ``filter`` must carry its placeholder, and ``stop.target_value`` only makes
    sense when ``primary`` names a metric — the geomean is a ratio, not a value.
    """
    problems: list[str] = []
    if config.filter is not None and FILTER_PLACEHOLDER not in config.filter:
        problems.append(
            invalid_value_message(
                "filter",
                f"a string containing the {FILTER_PLACEHOLDER} placeholder",
                config.filter,
            )
        )
    if (
        config.stop is not None
        and config.stop.target_value is not None
        and config.primary == GEOMEAN_PRIMARY
    ):
        problems.append(
            "Invalid config value for stop.target_value: it needs primary to name a metric, "
            f"not {json.dumps(GEOMEAN_PRIMARY)}"
        )
    return problems


def validate_loop_keys(config: BenchlessConfig) -> None:
    """Raise on the first cross-field violation."""
    problems = loop_key_problems(config)
    if problems:
        raise GymratError(problems[0])


def validate_config_dict(config: dict[str, object]) -> None:
    """Validate an in-memory config dict the same way a loaded ``gymrat.toml`` is.

    Runs the strict schema (``extra="forbid"``) and the cross-field loop-key
    checks over ``config``, raising a :class:`GymratError` on the first problem.
    Lets a writer (the init scaffold) reject a config before touching disk without
    a temp-file round-trip.
    """
    from gymrat.config.resolve import merge_config  # noqa: PLC0415 -- break validate↔resolve cycle

    config_file, problems = validate_and_convert(config)
    if problems:
        raise GymratError(problems[0])
    if config_file is None:
        return
    validate_loop_keys(merge_config(CliFlags(), config_file))


def runbook_problem(runbook: str, base_dir: str | Path | None) -> str | None:
    """Return a problem string when ``runbook`` does not name an existing file.

    Resolved against ``base_dir`` (or the cwd when ``None``), matching how the
    implicit ``gymrat.toml`` lookup is anchored — a runbook path is authored
    relative to the repository the config lives in.
    """
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    resolved = Path(os.path.normpath(base / runbook))
    try:
        info = resolved.stat()
    except FileNotFoundError:
        info = None
    except (OSError, ValueError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
        return f"Cannot read runbook path {json.dumps(runbook)}: {reason}"
    if info is None or not stat.S_ISREG(info.st_mode):
        return invalid_value_message("runbook", "a path to an existing file", runbook)
    return None


def assert_runbook_exists(runbook: str, base_dir: str | Path | None) -> None:
    """Raise when the runbook does not name an existing file."""
    problem = runbook_problem(runbook, base_dir)
    if problem is not None:
        raise GymratError(problem)
