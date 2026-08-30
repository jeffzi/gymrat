"""Collecting config pipeline for gymrat.

:func:`inspect_config` settles a configuration exactly like
:func:`gymrat.config.resolve_benchless_config`, but never raises: every flag,
env var, file, schema, and cross-field problem is gathered into one list so a
caller (a doctor/status command) can report them all at once. A settled config is
returned only when that list is empty.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path

from gymrat.config import (
    CONFIG_FILENAME,
    BenchlessConfig,
    CliFlags,
    ConfigFile,
    find_implicit_base,
    flag_problem,
    load_config_file_collecting,
    loop_key_problems,
    merge_config,
    runbook_problem,
)
from gymrat.config_env import (
    NUMBER_ENV_FIELDS,
    STRING_ENV_FIELDS,
    env_string_result,
)


@dataclass(frozen=True, slots=True)
class ConfigInspection:
    """Outcome of a collecting inspection.

    ``config`` and ``bench`` are populated only when ``problems`` is empty;
    ``bench`` lives here rather than on ``config`` because it has no default and a
    benchless settlement may legitimately lack one.
    """

    config_path: str | None
    problems: list[str]
    config: BenchlessConfig | None = None
    bench: str | None = None


def _collect_flag_problems(flags: CliFlags) -> list[str]:
    problems: list[str] = []
    for field_name in ("bench", "prepare", "adapter", "config"):
        problem = flag_problem(field_name, getattr(flags, field_name))
        if problem is not None:
            problems.append(problem)
    return problems


def _collect_env_flags(flags: CliFlags) -> tuple[CliFlags, list[str]]:
    """Read every ``GYMRAT_*`` field flag whose flag is unset, collecting problems.

    An env var is consulted only when its flag is ``None`` (a flag always wins
    without the env var's validation firing). ``GYMRAT_CONFIG`` is handled in
    :func:`_resolve_config_source` because it selects the file, not a field.
    """
    problems: list[str] = []
    strings: dict[str, str] = {}
    for field_name, env_var in STRING_ENV_FIELDS:
        if getattr(flags, field_name) is None:
            result = env_string_result(env_var)
            if result.problem is not None:
                problems.append(result.problem)
            if result.value is not None:
                strings[field_name] = str(result.value)
    numbers: dict[str, int] = {}
    for field_name, env_var, reader in NUMBER_ENV_FIELDS:
        if getattr(flags, field_name) is None:
            result = reader(env_var)
            if result.problem is not None:
                problems.append(result.problem)
            if isinstance(result.value, int):
                numbers[field_name] = result.value
    env_flags = CliFlags(
        bench=strings.get("bench"),
        prepare=strings.get("prepare"),
        adapter=strings.get("adapter"),
        samples=numbers.get("samples"),
        timeout=numbers.get("timeout"),
    )
    return env_flags, problems


def _build_effective_flags(flags: CliFlags, env_flags: CliFlags) -> CliFlags:
    """Layer flags over env values: flag > env, with empty strings ignored.

    An empty ``--bench``/``--prepare``/``--adapter`` never overrides the env value
    -- it is already recorded as a problem by :func:`_collect_flag_problems`.
    """

    def pick_string(flag_value: str | None, env_value: str | None) -> str | None:
        return flag_value if flag_value is not None and flag_value != "" else env_value

    return CliFlags(
        bench=pick_string(flags.bench, env_flags.bench),
        prepare=pick_string(flags.prepare, env_flags.prepare),
        adapter=pick_string(flags.adapter, env_flags.adapter),
        samples=flags.samples if flags.samples is not None else env_flags.samples,
        timeout=flags.timeout if flags.timeout is not None else env_flags.timeout,
    )


def _resolve_config_source(
    flags: CliFlags, base_dir: str | None
) -> tuple[str | None, ConfigFile | None, list[str]]:
    """Resolve which config file to load, load it, and report any problems.

    When the config source itself is broken (blank ``--config``, blank
    ``GYMRAT_CONFIG``), file loading is skipped and an empty ``ConfigFile`` is
    returned so the merge still yields defaults without probing the filesystem.
    """
    problems: list[str] = []

    env_config_path: str | None = None
    env_config_failed = False
    if flags.config is None:
        result = env_string_result("GYMRAT_CONFIG")
        if result.problem is not None:
            problems.append(result.problem)
            env_config_failed = True
        env_config_path = str(result.value) if result.value is not None else None

    # A whitespace-only --config is as blank as an empty one, and
    # `_collect_flag_problems` has already reported it; probing it on disk would
    # add a second problem for a path the user never named.
    config_flag_blank = flags.config is not None and not flags.config.strip()
    if config_flag_blank or env_config_failed:
        return None, ConfigFile(), problems

    explicit_config = flags.config if flags.config is not None else env_config_path
    if explicit_config is not None:
        resolved_path = explicit_config
    else:
        anchor = base_dir if base_dir is not None else find_implicit_base()
        resolved_path = str(Path(anchor) / CONFIG_FILENAME)
    required = explicit_config is not None
    file_result = load_config_file_collecting(resolved_path, required=required)
    problems.extend(file_result.problems)

    config_path = resolved_path if (required or file_result.exists) else None
    return config_path, file_result.config_file, problems


def _resolve_runbook(
    config: BenchlessConfig, config_path: str | None, problems: list[str]
) -> BenchlessConfig:
    """Settle ``config.runbook`` to an absolute path, or record why it cannot be.

    A runbook is checked only when a config path exists, since it is authored
    relative to the directory the config lives in.
    """
    if config.runbook is None or config_path is None:
        return config
    config_dir = Path(config_path).parent
    problem = runbook_problem(config.runbook, config_dir)
    if problem is not None:
        problems.append(problem)
        return config
    return replace(config, runbook=os.path.normpath(config_dir / config.runbook))


def inspect_config(flags: CliFlags, base_dir: str | None = None) -> ConfigInspection:
    """Settle a benchless configuration, collecting every problem instead of raising.

    Args:
        flags: Command-line overrides.
        base_dir: Anchor for the implicit ``gymrat.toml`` lookup; falls back to the
            git repository root or the cwd when ``None``.

    Returns:
        A :class:`ConfigInspection` whose ``config`` and ``bench`` are populated
        only when no problems were found.
    """
    problems = _collect_flag_problems(flags)

    env_flags, env_problems = _collect_env_flags(flags)
    problems.extend(env_problems)

    effective = _build_effective_flags(flags, env_flags)

    config_path, config_file, source_problems = _resolve_config_source(flags, base_dir)
    problems.extend(source_problems)

    if config_file is None:
        return ConfigInspection(config_path=config_path, problems=problems)

    config = merge_config(effective, config_file)
    problems.extend(loop_key_problems(config))
    config = _resolve_runbook(config, config_path, problems)

    if problems:
        return ConfigInspection(config_path=config_path, problems=problems)

    bench = effective.bench if effective.bench is not None else config_file.bench
    return ConfigInspection(
        config_path=config_path,
        problems=[],
        config=config,
        bench=bench,
    )
