"""Precedence pipeline: merge flags, env vars, config file, and defaults."""

import dataclasses
import os
from dataclasses import replace
from pathlib import Path

from gymrat.config.env import NUMBER_ENV_FIELDS, STRING_ENV_FIELDS, env_string_result
from gymrat.config.load import load_config_file
from gymrat.config.schema import validate_and_convert
from gymrat.config.types import (
    CONFIG_DEFAULTS,
    CONFIG_FILENAME,
    BenchlessConfig,
    CliFlags,
    ConfigFile,
    ResolvedConfig,
)
from gymrat.config.validate import (
    flag_problem,
    loop_key_problems,
    runbook_problem,
)
from gymrat.errors import GymratError
from gymrat.session.paths import repo_root


def _read_env_string(env_var: str) -> str | None:
    """Throwing wrapper around :func:`env_string_result`."""
    result = env_string_result(env_var)
    if result.problem is not None:
        raise GymratError(result.problem)
    return str(result.value) if result.value is not None else None


def _read_env_flags(flags: CliFlags) -> CliFlags:
    """Overlay ``GYMRAT_*`` values onto every flag left unset: flag > env > (later) file.

    An env var is consulted only when its flag is ``None``, so a flag always wins
    without the env var's validation ever firing. ``GYMRAT_CONFIG`` is handled in
    :func:`_settle_config` because it selects the file to load, not a field.
    """
    strings: dict[str, str] = {}
    for field_name, env_var in STRING_ENV_FIELDS:
        if getattr(flags, field_name) is None:
            value = _read_env_string(env_var)
            if value is not None:
                strings[field_name] = value
    numbers: dict[str, int] = {}
    for field_name, env_var, reader in NUMBER_ENV_FIELDS:
        if getattr(flags, field_name) is None:
            result = reader(env_var)
            if result.problem is not None:
                raise GymratError(result.problem)
            if isinstance(result.value, int):
                numbers[field_name] = result.value
    return CliFlags(
        bench=strings.get("bench", flags.bench),
        prepare=strings.get("prepare", flags.prepare),
        adapter=strings.get("adapter", flags.adapter),
        samples=numbers.get("samples", flags.samples),
        timeout=numbers.get("timeout", flags.timeout),
        config=flags.config,
    )


def find_implicit_base() -> str:
    """Return the anchor directory for the implicit ``gymrat.toml`` lookup.

    Inside a git repository the config lives at the repo root, so moving the cwd
    into a subdirectory must not lose it. Outside a repository — or when git is
    unavailable — the lookup falls back to the process cwd.
    """
    try:
        return repo_root()
    except GymratError:
        return str(Path.cwd())


def merge_config(flags: CliFlags, config_file: ConfigFile) -> BenchlessConfig:
    """Merge flags, config file, and defaults into every setting but ``bench``.

    ``bench`` is settled separately: benchmarking commands spread it into the
    result, and the rest never ask for it.
    """
    samples = (
        flags.samples
        if flags.samples is not None
        else config_file.samples
        if config_file.samples is not None
        else CONFIG_DEFAULTS.samples
    )
    timeout_seconds = (
        flags.timeout
        if flags.timeout is not None
        else config_file.timeout_seconds
        if config_file.timeout_seconds is not None
        else CONFIG_DEFAULTS.timeout_seconds
    )
    return BenchlessConfig(
        adapter=flags.adapter or config_file.adapter or CONFIG_DEFAULTS.adapter,
        samples=samples,
        timeout_seconds=timeout_seconds,
        unstable_noise_pct=(
            config_file.unstable_noise_pct
            if config_file.unstable_noise_pct is not None
            else CONFIG_DEFAULTS.unstable_noise_pct
        ),
        primary=config_file.primary or CONFIG_DEFAULTS.primary,
        prepare=flags.prepare if flags.prepare is not None else config_file.prepare,
        metrics=dict(config_file.metrics) if config_file.metrics is not None else None,
        kinds=dict(config_file.kinds) if config_file.kinds is not None else None,
        checks=config_file.checks,
        runbook=config_file.runbook,
        filter=config_file.filter,
        stop=config_file.stop,
        hooks=config_file.hooks,
        supervise=config_file.supervise,
    )


def validate_config_dict(config: dict[str, object]) -> None:
    """Validate an in-memory config dict the same way a loaded ``gymrat.toml`` is.

    Runs the strict schema (``extra="forbid"``) and the cross-field loop-key
    checks over ``config``, raising a :class:`GymratError` on the first problem.
    Lets a writer (the init scaffold) reject a config before touching disk without
    a temp-file round-trip.
    """
    config_file, problems = validate_and_convert(config)
    if problems:
        raise GymratError(problems[0])
    if config_file is None:
        return
    problems = loop_key_problems(merge_config(CliFlags(), config_file))
    if problems:
        raise GymratError(problems[0])


def _settle_config(
    flags: CliFlags, base_dir: str | Path | None
) -> tuple[BenchlessConfig, str | None]:
    for name, value in [
        ("bench", flags.bench),
        ("prepare", flags.prepare),
        ("adapter", flags.adapter),
        ("config", flags.config),
    ]:
        problem = flag_problem(name, value)
        if problem is not None:
            raise GymratError(problem)

    effective = _read_env_flags(flags)

    env_config_path = _read_env_string("GYMRAT_CONFIG") if flags.config is None else None
    explicit_config = flags.config if flags.config is not None else env_config_path
    if explicit_config is not None:
        config_path = Path(explicit_config)
    else:
        anchor = Path(base_dir) if base_dir is not None else Path(find_implicit_base())
        config_path = anchor / CONFIG_FILENAME
    config_file = load_config_file(config_path, required=explicit_config is not None)

    config = merge_config(effective, config_file)
    problems = loop_key_problems(config)
    if problems:
        raise GymratError(problems[0])

    if config.runbook is not None:
        config_dir = config_path.parent
        problem = runbook_problem(config.runbook, config_dir)
        if problem is not None:
            raise GymratError(problem)
        resolved_runbook = os.path.normpath(config_dir / config.runbook)
        config = replace(config, runbook=resolved_runbook)

    bench = effective.bench if effective.bench is not None else config_file.bench
    return config, bench


def resolve_benchless_config(
    flags: CliFlags, base_dir: str | Path | None = None
) -> BenchlessConfig:
    """Settle everything :func:`resolve_config` does except ``bench``.

    Use for a command that runs no benchmark: it settles the same values without
    demanding a bench command none of them would run.
    """
    config, _ = _settle_config(flags, base_dir)
    return config


def resolve_config(flags: CliFlags, base_dir: str | Path | None = None) -> ResolvedConfig:
    """Settle a run configuration from flags, env vars, config file, and defaults.

    ``bench`` has no default and must come from a flag or the config file — a run
    without it raises.
    """
    config, bench = _settle_config(flags, base_dir)
    if bench is None:
        message = "bench is required. Provide it via --bench flag or in config file."
        raise GymratError(message)
    parent_fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    return ResolvedConfig(bench=bench, **parent_fields)
