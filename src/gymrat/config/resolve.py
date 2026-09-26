"""Precedence pipeline: merge flags, env vars, config file, and defaults.

One settlement pipeline serves every caller. :func:`inspect_config` runs it and
never raises: every flag, env var, file, schema, and cross-field problem is
gathered into one list so a caller (a doctor/status command) can report them all
at once, and a settled config is returned only when that list is empty.
:func:`resolve_config` and :func:`resolve_benchless_config` run the same
pipeline and raise the first collected problem.
"""

import dataclasses
import os
from dataclasses import dataclass, replace
from pathlib import Path

from gymrat.config.env import NUMBER_ENV_FIELDS, STRING_ENV_FIELDS, env_string_result
from gymrat.config.load import load_config_file_collecting
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


def find_implicit_base() -> str:
    """Return the anchor directory for the implicit ``gymrat.toml`` lookup.

    Inside a git repository the config lives at the repo root, so moving the cwd
    into a subdirectory must not lose it. Outside a repository — or when git is
    unavailable — the lookup falls back to the process cwd.

    Returns:
        The git repository root, or the current working directory when no
        repository is found.
    """
    try:
        return repo_root()
    except GymratError:
        return str(Path.cwd())


def merge_config(flags: CliFlags, config_file: ConfigFile) -> BenchlessConfig:
    """Merge flags, config file, and defaults into every setting but ``bench``.

    ``bench`` is settled separately: benchmarking commands spread it into the
    result, and the rest never ask for it.

    Args:
        flags: CLI flags, taking precedence over the config file where set.
        config_file: Parsed config file, filling in fields the flags leave unset.

    Returns:
        The merged :class:`BenchlessConfig` with every field resolved from its
        highest-precedence source.
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
    checks over ``config``. Lets a writer (the init scaffold) reject a config
    before touching disk without a temp-file round-trip.

    Args:
        config: In-memory config data, shaped like a parsed ``gymrat.toml``.

    Raises:
        GymratError: On the first schema or cross-field validation problem.
    """
    config_file, problems = validate_and_convert(config)
    if problems:
        raise GymratError(problems[0])
    if config_file is None:
        return
    problems = loop_key_problems(merge_config(CliFlags(), config_file))
    if problems:
        raise GymratError(problems[0])


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

    Args:
        flags: The CLI flags to check for unset fields before reading env vars.

    Returns:
        A ``(env_flags, problems)`` pair: the flags populated from env vars and
        any validation problems encountered.
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
    """Layer flags over env values: flag > env, with empty strings ignored."""

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
    flags: CliFlags, base_dir: str | Path | None
) -> tuple[str | None, ConfigFile | None, list[str]]:
    """Resolve which config file to load, load it, and report any problems.

    When the config source itself is broken (blank ``--config``, blank
    ``GYMRAT_CONFIG``), file loading is skipped so the merge still yields
    defaults without probing the filesystem.

    Args:
        flags: Command-line overrides, consulted for an explicit ``--config``.
        base_dir: Anchor for the implicit ``gymrat.toml`` lookup; falls back to
            :func:`find_implicit_base` when ``None``.

    Returns:
        A ``(config_path, config_file, problems)`` triple: the resolved path
        (``None`` when no file applies), the parsed config (``None`` on fatal
        read/parse failure, an empty ``ConfigFile`` when the config source is
        blank), and any problems found.
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
    """Settle ``config.runbook`` against the config's directory, or record why it cannot be.

    A runbook is checked only when a config path exists, since it is authored
    relative to the directory the config lives in.

    Args:
        config: The settled config whose ``runbook`` field to resolve.
        config_path: Path to the loaded config file, or ``None`` when none applies.
        problems: The running problem list; appended to in place when the
            runbook cannot be resolved.

    Returns:
        The config with ``runbook`` joined onto the config file's directory and
        normalized (absolute only when ``config_path`` is), or unchanged when no
        resolution is needed or a problem was recorded.
    """
    if config.runbook is None or config_path is None:
        return config
    config_dir = Path(config_path).parent
    problem = runbook_problem(config.runbook, config_dir)
    if problem is not None:
        problems.append(problem)
        return config
    return replace(config, runbook=os.path.normpath(config_dir / config.runbook))


def inspect_config(flags: CliFlags, base_dir: str | Path | None = None) -> ConfigInspection:
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


def _settle_or_raise(
    flags: CliFlags, base_dir: str | Path | None
) -> tuple[BenchlessConfig, str | None]:
    inspection = inspect_config(flags, base_dir)
    if inspection.config is None:
        raise GymratError(inspection.problems[0])
    return inspection.config, inspection.bench


def resolve_benchless_config(
    flags: CliFlags, base_dir: str | Path | None = None
) -> BenchlessConfig:
    """Settle everything :func:`resolve_config` does except ``bench``.

    Use for a command that runs no benchmark: it settles the same values without
    demanding a bench command none of them would run.

    Args:
        flags: CLI flags, taking precedence over env vars and the config file.
        base_dir: Directory to anchor the implicit config-file lookup to, or
            ``None`` to use :func:`find_implicit_base`.

    Returns:
        The fully settled :class:`BenchlessConfig`.

    Raises:
        GymratError: With the first problem :func:`inspect_config` collects, when
            a flag, env var, config file, cross-field check, or runbook fails to
            validate.
    """
    config, _ = _settle_or_raise(flags, base_dir)
    return config


def resolve_config(flags: CliFlags, base_dir: str | Path | None = None) -> ResolvedConfig:
    """Settle a run configuration from flags, env vars, config file, and defaults.

    ``bench`` has no default and must come from a flag or the config file.

    Args:
        flags: CLI flags, taking precedence over env vars and the config file.
        base_dir: Directory to anchor the implicit config-file lookup to, or
            ``None`` to use :func:`find_implicit_base`.

    Returns:
        The fully settled :class:`ResolvedConfig` including ``bench``.

    Raises:
        GymratError: With the first problem :func:`inspect_config` collects, or
            when ``bench`` is missing from both flags and the config file.
    """
    config, bench = _settle_or_raise(flags, base_dir)
    if bench is None:
        message = "bench is required. Provide it via --bench flag or in config file."
        raise GymratError(message)
    parent_fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    return ResolvedConfig(bench=bench, **parent_fields)
