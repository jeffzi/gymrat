"""Precedence pipeline: merge flags, env vars, config file, and defaults.

One settlement pipeline serves every caller. :func:`inspect_config` runs it and
never raises: every flag, env var, file, schema, and cross-field problem is
gathered into one list so a caller (a doctor/status command) can report them all
at once, and a settled config is returned only when that list is empty.
:func:`resolve_config` and :func:`resolve_benchless_config` run the same
pipeline and raise the first collected problem.

The checks the pipeline applies beyond the file schema live here too: the blank
flag check, :func:`loop_key_problems` for cross-field rules, and
:func:`runbook_problem` for the runbook path. :func:`validate_config_dict`
runs the schema and the cross-field rules over an in-memory config.

The file side of the pipeline lives here as well. :func:`load_config_file_collecting`
reads and validates ``gymrat.toml``, collecting every problem.
:func:`validate_config_file` runs the frozen dataclasses from
:mod:`gymrat.config.types`, which carry the validation annotations, through a
pydantic ``TypeAdapter`` and words each failure as a gymrat problem string;
:func:`invalid_value_message` is the one wording that translator and the
cross-field checks share.
"""

import dataclasses
import json
import os
import stat
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

from gymrat.config.env import (
    NUMBER_ENV_FIELDS,
    STRING_ENV_FIELDS,
    env_positive_int_result,
    env_string_result,
)
from gymrat.config.types import (
    CONFIG_DEFAULTS,
    CONFIG_FILENAME,
    FILTER_PLACEHOLDER,
    GEOMEAN_PRIMARY,
    BenchlessConfig,
    CliFlags,
    ConfigFile,
    ConfigFileResult,
    ResolvedConfig,
)
from gymrat.errors import GymratError
from gymrat.pydantic_errors import (
    VALUE_ERROR_PREFIX,
    describe_key,
    drop_prefix_errors,
    phrase_for_error,
)
from gymrat.session.paths import repo_root

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


_CONFIG_ADAPTER = TypeAdapter(ConfigFile)


def invalid_value_message(field_name: str, expected_phrase: str, value: object) -> str:
    """Word an invalid-value problem.

    The single shape both the schema translator and the cross-field settlement
    checks report.

    Args:
        field_name: Dotted name of the offending config field.
        expected_phrase: Human-readable description of the expected shape.
        value: The actual value that failed validation.

    Returns:
        A human-readable problem string naming the field, its expected shape,
        and the actual value.
    """
    try:
        got = json.dumps(value)
    except TypeError:
        got = repr(value)
    return f"Invalid config value for {field_name}: expected {expected_phrase}, got {got}"


def _message_for_error(error: ErrorDetails) -> str:
    """Translate one pydantic error into a gymrat-worded problem string.

    A custom validator (the line-break key guard) already knows why it refused
    the value, so its own message is reported: a shape phrase would describe a
    fault the value does not have.

    Args:
        error: The pydantic error detail to translate.

    Returns:
        The problem string describing the validation failure.
    """
    key = describe_key(error["loc"])
    if error["type"] == "unexpected_keyword_argument":
        return f"Unknown config key: {key}"
    if error["type"] == "value_error":
        detail = error["msg"].removeprefix(VALUE_ERROR_PREFIX)
        return f"Invalid config value for {key}: {detail}"
    return invalid_value_message(key, phrase_for_error(error), error["input"])


def validate_config_file(data: dict[str, object]) -> tuple[ConfigFile | None, list[str]]:
    """Validate parsed config data into a :class:`ConfigFile`.

    Never raises: validation failures are returned as a problem list, not
    exceptions, so callers can collect and display all errors at once.

    Args:
        data: Raw config data, shaped like a parsed ``gymrat.toml``.

    Returns:
        A ``(config_file, problems)`` pair: the validated :class:`ConfigFile`
        (``None`` on failure) and any validation problems.
    """
    try:
        config_file = _CONFIG_ADAPTER.validate_python(data)
    except ValidationError as exc:
        return None, [_message_for_error(error) for error in drop_prefix_errors(exc.errors())]
    return config_file, []


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _reason(exc: OSError | ValueError) -> str:
    """Why a filesystem call failed: the OS's own wording when it gave one."""
    return exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)


def _read_source(path: Path) -> tuple[str | None, str | None]:
    """Read the config file, reporting a read failure as a problem rather than raising.

    Decoding as ``utf-8-sig`` drops the byte-order mark Windows editors prepend,
    which TOML parsing would otherwise reject.

    Args:
        path: Path to the config file to read.

    Returns:
        A ``(text, problem)`` pair: the file content and ``None`` on success,
        ``(None, None)`` when the file is absent, or ``(None, message)`` on
        read failure.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as exc:
        return None, f"Cannot read config file at {path}: {_reason(exc)}"
    return text, None


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

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        problem = f"Failed to parse config file at {config_path}: {exc}"
        return ConfigFileResult(config_file=None, exists=True, problems=[problem])
    config_file, problems = validate_config_file(data)
    return ConfigFileResult(config_file=config_file, exists=True, problems=problems)


# ---------------------------------------------------------------------------
# Settlement pipeline
# ---------------------------------------------------------------------------


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


def _first_set[T](*values: T | None, default: T) -> T:
    """The first value that is set, in precedence order, else ``default``."""
    return next((value for value in values if value is not None), default)


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
    return BenchlessConfig(
        adapter=flags.adapter or config_file.adapter or CONFIG_DEFAULTS.adapter,
        samples=_first_set(flags.samples, config_file.samples, default=CONFIG_DEFAULTS.samples),
        timeout_seconds=_first_set(
            flags.timeout, config_file.timeout_seconds, default=CONFIG_DEFAULTS.timeout_seconds
        ),
        unstable_noise_pct=_first_set(
            config_file.unstable_noise_pct, default=CONFIG_DEFAULTS.unstable_noise_pct
        ),
        primary=config_file.primary or CONFIG_DEFAULTS.primary,
        prepare=_first_set(flags.prepare, default=config_file.prepare),
        metrics=dict(config_file.metrics) if config_file.metrics is not None else None,
        kinds=dict(config_file.kinds) if config_file.kinds is not None else None,
        checks=config_file.checks,
        runbook=config_file.runbook,
        filter=config_file.filter,
        stop=config_file.stop,
        hooks=config_file.hooks,
        supervise=config_file.supervise,
    )


def loop_key_problems(config: BenchlessConfig) -> list[str]:
    """Return the cross-field violations the schema alone cannot express.

    ``filter`` must carry its placeholder, and ``stop.target_value`` only makes
    sense when ``primary`` names a metric — the geomean is a ratio, not a value.

    Args:
        config: The merged config to check for cross-field violations.

    Returns:
        The list of cross-field problem strings found, empty when none.
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


def runbook_problem(runbook: str, base_dir: Path) -> str | None:
    """Return a problem string when ``runbook`` does not name an existing file.

    The runbook is anchored the same way as the implicit ``gymrat.toml`` lookup,
    because a runbook path is authored relative to the repository the config
    lives in.

    Args:
        runbook: Path to the runbook, relative to ``base_dir``.
        base_dir: Directory the runbook path is resolved against.

    Returns:
        A problem string when ``runbook`` does not resolve to an existing
        regular file, or ``None`` when it does.
    """
    resolved = Path(os.path.normpath(base_dir / runbook))
    try:
        info = resolved.stat()
    except FileNotFoundError:
        info = None
    except (OSError, ValueError) as exc:
        return f"Cannot read runbook path {json.dumps(runbook)}: {_reason(exc)}"
    if info is None or not stat.S_ISREG(info.st_mode):
        return invalid_value_message("runbook", "a path to an existing file", runbook)
    return None


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
    config_file, problems = validate_config_file(config)
    if problems:
        raise GymratError(problems[0])
    assert config_file is not None  # noqa: S101 -- no problems means the schema produced a config
    problems = loop_key_problems(merge_config(CliFlags(), config_file))
    if problems:
        raise GymratError(problems[0])


def _collect_flag_problems(flags: CliFlags) -> list[str]:
    """Return one problem per blank (empty or whitespace-only) string flag.

    Flags bypass the file schema, so ``--bench ""`` or ``--bench "   "`` is the
    one way a blank string reaches a settled field. Each message names the flag,
    not the config key, because the flag is what the user typed.

    Args:
        flags: The command-line overrides to check.

    Returns:
        The problem strings, in flag order; empty when no flag is blank.
    """
    return [
        invalid_value_message(f"--{field_name}", "a non-empty string", value)
        for field_name in ("bench", "prepare", "adapter", "config")
        if (value := getattr(flags, field_name)) is not None and not value.strip()
    ]


def _collect_env_flags(flags: CliFlags) -> tuple[CliFlags, list[str]]:
    """Read every ``GYMRAT_*`` field flag whose flag is unset, collecting problems.

    An env var is consulted only when its flag is ``None`` (a flag always wins
    without the env var's validation firing). ``GYMRAT_CONFIG`` is handled in
    :func:`_resolve_config_source` because it selects the file, not a field.

    Args:
        flags: The CLI flags to check for unset fields before reading env vars.

    Returns:
        An ``(effective, problems)`` pair: the flags with every unset field
        filled from its env var, and any validation problems encountered.
    """
    problems: list[str] = []
    strings: dict[str, str] = {}
    for field_name, env_var in STRING_ENV_FIELDS:
        if getattr(flags, field_name) is None:
            result = env_string_result(env_var)
            if result.problem is not None:
                problems.append(result.problem)
            if result.value is not None:
                strings[field_name] = result.value
    numbers: dict[str, int] = {}
    for field_name, env_var, maximum in NUMBER_ENV_FIELDS:
        if getattr(flags, field_name) is None:
            result = env_positive_int_result(env_var, maximum)
            if result.problem is not None:
                problems.append(result.problem)
            if result.value is not None:
                numbers[field_name] = result.value
    effective = replace(
        flags,
        bench=_first_set(flags.bench, default=strings.get("bench")),
        prepare=_first_set(flags.prepare, default=strings.get("prepare")),
        adapter=_first_set(flags.adapter, default=strings.get("adapter")),
        samples=_first_set(flags.samples, default=numbers.get("samples")),
        timeout=_first_set(flags.timeout, default=numbers.get("timeout")),
    )
    return effective, problems


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
        env_config_path = result.value

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

    effective, env_problems = _collect_env_flags(flags)
    problems.extend(env_problems)

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
