import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from gymrat.cli.run_setup import SharedFlags
from gymrat.config import (
    MAX_SAFE_INTEGER,
    MAX_TIMEOUT_SECONDS,
    BenchlessConfig,
    CliFlags,
    HooksConfig,
    KindEntry,
    MetricEntry,
    ResolvedConfig,
    StopConfig,
    SuperviseConfig,
    config_trace_args,
    env_positive_int_result,
    inspect_config,
    is_positive_integer,
    resolve_benchless_config,
    resolve_config,
    runbook_problem,
)
from gymrat.errors import GymratError
from tests._config import benchless_config
from tests._git import run_git
from tests.config._toml import (
    LOOP_CONFIG,
    write_config,
    write_raw,
)

# Both resolvers settle the implicit gymrat.toml through the same lookup, so any
# behavior that depends on the lookup has to hold for either entry point.
Resolver = Callable[..., BenchlessConfig]
RESOLVERS = [
    pytest.param(resolve_config, id="resolve_config"),
    pytest.param(resolve_benchless_config, id="resolve_benchless_config"),
]


# ---------------------------------------------------------------------------
# No circular import at package load time
# ---------------------------------------------------------------------------


def test_config_module_when_imported_fresh_does_not_raise_import_error():
    result = subprocess.run(
        [sys.executable, "-c", "import gymrat.config"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, (
        f"Importing gymrat.config failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# GYMRAT_* environment variables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvCase:
    """One GYMRAT_* precedence case.

    Bundles the variable under test, the flags/config that compete with it, and
    the settled field to read back, so a parametrized test takes a single case
    argument instead of one parameter per column.
    """

    env_var: str
    env_value: str
    field: str
    expected: object
    flags: CliFlags = field(default_factory=CliFlags)
    config: dict[str, object] | None = None


def _arrange_env_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: EnvCase) -> None:
    if case.config is not None:
        write_config(tmp_path, case.config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(case.env_var, case.env_value)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            EnvCase("GYMRAT_BENCH", "env-bench", "bench", "env-bench"), id="flag-absent-bench"
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_PREPARE", "env-prepare", "prepare", "env-prepare", flags=CliFlags(bench="b")
            ),
            id="flag-absent-prepare",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_ADAPTER", "env-adapter", "adapter", "env-adapter", flags=CliFlags(bench="b")
            ),
            id="flag-absent-adapter",
        ),
        pytest.param(
            EnvCase("GYMRAT_SAMPLES", "42", "samples", 42, flags=CliFlags(bench="b")),
            id="flag-absent-samples",
        ),
        pytest.param(
            EnvCase("GYMRAT_TIMEOUT", "900", "timeout_seconds", 900, flags=CliFlags(bench="b")),
            id="flag-absent-timeout",
        ),
        pytest.param(
            EnvCase("GYMRAT_BENCH", "env-bench", "bench", "env-bench", config={"bench": "cfg"}),
            id="beats-config-bench",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_PREPARE",
                "env-prepare",
                "prepare",
                "env-prepare",
                config={"bench": "b", "prepare": "config-prepare"},
            ),
            id="beats-config-prepare",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_ADAPTER",
                "env-adapter",
                "adapter",
                "env-adapter",
                config={"bench": "b", "adapter": "config-adapter"},
            ),
            id="beats-config-adapter",
        ),
        pytest.param(
            EnvCase("GYMRAT_SAMPLES", "42", "samples", 42, config={"bench": "b", "samples": 20}),
            id="beats-config-samples",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_TIMEOUT",
                "900",
                "timeout_seconds",
                900,
                config={"bench": "b", "timeout_seconds": 3600},
            ),
            id="beats-config-timeout",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_BENCH",
                "env-bench",
                "bench",
                "flag-bench",
                flags=CliFlags(bench="flag-bench"),
            ),
            id="flag-wins-bench",
        ),
        pytest.param(
            EnvCase("GYMRAT_SAMPLES", "42", "samples", 7, flags=CliFlags(bench="b", samples=7)),
            id="flag-wins-samples",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_TIMEOUT",
                "42",
                "timeout_seconds",
                7,
                flags=CliFlags(bench="b", timeout=7),
            ),
            id="flag-wins-timeout",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_ADAPTER",
                "metric-lines",
                "adapter",
                "mitata",
                flags=CliFlags(bench="b", adapter="mitata"),
            ),
            id="flag-wins-adapter",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_PREPARE",
                "env-prepare",
                "prepare",
                "flag-prepare",
                flags=CliFlags(bench="b", prepare="flag-prepare"),
            ),
            id="flag-wins-prepare",
        ),
        pytest.param(
            EnvCase("GYMRAT_BENCH", "", "bench", "flag-bench", flags=CliFlags(bench="flag-bench")),
            id="flag-skips-validation-blank-bench",
        ),
        pytest.param(
            EnvCase("GYMRAT_SAMPLES", "abc", "samples", 5, flags=CliFlags(bench="b", samples=5)),
            id="flag-skips-validation-invalid-int",
        ),
    ],
)
def test_resolve_config_when_env_var_set_does_settle_by_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: EnvCase
):
    _arrange_env_case(tmp_path, monkeypatch, case)

    result = resolve_config(case.flags)

    assert getattr(result, case.field) == case.expected


def test_resolve_config_when_config_env_var_set_does_bypass_implicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "implicit-bench", "adapter": "implicit-adapter"})
    monkeypatch.chdir(tmp_path)
    env_config_path = write_config(tmp_path, {"bench": "alt-bench"}, name="alt-config.toml")
    monkeypatch.setenv("GYMRAT_CONFIG", str(env_config_path))

    result = resolve_config(CliFlags())

    assert result.bench == "alt-bench"
    assert result.adapter == "metric-lines"


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("abc", id="non-numeric"),
        pytest.param("1.5", id="non-integer"),
        pytest.param("", id="empty"),
        pytest.param("0x10", id="hex"),
        pytest.param(" 5", id="leading-space"),
        pytest.param("5 ", id="trailing-space"),
        pytest.param("+5", id="plus-sign"),
        pytest.param("1_000", id="underscore"),
        pytest.param("\N{ARABIC-INDIC DIGIT FIVE}", id="non-ascii-digit"),
        pytest.param("0", id="zero"),
        pytest.param("-1", id="negative"),
    ],
)
def test_is_positive_integer_when_not_bare_positive_digits_does_reject(raw: str):
    assert is_positive_integer(raw) is False


def test_env_positive_int_result_when_digit_string_exceeds_conversion_limit_does_report_problem(
    monkeypatch: pytest.MonkeyPatch,
):
    huge = "1" * 4301
    monkeypatch.setenv("GYMRAT_SAMPLES", huge)

    result = env_positive_int_result("GYMRAT_SAMPLES", maximum=MAX_SAFE_INTEGER)

    assert result.problem is not None
    assert result.value is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("1", 1, id="one"),
        pytest.param("10", 10, id="ceiling"),
        pytest.param("007", 7, id="leading-zeros"),
    ],
)
def test_env_positive_int_result_when_bare_digits_within_ceiling_does_accept(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
):
    monkeypatch.setenv("GYMRAT_SAMPLES", raw)

    result = env_positive_int_result("GYMRAT_SAMPLES", maximum=10)

    assert (result.value, result.problem) == (expected, None)


# ---------------------------------------------------------------------------
# settlement failures match inspect_config (both resolvers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_resolve_when_settlement_fails_does_raise_first_inspect_config_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve: Resolver
):
    write_config(tmp_path, {"samples": "bad"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GYMRAT_TIMEOUT", "abc")
    flags = CliFlags(bench="my-bench")
    first_problem = inspect_config(flags).problems[0]

    with pytest.raises(GymratError) as exc:
        resolve(flags)

    assert str(exc.value) == first_problem


# ---------------------------------------------------------------------------
# resolve_config — precedence and the bench requirement
# ---------------------------------------------------------------------------


def test_resolve_config_when_flags_given_does_beat_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    result = resolve_config(
        CliFlags(bench="flag-bench", adapter="flag-adapter", samples=25, timeout=900)
    )

    assert result == ResolvedConfig(
        bench="flag-bench",
        adapter="flag-adapter",
        samples=25,
        timeout_seconds=900,
        unstable_noise_pct=200,
        primary="geomean",
    )


def test_resolve_config_when_bench_missing_from_flags_and_config_does_raise_naming_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError) as exc:
        resolve_config(CliFlags())

    message = str(exc.value)
    assert "--bench" in message
    assert "config file" in message


def test_resolve_config_when_stop_sets_only_max_iterations_under_geomean_does_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "stop": {"max_iterations": 5}})
    monkeypatch.chdir(tmp_path)

    result = resolve_config(CliFlags())

    assert result.stop == StopConfig(max_iterations=5)


# ---------------------------------------------------------------------------
# problem helpers
# ---------------------------------------------------------------------------


def test_runbook_problem_when_path_cannot_be_read_does_name_the_path_and_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def failing_stat(_path: Path, **_kwargs: object) -> object:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "stat", failing_stat)

    assert (
        runbook_problem("runbook.md", tmp_path)
        == 'Cannot read runbook path "runbook.md": Permission denied'
    )


# ---------------------------------------------------------------------------
# inspect_config — shared helpers and fixtures
# ---------------------------------------------------------------------------

# The fully defaulted settled config: what inspect_config yields when neither
# flags nor a config file supply any value. bench lives on ConfigInspection, not
# on the settled BenchlessConfig, so it never appears here.
DEFAULT_CONFIG = benchless_config()


def has_problem(problems: list[str], pattern: str) -> bool:
    return any(re.search(pattern, problem) for problem in problems)


# ---------------------------------------------------------------------------
# inspect_config — settled configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "bench"),
    [
        pytest.param(CliFlags(), None, id="empty-flags"),
        pytest.param(CliFlags(bench="flag-bench"), "flag-bench", id="bench-flag"),
    ],
)
def test_inspect_config_when_no_file_does_settle_defaults_and_carry_any_bench_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: CliFlags, bench: str | None
):
    monkeypatch.chdir(tmp_path)

    result = inspect_config(flags)

    assert result.config_path is None
    assert result.problems == []
    assert result.config == DEFAULT_CONFIG
    assert result.bench == bench


def test_inspect_config_when_valid_file_provides_values_does_settle_config_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(
        tmp_path,
        {
            "bench": "config-bench",
            "adapter": "custom-adapter",
            "samples": 20,
            "timeout_seconds": 3600,
            "unstable_noise_pct": 150.5,
        },
    )
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.config_path == str(tmp_path / "gymrat.toml")
    assert result.problems == []
    assert result.config == BenchlessConfig(
        adapter="custom-adapter",
        samples=20,
        timeout_seconds=3600,
        unstable_noise_pct=150.5,
        primary="geomean",
    )
    assert result.bench == "config-bench"


def test_inspect_config_when_flags_override_file_does_use_flag_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(
        tmp_path,
        {"bench": "config-bench", "adapter": "config-adapter", "samples": 20},
    )
    monkeypatch.chdir(tmp_path)

    result = inspect_config(
        CliFlags(bench="flag-bench", adapter="flag-adapter", samples=5, timeout=30)
    )

    assert result.problems == []
    assert result.config == benchless_config(adapter="flag-adapter", samples=5, timeout_seconds=30)
    assert result.bench == "flag-bench"


def test_inspect_config_when_file_has_loop_and_override_keys_does_carry_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(
        tmp_path,
        {
            "bench": "config-bench",
            **LOOP_CONFIG,
            "metrics": {"decode/time": {"direction": "higher", "gating": False, "exact": True}},
            "kinds": {"memory": {"gating": False}},
            "supervise": {"model": "claude-sonnet", "effort": "high"},
        },
    )
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.problems == []
    assert result.config == benchless_config(
        primary="decode/time",
        checks="npm test",
        filter="npm run bench -- {names}",
        stop=StopConfig(target_value=1.5, max_iterations=20),
        hooks=HooksConfig(before="npm run warm-cache", after="npm run cool-down"),
        metrics={"decode/time": MetricEntry(direction="higher", gating=False, exact=True)},
        kinds={"memory": KindEntry(gating=False)},
        supervise=SuperviseConfig(model="claude-sonnet", effort="high"),
    )
    assert result.bench == "config-bench"


def test_inspect_config_when_file_names_existing_runbook_does_resolve_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "runbook": "RUNBOOK.md"})
    (tmp_path / "RUNBOOK.md").write_text("# Steps\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.problems == []
    assert result.config is not None
    assert result.config.runbook == str(tmp_path / "RUNBOOK.md")


def test_inspect_config_when_base_dir_given_does_read_base_dir_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    write_config(base_dir, {"bench": "base-bench", "checks": "base-checks"})
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    write_config(cwd_dir, {"bench": "cwd-bench", "checks": "cwd-checks"})
    monkeypatch.chdir(cwd_dir)

    result = inspect_config(CliFlags(), str(base_dir))

    assert result.bench == "base-bench"
    assert result.config_path == str(base_dir / "gymrat.toml")
    assert result.config is not None
    assert result.config.checks == "base-checks"


def test_inspect_config_when_config_flag_relative_to_cwd_does_read_named_file_over_base_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    nested_dir = tmp_path / "packages" / "core"
    nested_dir.mkdir(parents=True)
    write_config(tmp_path, {"bench": "a-bench", "checks": "base-checks"})
    write_config(nested_dir, {"bench": "a-bench", "checks": "cwd-checks"})
    write_config(nested_dir, {"bench": "a-bench", "checks": "named-checks"}, name="custom.toml")
    monkeypatch.chdir(nested_dir)

    result = inspect_config(CliFlags(config="custom.toml"), str(tmp_path))

    assert result.config is not None
    assert result.config.checks == "named-checks"


def test_inspect_config_when_cwd_inside_git_repo_does_find_config_at_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_git(["init"], str(tmp_path))
    write_config(tmp_path, {"bench": "repo-bench", "checks": "repo-checks"})
    nested_dir = tmp_path / "packages" / "core"
    nested_dir.mkdir(parents=True)
    monkeypatch.chdir(nested_dir)

    result = inspect_config(CliFlags(bench="flag-bench"))

    assert result.config is not None
    assert result.config.checks == "repo-checks"


@pytest.mark.parametrize(
    ("runbook", "expected"),
    [
        pytest.param("RUNBOOK.md", Path("settings") / "RUNBOOK.md", id="beside-config"),
        pytest.param("../docs/RUNBOOK.md", Path("docs") / "RUNBOOK.md", id="climbs-out-normalized"),
    ],
)
def test_inspect_config_when_config_flag_names_file_in_other_dir_does_resolve_runbook_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runbook: str, expected: Path
):
    config_dir = tmp_path / "settings"
    config_dir.mkdir()
    (tmp_path / "docs").mkdir()
    write_config(config_dir, {"bench": "a-bench", "runbook": runbook})
    (config_dir / runbook).write_text("# Steps\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags(config="settings/gymrat.toml"))

    assert result.problems == []
    assert result.config is not None
    assert result.config.runbook == str(expected)


# ---------------------------------------------------------------------------
# inspect_config — collected problems (never raises)
# ---------------------------------------------------------------------------


def test_inspect_config_when_config_flag_names_missing_path_does_report_and_omit_config(
    tmp_path: Path,
):
    missing_path = tmp_path / "typo.toml"

    result = inspect_config(CliFlags(bench="my-bench", config=str(missing_path)))

    assert result.config_path == str(missing_path)
    assert has_problem(result.problems, re.escape(str(missing_path)))
    assert result.config is None


def test_inspect_config_when_file_is_invalid_toml_does_report_naming_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = write_raw(tmp_path, "= invalid toml =")
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.config_path == str(config_path)
    assert has_problem(result.problems, re.escape(str(config_path)))
    assert result.config is None


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"primary": "geomean"}, id="geomean-explicit"),
        pytest.param({}, id="geomean-by-default"),
    ],
)
def test_inspect_config_when_target_value_with_geomean_primary_does_report_naming_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
):
    write_config(tmp_path, {"bench": "config-bench", "stop": {"target_value": 1.5}, **overrides})
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.problems == [
        'Invalid config value for stop.target_value: it needs primary to name a metric, not "geomean"'
    ]


@pytest.mark.parametrize(
    "runbook",
    [pytest.param("missing.md", id="missing"), pytest.param("docs", id="directory")],
)
def test_inspect_config_when_runbook_not_an_existing_file_does_report_naming_field_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runbook: str
):
    write_config(tmp_path, {"bench": "config-bench", "runbook": runbook})
    (tmp_path / "docs").mkdir()
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.problems == [
        f'Invalid config value for runbook: expected a path to an existing file, got "{runbook}"'
    ]


def _nul_path_reason() -> str:
    # The interpreter words this error itself, and the wording changed between
    # 3.12 patch releases ("embedded null byte" became "stat: embedded null
    # character in path"), so the expected reason comes from the running one.
    try:
        Path("a\0b").stat()
    except ValueError as exc:
        return str(exc)
    pytest.fail("stat accepted a path holding a NUL character")


def test_inspect_config_when_runbook_embeds_nul_does_report_problem_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_raw(tmp_path, 'bench = "config-bench"\nrunbook = "a\\u0000b"\n')
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.problems == [f'Cannot read runbook path "a\\u0000b": {_nul_path_reason()}']


@pytest.mark.parametrize(
    ("flags", "key", "got"),
    [
        pytest.param(CliFlags(bench=""), "bench", '""', id="bench"),
        pytest.param(CliFlags(bench="my-bench", prepare=""), "prepare", '""', id="prepare"),
        pytest.param(CliFlags(bench="my-bench", adapter=""), "adapter", '""', id="adapter"),
        pytest.param(CliFlags(bench="my-bench", config=""), "config", '""', id="config"),
        pytest.param(CliFlags(bench="\t"), "bench", '"\\t"', id="whitespace"),
    ],
)
def test_inspect_config_when_flag_holds_blank_string_does_report_naming_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: CliFlags, key: str, got: str
):
    monkeypatch.chdir(tmp_path)

    result = inspect_config(flags)

    assert result.problems == [
        f"Invalid config value for --{key}: expected a non-empty string, got {got}"
    ]


def test_inspect_config_when_multiple_flags_empty_does_collect_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags(bench="", adapter=""))

    assert result.problems == [
        'Invalid config value for --bench: expected a non-empty string, got ""',
        'Invalid config value for --adapter: expected a non-empty string, got ""',
    ]


_FLAG_AND_ENV_PROBLEMS = [
    'Invalid config value for --adapter: expected a non-empty string, got ""',
    'Invalid value for GYMRAT_SAMPLES: expected a positive integer, got "abc"',
]


@pytest.mark.parametrize(
    ("config", "file_problems"),
    [
        pytest.param(
            {"filter": "npm run bench", "runbook": "missing.md"},
            [
                (
                    "Invalid config value for filter: expected a string containing the {names} "
                    'placeholder, got "npm run bench"'
                ),
                (
                    "Invalid config value for runbook: expected a path to an existing file, "
                    'got "missing.md"'
                ),
            ],
            id="loop-keys-then-runbook",
        ),
        pytest.param(
            {"samples": "bad", "filter": "npm run bench", "runbook": "missing.md"},
            ['Invalid config value for samples: expected an integer, got "bad"'],
            id="schema-stops-before-loop-keys",
        ),
    ],
)
def test_inspect_config_when_every_step_fails_does_report_flags_then_env_then_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object],
    file_problems: list[str],
):
    write_config(tmp_path, config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GYMRAT_SAMPLES", "abc")

    result = inspect_config(CliFlags(bench="my-bench", adapter=""))

    assert result.problems == [*_FLAG_AND_ENV_PROBLEMS, *file_problems]
    assert result.config is None


def test_inspect_config_when_config_flag_blank_does_report_and_skip_file_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "cwd-bench"})
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags(config="   "))

    # A blank value is one mistake: probing it on disk would add a second,
    # spurious "file not found" problem for a path the user never named.
    assert len(result.problems) == 1
    assert has_problem(result.problems, r"--config.*non-empty")
    assert result.config_path is None
    assert result.config is None
    assert result.bench is None


# ---------------------------------------------------------------------------
# inspect_config — GYMRAT_* environment variables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env_var", "flags", "env_value"),
    [
        pytest.param("GYMRAT_BENCH", CliFlags(), "", id="bench-empty"),
        pytest.param("GYMRAT_PREPARE", CliFlags(bench="b"), " ", id="prepare-space"),
        pytest.param("GYMRAT_ADAPTER", CliFlags(bench="b"), "\t", id="adapter-tab"),
        pytest.param("GYMRAT_CONFIG", CliFlags(bench="b"), "  \n  ", id="config-padded-newline"),
    ],
)
def test_inspect_config_when_string_env_var_blank_does_report_naming_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    flags: CliFlags,
    env_value: str,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, env_value)

    result = inspect_config(flags)

    assert result.problems == [
        f"Invalid value for {env_var}: expected a non-empty string, got {json.dumps(env_value)}"
    ]


def test_inspect_config_when_config_env_var_names_missing_path_does_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    missing_path = tmp_path / "typo.toml"
    monkeypatch.setenv("GYMRAT_CONFIG", str(missing_path))

    result = inspect_config(CliFlags(bench="my-bench"))

    assert has_problem(result.problems, re.escape(str(missing_path)))


def test_inspect_config_when_every_field_env_var_invalid_does_report_each_in_field_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GYMRAT_BENCH", " ")
    monkeypatch.setenv("GYMRAT_PREPARE", "")
    monkeypatch.setenv("GYMRAT_ADAPTER", "")
    monkeypatch.setenv("GYMRAT_SAMPLES", "0")
    monkeypatch.setenv("GYMRAT_TIMEOUT", "0")

    result = inspect_config(CliFlags())

    assert result.problems == [
        'Invalid value for GYMRAT_BENCH: expected a non-empty string, got " "',
        'Invalid value for GYMRAT_PREPARE: expected a non-empty string, got ""',
        'Invalid value for GYMRAT_ADAPTER: expected a non-empty string, got ""',
        'Invalid value for GYMRAT_SAMPLES: expected a positive integer, got "0"',
        'Invalid value for GYMRAT_TIMEOUT: expected a positive integer, got "0"',
    ]


@pytest.mark.parametrize(
    ("env_var", "cap"),
    [
        pytest.param("GYMRAT_TIMEOUT", MAX_TIMEOUT_SECONDS, id="timeout"),
        pytest.param("GYMRAT_SAMPLES", MAX_SAFE_INTEGER, id="samples"),
    ],
)
def test_inspect_config_when_integer_env_var_exceeds_cap_does_reject_as_not_a_positive_integer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_var: str, cap: int
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, str(cap + 1))

    result = inspect_config(CliFlags(bench="my-bench"))

    assert result.problems == [
        f'Invalid value for {env_var}: expected a positive integer, got "{cap + 1}"'
    ]


# ---------------------------------------------------------------------------
# config_trace_args
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        pytest.param(CliFlags(), {}, id="all-unset"),
        pytest.param(
            CliFlags(
                bench="sh bench.sh",
                prepare="make",
                adapter="mitata",
                samples=5,
                timeout=30,
                config="gymrat.toml",
            ),
            {
                "bench": "sh bench.sh",
                "prepare": "make",
                "adapter": "mitata",
                "samples": 5,
                "timeout": 30,
                "config": "gymrat.toml",
            },
            id="all-set",
        ),
        pytest.param(CliFlags(samples=0, timeout=0), {"samples": 0, "timeout": 0}, id="zero-kept"),
        pytest.param(
            SharedFlags(samples=5, format="json"), {"samples": 5}, id="subclass-fields-left-out"
        ),
    ],
)
def test_config_trace_args_when_flags_vary_does_keep_only_the_set_overrides(
    flags: CliFlags, expected: dict[str, object]
):
    args = config_trace_args(flags)

    assert args == expected


def test_config_trace_args_when_given_extras_does_append_the_set_ones_in_order():
    flags = CliFlags(
        bench="sh bench.sh",
        prepare="make",
        adapter="mitata",
        samples=5,
        timeout=30,
        config="gymrat.toml",
    )

    args = config_trace_args(flags, baseline="main", message=None, allow_unimproved=True)

    assert list(args.items())[-2:] == [("baseline", "main"), ("allow_unimproved", True)]
    assert list(args)[:6] == ["bench", "prepare", "adapter", "samples", "timeout", "config"]
