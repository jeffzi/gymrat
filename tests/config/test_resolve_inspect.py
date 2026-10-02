import re
from pathlib import Path

import pytest
import tomli_w

from gymrat.config.env import MAX_SAFE_INTEGER, MAX_TIMEOUT_SECONDS
from gymrat.config.resolve import inspect_config
from gymrat.config.types import BenchlessConfig, CliFlags, HooksConfig, StopConfig

# ---------------------------------------------------------------------------
# inspect_config — shared helpers and fixtures
# ---------------------------------------------------------------------------

# The fully defaulted settled config: what inspect_config yields when neither
# flags nor a config file supply any value. bench lives on ConfigInspection, not
# on the settled BenchlessConfig, so it never appears here.
DEFAULT_CONFIG = BenchlessConfig(
    adapter="metric-lines",
    samples=10,
    timeout_seconds=1800,
    unstable_noise_pct=200,
    primary="geomean",
)

# Full loop configuration exercised as a config-file body; every loop key must
# survive into the settled config unchanged.
LOOP_CONFIG: dict[str, object] = {
    "checks": "npm test",
    "filter": "npm run bench -- {names}",
    "primary": "decode/time",
    "stop": {"target_value": 1.5, "max_iterations": 20},
    "hooks": {"before": "npm run warm-cache", "after": "npm run cool-down"},
}


def write_config(directory: Path, content: dict[str, object]) -> Path:
    config_path = directory / "gymrat.toml"
    config_path.write_text(tomli_w.dumps(content), encoding="utf-8")
    return config_path


def write_raw_config(directory: Path, content: str) -> Path:
    config_path = directory / "gymrat.toml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def has_problem(problems: list[str], pattern: str) -> bool:
    return any(re.search(pattern, problem) for problem in problems)


# ---------------------------------------------------------------------------
# inspect_config — settled configuration
# ---------------------------------------------------------------------------


def test_inspect_config_when_no_file_and_empty_flags_does_return_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.config_path is None

    assert result.problems == []
    assert result.config == DEFAULT_CONFIG
    assert result.bench is None


def test_inspect_config_when_flags_provide_bench_and_no_file_does_carry_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags(bench="flag-bench"))

    assert result.problems == []
    assert result.config == DEFAULT_CONFIG
    assert result.bench == "flag-bench"


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
    assert result.config == BenchlessConfig(
        adapter="flag-adapter",
        samples=5,
        timeout_seconds=30,
        unstable_noise_pct=200,
        primary="geomean",
    )
    assert result.bench == "flag-bench"


def test_inspect_config_when_file_has_loop_keys_does_carry_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", **LOOP_CONFIG})
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.problems == []
    assert result.config == BenchlessConfig(
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200,
        primary="decode/time",
        checks="npm test",
        filter="npm run bench -- {names}",
        stop=StopConfig(target_value=1.5, max_iterations=20),
        hooks=HooksConfig(before="npm run warm-cache", after="npm run cool-down"),
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
    config_path = write_raw_config(tmp_path, "= invalid toml =")
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.config_path == str(config_path)

    assert has_problem(result.problems, re.escape(str(config_path)))
    assert result.config is None


def test_inspect_config_when_file_has_multiple_schema_issues_does_collect_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": 42, "samples": "bad", "adapter": 123})
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    joined = "\n".join(result.problems)
    assert len(result.problems) >= 3
    assert "bench" in joined
    assert "samples" in joined
    assert "adapter" in joined
    assert result.config is None


def test_inspect_config_when_filter_omits_names_placeholder_does_report_naming_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "filter": "npm run bench"})
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert has_problem(result.problems, r"filter.*\{names\}")


def test_inspect_config_when_target_value_with_geomean_primary_does_report_naming_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "stop": {"target_value": 1.5}})
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert has_problem(result.problems, r"target_value.*geomean|geomean.*target_value")


def test_inspect_config_when_runbook_missing_does_report_naming_field_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "runbook": "missing.md"})
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert has_problem(result.problems, "runbook")
    assert "missing.md" in "\n".join(result.problems)


def test_inspect_config_when_runbook_embeds_nul_does_report_problem_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_raw_config(tmp_path, 'bench = "config-bench"\nrunbook = "a\\u0000b"\n')
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert has_problem(result.problems, "runbook")


@pytest.mark.parametrize(
    ("key", "flags"),
    [
        pytest.param("bench", CliFlags(bench=""), id="bench"),
        pytest.param("prepare", CliFlags(bench="my-bench", prepare=""), id="prepare"),
        pytest.param("adapter", CliFlags(bench="my-bench", adapter=""), id="adapter"),
        pytest.param("config", CliFlags(bench="my-bench", config=""), id="config"),
    ],
)
def test_inspect_config_when_flag_holds_empty_string_does_report_naming_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, flags: CliFlags
):
    monkeypatch.chdir(tmp_path)

    result = inspect_config(flags)

    assert has_problem(result.problems, rf"--{key}.*non-empty")


def test_inspect_config_when_multiple_flags_empty_does_collect_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags(bench="", adapter=""))

    assert len(result.problems) >= 2
    assert has_problem(result.problems, r"--bench.*non-empty")
    assert has_problem(result.problems, r"--adapter.*non-empty")


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


@pytest.mark.parametrize(
    "config_flag",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="spaces"),
        pytest.param("\t", id="tab"),
    ],
)
def test_inspect_config_when_config_flag_blank_does_report_and_skip_file_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_flag: str
):
    write_config(tmp_path, {"bench": "cwd-bench"})
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags(config=config_flag))

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
    ("env_var", "flags"),
    [
        pytest.param("GYMRAT_BENCH", CliFlags(), id="bench"),
        pytest.param("GYMRAT_PREPARE", CliFlags(bench="b"), id="prepare"),
        pytest.param("GYMRAT_ADAPTER", CliFlags(bench="b"), id="adapter"),
        pytest.param("GYMRAT_CONFIG", CliFlags(bench="b"), id="config"),
    ],
)
def test_inspect_config_when_string_env_var_blank_does_report_naming_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    flags: CliFlags,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, "")

    result = inspect_config(flags)

    assert has_problem(result.problems, rf"{env_var}.*non-empty")


def test_inspect_config_when_config_env_var_names_missing_path_does_report_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    missing_path = tmp_path / "typo.toml"
    monkeypatch.setenv("GYMRAT_CONFIG", str(missing_path))

    result = inspect_config(CliFlags(bench="my-bench"))

    assert has_problem(result.problems, re.escape(str(missing_path)))


@pytest.mark.parametrize("env_var", ["GYMRAT_SAMPLES", "GYMRAT_TIMEOUT"])
def test_inspect_config_when_integer_env_var_invalid_does_report_naming_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_var: str
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, "abc")

    result = inspect_config(CliFlags(bench="my-bench"))

    assert has_problem(result.problems, rf"{env_var}.*positive integer")


@pytest.mark.parametrize(
    ("env_var", "cap"),
    [
        pytest.param("GYMRAT_TIMEOUT", MAX_TIMEOUT_SECONDS, id="timeout"),
        pytest.param("GYMRAT_SAMPLES", MAX_SAFE_INTEGER, id="samples"),
    ],
)
def test_inspect_config_when_integer_env_var_exceeds_cap_does_report_naming_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_var: str, cap: int
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, str(cap + 1))

    result = inspect_config(CliFlags(bench="my-bench"))

    joined = "\n".join(result.problems)
    assert has_problem(result.problems, env_var)
    assert "a positive integer" in joined
