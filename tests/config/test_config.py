from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from gymrat.cli.options import OutputFormat
from gymrat.cli.run_setup import SharedFlags
from gymrat.config import (
    MAX_SAFE_INTEGER,
    MAX_TIMEOUT_SECONDS,
    BenchlessConfig,
    CliFlags,
    KindEntry,
    MetricEntry,
    StopConfig,
    config_trace_args,
    env_positive_int_result,
    find_implicit_base,
    inspect_config,
    is_positive_integer,
    resolve_config,
    runbook_problem,
)
from gymrat.errors import GymratError
from tests._config import benchless_config
from tests._git import run_git
from tests._imports import modules_imported_by
from tests.config._toml import write_config, write_raw

# ---------------------------------------------------------------------------
# No circular import at package load time
# ---------------------------------------------------------------------------


def test_config_module_when_imported_fresh_does_not_raise_import_error():
    modules_imported_by("gymrat.config")


# ---------------------------------------------------------------------------
# GYMRAT_* environment variables
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnvCase:
    """One GYMRAT_* precedence case.

    Bundles the variable under test, the flags/config that compete with it, and
    the settled field to read back, so a parametrized test takes a single case
    argument instead of one parameter per column.
    """

    env_var: str
    env_value: str
    settled_field: str
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
                "GYMRAT_PREPARE",
                "env-prepare",
                "prepare",
                "flag-prepare",
                flags=CliFlags(bench="b", prepare="flag-prepare"),
            ),
            id="flag-wins-prepare",
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

    assert getattr(result, case.settled_field) == case.expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("+5", id="plus-sign"),
        pytest.param("\N{ARABIC-INDIC DIGIT FIVE}", id="non-ascii-digit"),
        pytest.param("0", id="zero"),
    ],
)
def test_is_positive_integer_when_not_bare_positive_digits_does_reject(raw: str):
    accepted = is_positive_integer(raw)

    assert accepted is False


def test_env_positive_int_result_when_digit_string_exceeds_conversion_limit_does_report_problem(
    monkeypatch: pytest.MonkeyPatch,
):
    huge = "1" * 4301
    monkeypatch.setenv("GYMRAT_SAMPLES", huge)

    result = env_positive_int_result("GYMRAT_SAMPLES", maximum=MAX_SAFE_INTEGER)

    assert result.problem is not None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("10", 10, id="ceiling"),
        pytest.param("007", 7, id="leading-zeros"),
    ],
)
def test_env_positive_int_result_when_bare_digits_within_ceiling_does_accept(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
):
    monkeypatch.setenv("GYMRAT_SAMPLES", raw)

    result = env_positive_int_result("GYMRAT_SAMPLES", maximum=10)

    assert result.value == expected


# ---------------------------------------------------------------------------
# settlement failures match inspect_config
# ---------------------------------------------------------------------------


def test_resolve_config_when_settlement_fails_does_raise_first_inspect_config_problem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"samples": "bad"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GYMRAT_TIMEOUT", "abc")
    flags = CliFlags(bench="my-bench")
    first_problem = inspect_config(flags).problems[0]

    with pytest.raises(GymratError) as exc:
        resolve_config(flags)

    assert str(exc.value) == first_problem


# ---------------------------------------------------------------------------
# resolve_config — precedence and the bench requirement
# ---------------------------------------------------------------------------


def test_resolve_config_when_bench_missing_from_flags_and_config_does_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError):
        resolve_config(CliFlags())


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

    problem = runbook_problem("runbook.md", tmp_path)

    assert problem is not None
    assert '"runbook.md"' in problem
    assert problem.endswith(": Permission denied")


# ---------------------------------------------------------------------------
# inspect_config — shared helpers
# ---------------------------------------------------------------------------

# The fully defaulted settled config: what inspect_config yields when neither
# flags nor a config file supply any value. bench lives on ConfigInspection, not
# on the settled BenchlessConfig, so it never appears here.
DEFAULT_CONFIG = benchless_config()


# ---------------------------------------------------------------------------
# inspect_config — settled configuration
# ---------------------------------------------------------------------------


def test_inspect_config_when_no_file_or_flags_does_settle_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.config_path is None
    assert result.problems == []
    assert result.config == DEFAULT_CONFIG
    assert result.bench is None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            {"adapter": "custom-adapter", "samples": 20, "timeout_seconds": 3600},
            benchless_config(adapter="custom-adapter", samples=20, timeout_seconds=3600),
            id="flag-backed-keys",
        ),
        pytest.param(
            {
                "unstable_noise_pct": 150.5,
                "filter": "npm run bench -- {names}",
                "primary": "decode/time",
                "stop": {"target_value": 1.5},
                "metrics": {"decode/time": {"direction": "higher"}},
                "kinds": {"memory": {"gating": False}},
            },
            benchless_config(
                unstable_noise_pct=150.5,
                filter="npm run bench -- {names}",
                primary="decode/time",
                stop=StopConfig(target_value=1.5),
                metrics={"decode/time": MetricEntry(direction="higher")},
                kinds={"memory": KindEntry(gating=False)},
            ),
            id="file-only-keys",
        ),
    ],
)
def test_inspect_config_when_valid_file_provides_values_does_settle_from_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: dict[str, object],
    expected: BenchlessConfig,
):
    write_config(tmp_path, {"bench": "config-bench", **content})
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert result.problems == []
    assert result.config == expected
    assert result.bench == "config-bench"
    assert result.config_path == str(tmp_path / "gymrat.toml")


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


def test_inspect_config_when_base_dir_given_does_read_base_dir_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    write_config(base_dir, {"bench": "base-bench"})
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    write_config(cwd_dir, {"bench": "cwd-bench"})
    monkeypatch.chdir(cwd_dir)

    result = inspect_config(CliFlags(), str(base_dir))

    assert result.bench == "base-bench"


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
# find_implicit_base
# ---------------------------------------------------------------------------


def test_find_implicit_base_when_cwd_outside_any_repository_does_return_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    base = find_implicit_base()

    assert Path(base) == tmp_path


def test_find_implicit_base_when_git_declines_for_cwd_does_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".git").write_text("banana\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError):
        find_implicit_base()


# ---------------------------------------------------------------------------
# inspect_config — collected problems (a config problem never raises)
# ---------------------------------------------------------------------------


def _names_reported_once(problems: list[str], names: list[str]) -> list[str]:
    # The names exactly one problem carries; callers compare the problem count too,
    # so a missing or extra problem fails as well.
    return [name for name in names if sum(name in problem for problem in problems) == 1]


def _name_config_by_flag(_monkeypatch: pytest.MonkeyPatch, config_path: str) -> CliFlags:
    return CliFlags(bench="my-bench", config=config_path)


def _name_config_by_env_var(monkeypatch: pytest.MonkeyPatch, config_path: str) -> CliFlags:
    monkeypatch.setenv("GYMRAT_CONFIG", config_path)
    return CliFlags(bench="my-bench")


@pytest.mark.parametrize(
    "name_config",
    [
        pytest.param(_name_config_by_flag, id="flag"),
        pytest.param(_name_config_by_env_var, id="env-var"),
    ],
)
def test_inspect_config_when_config_source_names_missing_path_does_report_naming_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name_config: Callable[[pytest.MonkeyPatch, str], CliFlags],
):
    missing_path = tmp_path / "typo.toml"
    flags = name_config(monkeypatch, str(missing_path))

    result = inspect_config(flags)

    assert result.config_path == str(missing_path)
    assert len(result.problems) == 1
    assert str(missing_path) in result.problems[0]


def test_inspect_config_when_target_value_with_geomean_primary_does_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(
        tmp_path,
        {"bench": "config-bench", "stop": {"target_value": 1.5}, "primary": "geomean"},
    )
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags())

    assert len(result.problems) == 1


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

    assert len(result.problems) == 1
    assert result.problems[0].endswith(f'"{runbook}"')


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

    assert len(result.problems) == 1
    assert '"a\\u0000b"' in result.problems[0]
    assert result.problems[0].endswith(f": {_nul_path_reason()}")


def test_inspect_config_when_flags_hold_blank_strings_does_report_each_flag():
    names = ["--bench", "--prepare", "--adapter", "--config"]

    result = inspect_config(CliFlags(bench="\t", prepare="", adapter="", config="   "))

    assert (len(result.problems), _names_reported_once(result.problems, names)) == (
        len(names),
        names,
    )


def test_inspect_config_when_config_flag_blank_does_not_load_implicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "cwd-bench"})
    monkeypatch.chdir(tmp_path)

    result = inspect_config(CliFlags(bench="b", config="   "))

    # A blank --config is a broken source, not an absent one: the gymrat.toml
    # sitting in the working directory must not be picked up in its place.
    assert result.config_path is None


@pytest.mark.parametrize(
    ("config", "file_names"),
    [
        pytest.param(
            {"filter": "npm run bench", "runbook": "missing.md"},
            ["filter", "runbook"],
            id="loop-keys-then-runbook",
        ),
        pytest.param(
            {"samples": "bad", "filter": "npm run bench", "runbook": "missing.md"},
            ["samples"],
            id="schema-stops-before-loop-keys",
        ),
    ],
)
def test_inspect_config_when_every_step_fails_does_report_flags_env_and_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object],
    file_names: list[str],
):
    write_config(tmp_path, config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GYMRAT_SAMPLES", "abc")
    names = ["--adapter", "GYMRAT_SAMPLES", *file_names]

    result = inspect_config(CliFlags(bench="my-bench", adapter=""))

    assert (len(result.problems), _names_reported_once(result.problems, names)) == (
        len(names),
        names,
    )
    assert result.config is None


# ---------------------------------------------------------------------------
# inspect_config — GYMRAT_* environment variables
# ---------------------------------------------------------------------------


def test_inspect_config_when_config_env_var_blank_does_report_naming_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GYMRAT_CONFIG", "  \n  ")
    names = ["GYMRAT_CONFIG"]

    result = inspect_config(CliFlags(bench="b"))

    assert (len(result.problems), _names_reported_once(result.problems, names)) == (
        len(names),
        names,
    )


def test_inspect_config_when_every_field_env_var_invalid_does_report_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GYMRAT_BENCH", " ")
    monkeypatch.setenv("GYMRAT_PREPARE", "")
    monkeypatch.setenv("GYMRAT_ADAPTER", "")
    monkeypatch.setenv("GYMRAT_SAMPLES", "0")
    monkeypatch.setenv("GYMRAT_TIMEOUT", "0")
    names = [
        "GYMRAT_BENCH",
        "GYMRAT_PREPARE",
        "GYMRAT_ADAPTER",
        "GYMRAT_SAMPLES",
        "GYMRAT_TIMEOUT",
    ]

    result = inspect_config(CliFlags())

    assert (len(result.problems), _names_reported_once(result.problems, names)) == (
        len(names),
        names,
    )


def test_inspect_config_when_integer_env_var_exceeds_cap_does_reject_as_not_a_positive_integer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cap = MAX_TIMEOUT_SECONDS
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GYMRAT_TIMEOUT", str(cap + 1))
    names = ["GYMRAT_TIMEOUT"]

    result = inspect_config(CliFlags(bench="my-bench"))

    assert (len(result.problems), _names_reported_once(result.problems, names)) == (
        len(names),
        names,
    )
    assert result.problems[0].endswith(f'"{cap + 1}"')


# ---------------------------------------------------------------------------
# config_trace_args
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        pytest.param(CliFlags(samples=0, timeout=0), {"samples": 0, "timeout": 0}, id="zero-kept"),
        pytest.param(
            SharedFlags(samples=5, format=OutputFormat.json),
            {"samples": 5},
            id="subclass-fields-left-out",
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
