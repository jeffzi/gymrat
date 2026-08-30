import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import tomli_w

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
    resolve_benchless_config,
    resolve_config,
)
from gymrat.errors import GymratError

LOOP_CONFIG: dict[str, object] = {
    "checks": "npm test",
    "filter": "npm run bench -- {names}",
    "primary": "decode/time",
    "stop": {"target_value": 1.5, "max_iterations": 20},
    "hooks": {"before": "npm run warm-cache", "after": "npm run cool-down"},
}

# Values every positive-integer env var (GYMRAT_SAMPLES, GYMRAT_TIMEOUT) rejects.
INVALID_POSITIVE_INTEGER_VALUES = [
    pytest.param("abc", id="non-numeric"),
    pytest.param("1.5", id="non-integer"),
    pytest.param("0", id="zero"),
    pytest.param("-1", id="negative"),
    pytest.param("", id="empty"),
    pytest.param("0x10", id="hex"),
]

# Both resolvers settle the implicit gymrat.toml through the same lookup, so any
# behavior that depends on the lookup has to hold for either entry point.
Resolver = Callable[..., BenchlessConfig]
RESOLVERS = [
    pytest.param(resolve_config, id="resolve_config"),
    pytest.param(resolve_benchless_config, id="resolve_benchless_config"),
]


def write_toml(path: Path, content: dict[str, object]) -> Path:
    path.write_text(tomli_w.dumps(content), encoding="utf-8")
    return path


def write_config(directory: Path, content: dict[str, object]) -> Path:
    return write_toml(directory / "gymrat.toml", content)


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


def _settle_env_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: EnvCase) -> object:
    """Apply a case's config file and env var, settle, and return the field value."""
    if case.config is not None:
        write_config(tmp_path, case.config)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(case.env_var, case.env_value)

    result = resolve_config(case.flags)

    return getattr(result, case.field)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(EnvCase("GYMRAT_BENCH", "env-bench", "bench", "env-bench"), id="bench"),
        pytest.param(
            EnvCase(
                "GYMRAT_PREPARE", "env-prepare", "prepare", "env-prepare", flags=CliFlags(bench="b")
            ),
            id="prepare",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_ADAPTER", "env-adapter", "adapter", "env-adapter", flags=CliFlags(bench="b")
            ),
            id="adapter",
        ),
        pytest.param(
            EnvCase("GYMRAT_SAMPLES", "42", "samples", 42, flags=CliFlags(bench="b")), id="samples"
        ),
        pytest.param(
            EnvCase("GYMRAT_TIMEOUT", "900", "timeout_seconds", 900, flags=CliFlags(bench="b")),
            id="timeout",
        ),
    ],
)
def test_resolve_config_when_env_var_set_and_flag_absent_does_use_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: EnvCase
):
    assert _settle_env_case(tmp_path, monkeypatch, case) == case.expected


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            EnvCase("GYMRAT_BENCH", "env-bench", "bench", "env-bench", config={"bench": "cfg"}),
            id="bench",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_PREPARE",
                "env-prepare",
                "prepare",
                "env-prepare",
                config={"bench": "b", "prepare": "config-prepare"},
            ),
            id="prepare",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_ADAPTER",
                "env-adapter",
                "adapter",
                "env-adapter",
                config={"bench": "b", "adapter": "config-adapter"},
            ),
            id="adapter",
        ),
        pytest.param(
            EnvCase("GYMRAT_SAMPLES", "42", "samples", 42, config={"bench": "b", "samples": 20}),
            id="samples",
        ),
        pytest.param(
            EnvCase(
                "GYMRAT_TIMEOUT",
                "900",
                "timeout_seconds",
                900,
                config={"bench": "b", "timeout_seconds": 3600},
            ),
            id="timeout",
        ),
    ],
)
def test_resolve_config_when_env_var_set_and_config_provides_field_does_use_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: EnvCase
):
    assert _settle_env_case(tmp_path, monkeypatch, case) == case.expected


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            EnvCase(
                "GYMRAT_BENCH",
                "env-bench",
                "bench",
                "flag-bench",
                flags=CliFlags(bench="flag-bench"),
            ),
            id="bench",
        ),
        pytest.param(
            EnvCase("GYMRAT_SAMPLES", "42", "samples", 7, flags=CliFlags(bench="b", samples=7)),
            id="samples",
        ),
    ],
)
def test_resolve_config_when_flag_present_does_ignore_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: EnvCase
):
    assert _settle_env_case(tmp_path, monkeypatch, case) == case.expected


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            EnvCase("GYMRAT_BENCH", "", "bench", "flag-bench", flags=CliFlags(bench="flag-bench")),
            id="blank-bench",
        ),
        pytest.param(
            EnvCase("GYMRAT_SAMPLES", "abc", "samples", 5, flags=CliFlags(bench="b", samples=5)),
            id="invalid-int",
        ),
    ],
)
def test_resolve_config_when_flag_present_does_not_validate_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: EnvCase
):
    assert _settle_env_case(tmp_path, monkeypatch, case) == case.expected


@pytest.mark.parametrize(
    ("env_var", "flags"),
    [
        pytest.param("GYMRAT_BENCH", CliFlags(), id="bench"),
        pytest.param("GYMRAT_PREPARE", CliFlags(bench="b"), id="prepare"),
        pytest.param("GYMRAT_ADAPTER", CliFlags(bench="b"), id="adapter"),
    ],
)
@pytest.mark.parametrize(
    "env_value",
    [
        pytest.param("", id="empty"),
        pytest.param(" ", id="space"),
        pytest.param("\t", id="tab"),
        pytest.param("  \n  ", id="padded-newline"),
    ],
)
def test_resolve_config_when_string_env_var_blank_does_raise_naming_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    flags: CliFlags,
    env_value: str,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, env_value)

    with pytest.raises(GymratError, match=rf"{env_var}.*non-empty"):
        resolve_config(flags)


@pytest.mark.parametrize("env_var", ["GYMRAT_SAMPLES", "GYMRAT_TIMEOUT"])
@pytest.mark.parametrize("env_value", INVALID_POSITIVE_INTEGER_VALUES)
def test_resolve_config_when_integer_env_var_invalid_does_raise_naming_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_var: str,
    env_value: str,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, env_value)

    with pytest.raises(GymratError, match=rf"{env_var}.*positive integer"):
        resolve_config(CliFlags(bench="my-bench"))


@pytest.mark.parametrize(
    ("env_var", "cap"),
    [
        pytest.param("GYMRAT_TIMEOUT", MAX_TIMEOUT_SECONDS, id="timeout"),
        pytest.param("GYMRAT_SAMPLES", MAX_SAFE_INTEGER, id="samples"),
    ],
)
def test_resolve_config_when_integer_env_var_exceeds_cap_does_raise_naming_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_var: str, cap: int
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(env_var, str(cap + 1))

    with pytest.raises(GymratError) as exc:
        resolve_config(CliFlags(bench="my-bench"))

    message = str(exc.value)
    assert env_var in message
    assert f"no greater than {cap}" in message


def test_resolve_config_when_config_env_var_names_file_and_config_flag_absent_does_load_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    env_config_path = write_toml(tmp_path / "env-config.toml", {"bench": "env-config-bench"})
    monkeypatch.setenv("GYMRAT_CONFIG", str(env_config_path))

    result = resolve_config(CliFlags())

    assert result.bench == "env-config-bench"


def test_resolve_config_when_config_env_var_names_missing_file_does_raise_naming_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    missing_path = tmp_path / "typo.toml"
    monkeypatch.setenv("GYMRAT_CONFIG", str(missing_path))

    with pytest.raises(GymratError) as exc:
        resolve_config(CliFlags(bench="my-bench"))

    assert str(missing_path) in str(exc.value)


def test_resolve_config_when_config_env_var_set_does_bypass_implicit_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "implicit-bench", "adapter": "implicit-adapter"})
    monkeypatch.chdir(tmp_path)
    env_config_path = write_toml(tmp_path / "alt-config.toml", {"bench": "alt-bench"})
    monkeypatch.setenv("GYMRAT_CONFIG", str(env_config_path))

    result = resolve_config(CliFlags())

    assert result.bench == "alt-bench"
    assert result.adapter == "metric-lines"


def test_resolve_config_when_config_env_var_blank_does_raise_naming_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GYMRAT_CONFIG", "")

    with pytest.raises(GymratError, match=r"GYMRAT_CONFIG.*non-empty"):
        resolve_config(CliFlags(bench="my-bench"))


def test_resolve_benchless_config_when_config_flag_blank_does_raise_naming_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError, match=r"--config.*non-empty"):
        resolve_benchless_config(CliFlags(config=""))


# ---------------------------------------------------------------------------
# base directory lookup (both resolvers)
# ---------------------------------------------------------------------------


def _nested_config_dirs(
    base_dir: Path, base_config: dict[str, object], nested_config: dict[str, object]
) -> Path:
    nested_dir = base_dir / "packages" / "core"
    nested_dir.mkdir(parents=True)
    write_config(base_dir, base_config)
    write_config(nested_dir, nested_config)
    return nested_dir


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_resolve_when_base_dir_given_does_read_base_dir_config_over_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve: Resolver
):
    nested_dir = _nested_config_dirs(
        tmp_path,
        {"bench": "a-bench", "checks": "base-checks"},
        {"bench": "a-bench", "checks": "cwd-checks"},
    )
    monkeypatch.chdir(nested_dir)

    result = resolve(CliFlags(), tmp_path)

    assert result.checks == "base-checks"


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_resolve_when_config_flag_relative_to_cwd_does_read_named_file_over_base_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve: Resolver
):
    nested_dir = _nested_config_dirs(
        tmp_path,
        {"bench": "a-bench", "checks": "base-checks"},
        {"bench": "a-bench", "checks": "cwd-checks"},
    )
    write_toml(nested_dir / "custom.toml", {"bench": "a-bench", "checks": "named-checks"})
    monkeypatch.chdir(nested_dir)

    result = resolve(CliFlags(config="custom.toml"), tmp_path)

    assert result.checks == "named-checks"


# ---------------------------------------------------------------------------
# runbook resolution (both resolvers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_resolve_when_config_has_no_runbook_does_omit_runbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve: Resolver
):
    write_config(tmp_path, {"bench": "a-bench"})
    monkeypatch.chdir(tmp_path)

    result = resolve(CliFlags())

    assert result.runbook is None


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_resolve_when_runbook_names_existing_file_does_resolve_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve: Resolver
):
    write_config(tmp_path, {"bench": "a-bench", "runbook": "RUNBOOK.md"})
    (tmp_path / "RUNBOOK.md").write_text("# Steps\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = resolve(CliFlags())

    assert result.runbook == str(tmp_path / "RUNBOOK.md")


def _make_docs_dir(base: Path) -> None:
    (base / "docs").mkdir()


@pytest.mark.parametrize(
    ("runbook", "setup"),
    [
        pytest.param("missing.md", None, id="missing"),
        pytest.param("docs", _make_docs_dir, id="directory"),
    ],
)
@pytest.mark.parametrize("resolve", RESOLVERS)
def test_resolve_when_runbook_does_not_name_existing_file_does_raise_naming_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resolve: Resolver,
    runbook: str,
    setup: Callable[[Path], None] | None,
):
    write_config(tmp_path, {"bench": "a-bench", "runbook": runbook})
    if setup is not None:
        setup(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError) as exc:
        resolve(CliFlags())

    message = str(exc.value)
    assert "runbook" in message
    assert runbook in message


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_resolve_when_runbook_embeds_nul_does_raise_naming_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve: Resolver
):
    (tmp_path / "gymrat.toml").write_text(
        'bench = "a-bench"\nrunbook = "a\\u0000b"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError) as exc:
        resolve(CliFlags())

    assert "runbook" in str(exc.value)


# ---------------------------------------------------------------------------
# implicit lookup in a git repository (both resolvers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_resolve_when_cwd_inside_git_repo_does_find_config_at_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve: Resolver
):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)  # noqa: S607
    write_config(tmp_path, {"bench": "repo-bench", "checks": "repo-checks"})
    nested_dir = tmp_path / "packages" / "core"
    nested_dir.mkdir(parents=True)
    monkeypatch.chdir(nested_dir)

    result = resolve(CliFlags(bench="flag-bench"))

    assert result.checks == "repo-checks"


# ---------------------------------------------------------------------------
# resolve_config — precedence, flags, and loop keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "config", "expected"),
    [
        pytest.param(
            CliFlags(bench="my-bench"),
            None,
            ResolvedConfig(
                bench="my-bench",
                adapter="metric-lines",
                samples=10,
                timeout_seconds=1800,
                unstable_noise_pct=200,
                primary="geomean",
            ),
            id="defaults-when-flags-and-config-empty",
        ),
        pytest.param(
            CliFlags(),
            {
                "bench": "config-bench",
                "adapter": "custom-adapter",
                "samples": 20,
                "timeout_seconds": 3600,
                "unstable_noise_pct": 150.5,
            },
            ResolvedConfig(
                bench="config-bench",
                adapter="custom-adapter",
                samples=20,
                timeout_seconds=3600,
                unstable_noise_pct=150.5,
                primary="geomean",
            ),
            id="config-beats-defaults",
        ),
        pytest.param(
            CliFlags(bench="flag-bench", adapter="flag-adapter"),
            {"bench": "config-bench", "adapter": "config-adapter", "samples": 20},
            ResolvedConfig(
                bench="flag-bench",
                adapter="flag-adapter",
                samples=20,
                timeout_seconds=1800,
                unstable_noise_pct=200,
                primary="geomean",
            ),
            id="flags-beat-config",
        ),
        pytest.param(
            CliFlags(bench="flag-bench", adapter="flag-adapter", samples=25, timeout=900),
            None,
            ResolvedConfig(
                bench="flag-bench",
                adapter="flag-adapter",
                samples=25,
                timeout_seconds=900,
                unstable_noise_pct=200,
                primary="geomean",
            ),
            id="flags-beat-defaults",
        ),
    ],
)
def test_resolve_config_when_flags_and_config_vary_does_merge_by_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: CliFlags,
    config: dict[str, object] | None,
    expected: ResolvedConfig,
):
    if config is not None:
        write_config(tmp_path, config)
    monkeypatch.chdir(tmp_path)

    assert resolve_config(flags) == expected


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


def test_resolve_config_when_prepare_provided_does_include_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    result = resolve_config(CliFlags(bench="my-bench", prepare="prepare-cmd"))

    assert result.prepare == "prepare-cmd"


@pytest.mark.parametrize(
    ("key", "flags"),
    [
        pytest.param("bench", CliFlags(bench=""), id="bench"),
        pytest.param("prepare", CliFlags(bench="my-bench", prepare=""), id="prepare"),
        pytest.param("adapter", CliFlags(bench="my-bench", adapter=""), id="adapter"),
        pytest.param("config", CliFlags(bench="my-bench", config=""), id="config"),
    ],
)
def test_resolve_config_when_flag_holds_empty_string_does_raise_naming_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, flags: CliFlags
):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError, match=rf"--{key}.*non-empty"):
        resolve_config(flags)


def test_resolve_config_when_explicit_config_path_given_does_load_it(tmp_path: Path):
    config_path = tmp_path / "custom-config.toml"
    config_path.write_text(tomli_w.dumps({"bench": "custom-bench"}), encoding="utf-8")

    result = resolve_config(CliFlags(config=str(config_path)))

    assert result.bench == "custom-bench"


def test_resolve_config_when_explicit_config_path_missing_does_raise_naming_path(tmp_path: Path):
    missing = tmp_path / "typo.toml"

    with pytest.raises(GymratError) as exc:
        resolve_config(CliFlags(bench="my-bench", config=str(missing)))

    assert str(missing) in str(exc.value)


def test_resolve_config_when_config_has_metrics_does_propagate_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(
        tmp_path,
        {
            "bench": "config-bench",
            "metrics": {"decode/time": {"direction": "higher", "gating": False, "exact": True}},
        },
    )
    monkeypatch.chdir(tmp_path)

    result = resolve_config(CliFlags())

    assert result == ResolvedConfig(
        bench="config-bench",
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200,
        primary="geomean",
        metrics={"decode/time": MetricEntry(direction="higher", gating=False, exact=True)},
    )


def test_resolve_config_when_config_has_kinds_does_propagate_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "kinds": {"memory": {"gating": False}}})
    monkeypatch.chdir(tmp_path)

    result = resolve_config(CliFlags())

    assert result == ResolvedConfig(
        bench="config-bench",
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200,
        primary="geomean",
        kinds={"memory": KindEntry(gating=False)},
    )


def test_resolve_config_when_config_has_no_metrics_does_omit_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench"})
    monkeypatch.chdir(tmp_path)

    result = resolve_config(CliFlags())

    assert result.metrics is None


def test_resolve_config_when_timeout_flag_given_does_beat_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"timeout_seconds": 3600})
    monkeypatch.chdir(tmp_path)

    result = resolve_config(CliFlags(bench="my-bench", timeout=1200))

    assert result.timeout_seconds == 1200


def test_resolve_config_when_config_has_loop_keys_does_propagate_over_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", **LOOP_CONFIG})
    monkeypatch.chdir(tmp_path)

    result = resolve_config(CliFlags())

    assert result == ResolvedConfig(
        bench="config-bench",
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


def test_resolve_config_when_filter_omits_names_placeholder_does_raise_naming_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "filter": "npm run bench"})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError) as exc:
        resolve_config(CliFlags())

    message = str(exc.value)
    assert "filter" in message
    assert "{names}" in message


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"primary": "geomean"}, id="geomean-explicit"),
        pytest.param({}, id="geomean-by-default"),
    ],
)
def test_resolve_config_when_stop_target_value_with_geomean_does_raise_naming_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
):
    write_config(
        tmp_path,
        {"bench": "config-bench", "stop": {"target_value": 1.5}, **overrides},
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError) as exc:
        resolve_config(CliFlags())

    message = str(exc.value)
    assert "target_value" in message
    assert "geomean" in message


def test_resolve_config_when_stop_sets_only_max_iterations_under_geomean_does_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "stop": {"max_iterations": 5}})
    monkeypatch.chdir(tmp_path)

    result = resolve_config(CliFlags())

    assert result.stop == StopConfig(max_iterations=5)
