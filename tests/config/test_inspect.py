import re
from collections.abc import Callable
from pathlib import Path

import pytest
import tomli_w

from gymrat.adapters.types import Adapter, MetricDefaults
from gymrat.config import (
    MAX_SAFE_INTEGER,
    MAX_TIMEOUT_SECONDS,
    BenchlessConfig,
    CliFlags,
    HooksConfig,
    KindEntry,
    MetricEntry,
    StopConfig,
    resolve_metric_meta,
)
from gymrat.config.inspect import inspect_config
from gymrat.model import Direction, MetricUnit, ResolvedMetricMeta
from gymrat.warn import WarnSink, warn_to_stderr


def make_adapter(
    defaults_fn: Callable[[str], MetricDefaults] = lambda _name: MetricDefaults(direction="lower"),
) -> Adapter:
    """Build a mock adapter whose per-metric defaults come from ``defaults_fn``."""

    class MockAdapter:
        name = "test-adapter"

        def parse(self, stdout: str, warn: WarnSink = warn_to_stderr) -> dict[str, float]:
            return {}

        def defaults(self, metric_name: str) -> MetricDefaults:
            return defaults_fn(metric_name)

    return MockAdapter()


def metric_meta(
    short_name: str,
    *,
    direction: Direction = "lower",
    gating: bool = True,
    exact: bool = False,
    unit: MetricUnit | None = None,
    kind: str = "other",
) -> ResolvedMetricMeta:
    """A resolved meta defaulting to a lower-is-better, gating, non-exact "other" metric."""
    return ResolvedMetricMeta(
        direction=direction,
        gating=gating,
        exact=exact,
        unit=unit,
        kind=kind,
        short_name=short_name,
    )


# ---------------------------------------------------------------------------
# resolve_metric_meta — adapter defaults
# ---------------------------------------------------------------------------


def test_resolve_metric_meta_when_config_metrics_none_does_default_gating_exact_kind_short_name():
    adapter = make_adapter()

    result = resolve_metric_meta(["response-time"], None, adapter)

    assert result == {"response-time": metric_meta("response-time")}


def test_resolve_metric_meta_when_adapter_returns_unit_does_carry_unit():
    adapter = make_adapter(lambda _name: MetricDefaults(direction="lower", unit="ns"))

    result = resolve_metric_meta(["response-time"], None, adapter)

    assert result == {"response-time": metric_meta("response-time", unit="ns")}


def test_resolve_metric_meta_when_adapter_reports_kind_and_short_name_does_carry_both():
    adapter = make_adapter(
        lambda _name: MetricDefaults(direction="lower", kind="memory", short_name="heap")
    )

    result = resolve_metric_meta(["bench-a/heap"], None, adapter)

    assert result == {"bench-a/heap": metric_meta("heap", kind="memory")}


# ---------------------------------------------------------------------------
# resolve_metric_meta — per-metric config overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "entry", "expected"),
    [
        pytest.param(
            "throughput",
            MetricEntry(direction="higher"),
            metric_meta("throughput", direction="higher"),
            id="direction",
        ),
        pytest.param(
            "response-time",
            MetricEntry(gating=False),
            metric_meta("response-time", gating=False),
            id="gating",
        ),
        pytest.param(
            "response-time",
            MetricEntry(exact=True),
            metric_meta("response-time", exact=True),
            id="exact",
        ),
    ],
)
def test_resolve_metric_meta_when_config_sets_single_field_does_override(
    metric_name: str, entry: MetricEntry, expected: ResolvedMetricMeta
):
    adapter = make_adapter()
    config_metrics = {metric_name: entry}

    result = resolve_metric_meta([metric_name], config_metrics, adapter)

    assert result == {metric_name: expected}


def test_resolve_metric_meta_when_config_sets_only_gating_does_keep_direction_and_default_exact():
    adapter = make_adapter(lambda _name: MetricDefaults(direction="lower", unit="bytes"))
    config_metrics = {"memory-usage": MetricEntry(gating=False)}

    result = resolve_metric_meta(["memory-usage"], config_metrics, adapter)

    assert result == {"memory-usage": metric_meta("memory-usage", unit="bytes", gating=False)}


# ---------------------------------------------------------------------------
# resolve_metric_meta — multiple metrics and unused entries
# ---------------------------------------------------------------------------


def test_resolve_metric_meta_when_multiple_names_does_resolve_each_in_order():
    def defaults_fn(name: str) -> MetricDefaults:
        if name == "response-time":
            return MetricDefaults(direction="lower", unit="ns")
        if name == "throughput":
            return MetricDefaults(direction="higher")
        return MetricDefaults(direction="lower")

    adapter = make_adapter(defaults_fn)
    config_metrics = {
        "response-time": MetricEntry(gating=False),
        "throughput": MetricEntry(exact=True),
    }

    result = resolve_metric_meta(["response-time", "throughput"], config_metrics, adapter)

    assert list(result) == ["response-time", "throughput"]
    assert result == {
        "response-time": metric_meta("response-time", unit="ns", gating=False),
        "throughput": metric_meta("throughput", direction="higher", exact=True),
    }


def test_resolve_metric_meta_when_config_has_unused_entries_does_ignore_them():
    adapter = make_adapter()
    config_metrics = {
        "response-time": MetricEntry(gating=False),
        "unused": MetricEntry(gating=True, exact=True),
    }

    result = resolve_metric_meta(["response-time"], config_metrics, adapter)

    assert result == {"response-time": metric_meta("response-time", gating=False)}


# ---------------------------------------------------------------------------
# resolve_metric_meta — kind-level gating
# ---------------------------------------------------------------------------


def test_resolve_metric_meta_when_kind_sets_gating_does_apply_only_to_matching_kind():
    def defaults_fn(name: str) -> MetricDefaults:
        if name.endswith("/heap"):
            return MetricDefaults(direction="lower", kind="memory", short_name="heap")
        return MetricDefaults(direction="lower", kind="time", short_name="time")

    adapter = make_adapter(defaults_fn)
    config_kinds = {"memory": KindEntry(gating=False)}

    result = resolve_metric_meta(["bench-a/heap", "bench-a/time"], None, adapter, config_kinds)

    assert result == {
        "bench-a/heap": metric_meta("heap", gating=False, kind="memory"),
        "bench-a/time": metric_meta("time", kind="time"),
    }


def test_resolve_metric_meta_when_kind_entry_matches_no_metric_does_ignore_it():
    adapter = make_adapter(
        lambda _name: MetricDefaults(direction="lower", kind="memory", short_name="heap")
    )
    config_kinds = {"io": KindEntry(gating=False)}

    result = resolve_metric_meta(["bench-a/heap"], None, adapter, config_kinds)

    assert result == {"bench-a/heap": metric_meta("heap", kind="memory")}


def test_resolve_metric_meta_when_metric_and_kind_disagree_does_let_metric_win():
    adapter = make_adapter(
        lambda name: MetricDefaults(
            direction="lower", kind="memory", short_name=name.split("/")[-1]
        )
    )
    config_metrics = {"bench-a/heap": MetricEntry(gating=True)}
    config_kinds = {"memory": KindEntry(gating=False)}

    result = resolve_metric_meta(
        ["bench-a/heap", "bench-a/rss"], config_metrics, adapter, config_kinds
    )

    assert result == {
        "bench-a/heap": metric_meta("heap", kind="memory"),
        "bench-a/rss": metric_meta("rss", kind="memory", gating=False),
    }


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
