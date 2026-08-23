import json
from pathlib import Path

import pytest

from gymrat_py.config import (
    MAX_TIMEOUT_SECONDS,
    CliFlags,
    ConfigFile,
    ConfigFileResult,
    HooksConfig,
    KindEntry,
    MetricEntry,
    ResolvedConfig,
    StopConfig,
    load_config_file,
    load_config_file_collecting,
    resolve_config,
)
from gymrat_py.errors import GymratError

# Byte-order mark that editors on Windows prepend to UTF-8 files: EF BB BF.
UTF8_BOM = "﻿"

# Full loop configuration shared between the round-trip parsing tests.
LOOP_CONFIG: dict[str, object] = {
    "checks": "npm test",
    "filter": "npm run bench -- {names}",
    "primary": "decode/time",
    "stop": {"targetValue": 1.5, "maxIterations": 20},
    "hooks": {"before": "npm run warm-cache", "after": "npm run cool-down"},
}

# Characters JavaScript regexes treat as line terminators; a `^…$` key pattern
# stops at every one of them, so any of these embedded in a config key can
# smuggle the rest of the key past validation.
LINE_BREAKS = ["\n", "\r", "\u2028", "\u2029"]


def write_config(directory: Path, content: dict[str, object]) -> Path:
    return write_raw(directory, json.dumps(content))


def write_raw(directory: Path, text: str) -> Path:
    config_path = directory / "gymrat.json"
    config_path.write_text(text, encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# missing file
# ---------------------------------------------------------------------------


def test_load_config_file_when_file_missing_does_return_empty_config(tmp_path: Path):
    missing = tmp_path / "nonexistent.json"

    assert load_config_file(missing) == ConfigFile()


def test_load_config_file_when_file_missing_and_required_does_raise_naming_path(tmp_path: Path):
    missing = tmp_path / "nonexistent.json"

    with pytest.raises(GymratError) as exc:
        load_config_file(missing, required=True)

    assert str(missing) in str(exc.value)


# ---------------------------------------------------------------------------
# unreadable path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("required", [False, True])
def test_load_config_file_when_path_is_directory_does_raise_naming_path(
    tmp_path: Path, required: bool
):
    with pytest.raises(GymratError) as exc:
        load_config_file(tmp_path, required=required)

    assert str(tmp_path) in str(exc.value)
    assert "Cannot read config file" in str(exc.value)


# ---------------------------------------------------------------------------
# valid JSON with known keys
# ---------------------------------------------------------------------------


def test_load_config_file_when_only_bench_given_does_return_parsed_bench(tmp_path: Path):
    config_path = write_config(tmp_path, {"bench": "custom-bench"})

    assert load_config_file(config_path) == ConfigFile(bench="custom-bench")


def test_load_config_file_when_all_known_keys_given_does_round_trip(tmp_path: Path):
    config_path = write_config(
        tmp_path,
        {
            "bench": "bench-name",
            "prepare": "prepare-cmd",
            "adapter": "adapter-name",
            "samples": 10,
            "timeoutSeconds": 30,
            "unstableNoisePct": 150.5,
            "metrics": {
                "metric1": {"direction": "lower", "gating": True, "exact": False},
                "metric2": {"direction": "higher"},
            },
        },
    )

    assert load_config_file(config_path) == ConfigFile(
        bench="bench-name",
        prepare="prepare-cmd",
        adapter="adapter-name",
        samples=10,
        timeout_seconds=30,
        unstable_noise_pct=150.5,
        metrics={
            "metric1": MetricEntry(direction="lower", gating=True, exact=False),
            "metric2": MetricEntry(direction="higher"),
        },
    )


def test_load_config_file_when_partial_metrics_metadata_given_does_round_trip(tmp_path: Path):
    config_path = write_config(
        tmp_path,
        {"metrics": {"responseTime": {"direction": "lower"}, "throughput": {"gating": True}}},
    )

    assert load_config_file(config_path) == ConfigFile(
        metrics={
            "responseTime": MetricEntry(direction="lower"),
            "throughput": MetricEntry(gating=True),
        }
    )


# ---------------------------------------------------------------------------
# invalid JSON / BOM / non-finite literals
# ---------------------------------------------------------------------------


def test_load_config_file_when_json_invalid_does_raise_naming_path(tmp_path: Path):
    config_path = write_raw(tmp_path, "{ invalid json }")

    with pytest.raises(GymratError) as exc:
        load_config_file(config_path)

    assert str(config_path) in str(exc.value)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_load_config_file_when_non_finite_literal_does_raise_parse_failure(
    tmp_path: Path, literal: str
):
    config_path = write_raw(tmp_path, f'{{"unstableNoisePct": {literal}}}')

    with pytest.raises(GymratError) as exc:
        load_config_file(config_path)

    assert str(config_path) in str(exc.value)


def test_load_config_file_when_prefixed_with_bom_does_parse_as_if_absent(tmp_path: Path):
    config_path = write_raw(
        tmp_path, f"{UTF8_BOM}{json.dumps({'bench': 'bom-bench', 'samples': 5})}"
    )

    assert load_config_file(config_path) == ConfigFile(bench="bom-bench", samples=5)


# ---------------------------------------------------------------------------
# non-object JSON root
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("[]", id="array"),
        pytest.param('"bench"', id="string"),
        pytest.param("3", id="number"),
        pytest.param("true", id="boolean"),
        pytest.param("null", id="null"),
    ],
)
def test_load_config_file_when_root_not_object_does_raise_naming_json_object(
    tmp_path: Path, raw: str
):
    config_path = write_raw(tmp_path, raw)

    with pytest.raises(GymratError) as exc:
        load_config_file(config_path)

    assert str(config_path) in str(exc.value)
    assert "JSON object" in str(exc.value)


# ---------------------------------------------------------------------------
# unknown keys
# ---------------------------------------------------------------------------


def test_load_config_file_when_unknown_top_level_key_does_raise_naming_key(tmp_path: Path):
    config_path = write_config(tmp_path, {"unknownKey": "value"})

    with pytest.raises(GymratError, match="unknownKey"):
        load_config_file(config_path)


def test_load_config_file_when_unknown_key_mixed_with_known_does_raise_naming_unknown(
    tmp_path: Path,
):
    config_path = write_config(tmp_path, {"bench": "name", "badKey": "value"})

    with pytest.raises(GymratError, match="badKey"):
        load_config_file(config_path)


def test_load_config_file_when_empty_string_top_level_key_does_name_quoted_empty_key(
    tmp_path: Path,
):
    config_path = write_config(tmp_path, {"": 1})

    with pytest.raises(GymratError) as exc:
        load_config_file(config_path)

    assert 'Unknown config key: ""' in str(exc.value)
    assert "JSON object" not in str(exc.value)


# ---------------------------------------------------------------------------
# empty object
# ---------------------------------------------------------------------------


def test_load_config_file_when_empty_object_does_return_empty_config(tmp_path: Path):
    config_path = write_config(tmp_path, {})

    assert load_config_file(config_path) == ConfigFile()


# ---------------------------------------------------------------------------
# string-typed keys holding non-strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("bench", 42, id="bench-number"),
        pytest.param("bench", ["a"], id="bench-array"),
        pytest.param("prepare", True, id="prepare-boolean"),
        pytest.param("prepare", {"cmd": "x"}, id="prepare-object"),
        pytest.param("adapter", None, id="adapter-null"),
        pytest.param("checks", 42, id="checks-number"),
        pytest.param("filter", ["a"], id="filter-array"),
        pytest.param("primary", None, id="primary-null"),
    ],
)
def test_load_config_file_when_string_key_holds_non_string_does_name_key_and_string(
    tmp_path: Path, key: str, value: object
):
    config_path = write_config(tmp_path, {key: value})

    with pytest.raises(GymratError, match=rf"{key}.*string"):
        load_config_file(config_path)


# ---------------------------------------------------------------------------
# non-empty-string keys holding empty / whitespace-only strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["checks", "bench", "prepare", "adapter", "runbook", "primary"])
def test_load_config_file_when_non_empty_string_key_holds_empty_does_name_key_and_non_empty(
    tmp_path: Path, key: str
):
    config_path = write_config(tmp_path, {key: ""})

    with pytest.raises(GymratError, match=rf"{key}.*non-empty"):
        load_config_file(config_path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("checks", " ", id="space"),
        pytest.param("bench", "\t", id="tab"),
        pytest.param("prepare", "  \n  ", id="padded-newline"),
        pytest.param("adapter", "\u00a0", id="nbsp"),
    ],
)
def test_load_config_file_when_non_empty_string_key_holds_whitespace_does_name_key_and_non_empty(
    tmp_path: Path, key: str, value: str
):
    config_path = write_config(tmp_path, {key: value})

    with pytest.raises(GymratError, match=rf"{key}.*non-empty"):
        load_config_file(config_path)


# ---------------------------------------------------------------------------
# positive-integer keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("samples", "ten", id="samples-string"),
        pytest.param("samples", 1.5, id="samples-non-integer"),
        pytest.param("samples", 0, id="samples-zero"),
        pytest.param("timeoutSeconds", -1, id="timeout-negative"),
        pytest.param("timeoutSeconds", True, id="timeout-boolean"),
        pytest.param("timeoutSeconds", None, id="timeout-null"),
    ],
)
def test_load_config_file_when_integer_key_invalid_does_name_key_and_positive_integer(
    tmp_path: Path, key: str, value: object
):
    config_path = write_config(tmp_path, {key: value})

    with pytest.raises(GymratError, match=rf"{key}.*positive integer"):
        load_config_file(config_path)


def test_load_config_file_when_integer_key_given_integral_float_does_accept(tmp_path: Path):
    config_path = write_raw(tmp_path, '{"samples": 5.0}')

    assert load_config_file(config_path) == ConfigFile(samples=5)


def test_load_config_file_when_timeout_exceeds_cap_does_name_key_and_cap(tmp_path: Path):
    config_path = write_config(tmp_path, {"timeoutSeconds": MAX_TIMEOUT_SECONDS + 1})

    with pytest.raises(GymratError) as exc:
        load_config_file(config_path)

    assert "timeoutSeconds" in str(exc.value)
    assert "no greater than 2147483" in str(exc.value)


def test_load_config_file_when_timeout_on_cap_does_accept(tmp_path: Path):
    config_path = write_config(tmp_path, {"timeoutSeconds": MAX_TIMEOUT_SECONDS})

    assert load_config_file(config_path) == ConfigFile(timeout_seconds=2_147_483)


# ---------------------------------------------------------------------------
# unstableNoisePct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("loud", id="string"),
        pytest.param(0, id="zero"),
        pytest.param(-5, id="negative"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="null"),
        pytest.param(0.25, id="below-floor"),
    ],
)
def test_load_config_file_when_noise_pct_invalid_does_name_key_and_noise_floor(
    tmp_path: Path, value: object
):
    config_path = write_config(tmp_path, {"unstableNoisePct": value})

    with pytest.raises(GymratError) as exc:
        load_config_file(config_path)

    message = str(exc.value)
    assert "unstableNoisePct" in message
    assert "0.5" in message
    assert "noise floor" in message


def test_load_config_file_when_noise_pct_on_floor_does_accept(tmp_path: Path):
    config_path = write_config(tmp_path, {"unstableNoisePct": 0.5})

    assert load_config_file(config_path) == ConfigFile(unstable_noise_pct=0.5)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([], id="array"),
        pytest.param("latency", id="string"),
        pytest.param(3, id="number"),
        pytest.param(None, id="null"),
    ],
)
def test_load_config_file_when_metrics_not_object_does_name_metrics_and_object(
    tmp_path: Path, value: object
):
    config_path = write_config(tmp_path, {"metrics": value})

    with pytest.raises(GymratError, match=r"metrics.*object"):
        load_config_file(config_path)


def test_load_config_file_when_metrics_entry_not_object_does_name_entry_and_object(tmp_path: Path):
    config_path = write_config(tmp_path, {"metrics": {"latency": "lower"}})

    with pytest.raises(GymratError, match=r"metrics\.latency.*object"):
        load_config_file(config_path)


def test_load_config_file_when_metrics_entry_under_empty_key_invalid_does_quote_empty_key(
    tmp_path: Path,
):
    config_path = write_config(tmp_path, {"metrics": {"": 5}})

    with pytest.raises(GymratError, match=r'metrics\."".*object'):
        load_config_file(config_path)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("sideways", id="unknown-string"),
        pytest.param("Lower", id="wrong-case"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="null"),
    ],
)
def test_load_config_file_when_metrics_direction_invalid_does_name_direction_and_options(
    tmp_path: Path, value: object
):
    config_path = write_config(tmp_path, {"metrics": {"latency": {"direction": value}}})

    with pytest.raises(GymratError, match=r'metrics\.latency\.direction.*"lower".*"higher"'):
        load_config_file(config_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("gating", "yes", id="gating-string"),
        pytest.param("gating", None, id="gating-null"),
        pytest.param("exact", 1, id="exact-number"),
    ],
)
def test_load_config_file_when_metrics_flag_non_boolean_does_name_field_and_boolean(
    tmp_path: Path, field: str, value: object
):
    config_path = write_config(tmp_path, {"metrics": {"latency": {field: value}}})

    with pytest.raises(GymratError, match=rf"metrics\.latency\.{field}.*boolean"):
        load_config_file(config_path)


def test_load_config_file_when_metrics_entry_has_unknown_key_does_name_key(tmp_path: Path):
    config_path = write_config(
        tmp_path, {"metrics": {"latency": {"direction": "lower", "threshold": "higher"}}}
    )

    with pytest.raises(GymratError, match=r"metrics\.latency\.threshold"):
        load_config_file(config_path)


@pytest.mark.parametrize("char", LINE_BREAKS)
def test_load_config_file_when_metrics_key_embeds_line_break_does_name_metrics(
    tmp_path: Path, char: str
):
    smuggled = f"latency{char}direction: 999, gating: 0"
    config_path = write_config(tmp_path, {"metrics": {smuggled: {"direction": "lower"}}})

    with pytest.raises(GymratError, match="metrics"):
        load_config_file(config_path)


# ---------------------------------------------------------------------------
# kinds
# ---------------------------------------------------------------------------


def test_load_config_file_when_kinds_key_embeds_line_break_does_name_kinds(tmp_path: Path):
    config_path = write_config(tmp_path, {"kinds": {"memory\ngating: 999": {"gating": False}}})

    with pytest.raises(GymratError, match="kinds"):
        load_config_file(config_path)


def test_load_config_file_when_kinds_section_given_does_round_trip(tmp_path: Path):
    config_path = write_config(tmp_path, {"kinds": {"memory": {"gating": False}, "time": {}}})

    assert load_config_file(config_path) == ConfigFile(
        kinds={"memory": KindEntry(gating=False), "time": KindEntry()}
    )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([], id="array"),
        pytest.param("memory", id="string"),
        pytest.param(3, id="number"),
        pytest.param(None, id="null"),
    ],
)
def test_load_config_file_when_kinds_not_object_does_name_kinds_and_object(
    tmp_path: Path, value: object
):
    config_path = write_config(tmp_path, {"kinds": value})

    with pytest.raises(GymratError, match=r"kinds.*object"):
        load_config_file(config_path)


def test_load_config_file_when_kinds_entry_not_object_does_name_entry_and_object(tmp_path: Path):
    config_path = write_config(tmp_path, {"kinds": {"memory": False}})

    with pytest.raises(GymratError, match=r"kinds\.memory.*object"):
        load_config_file(config_path)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("yes", id="string"),
        pytest.param(1, id="number"),
        pytest.param(None, id="null"),
    ],
)
def test_load_config_file_when_kinds_gating_non_boolean_does_name_gating_and_boolean(
    tmp_path: Path, value: object
):
    config_path = write_config(tmp_path, {"kinds": {"memory": {"gating": value}}})

    with pytest.raises(GymratError, match=r"kinds\.memory\.gating.*boolean"):
        load_config_file(config_path)


def test_load_config_file_when_kinds_entry_has_unknown_key_does_name_dotted_path(tmp_path: Path):
    config_path = write_config(tmp_path, {"kinds": {"memory": {"gating": False, "threshold": 5}}})

    with pytest.raises(GymratError, match=r"Unknown config key: kinds\.memory\.threshold"):
        load_config_file(config_path)


# ---------------------------------------------------------------------------
# runbook and loop keys
# ---------------------------------------------------------------------------


def test_load_config_file_when_runbook_given_does_round_trip(tmp_path: Path):
    config_path = write_config(tmp_path, {"runbook": "RUNBOOK.md"})

    assert load_config_file(config_path) == ConfigFile(runbook="RUNBOOK.md")


def test_load_config_file_when_loop_keys_given_does_round_trip(tmp_path: Path):
    config_path = write_config(tmp_path, LOOP_CONFIG)

    assert load_config_file(config_path) == ConfigFile(
        checks="npm test",
        filter="npm run bench -- {names}",
        primary="decode/time",
        stop=StopConfig(target_value=1.5, max_iterations=20),
        hooks=HooksConfig(before="npm run warm-cache", after="npm run cool-down"),
    )


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hooks", "expected"),
    [
        pytest.param(
            {"before": "npm run warm-cache"},
            HooksConfig(before="npm run warm-cache"),
            id="only-before",
        ),
        pytest.param(
            {"after": "npm run cool-down"},
            HooksConfig(after="npm run cool-down"),
            id="only-after",
        ),
    ],
)
def test_load_config_file_when_hooks_partial_does_round_trip(
    tmp_path: Path, hooks: dict[str, str], expected: HooksConfig
):
    config_path = write_config(tmp_path, {"hooks": hooks})

    assert load_config_file(config_path) == ConfigFile(hooks=expected)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("gymrat.hooks", id="string"),
        pytest.param([], id="array"),
        pytest.param(True, id="boolean"),
        pytest.param(None, id="null"),
    ],
)
def test_load_config_file_when_hooks_not_object_does_name_hooks_and_object(
    tmp_path: Path, value: object
):
    config_path = write_config(tmp_path, {"hooks": value})

    with pytest.raises(GymratError, match=r"hooks.*object"):
        load_config_file(config_path)


@pytest.mark.parametrize(
    ("stage", "value"),
    [
        pytest.param("before", "", id="before-empty"),
        pytest.param("after", "", id="after-empty"),
        pytest.param("before", 42, id="before-number"),
        pytest.param("after", None, id="after-null"),
    ],
)
def test_load_config_file_when_hooks_command_not_non_empty_string_does_name_stage(
    tmp_path: Path, stage: str, value: object
):
    config_path = write_config(tmp_path, {"hooks": {stage: value}})

    with pytest.raises(GymratError, match=rf"hooks\.{stage}.*non-empty string"):
        load_config_file(config_path)


@pytest.mark.parametrize(
    ("stage", "value"),
    [
        pytest.param("before", " ", id="before-space"),
        pytest.param("after", "\t\n ", id="after-mixed-whitespace"),
    ],
)
def test_load_config_file_when_hooks_command_whitespace_only_does_name_stage(
    tmp_path: Path, stage: str, value: str
):
    config_path = write_config(tmp_path, {"hooks": {stage: value}})

    with pytest.raises(GymratError, match=rf"hooks\.{stage}.*non-empty"):
        load_config_file(config_path)


def test_load_config_file_when_hooks_has_unknown_key_does_name_dotted_path(tmp_path: Path):
    config_path = write_config(
        tmp_path, {"hooks": {"before": "npm run warm-cache", "during": "npm run mid"}}
    )

    with pytest.raises(GymratError, match=r"Unknown config key: hooks\.during"):
        load_config_file(config_path)


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        pytest.param("targetValue", "fast", r"stop\.targetValue.*number", id="target-string"),
        pytest.param("maxIterations", 0, r"stop\.maxIterations.*positive integer", id="max-zero"),
        pytest.param(
            "maxIterations", 1.5, r"stop\.maxIterations.*positive integer", id="max-non-integer"
        ),
    ],
)
def test_load_config_file_when_stop_field_invalid_does_name_field(
    tmp_path: Path, field: str, value: object, pattern: str
):
    config_path = write_config(tmp_path, {"stop": {field: value}})

    with pytest.raises(GymratError, match=pattern):
        load_config_file(config_path)


def test_load_config_file_when_stop_has_unknown_key_does_name_dotted_path(tmp_path: Path):
    config_path = write_config(tmp_path, {"stop": {"targetValue": 1, "patience": 3}})

    with pytest.raises(GymratError, match=r"Unknown config key: stop\.patience"):
        load_config_file(config_path)


# ---------------------------------------------------------------------------
# load_config_file_collecting
# ---------------------------------------------------------------------------


def test_load_config_file_collecting_when_file_missing_does_report_absent(tmp_path: Path):
    missing = tmp_path / "nonexistent.json"

    assert load_config_file_collecting(missing, required=False) == ConfigFileResult(
        config_file=ConfigFile(), exists=False, problems=[]
    )


def test_load_config_file_collecting_when_file_missing_and_required_does_report_problem(
    tmp_path: Path,
):
    missing = tmp_path / "nonexistent.json"

    result = load_config_file_collecting(missing, required=True)

    assert result.config_file is None
    assert result.exists is False
    assert any(str(missing) in problem for problem in result.problems)


def test_load_config_file_collecting_when_valid_does_report_config_and_no_problems(tmp_path: Path):
    config_path = write_config(tmp_path, {"bench": "custom-bench"})

    assert load_config_file_collecting(config_path, required=False) == ConfigFileResult(
        config_file=ConfigFile(bench="custom-bench"), exists=True, problems=[]
    )


def test_load_config_file_collecting_when_multiple_fields_invalid_does_report_every_problem(
    tmp_path: Path,
):
    config_path = write_config(tmp_path, {"bench": 42, "samples": 0})

    result = load_config_file_collecting(config_path, required=False)

    assert result.config_file is None
    assert result.exists is True
    assert len(result.problems) == 2
    joined = "\n".join(result.problems)
    assert "bench" in joined
    assert "samples" in joined


def test_load_config_file_collecting_when_path_is_directory_does_collect_read_failure(
    tmp_path: Path,
):
    result = load_config_file_collecting(tmp_path, required=False)

    assert result.config_file is None
    assert result.exists is True
    assert any(f"Cannot read config file at {tmp_path}: " in problem for problem in result.problems)


# ---------------------------------------------------------------------------
# resolve_config
# ---------------------------------------------------------------------------

# Every GYMRAT_* variable resolve_config consults, cleared before each test so
# an ambient value in the developer's shell cannot bleed into these cases.
GYMRAT_ENV_VARS = (
    "GYMRAT_BENCH",
    "GYMRAT_PREPARE",
    "GYMRAT_ADAPTER",
    "GYMRAT_SAMPLES",
    "GYMRAT_TIMEOUT",
    "GYMRAT_CONFIG",
)


@pytest.fixture(autouse=True)
def _clear_gymrat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in GYMRAT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


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
                "timeoutSeconds": 3600,
                "unstableNoisePct": 150.5,
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
    config_path = tmp_path / "custom-config.json"
    config_path.write_text(json.dumps({"bench": "custom-bench"}), encoding="utf-8")

    result = resolve_config(CliFlags(config=str(config_path)))

    assert result.bench == "custom-bench"


def test_resolve_config_when_explicit_config_path_missing_does_raise_naming_path(tmp_path: Path):
    missing = tmp_path / "typo.json"

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
    write_config(tmp_path, {"timeoutSeconds": 3600})
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
        {"bench": "config-bench", "stop": {"targetValue": 1.5}, **overrides},
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GymratError) as exc:
        resolve_config(CliFlags())

    message = str(exc.value)
    assert "targetValue" in message
    assert "geomean" in message


def test_resolve_config_when_stop_sets_only_max_iterations_under_geomean_does_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_config(tmp_path, {"bench": "config-bench", "stop": {"maxIterations": 5}})
    monkeypatch.chdir(tmp_path)

    result = resolve_config(CliFlags())

    assert result.stop == StopConfig(max_iterations=5)
