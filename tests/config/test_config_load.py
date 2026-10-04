import json
import sys
from collections.abc import Callable
from dataclasses import fields
from operator import attrgetter
from pathlib import Path

import pytest
import tomli_w

from gymrat.config import (
    MAX_SAFE_INTEGER,
    MAX_TIMEOUT_SECONDS,
    ConfigFile,
    ConfigFileResult,
    HooksConfig,
    KindEntry,
    MetricEntry,
    StopConfig,
    load_config_file_collecting,
    validate_config_dict,
)
from gymrat.errors import GymratError
from tests.adapters._inputs import LINE_BREAKS
from tests.config._toml import write_config, write_raw

# Byte-order mark that editors on Windows prepend to UTF-8 files: EF BB BF.
UTF8_BOM = "﻿"

# Full loop configuration shared between the round-trip parsing tests.
LOOP_CONFIG: dict[str, object] = {
    "checks": "npm test",
    "filter": "npm run bench -- {names}",
    "primary": "decode/time",
    "stop": {"target_value": 1.5, "max_iterations": 20},
    "hooks": {"before": "npm run warm-cache", "after": "npm run cool-down"},
}

# Every character `str.splitlines` breaks on. Any of these embedded in a config
# key would split the key, and every message naming it, across lines, so a key
# holding one is rejected.
LINE_BREAK_CHARS = [line_break.char for line_break in LINE_BREAKS]


def _unknown_line_break_key_param(char: str) -> object:
    key = f"bench{char}samples"
    return pytest.param(
        {key: 1}, [f"Unknown config key: {json.dumps(key)}"], id=f"line-break-{ord(char)}"
    )


#: Why reading a directory as the config file fails: the OS refuses it differently on Windows.
DIRECTORY_READ_REASON = "Permission denied" if sys.platform == "win32" else "Is a directory"


def load_config(config_path: Path) -> ConfigFile:
    """Load a config file that must be accepted and return what it parsed to."""
    result = load_config_file_collecting(config_path, required=False)
    assert result.problems == []
    assert result.config_file is not None
    return result.config_file


def load_error_message(config_path: Path) -> str:
    """Load a config file that must be rejected and return the first problem."""
    result = load_config_file_collecting(config_path, required=False)
    assert result.config_file is None
    return result.problems[0]


# ---------------------------------------------------------------------------
# valid TOML with known keys
# ---------------------------------------------------------------------------


def test_load_config_file_when_only_bench_given_does_return_parsed_bench(tmp_path: Path):
    config_path = write_config(tmp_path, {"bench": "custom-bench"})

    result = load_config(config_path)

    assert result == ConfigFile(bench="custom-bench")


def test_load_config_file_when_all_known_keys_given_does_round_trip(tmp_path: Path):
    config_path = write_config(
        tmp_path,
        {
            "bench": "bench-name",
            "prepare": "prepare-cmd",
            "adapter": "adapter-name",
            "samples": 10,
            "timeout_seconds": 30,
            "unstable_noise_pct": 150.5,
            "metrics": {
                "metric1": {"direction": "lower", "gating": True, "exact": False},
                "metric2": {"direction": "higher"},
            },
        },
    )

    assert load_config(config_path) == ConfigFile(
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

    result = load_config(config_path)

    assert result == ConfigFile(
        metrics={
            "responseTime": MetricEntry(direction="lower"),
            "throughput": MetricEntry(gating=True),
        }
    )


# ---------------------------------------------------------------------------
# invalid TOML / duplicate key / BOM / non-finite literals
# ---------------------------------------------------------------------------


def test_load_config_file_when_toml_invalid_does_raise_naming_path(tmp_path: Path):
    config_path = write_raw(tmp_path, "key = ")

    message = load_error_message(config_path)

    assert message == (
        f"Failed to parse config file at {config_path}: Invalid value (at end of document)"
    )


def test_load_config_file_when_duplicate_key_does_raise_naming_path(tmp_path: Path):
    config_path = write_raw(tmp_path, 'bench = "first"\nbench = "second"')

    message = load_error_message(config_path)

    assert message == (
        f"Failed to parse config file at {config_path}: Cannot overwrite a value (at end of document)"
    )


@pytest.mark.parametrize(
    ("literal", "token"),
    [
        pytest.param("nan", "NaN", id="nan"),
        pytest.param("inf", "Infinity", id="inf"),
        pytest.param("-inf", "-Infinity", id="negative-inf"),
    ],
)
@pytest.mark.parametrize(
    ("prefix", "key"),
    [
        pytest.param("unstable_noise_pct = ", "unstable_noise_pct", id="noise-pct"),
        pytest.param("[stop]\ntarget_value = ", "stop.target_value", id="stop-target-value"),
    ],
)
def test_load_config_file_when_number_key_non_finite_does_reject_as_not_a_number(
    tmp_path: Path, literal: str, token: str, prefix: str, key: str
):
    config_path = write_raw(tmp_path, f"{prefix}{literal}")

    message = load_error_message(config_path)

    assert message == f"Invalid config value for {key}: expected a number, got {token}"


def test_load_config_file_when_prefixed_with_bom_does_parse_as_if_absent(tmp_path: Path):
    config_path = write_raw(
        tmp_path, f"{UTF8_BOM}{tomli_w.dumps({'bench': 'bom-bench', 'samples': 5})}"
    )

    result = load_config(config_path)

    assert result == ConfigFile(bench="bom-bench", samples=5)


# ---------------------------------------------------------------------------
# unknown keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected_problems"),
    [
        pytest.param({"unknownKey": "value"}, ["Unknown config key: unknownKey"], id="only-key"),
        pytest.param(
            {"bench": "name", "badKey": "value"},
            ["Unknown config key: badKey"],
            id="mixed-with-known",
        ),
        pytest.param({"": 1}, ['Unknown config key: ""'], id="empty-string"),
        *[_unknown_line_break_key_param(char) for char in LINE_BREAK_CHARS],
    ],
)
def test_load_config_file_collecting_when_top_level_key_unknown_does_report_key_quoted_as_needed(
    tmp_path: Path, content: dict[str, object], expected_problems: list[str]
):
    config_path = write_config(tmp_path, content)

    result = load_config_file_collecting(config_path, required=False)

    assert result.config_file is None
    assert result.problems == expected_problems


# ---------------------------------------------------------------------------
# empty object
# ---------------------------------------------------------------------------


def test_load_config_file_when_empty_object_does_return_empty_config(tmp_path: Path):
    config_path = write_config(tmp_path, {})

    result = load_config(config_path)

    assert result == ConfigFile()


# ---------------------------------------------------------------------------
# sections and flags of the wrong type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "key_path", "got"),
    [
        pytest.param({"metrics": "latency"}, "metrics", '"latency"', id="metrics-string"),
        pytest.param(
            {"metrics": {"latency": "lower"}}, "metrics.latency", '"lower"', id="metrics-entry"
        ),
        pytest.param({"metrics": {"": 5}}, 'metrics.""', "5", id="metrics-entry-empty-key"),
        pytest.param({"kinds": "memory"}, "kinds", '"memory"', id="kinds-string"),
        pytest.param({"kinds": {"memory": False}}, "kinds.memory", "false", id="kinds-entry"),
        pytest.param({"hooks": "gymrat.hooks"}, "hooks", '"gymrat.hooks"', id="hooks-string"),
        pytest.param(
            {"supervise": "claude-sonnet"}, "supervise", '"claude-sonnet"', id="supervise-string"
        ),
    ],
)
def test_load_config_file_when_section_not_object_does_name_key_path_and_object(
    tmp_path: Path, content: dict[str, object], key_path: str, got: str
):
    config_path = write_config(tmp_path, content)

    message = load_error_message(config_path)

    assert message == f"Invalid config value for {key_path}: expected an object, got {got}"


@pytest.mark.parametrize(
    ("content", "key_path", "got"),
    [
        pytest.param(
            {"metrics": {"latency": {"gating": "yes"}}},
            "metrics.latency.gating",
            '"yes"',
            id="metrics-gating-string",
        ),
        pytest.param(
            {"metrics": {"latency": {"exact": 1}}},
            "metrics.latency.exact",
            "1",
            id="metrics-exact-number",
        ),
        pytest.param(
            {"kinds": {"memory": {"gating": "yes"}}},
            "kinds.memory.gating",
            '"yes"',
            id="kinds-gating-string",
        ),
    ],
)
def test_load_config_file_when_flag_non_boolean_does_name_key_path_and_boolean(
    tmp_path: Path, content: dict[str, object], key_path: str, got: str
):
    config_path = write_config(tmp_path, content)

    message = load_error_message(config_path)

    assert message == f"Invalid config value for {key_path}: expected a boolean, got {got}"


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
        pytest.param("checks", 42, id="checks-number"),
        pytest.param("filter", ["a"], id="filter-array"),
    ],
)
def test_load_config_file_when_string_key_holds_non_string_does_name_key_and_string(
    tmp_path: Path, key: str, value: object
):
    config_path = write_config(tmp_path, {key: value})

    message = load_error_message(config_path)

    assert message == f"Invalid config value for {key}: expected a string, got {json.dumps(value)}"


# ---------------------------------------------------------------------------
# non-empty-string keys holding empty / whitespace-only strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("checks", "", id="checks-empty"),
        pytest.param("bench", "", id="bench-empty"),
        pytest.param("prepare", "", id="prepare-empty"),
        pytest.param("adapter", "", id="adapter-empty"),
        pytest.param("runbook", "", id="runbook-empty"),
        pytest.param("primary", "", id="primary-empty"),
        pytest.param("checks", " ", id="space"),
        pytest.param("bench", "\t", id="tab"),
        pytest.param("adapter", "\u00a0", id="nbsp"),
    ],
)
def test_load_config_file_when_non_empty_string_key_holds_blank_does_name_key_and_non_empty(
    tmp_path: Path, key: str, value: str
):
    config_path = write_config(tmp_path, {key: value})

    message = load_error_message(config_path)

    assert message.startswith(f"Invalid config value for {key}: expected a non-empty string, got ")


# ---------------------------------------------------------------------------
# positive-integer keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value", "phrase"),
    [
        pytest.param("samples", "ten", "an integer", id="samples-string"),
        pytest.param("samples", 1.5, "an integer", id="samples-non-integer"),
        pytest.param("samples", 0, "a number at or above 1", id="samples-zero"),
        pytest.param("timeout_seconds", -1, "a number at or above 1", id="timeout-negative"),
        pytest.param("timeout_seconds", True, "an integer", id="timeout-boolean"),
    ],
)
def test_load_config_file_when_integer_key_invalid_does_name_key_and_expected_shape(
    tmp_path: Path, key: str, value: object, phrase: str
):
    config_path = write_config(tmp_path, {key: value})

    message = load_error_message(config_path)

    assert message.startswith(f"Invalid config value for {key}: expected {phrase}, got ")


def test_load_config_file_when_integer_key_given_integral_float_does_accept(tmp_path: Path):
    config_path = write_raw(tmp_path, "samples = 5.0")

    result = load_config(config_path)

    assert result == ConfigFile(samples=5)


@pytest.mark.parametrize(
    ("key", "cap"),
    [
        pytest.param("timeout_seconds", MAX_TIMEOUT_SECONDS, id="timeout-seconds"),
        pytest.param("samples", MAX_SAFE_INTEGER, id="samples"),
    ],
)
def test_load_config_file_when_integer_key_exceeds_cap_does_name_key_and_cap(
    tmp_path: Path, key: str, cap: int
):
    config_path = write_config(tmp_path, {key: cap + 1})

    message = load_error_message(config_path)

    assert message == (
        f"Invalid config value for {key}: expected a number at or below {cap}, got {cap + 1}"
    )


@pytest.mark.parametrize(
    ("key", "cap", "expected"),
    [
        pytest.param(
            "timeout_seconds",
            MAX_TIMEOUT_SECONDS,
            ConfigFile(timeout_seconds=MAX_TIMEOUT_SECONDS),
            id="timeout-seconds",
        ),
        pytest.param(
            "samples", MAX_SAFE_INTEGER, ConfigFile(samples=MAX_SAFE_INTEGER), id="samples"
        ),
    ],
)
def test_load_config_file_when_integer_key_on_cap_does_accept(
    tmp_path: Path, key: str, cap: int, expected: ConfigFile
):
    config_path = write_config(tmp_path, {key: cap})

    assert load_config(config_path) == expected


# ---------------------------------------------------------------------------
# unstable_noise_pct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "phrase"),
    [
        pytest.param("loud", "a number", id="string"),
        pytest.param(0.25, "a number at or above 0.5", id="below-floor"),
    ],
)
def test_load_config_file_when_noise_pct_invalid_does_name_key_and_expected_shape(
    tmp_path: Path, value: object, phrase: str
):
    config_path = write_config(tmp_path, {"unstable_noise_pct": value})

    message = load_error_message(config_path)

    assert message.startswith(
        f"Invalid config value for unstable_noise_pct: expected {phrase}, got "
    )


def test_load_config_file_when_noise_pct_on_floor_does_accept(tmp_path: Path):
    config_path = write_config(tmp_path, {"unstable_noise_pct": 0.5})

    result = load_config(config_path)

    assert result == ConfigFile(unstable_noise_pct=0.5)


# ---------------------------------------------------------------------------
# number-typed keys: integers accepted, booleans rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "read", "expected"),
    [
        pytest.param(
            "unstable_noise_pct = 3", attrgetter("unstable_noise_pct"), 3.0, id="noise-pct"
        ),
        pytest.param(
            "[stop]\ntarget_value = -2",
            attrgetter("stop.target_value"),
            -2.0,
            id="stop-target-value",
        ),
    ],
)
def test_load_config_file_when_number_key_given_integer_does_accept_as_float(
    tmp_path: Path, text: str, read: Callable[[ConfigFile], object], expected: float
):
    config_path = write_raw(tmp_path, text)

    value = read(load_config(config_path))

    assert (type(value), value) == (float, expected)


def test_load_config_file_when_stop_target_value_boolean_does_reject_with_exact_message(
    tmp_path: Path,
):
    config_path = write_raw(tmp_path, "[stop]\ntarget_value = false")

    message = load_error_message(config_path)

    assert message == "Invalid config value for stop.target_value: expected a number, got false"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("sideways", id="unknown-string"),
        pytest.param("Lower", id="wrong-case"),
        pytest.param(True, id="boolean"),
    ],
)
def test_load_config_file_when_metrics_direction_invalid_does_name_direction_and_options(
    tmp_path: Path, value: object
):
    config_path = write_config(tmp_path, {"metrics": {"latency": {"direction": value}}})

    message = load_error_message(config_path)

    assert message.startswith(
        "Invalid config value for metrics.latency.direction: expected 'lower' or 'higher', got "
    )


def test_load_config_file_when_metrics_entry_has_unknown_key_does_name_key(tmp_path: Path):
    config_path = write_config(
        tmp_path, {"metrics": {"latency": {"direction": "lower", "threshold": "higher"}}}
    )

    assert load_error_message(config_path) == "Unknown config key: metrics.latency.threshold"


@pytest.mark.parametrize("char", LINE_BREAK_CHARS)
def test_load_config_file_when_metrics_key_embeds_line_break_does_report_key_not_shape(
    tmp_path: Path, char: str
):
    smuggled = f"latency{char}direction: 999, gating: 0"
    config_path = write_config(tmp_path, {"metrics": {smuggled: {"direction": "lower"}}})

    message = load_error_message(config_path)

    # The rejected value is an object; only its key is bad, so the message names
    # the key instead of demanding an object.
    assert message == (
        f"Invalid config value for metrics: key {json.dumps(smuggled)} must not embed a line break"
    )


@pytest.mark.parametrize(
    ("metric_name", "expected_path"),
    [
        pytest.param("decode.time", 'metrics."decode.time".direction', id="dot"),
        pytest.param("decode time", 'metrics."decode time".direction', id="space"),
        pytest.param('decode"time', 'metrics."decode\\"time".direction', id="quote"),
    ],
)
def test_load_config_file_when_metric_name_needs_quoting_does_quote_it_in_key_path(
    tmp_path: Path, metric_name: str, expected_path: str
):
    config_path = write_config(tmp_path, {"metrics": {metric_name: {"direction": "sideways"}}})

    message = load_error_message(config_path)

    assert expected_path in message


# ---------------------------------------------------------------------------
# kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("char", LINE_BREAK_CHARS)
def test_load_config_file_when_kinds_key_embeds_line_break_does_name_kinds(
    tmp_path: Path, char: str
):
    config_path = write_config(tmp_path, {"kinds": {f"memory{char}gating: 999": {"gating": False}}})

    assert "kinds" in load_error_message(config_path)


def test_load_config_file_when_kinds_section_given_does_round_trip(tmp_path: Path):
    config_path = write_config(tmp_path, {"kinds": {"memory": {"gating": False}, "time": {}}})

    result = load_config(config_path)

    assert result == ConfigFile(kinds={"memory": KindEntry(gating=False), "time": KindEntry()})


def test_load_config_file_when_kinds_entry_has_unknown_key_does_name_dotted_path(tmp_path: Path):
    config_path = write_config(tmp_path, {"kinds": {"memory": {"gating": False, "threshold": 5}}})

    assert "Unknown config key: kinds.memory.threshold" in load_error_message(config_path)


# ---------------------------------------------------------------------------
# runbook and loop keys
# ---------------------------------------------------------------------------


def test_load_config_file_when_runbook_given_does_round_trip(tmp_path: Path):
    config_path = write_config(tmp_path, {"runbook": "RUNBOOK.md"})

    result = load_config(config_path)

    assert result == ConfigFile(runbook="RUNBOOK.md")


def test_load_config_file_when_loop_keys_given_does_round_trip(tmp_path: Path):
    config_path = write_config(tmp_path, LOOP_CONFIG)

    result = load_config(config_path)

    assert result == ConfigFile(
        checks="npm test",
        filter="npm run bench -- {names}",
        primary="decode/time",
        stop=StopConfig(target_value=1.5, max_iterations=20),
        hooks=HooksConfig(before="npm run warm-cache", after="npm run cool-down"),
    )


# ---------------------------------------------------------------------------
# explicit None and empty values
# ---------------------------------------------------------------------------


def test_validate_config_dict_when_optional_keys_explicitly_none_does_accept():
    config: dict[str, object] = {field.name: None for field in fields(ConfigFile)}

    result = validate_config_dict(config)

    assert result is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        pytest.param(
            {"samples": "bad", "timeout_seconds": "bad", "filter": "npm run bench"},
            'Invalid config value for samples: expected an integer, got "bad"',
            id="schema",
        ),
        pytest.param(
            {"filter": "npm run bench", "primary": "geomean", "stop": {"target_value": 1.5}},
            (
                "Invalid config value for filter: expected a string containing the {names} "
                'placeholder, got "npm run bench"'
            ),
            id="loop-keys",
        ),
    ],
)
def test_validate_config_dict_when_several_problems_does_raise_the_first(
    config: dict[str, object], message: str
):
    with pytest.raises(GymratError) as exc_info:
        validate_config_dict(config)

    assert exc_info.value.args[0] == message


def test_load_config_file_when_filter_empty_does_keep_empty_filter(tmp_path: Path):
    config_path = write_config(tmp_path, {"filter": ""})

    result = load_config(config_path)

    assert result == ConfigFile(filter="")


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

    assert load_config(config_path) == ConfigFile(hooks=expected)


@pytest.mark.parametrize(
    ("stage", "value", "phrase"),
    [
        pytest.param("before", "", "a non-empty string", id="before-empty"),
        pytest.param("after", "", "a non-empty string", id="after-empty"),
        pytest.param("before", 42, "a string", id="before-number"),
        pytest.param("before", " ", "a non-empty string", id="before-space"),
        pytest.param("after", "\t\n ", "a non-empty string", id="after-mixed-whitespace"),
    ],
)
def test_load_config_file_when_hooks_command_not_non_empty_string_does_name_stage(
    tmp_path: Path, stage: str, value: object, phrase: str
):
    config_path = write_config(tmp_path, {"hooks": {stage: value}})

    message = load_error_message(config_path)

    assert message == (
        f"Invalid config value for hooks.{stage}: expected {phrase}, got {json.dumps(value)}"
    )


def test_load_config_file_when_hooks_has_unknown_key_does_name_dotted_path(tmp_path: Path):
    config_path = write_config(
        tmp_path, {"hooks": {"before": "npm run warm-cache", "during": "npm run mid"}}
    )

    assert "Unknown config key: hooks.during" in load_error_message(config_path)


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        pytest.param(
            "target_value",
            "fast",
            'Invalid config value for stop.target_value: expected a number, got "fast"',
            id="target-string",
        ),
        pytest.param(
            "max_iterations",
            0,
            "Invalid config value for stop.max_iterations: expected a number at or above 1, got 0",
            id="max-zero",
        ),
        pytest.param(
            "max_iterations",
            1.5,
            "Invalid config value for stop.max_iterations: expected an integer, got 1.5",
            id="max-non-integer",
        ),
    ],
)
def test_load_config_file_when_stop_field_invalid_does_name_field_and_expected_shape(
    tmp_path: Path, field: str, value: object, expected: str
):
    config_path = write_config(tmp_path, {"stop": {field: value}})

    message = load_error_message(config_path)

    assert message == expected


def test_load_config_file_when_stop_has_unknown_key_does_name_dotted_path(tmp_path: Path):
    config_path = write_config(tmp_path, {"stop": {"target_value": 1, "patience": 3}})

    assert "Unknown config key: stop.patience" in load_error_message(config_path)


# ---------------------------------------------------------------------------
# supervise table — absent / present / rejection
# ---------------------------------------------------------------------------


def test_load_config_file_when_no_supervise_table_does_return_config_without_supervise(
    tmp_path: Path,
):
    config_path = write_config(tmp_path, {"bench": "my-bench"})

    result = load_config(config_path)

    assert result.supervise is None


def test_load_config_file_when_supervise_has_unknown_key_does_name_dotted_path(tmp_path: Path):
    config_path = write_config(
        tmp_path, {"supervise": {"model": "claude-sonnet", "temperature": 0.7}}
    )

    assert "Unknown config key: supervise.temperature" in load_error_message(config_path)


def test_load_config_file_when_supervise_model_blank_does_name_model_and_non_empty(
    tmp_path: Path,
):
    config_path = write_config(tmp_path, {"supervise": {"model": ""}})

    message = load_error_message(config_path)

    assert message == (
        'Invalid config value for supervise.model: expected a non-empty string, got ""'
    )


@pytest.mark.parametrize(
    "effort",
    [
        pytest.param("low", id="low"),
        pytest.param("medium", id="medium"),
        pytest.param("high", id="high"),
        pytest.param("xhigh", id="xhigh"),
        pytest.param("max", id="max"),
    ],
)
def test_load_config_file_when_supervise_effort_valid_does_accept(tmp_path: Path, effort: str):
    config_path = write_config(tmp_path, {"supervise": {"effort": effort}})

    result = load_config(config_path)

    assert result.supervise is not None
    assert result.supervise.effort == effort


@pytest.mark.parametrize(
    "effort",
    [
        pytest.param("turbo", id="unknown"),
        pytest.param("High", id="wrong-case"),
        pytest.param("", id="empty"),
    ],
)
def test_load_config_file_when_supervise_effort_invalid_does_name_effort_and_all_five_levels(
    tmp_path: Path, effort: str
):
    config_path = write_config(tmp_path, {"supervise": {"effort": effort}})

    message = load_error_message(config_path)

    assert message == (
        "Invalid config value for supervise.effort: "
        f"expected 'low', 'medium', 'high', 'xhigh' or 'max', got \"{effort}\""
    )


# ---------------------------------------------------------------------------
# camelCase keys rejected as unknown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "camel_key"),
    [
        pytest.param({"timeoutSeconds": 30}, "timeoutSeconds", id="timeout-seconds"),
        pytest.param({"unstableNoisePct": 150.5}, "unstableNoisePct", id="unstable-noise-pct"),
        pytest.param({"stop": {"targetValue": 1.5}}, "stop.targetValue", id="stop-target-value"),
        pytest.param(
            {"stop": {"maxIterations": 20}}, "stop.maxIterations", id="stop-max-iterations"
        ),
    ],
)
def test_load_config_file_when_camel_case_key_given_does_reject_as_unknown(
    tmp_path: Path, content: dict[str, object], camel_key: str
):
    config_path = write_config(tmp_path, content)

    assert f"Unknown config key: {camel_key}" in load_error_message(config_path)


# ---------------------------------------------------------------------------
# load_config_file_collecting
# ---------------------------------------------------------------------------


def test_load_config_file_collecting_when_file_missing_does_report_absent(tmp_path: Path):
    missing = tmp_path / "nonexistent.toml"

    result = load_config_file_collecting(missing, required=False)

    assert result == ConfigFileResult(config_file=ConfigFile(), exists=False, problems=[])


def test_load_config_file_collecting_when_file_missing_and_required_does_report_problem(
    tmp_path: Path,
):
    missing = tmp_path / "nonexistent.toml"

    result = load_config_file_collecting(missing, required=True)

    assert result == ConfigFileResult(
        config_file=None, exists=False, problems=[f"Config file not found at {missing}"]
    )


def test_load_config_file_collecting_when_valid_does_report_config_and_no_problems(tmp_path: Path):
    config_path = write_config(tmp_path, {"bench": "custom-bench"})

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(
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

    assert result == ConfigFileResult(
        config_file=None,
        exists=True,
        problems=[f"Cannot read config file at {tmp_path}: {DIRECTORY_READ_REASON}"],
    )


def test_load_config_file_collecting_when_value_is_toml_date_does_report_problem_not_crash(
    tmp_path: Path,
):
    config_path = write_raw(tmp_path, "samples = 1979-05-27")

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(
        config_file=None,
        exists=True,
        problems=[
            "Invalid config value for samples: expected an integer, got datetime.date(1979, 5, 27)"
        ],
    )


def test_load_config_file_collecting_when_file_is_utf16_does_report_read_failure(
    tmp_path: Path,
):
    config_path = tmp_path / "gymrat.toml"
    config_path.write_bytes('bench = "hello"'.encode("utf-16"))

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(
        config_file=None,
        exists=True,
        problems=[
            (
                f"Cannot read config file at {config_path}: "
                "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"
            )
        ],
    )


def test_load_config_file_collecting_when_toml_invalid_does_report_parse_problem(tmp_path: Path):
    config_path = write_raw(tmp_path, "key = ")

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(
        config_file=None,
        exists=True,
        problems=[
            f"Failed to parse config file at {config_path}: Invalid value (at end of document)"
        ],
    )
