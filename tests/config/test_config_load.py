import json
import math
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
    SuperviseConfig,
    load_config_file_collecting,
    validate_config_dict,
)
from gymrat.errors import GymratError
from tests.config._toml import (
    DEEP_NESTING_DOCUMENT,
    DIGIT_LIMIT_DOCUMENT,
    HUGE_HEX_BITS,
    HUGE_HEX_LITERAL,
    LOOP_CONFIG,
    LOOP_FIELDS,
    write_config,
    write_raw,
)

# Byte-order mark that editors on Windows prepend to UTF-8 files: EF BB BF.
UTF8_BOM = "\N{BYTE ORDER MARK}"

# Two characters `str.splitlines` breaks on: an ASCII one and a non-ASCII one, so
# a key holding either is rejected and its message escapes it. Which characters
# count as line breaks is pinned in tests/test_metric_name.py.
LINE_BREAK_CHARS = ["\n", "\u2028"]


#: Why reading a directory as the config file fails: the OS refuses it differently on Windows.
DIRECTORY_READ_REASON = "Permission denied" if sys.platform == "win32" else "Is a directory"


# ---------------------------------------------------------------------------
# valid TOML with known keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        pytest.param(
            tomli_w.dumps({
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
            }),
            ConfigFile(
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
            ),
            id="flag-backed-keys-and-metrics",
        ),
        pytest.param(
            tomli_w.dumps({
                "metrics": {"responseTime": {"direction": "lower"}, "throughput": {"gating": True}}
            }),
            ConfigFile(
                metrics={
                    "responseTime": MetricEntry(direction="lower"),
                    "throughput": MetricEntry(gating=True),
                }
            ),
            id="partial-metrics-metadata",
        ),
        pytest.param(
            f"{UTF8_BOM}{tomli_w.dumps({'bench': 'bom-bench', 'samples': 5})}",
            ConfigFile(bench="bom-bench", samples=5),
            id="byte-order-mark",
        ),
        pytest.param("", ConfigFile(), id="empty"),
        pytest.param("samples = 5.0", ConfigFile(samples=5), id="integral-float-integer"),
        pytest.param(
            f"timeout_seconds = {MAX_TIMEOUT_SECONDS}",
            ConfigFile(timeout_seconds=MAX_TIMEOUT_SECONDS),
            id="timeout-seconds-on-cap",
        ),
        pytest.param(
            f"samples = {MAX_SAFE_INTEGER}",
            ConfigFile(samples=MAX_SAFE_INTEGER),
            id="samples-on-cap",
        ),
        pytest.param(
            "unstable_noise_pct = 0.5", ConfigFile(unstable_noise_pct=0.5), id="noise-pct-on-floor"
        ),
        pytest.param(
            tomli_w.dumps({"kinds": {"memory": {"gating": False}, "time": {}}}),
            ConfigFile(kinds={"memory": KindEntry(gating=False), "time": KindEntry()}),
            id="kinds",
        ),
        pytest.param('runbook = "RUNBOOK.md"', ConfigFile(runbook="RUNBOOK.md"), id="runbook"),
        pytest.param(
            tomli_w.dumps(LOOP_CONFIG),
            ConfigFile(**LOOP_FIELDS),
            id="loop-keys",
        ),
        pytest.param('filter = ""', ConfigFile(filter=""), id="empty-filter"),
        pytest.param(
            '[hooks]\nbefore = "npm run warm-cache"',
            ConfigFile(hooks=HooksConfig(before="npm run warm-cache")),
            id="hooks-only-before",
        ),
        pytest.param(
            '[hooks]\nafter = "npm run cool-down"',
            ConfigFile(hooks=HooksConfig(after="npm run cool-down")),
            id="hooks-only-after",
        ),
        *(
            pytest.param(
                f'[supervise]\neffort = "{effort}"',
                ConfigFile(supervise=SuperviseConfig(effort=effort)),
                id=f"supervise-effort-{effort}",
            )
            for effort in ("low", "medium", "high", "xhigh", "max")
        ),
    ],
)
def test_load_config_file_collecting_when_document_valid_does_parse_it(
    tmp_path: Path, document: str, expected: ConfigFile
):
    config_path = write_raw(tmp_path, document)

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(config_file=expected, exists=True, problems=[])


# ---------------------------------------------------------------------------
# invalid TOML / non-finite literals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        pytest.param(
            'bench = "first"\nbench = "second"',
            "Cannot overwrite a value (at end of document)",
            id="duplicate-key",
        ),
        pytest.param("key = ", "Invalid value (at end of document)", id="missing-value"),
    ],
)
def test_load_config_file_collecting_when_toml_invalid_does_report_parse_problem(
    tmp_path: Path, document: str, reason: str
):
    config_path = write_raw(tmp_path, document)

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(
        config_file=None,
        exists=True,
        problems=[f"Failed to parse config file at {config_path}: {reason}"],
    )


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        pytest.param(DIGIT_LIMIT_DOCUMENT, "Exceeds the limit", id="integer-past-digit-limit"),
        pytest.param(DEEP_NESTING_DOCUMENT, "", id="nesting-past-recursion-limit"),
    ],
)
def test_load_config_file_collecting_when_parser_hits_interpreter_limit_does_report_parse_problem(
    tmp_path: Path, document: str, reason: str
):
    config_path = write_raw(tmp_path, document)

    result = load_config_file_collecting(config_path, required=False)

    assert (result.config_file, result.exists, len(result.problems)) == (None, True, 1)
    assert result.problems[0].startswith(f"Failed to parse config file at {config_path}: {reason}")


@pytest.mark.parametrize(
    ("document", "phrase", "got"),
    [
        pytest.param(
            f"samples = {HUGE_HEX_LITERAL}",
            f"a number at or below {MAX_SAFE_INTEGER}",
            f"a {HUGE_HEX_BITS}-bit integer",
            id="integer-key",
        ),
        pytest.param(
            f"samples = [{HUGE_HEX_LITERAL}]",
            "an integer",
            "a list too large to display",
            id="array-holding-the-integer",
        ),
    ],
)
def test_load_config_file_collecting_when_integer_too_long_to_print_does_report_problem_naming_key(
    tmp_path: Path, document: str, phrase: str, got: str
):
    config_path = write_raw(tmp_path, document)

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(
        config_file=None,
        exists=True,
        problems=[f"Invalid config value for samples: expected {phrase}, got {got}"],
    )


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
        *(
            pytest.param(
                {f"bench{char}samples": 1},
                [f"Unknown config key: {json.dumps(f'bench{char}samples')}"],
                id=f"line-break-{ord(char)}",
            )
            for char in LINE_BREAK_CHARS
        ),
        pytest.param(
            {"metrics": {"latency": {"direction": "lower", "threshold": "higher"}}},
            ["Unknown config key: metrics.latency.threshold"],
            id="metrics-entry",
        ),
        pytest.param(
            {"kinds": {"memory": {"gating": False, "threshold": 5}}},
            ["Unknown config key: kinds.memory.threshold"],
            id="kinds-entry",
        ),
        pytest.param(
            {"hooks": {"before": "npm run warm-cache", "during": "npm run mid"}},
            ["Unknown config key: hooks.during"],
            id="hooks",
        ),
        pytest.param(
            {"stop": {"target_value": 1, "patience": 3}},
            ["Unknown config key: stop.patience"],
            id="stop",
        ),
        pytest.param(
            {"supervise": {"model": "claude-sonnet", "temperature": 0.7}},
            ["Unknown config key: supervise.temperature"],
            id="supervise",
        ),
    ],
)
def test_load_config_file_collecting_when_key_unknown_does_report_its_path_quoted_as_needed(
    tmp_path: Path, content: dict[str, object], expected_problems: list[str]
):
    config_path = write_config(tmp_path, content)

    result = load_config_file_collecting(config_path, required=False)

    assert result.config_file is None
    assert result.problems == expected_problems


# ---------------------------------------------------------------------------
# values of the wrong type or out of range
# ---------------------------------------------------------------------------


def _wrong_type(content: dict[str, object], key_path: str, expected: str, got: str) -> object:
    return pytest.param(
        content,
        f"Invalid config value for {key_path}: expected {expected}, got {got}",
        id=f"{key_path}-{got}",
    )


_EFFORT_LEVELS = "'low', 'medium', 'high', 'xhigh' or 'max'"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        # sections that must be tables
        _wrong_type({"metrics": "latency"}, "metrics", "an object", '"latency"'),
        _wrong_type({"metrics": {"latency": "lower"}}, "metrics.latency", "an object", '"lower"'),
        _wrong_type({"metrics": {"": 5}}, 'metrics.""', "an object", "5"),
        _wrong_type({"kinds": "memory"}, "kinds", "an object", '"memory"'),
        _wrong_type({"kinds": {"memory": False}}, "kinds.memory", "an object", "false"),
        _wrong_type({"hooks": "gymrat.hooks"}, "hooks", "an object", '"gymrat.hooks"'),
        _wrong_type({"supervise": "claude-sonnet"}, "supervise", "an object", '"claude-sonnet"'),
        # flags that must be booleans
        _wrong_type(
            {"metrics": {"latency": {"gating": "yes"}}},
            "metrics.latency.gating",
            "a boolean",
            '"yes"',
        ),
        _wrong_type(
            {"metrics": {"latency": {"exact": 1}}}, "metrics.latency.exact", "a boolean", "1"
        ),
        _wrong_type(
            {"kinds": {"memory": {"gating": "yes"}}}, "kinds.memory.gating", "a boolean", '"yes"'
        ),
        # keys that must be strings
        _wrong_type({"bench": 42}, "bench", "a string", "42"),
        _wrong_type({"prepare": True}, "prepare", "a string", "true"),
        _wrong_type({"checks": 42}, "checks", "a string", "42"),
        _wrong_type({"filter": ["a"]}, "filter", "a string", '["a"]'),
        _wrong_type({"hooks": {"before": 42}}, "hooks.before", "a string", "42"),
        # keys that must be non-empty strings
        _wrong_type({"hooks": {"before": ""}}, "hooks.before", "a non-empty string", '""'),
        _wrong_type({"hooks": {"after": ""}}, "hooks.after", "a non-empty string", '""'),
        _wrong_type({"supervise": {"model": ""}}, "supervise.model", "a non-empty string", '""'),
        _wrong_type({"checks": ""}, "checks", "a non-empty string", '""'),
        _wrong_type({"bench": ""}, "bench", "a non-empty string", '""'),
        _wrong_type({"prepare": ""}, "prepare", "a non-empty string", '""'),
        _wrong_type({"adapter": ""}, "adapter", "a non-empty string", '""'),
        _wrong_type({"runbook": ""}, "runbook", "a non-empty string", '""'),
        _wrong_type({"primary": ""}, "primary", "a non-empty string", '""'),
        _wrong_type({"bench": " "}, "bench", "a non-empty string", '" "'),
        _wrong_type({"bench": "\t\n "}, "bench", "a non-empty string", '"\\t\\n "'),
        _wrong_type({"bench": "\u00a0"}, "bench", "a non-empty string", '"\\u00a0"'),
        # positive-integer keys
        _wrong_type({"samples": "ten"}, "samples", "an integer", '"ten"'),
        _wrong_type({"samples": 1.5}, "samples", "an integer", "1.5"),
        _wrong_type({"samples": 0}, "samples", "a number at or above 1", "0"),
        _wrong_type({"timeout_seconds": -1}, "timeout_seconds", "a number at or above 1", "-1"),
        _wrong_type({"timeout_seconds": True}, "timeout_seconds", "an integer", "true"),
        # unstable_noise_pct
        _wrong_type({"unstable_noise_pct": "loud"}, "unstable_noise_pct", "a number", '"loud"'),
        *(
            _wrong_type({"unstable_noise_pct": value}, "unstable_noise_pct", "a number", got)
            for value, got in (
                (math.nan, "NaN"),
                (False, "false"),
            )
        ),
        _wrong_type(
            {"unstable_noise_pct": 0.25}, "unstable_noise_pct", "a number at or above 0.5", "0.25"
        ),
        # metric direction
        *(
            _wrong_type(
                {"metrics": {"latency": {"direction": value}}},
                "metrics.latency.direction",
                "'lower' or 'higher'",
                got,
            )
            for value, got in (("sideways", '"sideways"'), ("Lower", '"Lower"'), (True, "true"))
        ),
        # integers past their cap
        _wrong_type(
            {"timeout_seconds": MAX_TIMEOUT_SECONDS + 1},
            "timeout_seconds",
            f"a number at or below {MAX_TIMEOUT_SECONDS}",
            str(MAX_TIMEOUT_SECONDS + 1),
        ),
        _wrong_type(
            {"samples": MAX_SAFE_INTEGER + 1},
            "samples",
            f"a number at or below {MAX_SAFE_INTEGER}",
            str(MAX_SAFE_INTEGER + 1),
        ),
        # stop fields
        _wrong_type({"stop": {"target_value": "fast"}}, "stop.target_value", "a number", '"fast"'),
        _wrong_type({"stop": {"target_value": math.nan}}, "stop.target_value", "a number", "NaN"),
        _wrong_type(
            {"stop": {"max_iterations": 0}}, "stop.max_iterations", "a number at or above 1", "0"
        ),
        _wrong_type({"stop": {"max_iterations": 1.5}}, "stop.max_iterations", "an integer", "1.5"),
        # supervise effort
        *(
            _wrong_type(
                {"supervise": {"effort": effort}}, "supervise.effort", _EFFORT_LEVELS, f'"{effort}"'
            )
            for effort in ("turbo", "High", "")
        ),
    ],
)
def test_load_config_file_collecting_when_value_invalid_does_name_key_path_and_expected_shape(
    tmp_path: Path, content: dict[str, object], message: str
):
    config_path = write_config(tmp_path, content)

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(config_file=None, exists=True, problems=[message])


def test_load_config_file_collecting_when_multiple_fields_invalid_does_report_every_problem(
    tmp_path: Path,
):
    config_path = write_config(tmp_path, {"bench": 42, "samples": 0})

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(
        config_file=None,
        exists=True,
        problems=[
            "Invalid config value for bench: expected a string, got 42",
            "Invalid config value for samples: expected a number at or above 1, got 0",
        ],
    )


# ---------------------------------------------------------------------------
# number-typed keys: integers accepted as floats
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
def test_load_config_file_collecting_when_number_key_given_integer_does_accept_as_float(
    tmp_path: Path, text: str, read: Callable[[ConfigFile], object], expected: float
):
    config_path = write_raw(tmp_path, text)

    result = load_config_file_collecting(config_path, required=False)

    assert result.problems == []
    assert result.config_file is not None
    value = read(result.config_file)
    assert (type(value), value) == (float, expected)


# ---------------------------------------------------------------------------
# section keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "char"),
    [
        *(pytest.param("metrics", char, id=f"metrics-{ord(char)}") for char in LINE_BREAK_CHARS),
        pytest.param("kinds", "\n", id="kinds-10"),
    ],
)
def test_load_config_file_collecting_when_section_key_embeds_line_break_does_report_key_not_shape(
    tmp_path: Path, section: str, char: str
):
    smuggled = f"latency{char}direction: 999, gating: 0"
    config_path = write_config(tmp_path, {section: {smuggled: {"gating": False}}})

    result = load_config_file_collecting(config_path, required=False)

    # The rejected value is an object; only its key is bad, so the message names
    # the key instead of demanding an object.
    assert result == ConfigFileResult(
        config_file=None,
        exists=True,
        problems=[
            (
                f"Invalid config value for {section}: "
                f"key {json.dumps(smuggled)} must not embed a line break"
            )
        ],
    )


@pytest.mark.parametrize(
    ("metric_name", "expected_path"),
    [
        pytest.param("decode.time", 'metrics."decode.time".direction', id="dot"),
        pytest.param("decode time", 'metrics."decode time".direction', id="space"),
        pytest.param('decode"time', 'metrics."decode\\"time".direction', id="quote"),
    ],
)
def test_load_config_file_collecting_when_metric_name_needs_quoting_does_quote_it_in_key_path(
    tmp_path: Path, metric_name: str, expected_path: str
):
    config_path = write_config(tmp_path, {"metrics": {metric_name: {"direction": "sideways"}}})

    result = load_config_file_collecting(config_path, required=False)

    assert result == ConfigFileResult(
        config_file=None,
        exists=True,
        problems=[
            (
                f"Invalid config value for {expected_path}: expected 'lower' or 'higher', "
                'got "sideways"'
            )
        ],
    )


# ---------------------------------------------------------------------------
# validate_config_dict
# ---------------------------------------------------------------------------


def test_validate_config_dict_when_optional_keys_explicitly_none_does_accept():
    config: dict[str, object] = {field.name: None for field in fields(ConfigFile)}

    validate_config_dict(config)


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


# ---------------------------------------------------------------------------
# missing, unreadable and undecodable files
# ---------------------------------------------------------------------------


def test_load_config_file_collecting_when_optional_file_missing_does_return_empty_config(
    tmp_path: Path,
):
    missing = tmp_path / "nonexistent.toml"

    result = load_config_file_collecting(missing, required=False)

    assert result == ConfigFileResult(config_file=ConfigFile(), exists=False, problems=[])


def test_load_config_file_collecting_when_path_is_directory_does_collect_read_failure(
    tmp_path: Path,
):
    result = load_config_file_collecting(tmp_path, required=False)

    assert result == ConfigFileResult(
        config_file=None,
        exists=True,
        problems=[f"Cannot read config file at {tmp_path}: {DIRECTORY_READ_REASON}"],
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
