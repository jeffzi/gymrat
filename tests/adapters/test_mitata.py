import json
from pathlib import Path
from typing import Any

import pytest

from gymrat.adapters import AdapterError, MetricDefaults, mitata_adapter
from gymrat.model import MetricUnit
from tests.adapters._inputs import LINE_BREAKS, build_stdout

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mitata.json"

# ---------------------------------------------------------------------------
# adapter shape
# ---------------------------------------------------------------------------


def test_mitata_adapter_when_inspected_does_expose_name():
    assert mitata_adapter.name == "mitata"


# ---------------------------------------------------------------------------
# basic JSON parsing
# ---------------------------------------------------------------------------

_BASIC_FIXTURE = build_stdout([
    {"alias": "encode", "runs": [{"name": "encode", "args": {}, "stats": {"p50": 42}}]}
])


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param(_BASIC_FIXTURE, id="no-preamble-or-trailer"),
        pytest.param(f"some preamble\nmore output\n{_BASIC_FIXTURE}", id="preamble"),
        pytest.param(f"{_BASIC_FIXTURE}\ntrailing output\nmore output", id="trailer"),
        pytest.param(f"preamble\n{_BASIC_FIXTURE}\ntrailer", id="preamble-and-trailer"),
    ],
)
def test_parse_when_json_surrounded_by_text_does_extract_metrics(stdout: str):
    assert mitata_adapter.parse(stdout) == {"encode#time": 42}


# ---------------------------------------------------------------------------
# metric naming for parameterized benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "args", "p50", "metric_name"),
    [
        pytest.param(
            "decode/$text", {"text": "digits"}, 42, "decode/text=digits#time", id="single"
        ),
        pytest.param("op/$a/$b", {"a": "x", "b": "y"}, 50, "op/a=x/b=y#time", id="multiple"),
        pytest.param("test/$x/sep/$x", {"x": "1"}, 99, "test/x=1/sep/x=1#time", id="repeated"),
        pytest.param(
            "test/$x/$unknown", {"x": "1"}, 77, "test/x=1/$unknown#time", id="stray-dollar"
        ),
        pytest.param("$ab", {"a": "x", "ab": "y"}, 8, "ab=y#time", id="longest-key-first"),
    ],
)
def test_parse_when_alias_has_placeholders_does_substitute_arg_values(
    alias: str, args: dict[str, Any], p50: int, metric_name: str
):
    stdout = build_stdout([
        {"alias": alias, "runs": [{"name": alias, "args": args, "stats": {"p50": p50}}]}
    ])

    assert mitata_adapter.parse(stdout) == {metric_name: p50}


def test_parse_when_alias_has_placeholders_and_args_empty_does_keep_placeholders_literal():
    stdout = build_stdout([
        {"alias": "test/$x", "runs": [{"name": "test", "args": {}, "stats": {"p50": 42}}]}
    ])

    assert mitata_adapter.parse(stdout) == {"test/$x#time": 42}


@pytest.mark.parametrize(
    ("value", "serialized"),
    [
        pytest.param("digits", "digits", id="string"),
        pytest.param(5, "5", id="int"),
        pytest.param(5.0, "5", id="integral-float"),
        pytest.param(1.5, "1.5", id="float"),
        pytest.param(True, "true", id="bool-true"),
        pytest.param(False, "false", id="bool-false"),
        pytest.param(None, "null", id="none"),
    ],
)
def test_parse_when_arg_is_primitive_does_serialize_js_style(value: Any, serialized: str):
    stdout = build_stdout([
        {"alias": "b/$v", "runs": [{"name": "b", "args": {"v": value}, "stats": {"p50": 1}}]}
    ])

    assert mitata_adapter.parse(stdout) == {f"b/v={serialized}#time": 1}


# ---------------------------------------------------------------------------
# alias substitution with hostile argument values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("a$&b", id="whole-match-reference"),
        pytest.param("a$`b", id="prefix-reference"),
        pytest.param("a$'b", id="suffix-reference"),
        pytest.param("a$$b", id="escaped-dollar"),
    ],
)
def test_parse_when_arg_value_holds_regex_replacement_syntax_does_keep_it_literal(value: str):
    stdout = build_stdout([
        {
            "alias": "decode/$text",
            "runs": [{"name": "d", "args": {"text": value}, "stats": {"p50": 42}}],
        }
    ])

    assert mitata_adapter.parse(stdout) == {f"decode/text={value}#time": 42}


def test_parse_when_arg_value_introduces_a_placeholder_does_not_re_substitute_it():
    stdout = build_stdout([
        {
            "alias": "op/$a/$b",
            "runs": [{"name": "op", "args": {"a": "$b", "b": "y"}, "stats": {"p50": 7}}],
        }
    ])

    assert mitata_adapter.parse(stdout) == {"op/a=$b/b=y#time": 7}


# ---------------------------------------------------------------------------
# metric names carrying a line terminator
# ---------------------------------------------------------------------------

_ESCAPED_LINE_TERMINATORS = [
    pytest.param(line_break.char, line_break.escaped, id=line_break.name)
    for line_break in LINE_BREAKS
]


@pytest.mark.parametrize(("terminator", "escaped"), _ESCAPED_LINE_TERMINATORS)
def test_parse_when_alias_holds_line_terminator_does_warn_on_one_line_and_skip(
    terminator: str, escaped: str
):
    offending = f"enc{terminator}ode"
    stdout = build_stdout([
        {
            "alias": offending,
            "runs": [{"name": "e", "args": {}, "stats": {"p50": 42, "heap": {"avg": "bad"}}}],
        },
        {"alias": "valid", "runs": [{"name": "v", "args": {}, "stats": {"p50": 1}}]},
    ])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid#time": 1}
    assert warnings == [
        (
            f'Skipping run with a line terminator in its metric name: "enc{escaped}ode" '
            "(the alias or one of its argument values carries one)"
        )
    ]


@pytest.mark.parametrize(
    "terminator", [pytest.param(line_break.char, id=line_break.name) for line_break in LINE_BREAKS]
)
def test_parse_when_arg_value_holds_line_terminator_does_warn_and_skip(terminator: str):
    stdout = build_stdout([
        {
            "alias": "decode/$text",
            "runs": [
                {"name": "d1", "args": {"text": f"di{terminator}gits"}, "stats": {"p50": 10}},
                {"name": "d2", "args": {"text": "words"}, "stats": {"p50": 20}},
            ],
        }
    ])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"decode/text=words#time": 20}
    assert warnings == [
        (
            'Skipping run with a line terminator in its metric name: "decode/$text" '
            "(the alias or one of its argument values carries one)"
        )
    ]


_LINE_TERMINATOR_ALIASES = [
    pytest.param(f"enc{line_break.char}ode", id=line_break.name) for line_break in LINE_BREAKS
]


@pytest.mark.parametrize("alias", _LINE_TERMINATOR_ALIASES)
@pytest.mark.parametrize(
    ("fields", "warning_template"),
    [
        pytest.param(
            {"runs": [{"args": {}, "error": "boom"}]},
            "Skipping run with an error: {alias} (boom)",
            id="errored-run",
        ),
        pytest.param(
            {"runs": [{"args": "bad", "stats": {"p50": 1}}]},
            'Skipping run of {alias} with invalid args: expected an object, got "bad"',
            id="invalid-run",
        ),
        pytest.param(
            {"runs": [{"args": {}, "stats": {"p50": float("nan")}}]},
            "Skipping run of {alias} with invalid stats.p50: expected a finite number, got NaN",
            id="non-finite-p50",
        ),
        pytest.param(
            {"runs": [None]},
            "Skipping run of {alias}: expected an object, got null",
            id="run-not-object",
        ),
        pytest.param({}, "Skipping benchmark {alias} with missing runs", id="runs-missing"),
        pytest.param(
            {"runs": 5},
            "Skipping benchmark {alias} with invalid runs: expected an array, got 5",
            id="runs-not-array",
        ),
    ],
)
def test_parse_when_skip_warning_names_alias_holding_line_terminator_does_escape_it(
    alias: str, fields: dict[str, Any], warning_template: str
):
    stdout = build_stdout([
        {"alias": alias, **fields},
        {"alias": "valid", "runs": [{"args": {}, "stats": {"p50": 1}}]},
    ])
    warnings: list[str] = []

    mitata_adapter.parse(stdout, warnings.append)

    assert warnings == [warning_template.format(alias=json.dumps(alias))]


@pytest.mark.parametrize(
    ("alias", "args"),
    [
        *(
            pytest.param(f"en{line_break.char}c#ode", {}, id=f"alias-{line_break.name}")
            for line_break in LINE_BREAKS
        ),
        *(
            pytest.param(
                "enc#$text", {"text": f"a{line_break.char}b"}, id=f"arg-value-{line_break.name}"
            )
            for line_break in LINE_BREAKS
        ),
    ],
)
def test_parse_when_reserved_hash_name_holds_line_terminator_does_raise_on_one_line(
    alias: str, args: dict[str, str]
):
    stdout = build_stdout([{"alias": alias, "runs": [{"args": args, "stats": {"p50": 42}}]}])

    with pytest.raises(AdapterError) as exc_info:
        mitata_adapter.parse(stdout)

    message = str(exc_info.value)
    assert json.dumps(alias) in message
    assert message.splitlines() == [message]


# ---------------------------------------------------------------------------
# alias containing '#' is a hard error
# ---------------------------------------------------------------------------


def test_parse_when_alias_contains_hash_does_raise_adapter_error():
    stdout = build_stdout([
        {"alias": "enc#ode", "runs": [{"name": "e", "args": {}, "stats": {"p50": 42}}]}
    ])

    with pytest.raises(AdapterError, match="enc#ode"):
        mitata_adapter.parse(stdout)


def test_parse_when_substituted_arg_introduces_hash_does_raise_adapter_error():
    stdout = build_stdout([
        {
            "alias": "op/$v",
            "runs": [{"name": "o", "args": {"v": "a#b"}, "stats": {"p50": 42}}],
        }
    ])

    with pytest.raises(AdapterError):
        mitata_adapter.parse(stdout)


# ---------------------------------------------------------------------------
# malformed run and benchmark shapes
# ---------------------------------------------------------------------------


_INVALID_RUN_PREFIX = 'Skipping run of "test" with invalid'


@pytest.mark.parametrize(
    ("bad_run", "warning"),
    [
        pytest.param(
            {"args": "not-a-record", "stats": {"p50": 1}},
            f'{_INVALID_RUN_PREFIX} args: expected an object, got "not-a-record"',
            id="args-not-object",
        ),
        pytest.param(
            {"args": None, "stats": {"p50": 1}},
            f"{_INVALID_RUN_PREFIX} args: expected an object, got null",
            id="args-null",
        ),
        pytest.param(
            {"args": {}, "stats": "not-a-record"},
            f'{_INVALID_RUN_PREFIX} stats: expected an object, got "not-a-record"',
            id="stats-not-object",
        ),
        pytest.param({"args": {}}, 'Skipping run of "test" with missing stats', id="stats-missing"),
        pytest.param(
            {"args": {}, "stats": {}},
            'Skipping run of "test" with missing stats.p50',
            id="p50-missing",
        ),
        pytest.param(
            {"args": {}, "stats": {"p50": True}},
            f"{_INVALID_RUN_PREFIX} stats.p50: expected a number, got true",
            id="p50-bool",
        ),
        pytest.param(
            {"args": {}, "stats": {"p50": "fast"}},
            f'{_INVALID_RUN_PREFIX} stats.p50: expected a number, got "fast"',
            id="p50-string",
        ),
        pytest.param(
            {"args": {}, "stats": {"p50": None}},
            f"{_INVALID_RUN_PREFIX} stats.p50: expected a number, got null",
            id="p50-null",
        ),
        pytest.param(
            {"args": {}, "stats": {"p50": float("inf")}},
            f"{_INVALID_RUN_PREFIX} stats.p50: expected a finite number, got Infinity",
            id="p50-positive-infinity",
        ),
        pytest.param(
            {"args": {}, "stats": {"p50": float("-inf")}},
            f"{_INVALID_RUN_PREFIX} stats.p50: expected a finite number, got -Infinity",
            id="p50-negative-infinity",
        ),
        pytest.param(
            {"args": {}, "stats": {"p50": float("nan")}},
            f"{_INVALID_RUN_PREFIX} stats.p50: expected a finite number, got NaN",
            id="p50-nan",
        ),
        pytest.param(
            {"args": {}, "stats": {"p50": 10**400}},
            f"{_INVALID_RUN_PREFIX} stats.p50: expected a number, got {10**400}",
            id="p50-integer-overflows-float",
        ),
        pytest.param(
            {"args": 1, "stats": 2},
            f"{_INVALID_RUN_PREFIX} args: expected an object, got 1",
            id="args-and-stats-invalid",
        ),
        pytest.param(None, 'Skipping run of "test": expected an object, got null', id="run-null"),
        pytest.param(42, 'Skipping run of "test": expected an object, got 42', id="run-number"),
    ],
)
def test_parse_when_run_is_malformed_does_warn_once_and_keep_other_runs(
    bad_run: object, warning: str
):
    stdout = build_stdout([{"alias": "test", "runs": [bad_run, {"args": {}, "stats": {"p50": 5}}]}])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 5}
    assert warnings == [warning]


def test_parse_when_run_args_missing_does_treat_as_empty_and_not_warn():
    stdout = build_stdout([{"alias": "test", "runs": [{"stats": {"p50": 5}}]}])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 5}
    assert warnings == []


_INVALID_ALIAS_WARNING = "Skipping benchmark with invalid alias: expected a string, got 42"


@pytest.mark.parametrize(
    ("bad_benchmark", "warning"),
    [
        pytest.param(
            {"alias": 42, "runs": [{"args": {}, "stats": {"p50": 1}}]},
            _INVALID_ALIAS_WARNING,
            id="alias-not-string",
        ),
        pytest.param(
            {"runs": [{"args": {}, "stats": {"p50": 1}}]},
            "Skipping benchmark with missing alias",
            id="alias-missing",
        ),
        pytest.param(
            {"alias": "orphan"},
            'Skipping benchmark "orphan" with missing runs',
            id="runs-missing",
        ),
        pytest.param(
            {"alias": "orphan", "runs": 5},
            'Skipping benchmark "orphan" with invalid runs: expected an array, got 5',
            id="runs-not-array",
        ),
        pytest.param(
            {"alias": 42, "runs": 5},
            _INVALID_ALIAS_WARNING,
            id="alias-and-runs-invalid",
        ),
        pytest.param(None, "Skipping benchmark: expected an object, got null", id="benchmark-null"),
        pytest.param(
            "string",
            'Skipping benchmark: expected an object, got "string"',
            id="benchmark-string",
        ),
    ],
)
def test_parse_when_benchmark_is_malformed_does_warn_once_and_keep_other_benchmarks(
    bad_benchmark: object, warning: str
):
    stdout = build_stdout([
        bad_benchmark,
        {"alias": "valid", "runs": [{"args": {}, "stats": {"p50": 1}}]},
    ])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid#time": 1}
    assert warnings == [warning]


# ---------------------------------------------------------------------------
# metric-name collisions
# ---------------------------------------------------------------------------

_ALIAS_MISSING_PLACEHOLDER = build_stdout([
    {
        "alias": "decode",
        "runs": [
            {"name": "decode/digits", "args": {"text": "digits"}, "stats": {"p50": 10}},
            {"name": "decode/words", "args": {"text": "words"}, "stats": {"p50": 20}},
        ],
    }
])


def test_parse_when_metric_names_collide_does_warn_and_keep_last(
    capsys: pytest.CaptureFixture[str],
):
    result = mitata_adapter.parse(_ALIAS_MISSING_PLACEHOLDER)

    assert result == {"decode#time": 20}
    assert "Duplicate metric name: decode#time" in capsys.readouterr().err


def test_parse_when_collision_and_sink_given_does_route_warning_off_stderr(
    capsys: pytest.CaptureFixture[str],
):
    warnings: list[str] = []

    mitata_adapter.parse(_ALIAS_MISSING_PLACEHOLDER, warnings.append)

    assert any("Duplicate metric name: decode#time" in w for w in warnings)
    assert capsys.readouterr().err == ""


def test_parse_when_two_benchmarks_share_alias_does_warn_collision(
    capsys: pytest.CaptureFixture[str],
):
    stdout = build_stdout([
        {"alias": "encode", "runs": [{"name": "encode", "args": {}, "stats": {"p50": 1}}]},
        {"alias": "encode", "runs": [{"name": "encode", "args": {}, "stats": {"p50": 2}}]},
    ])

    mitata_adapter.parse(stdout)

    assert "Duplicate metric name: encode#time" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# p50 value extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p50",
    [pytest.param(123.456, id="decimal"), pytest.param(0.0791015625, id="high-precision")],
)
def test_parse_when_p50_present_does_use_it_as_time_metric(p50: float):
    stdout = build_stdout([
        {"alias": "test", "runs": [{"name": "test", "args": {}, "stats": {"p50": p50}}]}
    ])

    assert mitata_adapter.parse(stdout) == {"test#time": p50}


# ---------------------------------------------------------------------------
# heap metric emission
# ---------------------------------------------------------------------------


def test_parse_when_heap_avg_present_does_emit_heap_metric_keeping_integers():
    stdout = build_stdout([
        {
            "alias": "test",
            "runs": [{"name": "test", "args": {}, "stats": {"p50": 42, "heap": {"avg": 1024}}}],
        }
    ])

    result = mitata_adapter.parse(stdout)

    assert result == {"test#time": 42, "test#heap": 1024}
    assert [type(v) for v in result.values()] == [int, int]


def test_parse_when_heap_avg_present_on_parameterized_bench_does_emit_named_heap_metric():
    stdout = build_stdout([
        {
            "alias": "decode/$text",
            "runs": [
                {
                    "name": "d",
                    "args": {"text": "digits"},
                    "stats": {"p50": 10, "heap": {"avg": 256}},
                }
            ],
        }
    ])

    assert mitata_adapter.parse(stdout) == {
        "decode/text=digits#time": 10,
        "decode/text=digits#heap": 256,
    }


def test_parse_when_heap_avg_missing_does_skip_heap_metric():
    stdout = build_stdout([
        {
            "alias": "test",
            "runs": [{"name": "test", "args": {}, "stats": {"p50": 42, "heap": {"total": 1024}}}],
        }
    ])

    assert mitata_adapter.parse(stdout) == {"test#time": 42}


_INVALID_HEAP_PREFIX = 'Skipping heap metric of "test" with invalid'


@pytest.mark.parametrize(
    ("heap_value", "warning"),
    [
        pytest.param(
            42, f"{_INVALID_HEAP_PREFIX} stats.heap: expected an object, got 42", id="integer"
        ),
        pytest.param(
            "bad", f'{_INVALID_HEAP_PREFIX} stats.heap: expected an object, got "bad"', id="string"
        ),
        pytest.param(
            [1, 2], f"{_INVALID_HEAP_PREFIX} stats.heap: expected an object, got [1, 2]", id="array"
        ),
        pytest.param(
            True, f"{_INVALID_HEAP_PREFIX} stats.heap: expected an object, got true", id="boolean"
        ),
        pytest.param(
            {"avg": "bad"},
            f'{_INVALID_HEAP_PREFIX} stats.heap.avg: expected a number, got "bad"',
            id="avg-string",
        ),
        pytest.param(
            {"avg": True},
            f"{_INVALID_HEAP_PREFIX} stats.heap.avg: expected a number, got true",
            id="avg-boolean",
        ),
        pytest.param(
            {"avg": float("inf")},
            f"{_INVALID_HEAP_PREFIX} stats.heap.avg: expected a finite number, got Infinity",
            id="avg-infinity",
        ),
        pytest.param(
            {"avg": float("nan")},
            f"{_INVALID_HEAP_PREFIX} stats.heap.avg: expected a finite number, got NaN",
            id="avg-nan",
        ),
    ],
)
def test_parse_when_heap_is_malformed_does_warn_once_and_keep_time_metric(
    heap_value: object, warning: str
):
    stdout = build_stdout([
        {
            "alias": "test",
            "runs": [{"name": "test", "args": {}, "stats": {"p50": 42, "heap": heap_value}}],
        }
    ])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 42}
    assert warnings == [warning]


def test_parse_when_heap_absent_does_not_warn():
    stdout = build_stdout([
        {"alias": "test", "runs": [{"name": "test", "args": {}, "stats": {"p50": 42}}]}
    ])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 42}
    assert warnings == []


# ---------------------------------------------------------------------------
# non-finite statistics
# ---------------------------------------------------------------------------


def test_parse_when_every_p50_non_finite_does_raise_no_valid_runs():
    stdout = (
        '{"benchmarks":[{"alias":"test","runs":[{"name":"t","args":{},"stats":{"p50":1e999}}]}]}'
    )

    with pytest.raises(AdapterError, match=r"^No valid benchmark runs found$"):
        mitata_adapter.parse(stdout)


# ---------------------------------------------------------------------------
# multiple runs and benchmarks
# ---------------------------------------------------------------------------


def test_parse_when_benchmark_has_multiple_runs_does_emit_metric_per_run():
    stdout = build_stdout([
        {
            "alias": "decode/$text",
            "runs": [
                {"name": "d", "args": {"text": "digits"}, "stats": {"p50": 10}},
                {"name": "w", "args": {"text": "words"}, "stats": {"p50": 20}},
            ],
        }
    ])

    assert mitata_adapter.parse(stdout) == {
        "decode/text=digits#time": 10,
        "decode/text=words#time": 20,
    }


def test_parse_when_runs_have_heap_does_emit_heap_metric_per_run():
    stdout = build_stdout([
        {
            "alias": "decode/$text",
            "runs": [
                {
                    "name": "d",
                    "args": {"text": "digits"},
                    "stats": {"p50": 10, "heap": {"avg": 256}},
                },
                {
                    "name": "w",
                    "args": {"text": "words"},
                    "stats": {"p50": 20, "heap": {"avg": 512}},
                },
            ],
        }
    ])

    assert mitata_adapter.parse(stdout) == {
        "decode/text=digits#time": 10,
        "decode/text=digits#heap": 256,
        "decode/text=words#time": 20,
        "decode/text=words#heap": 512,
    }


def test_parse_when_multiple_benchmarks_does_emit_metrics_for_all():
    stdout = build_stdout([
        {"alias": "encode", "runs": [{"name": "encode", "args": {}, "stats": {"p50": 42}}]},
        {"alias": "decode", "runs": [{"name": "decode", "args": {}, "stats": {"p50": 100}}]},
    ])

    assert mitata_adapter.parse(stdout) == {"encode#time": 42, "decode#time": 100}


# ---------------------------------------------------------------------------
# name-derived defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_name",
    ["test#time", "encode#time", "decode/x=1#time", "complex/a=1/b=2#time"],
)
def test_defaults_when_time_metric_does_describe_as_lower_ns_time(metric_name: str):
    short_name = metric_name.removesuffix("#time")

    assert mitata_adapter.defaults(metric_name) == MetricDefaults(
        direction="lower", unit="ns", kind="time", short_name=short_name
    )


@pytest.mark.parametrize(
    "metric_name",
    ["test#heap", "encode#heap", "decode/x=1#heap", "complex/a=1/b=2#heap"],
)
def test_defaults_when_heap_metric_does_describe_as_lower_bytes_memory(metric_name: str):
    short_name = metric_name.removesuffix("#heap")

    assert mitata_adapter.defaults(metric_name) == MetricDefaults(
        direction="lower", unit="bytes", kind="memory", short_name=short_name
    )


@pytest.mark.parametrize("metric_name", ["custom_metric", "test", "test/throughput", "test/ops"])
def test_defaults_when_metric_unrecognized_does_return_direction_only(metric_name: str):
    assert mitata_adapter.defaults(metric_name) == MetricDefaults(direction="lower")


@pytest.mark.parametrize(
    ("metric_name", "unit", "kind"),
    [
        pytest.param("#time", "ns", "time", id="time"),
        pytest.param("#heap", "bytes", "memory", id="heap"),
    ],
)
def test_defaults_when_prefix_empty_does_fall_back_to_full_metric_name(
    metric_name: str, unit: MetricUnit, kind: str
):
    assert mitata_adapter.defaults(metric_name) == MetricDefaults(
        direction="lower", unit=unit, kind=kind, short_name=metric_name
    )


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_parse_when_no_json_object_found_does_raise():
    with pytest.raises(AdapterError, match=r"^No JSON object found in stdout$"):
        mitata_adapter.parse("not valid json at all")


def test_parse_when_json_between_braces_malformed_does_raise():
    with pytest.raises(AdapterError, match=r"^Failed to parse JSON: "):
        mitata_adapter.parse("preamble { invalid json } trailer")


def test_parse_when_benchmarks_array_missing_does_raise():
    with pytest.raises(AdapterError, match=r"^JSON missing benchmarks array$"):
        mitata_adapter.parse(json.dumps({"something": "else"}))


def test_parse_when_benchmarks_array_empty_does_raise():
    with pytest.raises(AdapterError, match=r"^benchmarks array is empty$"):
        mitata_adapter.parse(json.dumps({"benchmarks": []}))


def test_parse_when_no_run_has_valid_stats_does_raise():
    stdout = build_stdout([{"alias": "test", "runs": [{"name": "test", "args": {}, "stats": {}}]}])

    with pytest.raises(AdapterError, match=r"^No valid benchmark runs found$"):
        mitata_adapter.parse(stdout)


# ---------------------------------------------------------------------------
# error field handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "runs", "metrics", "warning"),
    [
        pytest.param(
            "test/$x",
            [
                {
                    "name": "a",
                    "args": {"x": "a"},
                    "error": "something went wrong",
                    "stats": {"p50": 10},
                },
                {"name": "b", "args": {"x": "b"}, "stats": {"p50": 20}},
            ],
            {"test/x=b#time": 20},
            'Skipping run with an error: "test/$x" (something went wrong)',
            id="error-string",
        ),
        pytest.param(
            "test",
            [{"args": {}, "error": "boom"}, {"args": {}, "stats": {"p50": 20}}],
            {"test#time": 20},
            'Skipping run with an error: "test" (boom)',
            id="error-without-stats",
        ),
        pytest.param(
            "test",
            [
                {"name": "a", "args": {}, "error": {"code": 7}, "stats": {"p50": 10}},
                {"name": "b", "args": {}, "stats": {"p50": 20}},
            ],
            {"test#time": 20},
            'Skipping run with an error: "test" ({"code": 7})',
            id="error-object",
        ),
    ],
)
def test_parse_when_run_has_error_field_does_warn_once_and_keep_other_runs(
    alias: str, runs: list[dict[str, Any]], metrics: dict[str, float], warning: str
):
    stdout = build_stdout([{"alias": alias, "runs": runs}])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == metrics
    assert warnings == [warning]


@pytest.mark.parametrize(("terminator", "escaped"), _ESCAPED_LINE_TERMINATORS)
def test_parse_when_error_string_holds_line_terminator_does_escape_only_the_terminator(
    terminator: str, escaped: str
):
    stdout = build_stdout([
        {
            "alias": "test",
            "runs": [
                {"args": {}, "error": f'naïve "x"{terminator}boom'},
                {"args": {}, "stats": {"p50": 20}},
            ],
        }
    ])
    warnings: list[str] = []

    mitata_adapter.parse(stdout, warnings.append)

    assert warnings == [f'Skipping run with an error: "test" (naïve "x"{escaped}boom)']


def test_parse_when_all_runs_have_errors_does_raise():
    stdout = build_stdout([
        {
            "alias": "test",
            "runs": [
                {"name": "test", "args": {}, "error": "something failed", "stats": {"p50": 10}}
            ],
        }
    ])

    with pytest.raises(AdapterError, match=r"^No valid benchmark runs found$"):
        mitata_adapter.parse(stdout)


def test_parse_when_error_field_is_null_does_process_run_normally():
    stdout = build_stdout([
        {
            "alias": "test",
            "runs": [{"name": "test", "args": {}, "error": None, "stats": {"p50": 10}}],
        }
    ])

    assert mitata_adapter.parse(stdout) == {"test#time": 10}


# ---------------------------------------------------------------------------
# non-primitive run-argument serialization
# ---------------------------------------------------------------------------


def test_parse_when_arg_value_is_object_does_serialize_via_json():
    stdout = build_stdout([
        {
            "alias": "bench/$opts",
            "runs": [{"name": "cfg", "args": {"opts": {"size": 100}}, "stats": {"p50": 5}}],
        }
    ])

    assert mitata_adapter.parse(stdout) == {'bench/opts={"size":100}#time': 5}


def test_parse_when_object_arg_values_differ_does_keep_distinct_names():
    stdout = build_stdout([
        {
            "alias": "bench/$opts",
            "runs": [
                {"name": "a", "args": {"opts": {"size": 100}}, "stats": {"p50": 5}},
                {"name": "b", "args": {"opts": {"size": 200}}, "stats": {"p50": 10}},
            ],
        }
    ])

    assert mitata_adapter.parse(stdout) == {
        'bench/opts={"size":100}#time': 5,
        'bench/opts={"size":200}#time': 10,
    }


def test_parse_when_object_arg_has_unsorted_keys_does_serialize_in_sorted_order():
    stdout = build_stdout([
        {
            "alias": "bench/$opts",
            "runs": [{"name": "cfg", "args": {"opts": {"z": 1, "a": 2}}, "stats": {"p50": 5}}],
        }
    ])

    assert mitata_adapter.parse(stdout) == {'bench/opts={"a":2,"z":1}#time': 5}


def test_parse_when_arg_value_is_array_does_serialize_via_json():
    stdout = build_stdout([
        {
            "alias": "bench/$items",
            "runs": [{"name": "list", "args": {"items": [1, 2, 3]}, "stats": {"p50": 7}}],
        }
    ])

    assert mitata_adapter.parse(stdout) == {"bench/items=[1,2,3]#time": 7}


# ---------------------------------------------------------------------------
# skip warnings and the warning sink
# ---------------------------------------------------------------------------


def test_parse_when_skip_warning_and_sink_given_does_route_off_stderr(
    capsys: pytest.CaptureFixture[str],
):
    stdout = (
        '{"benchmarks":[{"alias":"test","runs":['
        '{"name":"test","args":{},"stats":{"p50":1e999}},'
        '{"name":"test2","args":{},"stats":{"p50":20}}]}]}'
    )
    warnings: list[str] = []

    mitata_adapter.parse(stdout, warnings.append)

    assert len(warnings) == 1
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# real fixture
# ---------------------------------------------------------------------------


def test_parse_when_given_real_fixture_does_extract_all_metrics():
    fixture = json.loads(_FIXTURE_PATH.read_text())

    result = mitata_adapter.parse(json.dumps(fixture))

    assert result == {
        "decode/text=digits#time": 4.0791015625,
        "decode/text=digits#heap": 0.13420623129857714,
        "decode/text=words#time": 7.8125,
        "decode/text=words#heap": 0.14746411878141288,
        "encode#time": 42.66357421875,
        "encode#heap": 80.1967411655276,
    }
