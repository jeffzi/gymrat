import json
from pathlib import Path
from typing import Any

import pytest

from gymrat.adapters.mitata import find_json_candidates, mitata_adapter
from gymrat.adapters.types import Adapter, AdapterError, MetricDefaults
from gymrat.model.metrics import MetricUnit
from tests.adapters._inputs import build_stdout

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mitata.json"

# ---------------------------------------------------------------------------
# adapter shape
# ---------------------------------------------------------------------------


def test_mitata_adapter_when_inspected_does_expose_name():
    assert mitata_adapter.name == "mitata"


def test_mitata_adapter_when_checked_does_satisfy_adapter_protocol():
    assert isinstance(mitata_adapter, Adapter)


# ---------------------------------------------------------------------------
# basic JSON parsing
# ---------------------------------------------------------------------------

_BASIC_FIXTURE = build_stdout(
    [{"alias": "encode", "runs": [{"name": "encode", "args": {}, "stats": {"p50": 42}}]}]
)


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
    stdout = build_stdout(
        [{"alias": alias, "runs": [{"name": alias, "args": args, "stats": {"p50": p50}}]}]
    )

    assert mitata_adapter.parse(stdout) == {metric_name: p50}


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
    stdout = build_stdout(
        [{"alias": "b/$v", "runs": [{"name": "b", "args": {"v": value}, "stats": {"p50": 1}}]}]
    )

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
    stdout = build_stdout(
        [
            {
                "alias": "decode/$text",
                "runs": [{"name": "d", "args": {"text": value}, "stats": {"p50": 42}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {f"decode/text={value}#time": 42}


def test_parse_when_arg_value_introduces_a_placeholder_does_not_re_substitute_it():
    stdout = build_stdout(
        [
            {
                "alias": "op/$a/$b",
                "runs": [{"name": "op", "args": {"a": "$b", "b": "y"}, "stats": {"p50": 7}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {"op/a=$b/b=y#time": 7}


# ---------------------------------------------------------------------------
# metric names carrying a line terminator
# ---------------------------------------------------------------------------

_LINE_TERMINATORS = [
    pytest.param(0x0A, id="line-feed"),
    pytest.param(0x0D, id="carriage-return"),
    pytest.param(0x2028, id="line-separator-u2028"),
    pytest.param(0x2029, id="paragraph-separator-u2029"),
]


@pytest.mark.parametrize("code_point", _LINE_TERMINATORS)
def test_parse_when_alias_holds_line_terminator_does_warn_and_skip(code_point: int):
    offending = f"enc{chr(code_point)}ode"
    stdout = build_stdout(
        [
            {"alias": offending, "runs": [{"name": "e", "args": {}, "stats": {"p50": 42}}]},
            {"alias": "valid", "runs": [{"name": "v", "args": {}, "stats": {"p50": 1}}]},
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid#time": 1}
    assert any(
        f"Skipping run with a line terminator in its metric name: {offending}" in w
        for w in warnings
    )


@pytest.mark.parametrize("code_point", _LINE_TERMINATORS)
def test_parse_when_arg_value_holds_line_terminator_does_warn_and_skip(code_point: int):
    stdout = build_stdout(
        [
            {
                "alias": "decode/$text",
                "runs": [
                    {
                        "name": "d1",
                        "args": {"text": f"di{chr(code_point)}gits"},
                        "stats": {"p50": 10},
                    },
                    {"name": "d2", "args": {"text": "words"}, "stats": {"p50": 20}},
                ],
            }
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"decode/text=words#time": 20}
    assert any(
        "Skipping run with a line terminator in its metric name: decode/$text" in w
        for w in warnings
    )


# ---------------------------------------------------------------------------
# alias containing '#' is a hard error
# ---------------------------------------------------------------------------


def test_parse_when_alias_contains_hash_does_raise_adapter_error():
    stdout = build_stdout(
        [{"alias": "enc#ode", "runs": [{"name": "e", "args": {}, "stats": {"p50": 42}}]}]
    )

    with pytest.raises(AdapterError, match="enc#ode"):
        mitata_adapter.parse(stdout)


def test_parse_when_substituted_arg_introduces_hash_does_raise_adapter_error():
    stdout = build_stdout(
        [
            {
                "alias": "op/$v",
                "runs": [{"name": "o", "args": {"v": "a#b"}, "stats": {"p50": 42}}],
            }
        ]
    )

    with pytest.raises(AdapterError):
        mitata_adapter.parse(stdout)


# ---------------------------------------------------------------------------
# malformed run and benchmark shapes
# ---------------------------------------------------------------------------


def test_parse_when_run_args_is_not_a_record_does_warn_and_skip():
    stdout = build_stdout(
        [
            {
                "alias": "test",
                "runs": [
                    {"args": "not-a-record", "stats": {"p50": 1}},
                    {"args": {}, "stats": {"p50": 5}},
                ],
            }
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 5}
    assert any("Skipping run with malformed args shape: test" in w for w in warnings)


def test_parse_when_run_stats_is_not_a_record_does_warn_and_skip():
    stdout = build_stdout(
        [
            {
                "alias": "test",
                "runs": [{"args": {}, "stats": "not-a-record"}, {"args": {}, "stats": {"p50": 5}}],
            }
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 5}
    assert any("Skipping run with malformed stats shape: test" in w for w in warnings)


def test_parse_when_run_args_missing_does_treat_as_empty_and_not_warn():
    stdout = build_stdout([{"alias": "test", "runs": [{"stats": {"p50": 5}}]}])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 5}
    assert warnings == []


def test_parse_when_benchmark_alias_is_not_a_string_does_warn_and_skip():
    stdout = build_stdout(
        [
            {"alias": 42, "runs": [{"args": {}, "stats": {"p50": 1}}]},
            {"alias": "valid", "runs": [{"args": {}, "stats": {"p50": 1}}]},
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid#time": 1}
    assert any("Skipping benchmark with malformed alias: 42" in w for w in warnings)


def test_parse_when_benchmark_runs_is_not_an_array_does_warn_and_skip():
    stdout = build_stdout(
        [{"alias": "orphan"}, {"alias": "valid", "runs": [{"args": {}, "stats": {"p50": 1}}]}]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid#time": 1}
    assert any("Skipping benchmark with malformed runs shape: orphan" in w for w in warnings)


def test_parse_when_run_has_error_does_warn_and_skip():
    stdout = build_stdout(
        [
            {
                "alias": "test",
                "runs": [
                    {"args": {}, "error": "boom", "stats": {"p50": 10}},
                    {"args": {}, "stats": {"p50": 20}},
                ],
            }
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 20}
    assert any("Skipping run with an error: test" in w and "boom" in w for w in warnings)


# ---------------------------------------------------------------------------
# metric-name collisions
# ---------------------------------------------------------------------------

_ALIAS_MISSING_PLACEHOLDER = build_stdout(
    [
        {
            "alias": "decode",
            "runs": [
                {"name": "decode/digits", "args": {"text": "digits"}, "stats": {"p50": 10}},
                {"name": "decode/words", "args": {"text": "words"}, "stats": {"p50": 20}},
            ],
        }
    ]
)


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
    stdout = build_stdout(
        [
            {"alias": "encode", "runs": [{"name": "encode", "args": {}, "stats": {"p50": 1}}]},
            {"alias": "encode", "runs": [{"name": "encode", "args": {}, "stats": {"p50": 2}}]},
        ]
    )

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
    stdout = build_stdout(
        [{"alias": "test", "runs": [{"name": "test", "args": {}, "stats": {"p50": p50}}]}]
    )

    assert mitata_adapter.parse(stdout) == {"test#time": p50}


def test_parse_when_p50_is_bool_does_warn_and_skip():
    stdout = (
        '{"benchmarks":[{"alias":"test","runs":['
        '{"name":"a","args":{},"stats":{"p50":true}},'
        '{"name":"b","args":{},"stats":{"p50":5}}]}]}'
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 5}
    assert any("stats.p50 is not a number" in w for w in warnings)


# ---------------------------------------------------------------------------
# heap metric emission
# ---------------------------------------------------------------------------


def test_parse_when_heap_avg_present_does_emit_heap_metric():
    stdout = build_stdout(
        [
            {
                "alias": "test",
                "runs": [{"name": "test", "args": {}, "stats": {"p50": 42, "heap": {"avg": 1024}}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {"test#time": 42, "test#heap": 1024}


def test_parse_when_heap_avg_present_on_parameterized_bench_does_emit_named_heap_metric():
    stdout = build_stdout(
        [
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
        ]
    )

    assert mitata_adapter.parse(stdout) == {
        "decode/text=digits#time": 10,
        "decode/text=digits#heap": 256,
    }


def test_parse_when_heap_avg_missing_does_skip_heap_metric():
    stdout = build_stdout(
        [
            {
                "alias": "test",
                "runs": [
                    {"name": "test", "args": {}, "stats": {"p50": 42, "heap": {"total": 1024}}}
                ],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {"test#time": 42}


@pytest.mark.parametrize(
    "heap_value",
    [
        pytest.param(42, id="integer"),
        pytest.param("bad", id="string"),
        pytest.param([1, 2], id="array"),
        pytest.param(True, id="boolean"),
    ],
)
def test_parse_when_heap_is_not_an_object_does_warn_and_skip_heap(heap_value: object):
    stdout = build_stdout(
        [
            {
                "alias": "test",
                "runs": [{"name": "test", "args": {}, "stats": {"p50": 42, "heap": heap_value}}],
            }
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 42}
    assert len(warnings) == 1
    assert "stats.heap is not an object" in warnings[0]


def test_parse_when_heap_absent_does_not_warn():
    stdout = build_stdout(
        [{"alias": "test", "runs": [{"name": "test", "args": {}, "stats": {"p50": 42}}]}]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 42}
    assert warnings == []


# ---------------------------------------------------------------------------
# non-finite statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "literal",
    [pytest.param("1e999", id="positive-infinity"), pytest.param("-1e999", id="negative-infinity")],
)
def test_parse_when_p50_is_non_finite_does_warn_and_skip(literal: str):
    stdout = (
        '{"benchmarks":[{"alias":"test/$x","runs":['
        f'{{"name":"a","args":{{"x":"a"}},"stats":{{"p50":{literal}}}}},'
        '{"name":"b","args":{"x":"b"},"stats":{"p50":20}}]}]}'
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test/x=b#time": 20}
    assert any("test/$x" in w and "non-finite" in w.lower() for w in warnings)


def test_parse_when_every_p50_non_finite_does_raise_no_valid_runs():
    stdout = (
        '{"benchmarks":[{"alias":"test","runs":[{"name":"t","args":{},"stats":{"p50":1e999}}]}]}'
    )

    with pytest.raises(AdapterError, match=r"^No valid benchmark runs found$"):
        mitata_adapter.parse(stdout)


def test_parse_when_heap_avg_non_finite_does_warn_and_skip_heap():
    stdout = '{"benchmarks":[{"alias":"test","runs":[{"name":"t","args":{},"stats":{"p50":42,"heap":{"avg":1e999}}}]}]}'
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 42}
    assert any("stats.heap.avg is not a finite number" in w for w in warnings)


# ---------------------------------------------------------------------------
# multiple runs and benchmarks
# ---------------------------------------------------------------------------


def test_parse_when_benchmark_has_multiple_runs_does_emit_metric_per_run():
    stdout = build_stdout(
        [
            {
                "alias": "decode/$text",
                "runs": [
                    {"name": "d", "args": {"text": "digits"}, "stats": {"p50": 10}},
                    {"name": "w", "args": {"text": "words"}, "stats": {"p50": 20}},
                ],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {
        "decode/text=digits#time": 10,
        "decode/text=words#time": 20,
    }


def test_parse_when_runs_have_heap_does_emit_heap_metric_per_run():
    stdout = build_stdout(
        [
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
        ]
    )

    assert mitata_adapter.parse(stdout) == {
        "decode/text=digits#time": 10,
        "decode/text=digits#heap": 256,
        "decode/text=words#time": 20,
        "decode/text=words#heap": 512,
    }


def test_parse_when_multiple_benchmarks_does_emit_metrics_for_all():
    stdout = build_stdout(
        [
            {"alias": "encode", "runs": [{"name": "encode", "args": {}, "stats": {"p50": 42}}]},
            {"alias": "decode", "runs": [{"name": "decode", "args": {}, "stats": {"p50": 100}}]},
        ]
    )

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
# find_json_candidates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param('{"key": "value"}', ['{"key": "value"}'], id="single-object"),
        pytest.param('{"a": 1} some text {"b": 2}', ['{"a": 1}', '{"b": 2}'], id="two-top-level"),
        pytest.param(
            '{"key": "value with {braces} inside"}',
            ['{"key": "value with {braces} inside"}'],
            id="braces-in-string",
        ),
        pytest.param(
            r'{"key": "value with \"escaped\" quotes and {braces}"}',
            [r'{"key": "value with \"escaped\" quotes and {braces}"}'],
            id="escaped-quotes",
        ),
        pytest.param(
            '{"outer": {"inner": {"deep": 1}}}',
            ['{"outer": {"inner": {"deep": 1}}}'],
            id="nested-braces",
        ),
        pytest.param("no braces here at all", [], id="no-braces"),
        pytest.param("prefix { incomplete", [], id="unbalanced"),
        pytest.param(
            'weight: 5" tall\n{"benchmarks": []}', ['{"benchmarks": []}'], id="stray-quote-outside"
        ),
        pytest.param(
            'cpu: {model}\n{"benchmarks": []}\nfooter: {info}',
            ['{"benchmarks": []}'],
            id="non-json-braces-around-payload",
        ),
    ],
)
def test_find_json_candidates_when_scanning_does_return_valid_json_objects(
    text: str, expected: list[str]
):
    assert find_json_candidates(text) == expected


# ---------------------------------------------------------------------------
# truncated JSON diagnostic
# ---------------------------------------------------------------------------


def test_parse_when_truncated_json_has_nested_object_does_report_decode_failure():
    # Truncated outer JSON — raw_decode at position 0 fails. The inner
    # {"alias":"encode"} is valid JSON but carries no "benchmarks" key.
    # The adapter should prefer the decode failure (explaining WHY the real
    # payload could not parse) over the generic "JSON missing benchmarks array".
    truncated = '{"benchmarks":[{"alias":"encode"}],"extra":'

    with pytest.raises(AdapterError, match=r"^Failed to parse JSON:"):
        mitata_adapter.parse(truncated)


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


def test_parse_when_benchmark_entries_not_objects_does_warn_and_skip():
    stdout = build_stdout(
        [
            None,
            42,
            "string",
            {"alias": "valid", "runs": [{"name": "valid", "args": {}, "stats": {"p50": 1}}]},
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid#time": 1}
    assert len(warnings) == 3


def test_parse_when_benchmarks_have_non_string_alias_or_missing_runs_does_warn_and_skip():
    stdout = build_stdout(
        [
            {"alias": 42, "runs": []},
            {"alias": "orphan"},
            {"runs": [{"args": {}, "stats": {"p50": 1}}]},
            {"alias": "valid", "runs": [{"name": "valid", "args": {}, "stats": {"p50": 1}}]},
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid#time": 1}
    assert len(warnings) == 3
    assert "alias is not a string" in warnings[0]
    assert "runs is not an array" in warnings[1]
    assert "alias is not a string" in warnings[2]


def test_parse_when_runs_are_not_objects_does_warn_and_skip():
    stdout = build_stdout(
        [{"alias": "test", "runs": [None, 42, {"args": {}, "stats": {"p50": 1}}]}]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 1}
    assert len(warnings) == 2
    assert all("test" in w for w in warnings)


# ---------------------------------------------------------------------------
# error field handling
# ---------------------------------------------------------------------------


def test_parse_when_run_has_error_field_does_warn_and_skip_that_run():
    stdout = build_stdout(
        [
            {
                "alias": "test/$x",
                "runs": [
                    {
                        "name": "a",
                        "args": {"x": "a"},
                        "error": "something went wrong",
                        "stats": {"p50": 10},
                    },
                    {"name": "b", "args": {"x": "b"}, "stats": {"p50": 20}},
                ],
            }
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test/x=b#time": 20}
    assert len(warnings) == 1
    assert "Skipping run with an error" in warnings[0]
    assert "something went wrong" in warnings[0]


def test_parse_when_all_runs_have_errors_does_raise():
    stdout = build_stdout(
        [
            {
                "alias": "test",
                "runs": [
                    {"name": "test", "args": {}, "error": "something failed", "stats": {"p50": 10}}
                ],
            }
        ]
    )

    with pytest.raises(AdapterError, match=r"^No valid benchmark runs found$"):
        mitata_adapter.parse(stdout)


def test_parse_when_error_field_is_null_does_process_run_normally():
    stdout = build_stdout(
        [
            {
                "alias": "test",
                "runs": [{"name": "test", "args": {}, "error": None, "stats": {"p50": 10}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {"test#time": 10}


def test_parse_when_error_field_is_object_does_render_as_json_in_warning():
    stdout = build_stdout(
        [
            {
                "alias": "test",
                "runs": [
                    {"name": "a", "args": {}, "error": {"code": 7}, "stats": {"p50": 10}},
                    {"name": "b", "args": {}, "stats": {"p50": 20}},
                ],
            }
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 20}
    assert any('{"code": 7}' in w for w in warnings)


# ---------------------------------------------------------------------------
# non-primitive run-argument serialization
# ---------------------------------------------------------------------------


def test_parse_when_arg_value_is_object_does_serialize_via_json():
    stdout = build_stdout(
        [
            {
                "alias": "bench/$opts",
                "runs": [{"name": "cfg", "args": {"opts": {"size": 100}}, "stats": {"p50": 5}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {'bench/opts={"size":100}#time': 5}


def test_parse_when_object_arg_values_differ_does_keep_distinct_names():
    stdout = build_stdout(
        [
            {
                "alias": "bench/$opts",
                "runs": [
                    {"name": "a", "args": {"opts": {"size": 100}}, "stats": {"p50": 5}},
                    {"name": "b", "args": {"opts": {"size": 200}}, "stats": {"p50": 10}},
                ],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {
        'bench/opts={"size":100}#time': 5,
        'bench/opts={"size":200}#time': 10,
    }


def test_parse_when_object_arg_has_unsorted_keys_does_serialize_in_sorted_order():
    stdout = build_stdout(
        [
            {
                "alias": "bench/$opts",
                "runs": [{"name": "cfg", "args": {"opts": {"z": 1, "a": 2}}, "stats": {"p50": 5}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {'bench/opts={"a":2,"z":1}#time': 5}


def test_parse_when_arg_value_is_array_does_serialize_via_json():
    stdout = build_stdout(
        [
            {
                "alias": "bench/$items",
                "runs": [{"name": "list", "args": {"items": [1, 2, 3]}, "stats": {"p50": 7}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {"bench/items=[1,2,3]#time": 7}


# ---------------------------------------------------------------------------
# non-finite p50 or malformed shape warnings
# ---------------------------------------------------------------------------


def test_parse_when_p50_missing_does_warn_and_skip():
    stdout = build_stdout(
        [
            {
                "alias": "test/$x",
                "runs": [
                    {"name": "a", "args": {"x": "a"}, "stats": {}},
                    {"name": "b", "args": {"x": "b"}, "stats": {"p50": 20}},
                ],
            }
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test/x=b#time": 20}
    assert any("test/$x" in w for w in warnings)


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

    assert warnings != []
    assert "non-finite" in warnings[0].lower()
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# brace-aware extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        pytest.param("cpu: {{model}}\nruntime: bun {{version}}\n\n{json}", id="braces-before"),
        pytest.param("{json}\nfooter: {{info}}", id="braces-after"),
        pytest.param("cpu: {{model}}\n{json}\nfooter: {{info}}", id="braces-both-sides"),
        pytest.param('weight: 5" tall\n{json}', id="stray-quote-before"),
    ],
)
def test_parse_when_banner_text_carries_braces_or_quotes_does_still_extract(template: str):
    payload = build_stdout([{"alias": "encode", "runs": [{"args": {}, "stats": {"p50": 42}}]}])

    result = mitata_adapter.parse(template.format(json=payload))

    assert result == {"encode#time": 42}


def test_parse_when_only_incomplete_brace_fragments_present_does_raise():
    with pytest.raises(AdapterError, match=r"^Failed to parse JSON:"):
        mitata_adapter.parse("cpu: {model}\nno json here\nfooter: {info}")


# ---------------------------------------------------------------------------
# unbalanced brace before payload
# ---------------------------------------------------------------------------


def test_parse_when_unbalanced_brace_precedes_json_does_still_find_payload():
    payload = build_stdout([{"alias": "encode", "runs": [{"args": {}, "stats": {"p50": 42}}]}])
    stdout = f"cpu: {{model\n{payload}"

    result = mitata_adapter.parse(stdout)

    assert result == {"encode#time": 42}


def test_parse_when_pathological_nesting_does_raise_adapter_error_not_recursion_error():
    stdout = "{" * 5000

    with pytest.raises(AdapterError, match=r"^Failed to parse JSON:"):
        mitata_adapter.parse(stdout)


# ---------------------------------------------------------------------------
# candidate selection among multiple JSON objects
# ---------------------------------------------------------------------------


def test_parse_when_decoy_precedes_real_object_does_prefer_the_benchmarks_carrier():
    decoy = json.dumps({"foo": "bar"})
    real = build_stdout([{"alias": "a", "runs": [{"args": {}, "stats": {"p50": 1}}]}])

    assert mitata_adapter.parse(f"{decoy}\n{real}") == {"a#time": 1}


def test_parse_when_no_candidate_carries_benchmarks_does_report_missing_array():
    stdout = f"{json.dumps({'foo': 'bar'})}\n{json.dumps({'baz': 1})}"

    with pytest.raises(AdapterError, match=r"^JSON missing benchmarks array$"):
        mitata_adapter.parse(stdout)


def _decode_error_reason(text: str, pos: int = 0) -> str:
    """Return the JSONDecodeError message from attempting ``raw_decode`` at *pos*.

    Uses :meth:`json.JSONDecoder.raw_decode` to match how the adapter now
    discovers candidates. The resulting char offset is absolute within *text*,
    not relative to a pre-sliced candidate.
    """
    try:
        json.JSONDecoder().raw_decode(text, pos)
    except json.JSONDecodeError as exc:
        return str(exc)
    msg = f"expected raw_decode at pos {pos} in {text!r} to fail"
    raise AssertionError(msg)


def test_parse_when_several_candidates_fail_does_report_longest_candidates_error():
    long_bad = '{"padding":"' + ("x" * 100) + '","bad":@}'
    short_bad = "{!}"
    stdout = f"{long_bad} noise {short_bad}"

    with pytest.raises(AdapterError) as exc_info:
        mitata_adapter.parse(stdout)

    longest_pos = stdout.index("{")
    assert (
        str(exc_info.value) == f"Failed to parse JSON: {_decode_error_reason(stdout, longest_pos)}"
    )


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
