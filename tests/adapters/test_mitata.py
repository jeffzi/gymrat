from typing import Any

import pytest

from gymrat.adapters.mitata import find_json_candidates, mitata_adapter
from gymrat.adapters.types import Adapter, AdapterError, MetricDefaults
from gymrat.model.metrics import MetricUnit
from tests.adapters._inputs import build_stdout

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
    assert mitata_adapter.parse(stdout) == {"encode/time": 42}


# ---------------------------------------------------------------------------
# metric naming for parameterized benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "args", "p50", "metric_name"),
    [
        pytest.param(
            "decode/$text", {"text": "digits"}, 42, "decode/text=digits/time", id="single"
        ),
        pytest.param("op/$a/$b", {"a": "x", "b": "y"}, 50, "op/a=x/b=y/time", id="multiple"),
        pytest.param("test/$x/sep/$x", {"x": "1"}, 99, "test/x=1/sep/x=1/time", id="repeated"),
        pytest.param(
            "test/$x/$unknown", {"x": "1"}, 77, "test/x=1/$unknown/time", id="stray-dollar"
        ),
        pytest.param("$ab", {"a": "x", "ab": "y"}, 8, "ab=y/time", id="longest-key-first"),
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

    assert mitata_adapter.parse(stdout) == {f"b/v={serialized}/time": 1}


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

    assert mitata_adapter.parse(stdout) == {f"decode/text={value}/time": 42}


def test_parse_when_arg_value_introduces_a_placeholder_does_not_re_substitute_it():
    stdout = build_stdout(
        [
            {
                "alias": "op/$a/$b",
                "runs": [{"name": "op", "args": {"a": "$b", "b": "y"}, "stats": {"p50": 7}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {"op/a=$b/b=y/time": 7}


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

    assert result == {"valid/time": 1}
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

    assert result == {"decode/text=words/time": 20}
    assert any(
        "Skipping run with a line terminator in its metric name: decode/$text" in w
        for w in warnings
    )


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

    assert result == {"test/time": 5}
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

    assert result == {"test/time": 5}
    assert any("Skipping run with malformed stats shape: test" in w for w in warnings)


def test_parse_when_run_args_missing_does_treat_as_empty_and_not_warn():
    stdout = build_stdout([{"alias": "test", "runs": [{"stats": {"p50": 5}}]}])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test/time": 5}
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

    assert result == {"valid/time": 1}
    assert any("Skipping benchmark with malformed alias: 42" in w for w in warnings)


def test_parse_when_benchmark_runs_is_not_an_array_does_warn_and_skip():
    stdout = build_stdout(
        [{"alias": "orphan"}, {"alias": "valid", "runs": [{"args": {}, "stats": {"p50": 1}}]}]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid/time": 1}
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

    assert result == {"test/time": 20}
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

    assert result == {"decode/time": 20}
    assert "Duplicate metric name: decode/time" in capsys.readouterr().err


def test_parse_when_collision_and_sink_given_does_route_warning_off_stderr(
    capsys: pytest.CaptureFixture[str],
):
    warnings: list[str] = []

    mitata_adapter.parse(_ALIAS_MISSING_PLACEHOLDER, warnings.append)

    assert any("Duplicate metric name: decode/time" in w for w in warnings)
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

    assert "Duplicate metric name: encode/time" in capsys.readouterr().err


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

    assert mitata_adapter.parse(stdout) == {"test/time": p50}


def test_parse_when_p50_is_bool_does_warn_and_skip():
    stdout = (
        '{"benchmarks":[{"alias":"test","runs":['
        '{"name":"a","args":{},"stats":{"p50":true}},'
        '{"name":"b","args":{},"stats":{"p50":5}}]}]}'
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test/time": 5}
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

    assert mitata_adapter.parse(stdout) == {"test/time": 42, "test/heap": 1024}


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
        "decode/text=digits/time": 10,
        "decode/text=digits/heap": 256,
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

    assert mitata_adapter.parse(stdout) == {"test/time": 42}


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

    assert result == {"test/time": 42}
    assert len(warnings) == 1
    assert str(heap_value) in warnings[0] or repr(heap_value) in warnings[0]


def test_parse_when_heap_absent_does_not_warn():
    stdout = build_stdout(
        [{"alias": "test", "runs": [{"name": "test", "args": {}, "stats": {"p50": 42}}]}]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test/time": 42}
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

    assert result == {"test/x=b/time": 20}
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

    assert result == {"test/time": 42}
    assert any("test" in w for w in warnings)


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
        "decode/text=digits/time": 10,
        "decode/text=words/time": 20,
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
        "decode/text=digits/time": 10,
        "decode/text=digits/heap": 256,
        "decode/text=words/time": 20,
        "decode/text=words/heap": 512,
    }


def test_parse_when_multiple_benchmarks_does_emit_metrics_for_all():
    stdout = build_stdout(
        [
            {"alias": "encode", "runs": [{"name": "encode", "args": {}, "stats": {"p50": 42}}]},
            {"alias": "decode", "runs": [{"name": "decode", "args": {}, "stats": {"p50": 100}}]},
        ]
    )

    assert mitata_adapter.parse(stdout) == {"encode/time": 42, "decode/time": 100}


# ---------------------------------------------------------------------------
# name-derived defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_name",
    ["test/time", "encode/time", "decode/x=1/time", "complex/a=1/b=2/time"],
)
def test_defaults_when_time_metric_does_describe_as_lower_ns_time(metric_name: str):
    short_name = metric_name.removesuffix("/time")

    assert mitata_adapter.defaults(metric_name) == MetricDefaults(
        direction="lower", unit="ns", kind="time", short_name=short_name
    )


@pytest.mark.parametrize(
    "metric_name",
    ["test/heap", "encode/heap", "decode/x=1/heap", "complex/a=1/b=2/heap"],
)
def test_defaults_when_heap_metric_does_describe_as_lower_bytes_memory(metric_name: str):
    short_name = metric_name.removesuffix("/heap")

    assert mitata_adapter.defaults(metric_name) == MetricDefaults(
        direction="lower", unit="bytes", kind="memory", short_name=short_name
    )


@pytest.mark.parametrize("metric_name", ["custom_metric", "test", "test/throughput", "test/ops"])
def test_defaults_when_metric_unrecognized_does_return_direction_only(metric_name: str):
    assert mitata_adapter.defaults(metric_name) == MetricDefaults(direction="lower")


@pytest.mark.parametrize(
    ("metric_name", "unit", "kind"),
    [
        pytest.param("/time", "ns", "time", id="time"),
        pytest.param("/heap", "bytes", "memory", id="heap"),
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
