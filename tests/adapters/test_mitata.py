import json
from pathlib import Path
from typing import Any

import pytest

from gymrat.adapters import AdapterError, mitata_adapter
from tests.adapters._inputs import LINE_BREAKS, VALID_BENCHMARK, benchmark, build_stdout

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mitata.json"

# ---------------------------------------------------------------------------
# basic JSON parsing
# ---------------------------------------------------------------------------

_BASIC_FIXTURE = build_stdout([benchmark("encode", p50=42)])


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param(f"preamble\n{_BASIC_FIXTURE}\ntrailer", id="preamble-and-trailer"),
        pytest.param(f"cpu: {{model}}\n{_BASIC_FIXTURE}\nfooter: {{info}}", id="braces-both-sides"),
        pytest.param(f'weight: 5" tall\n{_BASIC_FIXTURE}', id="stray-quote-before"),
        pytest.param(
            f'cpu: {{model 5" tall\n{_BASIC_FIXTURE}', id="stray-quote-inside-unclosed-brace"
        ),
        pytest.param(f"cpu: {{model\n{_BASIC_FIXTURE}", id="unbalanced-brace-before"),
    ],
)
def test_parse_when_json_surrounded_by_text_does_extract_metrics(stdout: str):
    assert mitata_adapter.parse(stdout) == {"encode#time": 42}


@pytest.mark.parametrize(
    "alias",
    [
        pytest.param("value with {braces} inside", id="braces-in-string"),
        pytest.param('value with "escaped" quotes and {braces}', id="escaped-quotes"),
    ],
)
def test_parse_when_a_payload_string_carries_braces_or_quotes_does_read_the_whole_object(
    alias: str,
):
    payload = build_stdout([benchmark(alias, p50=42)])

    assert mitata_adapter.parse(payload) == {f"{alias}#time": 42}


def test_parse_when_truncated_json_has_nested_object_does_report_decode_failure():
    # Truncated outer JSON — raw_decode at position 0 fails. The inner
    # {"alias":"encode"} is valid JSON but carries no "benchmarks" key.
    # The adapter should prefer the decode failure (explaining WHY the real
    # payload could not parse) over the generic "JSON missing benchmarks array".
    truncated = '{"benchmarks":[{"alias":"encode"}],"extra":'

    with pytest.raises(AdapterError, match=r"^Failed to parse JSON:"):
        mitata_adapter.parse(truncated)


def test_parse_when_pathological_nesting_does_raise_adapter_error_not_recursion_error():
    # A bare run of ``{`` is refused at the first key without descending, so the
    # depth comes from arrays under one key; one ``{`` keeps it to one attempt.
    stdout = '{"a":' + "[" * 500_000

    with pytest.raises(AdapterError, match=r"^Failed to parse JSON: Exceeded maximum recursion"):
        mitata_adapter.parse(stdout)


def test_parse_when_decoy_precedes_real_object_does_prefer_the_benchmarks_carrier():
    decoy = json.dumps({"foo": "bar"})
    real = build_stdout([benchmark("a")])

    assert mitata_adapter.parse(f"{decoy}\n{real}") == {"a#time": 1}


def test_parse_when_several_candidates_fail_does_report_longest_candidates_error():
    long_bad = '{"padding":"' + ("x" * 100) + '","bad":@}'
    stdout = f"{long_bad} noise {{!}}"

    with pytest.raises(AdapterError) as exc_info:
        mitata_adapter.parse(stdout)

    assert str(exc_info.value) == (
        "Failed to parse JSON: Expecting value: line 1 column 121 (char 120)"
    )


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
        pytest.param("test/$x", {}, 42, "test/$x#time", id="no-args-keeps-placeholder"),
        pytest.param(
            "decode/$text", {"text": "a\\1b"}, 42, "decode/text=a\\1b#time", id="backreference"
        ),
        pytest.param(
            "decode/$text",
            {"text": "\\g<0>"},
            42,
            "decode/text=\\g<0>#time",
            id="named-group-reference",
        ),
        pytest.param(
            "op/$a/$b",
            {"a": "$b", "b": "y"},
            7,
            "op/a=$b/b=y#time",
            id="value-is-later-placeholder",
        ),
        pytest.param(
            "op/$a/$b",
            {"a": "y", "b": "$a"},
            7,
            "op/a=y/b=$a#time",
            id="value-is-earlier-placeholder",
        ),
    ],
)
def test_parse_when_alias_has_placeholders_does_substitute_arg_values(
    alias: str, args: dict[str, Any], p50: int, metric_name: str
):
    stdout = build_stdout([benchmark(alias, p50=p50, args=args)])

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
        pytest.param({"size": 100}, '{"size":100}', id="object"),
        pytest.param({"z": 1, "a": 2}, '{"a":2,"z":1}', id="object-keys-sorted"),
        pytest.param([1, 2, 3], "[1,2,3]", id="array"),
    ],
)
def test_parse_when_arg_value_given_does_serialize_js_style(value: Any, serialized: str):
    stdout = build_stdout([benchmark("b/$v", args={"v": value})])

    assert mitata_adapter.parse(stdout) == {f"b/v={serialized}#time": 1}


# ---------------------------------------------------------------------------
# metric names carrying a line terminator
# ---------------------------------------------------------------------------

# One non-ASCII terminator per skip site catches both a missing escape and an
# ``ensure_ascii=False`` regression; the per-character escape table is pinned once,
# on the error-string warning below. Which characters count as line terminators is
# pinned in tests/test_metric_name.py.
_LINE_TERMINATOR_ALIAS = "enc\u2028ode"

_ESCAPED_LINE_TERMINATORS = [
    pytest.param(line_break.char, line_break.escaped, id=line_break.name)
    for line_break in LINE_BREAKS
]


@pytest.mark.parametrize(
    ("alias", "args", "warned_alias"),
    [
        pytest.param(_LINE_TERMINATOR_ALIAS, {}, '"enc\\u2028ode"', id="alias"),
        pytest.param("decode/$text", {"text": "di\u2028gits"}, '"decode/$text"', id="arg-value"),
    ],
)
def test_parse_when_metric_name_holds_line_terminator_does_skip_run_with_one_line_warning(
    alias: str, args: dict[str, str], warned_alias: str
):
    stdout = build_stdout([benchmark(alias, p50=42, args=args), VALID_BENCHMARK])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid#time": 1}
    assert warnings == [
        (
            f"Skipping run with a line terminator in its metric name: {warned_alias} "
            "(the alias or one of its argument values carries one)"
        )
    ]


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
        pytest.param({}, "Skipping benchmark {alias} with missing runs", id="runs-missing"),
    ],
)
def test_parse_when_skip_warning_names_alias_holding_line_terminator_does_escape_it(
    fields: dict[str, Any], warning_template: str
):
    stdout = build_stdout([
        {"alias": _LINE_TERMINATOR_ALIAS, **fields},
        VALID_BENCHMARK,
    ])
    warnings: list[str] = []

    mitata_adapter.parse(stdout, warnings.append)

    assert warnings == [warning_template.format(alias='"enc\\u2028ode"')]


@pytest.mark.parametrize(
    ("alias", "args"),
    [
        pytest.param("en\u2028c#ode", {}, id="alias"),
        pytest.param("enc#$text", {"text": "a\u2028b"}, id="arg-value"),
    ],
)
def test_parse_when_reserved_hash_name_holds_line_terminator_does_raise_on_one_line(
    alias: str, args: dict[str, str]
):
    stdout = build_stdout([benchmark(alias, p50=42, args=args)])

    with pytest.raises(AdapterError) as exc_info:
        mitata_adapter.parse(stdout)

    message = str(exc_info.value)
    assert json.dumps(alias) in message
    assert message.splitlines() == [message]


# ---------------------------------------------------------------------------
# alias containing '#' is a hard error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "args", "prefix"),
    [
        pytest.param("enc#ode", {}, "enc#ode", id="alias"),
        pytest.param("op/$v", {"v": "a#b"}, "op/v=a#b", id="substituted-arg"),
    ],
)
def test_parse_when_metric_prefix_contains_hash_does_raise_adapter_error(
    alias: str, args: dict[str, str], prefix: str
):
    stdout = build_stdout([benchmark(alias, p50=42, args=args)])

    with pytest.raises(AdapterError) as exc_info:
        mitata_adapter.parse(stdout)

    assert str(exc_info.value) == (
        f"Metric prefix \"{prefix}\" contains '#', which is reserved as the "
        f'metric-type separator (alias: "{alias}")'
    )


# ---------------------------------------------------------------------------
# metric names with an empty path segment
# ---------------------------------------------------------------------------


# Which name shape leaves an empty segment is pinned in tests/test_metric_name.py;
# here the alias case and the substituted-argument case prove the adapter checks the
# prefix after substitution.
@pytest.mark.parametrize(
    ("alias", "args", "prefix"),
    [
        pytest.param("a//b", {}, "a//b", id="alias-with-empty-segment"),
        pytest.param("op/$v", {"v": "a//b"}, "op/v=a//b", id="arg-value-with-empty-segment"),
    ],
)
def test_parse_when_metric_name_has_empty_path_segment_does_skip_run_with_one_warning(
    alias: str, args: dict[str, str], prefix: str
):
    stdout = build_stdout([
        benchmark(alias, args=args, stats={"p50": 42, "heap": {"avg": 7}}),
        VALID_BENCHMARK,
    ])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid#time": 1}
    assert warnings == [f'Skipping run with an empty path segment in its metric name: "{prefix}"']


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
    ],
)
def test_parse_when_run_is_malformed_does_skip_only_the_bad_run_with_one_warning(
    bad_run: object, warning: str
):
    stdout = build_stdout([{"alias": "test", "runs": [bad_run, {"args": {}, "stats": {"p50": 5}}]}])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 5}
    assert warnings == [warning]


def test_parse_when_run_args_missing_does_read_args_as_empty_without_warning():
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
def test_parse_when_benchmark_is_malformed_does_skip_only_the_bad_benchmark_with_one_warning(
    bad_benchmark: object, warning: str
):
    stdout = build_stdout([bad_benchmark, VALID_BENCHMARK])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid#time": 1}
    assert warnings == [warning]


# ---------------------------------------------------------------------------
# metric-name collisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stdout", "name", "kept"),
    [
        pytest.param(
            build_stdout([
                {
                    "alias": "decode",
                    "runs": [
                        {"name": "decode/digits", "args": {"text": "digits"}, "stats": {"p50": 10}},
                        {"name": "decode/words", "args": {"text": "words"}, "stats": {"p50": 20}},
                    ],
                }
            ]),
            "decode#time",
            20,
            id="alias-missing-placeholder",
        ),
        pytest.param(
            build_stdout([benchmark("encode", p50=1), benchmark("encode", p50=2)]),
            "encode#time",
            2,
            id="two-benchmarks-share-alias",
        ),
    ],
)
def test_parse_when_metric_names_collide_does_keep_last_value_with_warning(
    stdout: str, name: str, kept: int
):
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {name: kept}
    assert warnings == [
        (
            f"Duplicate metric name: {name} (keeping the last value; "
            "give the benchmark aliases distinct $placeholders to separate the runs)"
        )
    ]


# ---------------------------------------------------------------------------
# heap metric emission
# ---------------------------------------------------------------------------


def test_parse_when_heap_avg_present_does_emit_heap_metric_keeping_integers():
    stdout = build_stdout([benchmark("test", stats={"p50": 42, "heap": {"avg": 1024}})])

    result = mitata_adapter.parse(stdout)

    assert result == {"test#time": 42, "test#heap": 1024}
    assert [type(v) for v in result.values()] == [int, int]


@pytest.mark.parametrize(
    "stats",
    [
        pytest.param({"p50": 42, "heap": {"total": 1024}}, id="heap-avg-missing"),
        pytest.param({"p50": 42}, id="heap-absent"),
    ],
)
def test_parse_when_heap_avg_not_given_does_skip_heap_metric_silently(stats: dict[str, Any]):
    stdout = build_stdout([benchmark("test", stats=stats)])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 42}
    assert warnings == []


_INVALID_HEAP_PREFIX = 'Skipping heap metric of "test" with invalid'


@pytest.mark.parametrize(
    ("heap_value", "warning"),
    [
        pytest.param(
            42, f"{_INVALID_HEAP_PREFIX} stats.heap: expected an object, got 42", id="integer"
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
def test_parse_when_heap_is_malformed_does_keep_time_metric_with_one_warning(
    heap_value: object, warning: str
):
    stdout = build_stdout([benchmark("test", stats={"p50": 42, "heap": heap_value})])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test#time": 42}
    assert warnings == [warning]


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        pytest.param(
            "not valid json at all", r"^No JSON object found in stdout$", id="no-json-object"
        ),
        pytest.param(
            json.dumps({"something": "else"}),
            r"^JSON missing benchmarks array$",
            id="benchmarks-array-missing",
        ),
        pytest.param(
            json.dumps({"benchmarks": []}),
            r"^benchmarks array is empty$",
            id="benchmarks-array-empty",
        ),
        pytest.param(
            build_stdout([benchmark("test", stats={})]),
            r"^No valid benchmark runs found$",
            id="no-valid-stats",
        ),
    ],
)
def test_parse_when_stdout_has_no_usable_payload_does_raise_adapter_error(
    stdout: str, message: str
):
    with pytest.raises(AdapterError, match=message):
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
def test_parse_when_run_has_error_field_does_skip_only_the_errored_run_with_one_warning(
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


def test_parse_when_error_field_is_null_does_process_run_normally():
    stdout = build_stdout([
        {
            "alias": "test",
            "runs": [{"name": "test", "args": {}, "error": None, "stats": {"p50": 10}}],
        }
    ])

    assert mitata_adapter.parse(stdout) == {"test#time": 10}


# ---------------------------------------------------------------------------
# real fixture
# ---------------------------------------------------------------------------


def test_parse_when_given_real_fixture_does_extract_all_metrics():
    stdout = _FIXTURE_PATH.read_text()

    result = mitata_adapter.parse(stdout)

    assert result == {
        "decode/text=digits#time": 4.0791015625,
        "decode/text=digits#heap": 0.13420623129857714,
        "decode/text=words#time": 7.8125,
        "decode/text=words#heap": 0.14746411878141288,
        "encode#time": 42.66357421875,
        "encode#heap": 80.1967411655276,
    }
