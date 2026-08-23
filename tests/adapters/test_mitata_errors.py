import json
from pathlib import Path
from typing import Any

import pytest

from gymrat_py.adapters.mitata import mitata_adapter
from gymrat_py.adapters.types import AdapterError

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mitata.json"


def _stdout(benchmarks: list[Any]) -> str:
    return json.dumps({"benchmarks": benchmarks})


def _decode_error_reason(candidate: str) -> str:
    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        return str(exc)
    msg = f"expected {candidate!r} to fail json parsing"
    raise AssertionError(msg)


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
    stdout = _stdout([{"alias": "test", "runs": [{"name": "test", "args": {}, "stats": {}}]}])

    with pytest.raises(AdapterError, match=r"^No valid benchmark runs found$"):
        mitata_adapter.parse(stdout)


def test_parse_when_benchmark_entries_not_objects_does_skip_silently():
    stdout = _stdout(
        [
            None,
            42,
            "string",
            {"alias": "valid", "runs": [{"name": "valid", "args": {}, "stats": {"p50": 1}}]},
        ]
    )
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"valid/time": 1}
    assert warnings == []


def test_parse_when_benchmarks_have_non_string_alias_or_missing_runs_does_skip():
    stdout = _stdout(
        [
            {"alias": 42, "runs": []},
            {"alias": "orphan"},
            {"runs": [{"args": {}, "stats": {"p50": 1}}]},
            {"alias": "valid", "runs": [{"name": "valid", "args": {}, "stats": {"p50": 1}}]},
        ]
    )

    assert mitata_adapter.parse(stdout) == {"valid/time": 1}


def test_parse_when_runs_have_non_object_args_or_stats_does_skip():
    stdout = _stdout(
        [
            {
                "alias": "test",
                "runs": [
                    {"args": "not-object", "stats": {"p50": 1}},
                    {"args": {}, "stats": "not-object"},
                    {"args": {}, "stats": {"p50": 1}},
                ],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {"test/time": 1}


def test_parse_when_runs_are_not_objects_does_skip_silently():
    stdout = _stdout([{"alias": "test", "runs": [None, 42, {"args": {}, "stats": {"p50": 1}}]}])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {"test/time": 1}
    assert warnings == []


# ---------------------------------------------------------------------------
# error field handling
# ---------------------------------------------------------------------------


def test_parse_when_run_has_error_field_does_skip_that_run():
    stdout = _stdout(
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

    assert mitata_adapter.parse(stdout) == {"test/x=b/time": 20}


def test_parse_when_all_runs_have_errors_does_raise():
    stdout = _stdout(
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
    stdout = _stdout(
        [
            {
                "alias": "test",
                "runs": [{"name": "test", "args": {}, "error": None, "stats": {"p50": 10}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {"test/time": 10}


def test_parse_when_error_field_is_object_does_render_as_json_in_warning():
    stdout = _stdout(
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

    assert result == {"test/time": 20}
    assert any('{"code": 7}' in w for w in warnings)


# ---------------------------------------------------------------------------
# non-primitive run-argument serialization
# ---------------------------------------------------------------------------


def test_parse_when_arg_value_is_object_does_serialize_via_json():
    stdout = _stdout(
        [
            {
                "alias": "bench/$opts",
                "runs": [{"name": "cfg", "args": {"opts": {"size": 100}}, "stats": {"p50": 5}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {'bench/opts={"size":100}/time': 5}


def test_parse_when_object_arg_values_differ_does_keep_distinct_names():
    stdout = _stdout(
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
        'bench/opts={"size":100}/time': 5,
        'bench/opts={"size":200}/time': 10,
    }


def test_parse_when_object_arg_has_unsorted_keys_does_serialize_in_sorted_order():
    stdout = _stdout(
        [
            {
                "alias": "bench/$opts",
                "runs": [{"name": "cfg", "args": {"opts": {"z": 1, "a": 2}}, "stats": {"p50": 5}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {'bench/opts={"a":2,"z":1}/time': 5}


def test_parse_when_arg_value_is_array_does_serialize_via_json():
    stdout = _stdout(
        [
            {
                "alias": "bench/$items",
                "runs": [{"name": "list", "args": {"items": [1, 2, 3]}, "stats": {"p50": 7}}],
            }
        ]
    )

    assert mitata_adapter.parse(stdout) == {"bench/items=[1,2,3]/time": 7}


# ---------------------------------------------------------------------------
# non-finite p50 or malformed shape warnings
# ---------------------------------------------------------------------------


def test_parse_when_p50_missing_does_warn_and_skip():
    stdout = _stdout(
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

    assert result == {"test/x=b/time": 20}
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
    payload = _stdout([{"alias": "encode", "runs": [{"args": {}, "stats": {"p50": 42}}]}])

    result = mitata_adapter.parse(template.format(json=payload))

    assert result == {"encode/time": 42}


def test_parse_when_only_incomplete_brace_fragments_present_does_raise():
    with pytest.raises(AdapterError, match=r"^Failed to parse JSON:"):
        mitata_adapter.parse("cpu: {model}\nno json here\nfooter: {info}")


# ---------------------------------------------------------------------------
# candidate selection among multiple JSON objects
# ---------------------------------------------------------------------------


def test_parse_when_decoy_precedes_real_object_does_prefer_the_benchmarks_carrier():
    decoy = json.dumps({"foo": "bar"})
    real = _stdout([{"alias": "a", "runs": [{"args": {}, "stats": {"p50": 1}}]}])

    assert mitata_adapter.parse(f"{decoy}\n{real}") == {"a/time": 1}


def test_parse_when_no_candidate_carries_benchmarks_does_report_missing_array():
    stdout = f"{json.dumps({'foo': 'bar'})}\n{json.dumps({'baz': 1})}"

    with pytest.raises(AdapterError, match=r"^JSON missing benchmarks array$"):
        mitata_adapter.parse(stdout)


def test_parse_when_several_candidates_fail_does_report_longest_candidates_error():
    long_bad = '{"padding":"' + ("x" * 100) + '","bad":@}'
    short_bad = "{!}"
    stdout = f"{long_bad} noise {short_bad}"

    with pytest.raises(AdapterError) as exc_info:
        mitata_adapter.parse(stdout)

    assert str(exc_info.value) == f"Failed to parse JSON: {_decode_error_reason(long_bad)}"


# ---------------------------------------------------------------------------
# real fixture
# ---------------------------------------------------------------------------


def test_parse_when_given_real_fixture_does_extract_all_metrics():
    fixture = json.loads(_FIXTURE_PATH.read_text())

    result = mitata_adapter.parse(json.dumps(fixture))

    assert result == {
        "decode/text=digits/time": 4.0791015625,
        "decode/text=digits/heap": 0.13420623129857714,
        "decode/text=words/time": 7.8125,
        "decode/text=words/heap": 0.14746411878141288,
        "encode/time": 42.66357421875,
        "encode/heap": 80.1967411655276,
    }
