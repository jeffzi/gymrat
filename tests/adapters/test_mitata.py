import re
from typing import Any

import pytest

from gymrat.adapters.contract import MEMORY_SUFFIX, TIME_SUFFIX, AdapterError
from gymrat.adapters.mitata import mitata_adapter
from tests.adapters._inputs import (
    VALID_BENCHMARK,
    benchmark,
    build_stdout,
)

# ---------------------------------------------------------------------------
# metric naming for parameterized benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alias", "args", "p50", "metric_name"),
    [
        pytest.param(
            "op/$a/$b", {"a": "x", "b": "y"}, 50, f"op/a=x/b=y{TIME_SUFFIX}", id="multiple"
        ),
        pytest.param(
            "test/$x/$unknown", {"x": "1"}, 77, f"test/x=1/$unknown{TIME_SUFFIX}", id="stray-dollar"
        ),
        pytest.param("$ab", {"a": "x", "ab": "y"}, 8, f"ab=y{TIME_SUFFIX}", id="longest-key-first"),
        pytest.param("test/$x", {}, 42, f"test/$x{TIME_SUFFIX}", id="no-args-keeps-placeholder"),
        pytest.param(
            "decode/$text",
            {"text": "a\\1b"},
            42,
            f"decode/text=a\\1b{TIME_SUFFIX}",
            id="backreference",
        ),
        pytest.param(
            "op/$a/$b",
            {"a": "$b", "b": "y"},
            7,
            f"op/a=$b/b=y{TIME_SUFFIX}",
            id="value-is-later-placeholder",
        ),
        pytest.param(
            "op/$a/$b",
            {"a": "y", "b": "$a"},
            7,
            f"op/a=y/b=$a{TIME_SUFFIX}",
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
        pytest.param(5.0, "5", id="integral-float"),
        pytest.param(1.5, "1.5", id="float"),
        pytest.param(True, "true", id="bool-true"),
        pytest.param(False, "false", id="bool-false"),
        pytest.param(None, "null", id="none"),
        pytest.param({"a": 2}, '{"a":2}', id="object"),
    ],
)
def test_parse_when_arg_value_given_does_serialize_js_style(value: Any, serialized: str):
    stdout = build_stdout([benchmark("b/$v", args={"v": value})])

    assert mitata_adapter.parse(stdout) == {f"b/v={serialized}{TIME_SUFFIX}": 1}


# ---------------------------------------------------------------------------
# metric names carrying a line terminator
# ---------------------------------------------------------------------------

# One non-ASCII terminator per skip site catches both a missing escape and an
# ``ensure_ascii=False`` regression; the per-character escape table is pinned once,
# on describe_error in tests/utils/test_utils.py, which also pins which characters
# count as line terminators.
_LINE_TERMINATOR_ALIAS = "enc\u2028ode"


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

    assert result == {f"valid{TIME_SUFFIX}": 1}
    assert len(warnings) == 1
    assert warned_alias in warnings[0]


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"runs": [{"args": {}, "error": "boom"}]}, id="errored-run"),
        pytest.param({"runs": [{"args": "bad", "stats": {"p50": 1}}]}, id="invalid-run"),
    ],
)
def test_parse_when_skip_warning_names_alias_holding_line_terminator_does_escape_it(
    fields: dict[str, Any],
):
    stdout = build_stdout([
        {"alias": _LINE_TERMINATOR_ALIAS, **fields},
        VALID_BENCHMARK,
    ])
    warnings: list[str] = []

    mitata_adapter.parse(stdout, warnings.append)

    assert len(warnings) == 1
    assert '"enc\\u2028ode"' in warnings[0]


# ---------------------------------------------------------------------------
# alias containing '#' is a hard error
# ---------------------------------------------------------------------------


def test_parse_when_metric_prefix_contains_hash_does_raise_adapter_error():
    alias = "op/$v"
    stdout = build_stdout([benchmark(alias, p50=42, args={"v": "a#b"})])

    with pytest.raises(AdapterError) as exc_info:
        mitata_adapter.parse(stdout)

    message = str(exc_info.value)
    assert '"op/v=a#b"' in message
    assert f'"{alias}"' in message


# ---------------------------------------------------------------------------
# metric names with an empty path segment
# ---------------------------------------------------------------------------


# Which name shape leaves an empty segment is pinned in tests/test_metric_name.py;
# here the empty segment arrives through a substituted argument, proving the adapter
# checks the prefix after substitution.
def test_parse_when_metric_name_has_empty_path_segment_does_skip_run_with_one_warning():
    stdout = build_stdout([
        benchmark("op/$v", args={"v": "a//b"}, stats={"p50": 42, "heap": {"avg": 7}}),
        VALID_BENCHMARK,
    ])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {f"valid{TIME_SUFFIX}": 1}
    assert len(warnings) == 1
    assert '"op/v=a//b"' in warnings[0]


# ---------------------------------------------------------------------------
# malformed run and benchmark shapes
# ---------------------------------------------------------------------------


# Each row pairs a run with a regex for the computed parts of its skip warning.
@pytest.mark.parametrize(
    ("bad_run", "pattern"),
    [
        pytest.param(
            {"args": {}, "stats": {"p50": float("inf")}},
            r'"test".*stats\.p50',
            id="p50-positive-infinity",
        ),
        pytest.param(None, r'"test"', id="run-null"),
    ],
)
def test_parse_when_run_is_malformed_does_skip_only_the_bad_run_with_one_warning(
    bad_run: object, pattern: str
):
    stdout = build_stdout([{"alias": "test", "runs": [bad_run, {"args": {}, "stats": {"p50": 5}}]}])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {f"test{TIME_SUFFIX}": 5}
    assert len(warnings) == 1
    assert re.search(pattern, warnings[0])


def test_parse_when_benchmark_is_malformed_does_skip_only_the_bad_benchmark_with_one_warning(
    capsys: pytest.CaptureFixture[str],
):
    stdout = build_stdout([{"alias": "orphan"}, VALID_BENCHMARK])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert (result, len(warnings), capsys.readouterr().err) == ({f"valid{TIME_SUFFIX}": 1}, 1, "")
    assert re.search(r'"orphan".* runs$', warnings[0])


# ---------------------------------------------------------------------------
# memory metric emission
# ---------------------------------------------------------------------------


def test_parse_when_heap_avg_present_does_emit_memory_metric():
    stdout = build_stdout([benchmark("test", stats={"p50": 42, "heap": {"avg": 1024}})])

    assert mitata_adapter.parse(stdout) == {f"test{TIME_SUFFIX}": 42, f"test{MEMORY_SUFFIX}": 1024}


def test_parse_when_heap_avg_not_given_does_skip_memory_metric_silently():
    stdout = build_stdout([benchmark("test", stats={"p50": 42, "heap": {"total": 1024}})])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {f"test{TIME_SUFFIX}": 42}
    assert warnings == []


# Each row pairs a heap value with a regex for the computed parts of its skip warning.
@pytest.mark.parametrize(
    ("heap_value", "pattern"),
    [
        pytest.param({"avg": "bad"}, r'"test".*stats\.heap\.avg', id="avg-string"),
        pytest.param({"avg": float("inf")}, r'"test".*stats\.heap\.avg', id="avg-infinity"),
    ],
)
def test_parse_when_heap_is_malformed_does_keep_time_metric_with_one_warning(
    heap_value: object, pattern: str
):
    stdout = build_stdout([benchmark("test", stats={"p50": 42, "heap": heap_value})])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {f"test{TIME_SUFFIX}": 42}
    assert len(warnings) == 1
    assert re.search(pattern, warnings[0])


# ---------------------------------------------------------------------------
# error field handling
# ---------------------------------------------------------------------------


def test_parse_when_run_has_error_field_does_skip_only_the_errored_run_with_one_warning():
    stdout = build_stdout([
        {
            "alias": "test",
            "runs": [{"args": {}, "error": "boom"}, {"args": {}, "stats": {"p50": 20}}],
        }
    ])
    warnings: list[str] = []

    result = mitata_adapter.parse(stdout, warnings.append)

    assert result == {f"test{TIME_SUFFIX}": 20}
    assert len(warnings) == 1
    assert re.search(r'"test".*boom', warnings[0])


def test_parse_when_error_field_is_null_does_process_run_normally():
    stdout = build_stdout([
        {
            "alias": "test",
            "runs": [{"name": "test", "args": {}, "error": None, "stats": {"p50": 10}}],
        }
    ])

    assert mitata_adapter.parse(stdout) == {f"test{TIME_SUFFIX}": 10}
