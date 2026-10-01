"""Tests for how the mitata adapter finds its JSON payload in benchmark stdout.

Banner text around the payload may carry braces, quotes, or decoy JSON objects;
the adapter scans for balanced candidates, prefers the one carrying a
``benchmarks`` array, and reports the most informative decode failure when none
parses.
"""

import json

import pytest

from gymrat.adapters.mitata import find_json_candidates, mitata_adapter
from gymrat.adapters.types import AdapterError
from tests.adapters._inputs import build_stdout

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

    Uses :meth:`json.JSONDecoder.raw_decode` to match how the adapter
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
