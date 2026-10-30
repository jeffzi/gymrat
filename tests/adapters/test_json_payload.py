import json

import pytest

from gymrat.adapters.contract import AdapterError
from gymrat.adapters.json_payload import JsonArrayAdapter, NameWording, accepts_metric_prefix
from gymrat.utils import WarnSink


def _copy_entry(entry: object, metrics: dict[str, float], _warn: WarnSink) -> None:
    if isinstance(entry, dict):
        metrics.update(entry)


_FIRST = JsonArrayAdapter(
    name="items", key="items", last=False, extract=_copy_entry, empty_msg="No items yielded"
)
_LAST = JsonArrayAdapter(
    name="items", key="items", last=True, extract=_copy_entry, empty_msg="No items yielded"
)

_WORDING = NameWording(
    skipped="entry",
    label="Entry name",
    origin='"origin"',
    hash_note=" (hash note)",
    terminator_note=" (terminator note)",
)


def _payload(*entries: object) -> str:
    return json.dumps({"items": list(entries)})


# ---------------------------------------------------------------------------
# locating the payload in stdout
# ---------------------------------------------------------------------------

_PAYLOAD = _payload({"a": 1})


def test_parse_when_payload_surrounded_by_text_does_extract_metrics():
    stdout = f"cpu: {{model}}\n{_PAYLOAD}\nfooter: {{info}}"

    assert _FIRST.parse(stdout) == {"a": 1}


def test_parse_when_key_holds_no_list_does_skip_that_object():
    stdout = f"{json.dumps({'items': 'done'})}\n{_PAYLOAD}"

    assert _FIRST.parse(stdout) == {"a": 1}


@pytest.mark.parametrize(
    ("adapter", "expected"),
    [
        pytest.param(_FIRST, {"first": 1}, id="first"),
        pytest.param(_LAST, {"second": 2}, id="last"),
    ],
)
def test_parse_when_several_objects_carry_the_key_does_pick_by_last_flag(
    adapter: JsonArrayAdapter, expected: dict[str, float]
):
    stdout = f"{_payload({'first': 1})}\n{_payload({'second': 2})}"

    assert adapter.parse(stdout) == expected


# ---------------------------------------------------------------------------
# decode failures
# ---------------------------------------------------------------------------


def test_parse_when_truncated_json_has_nested_object_does_report_decode_failure():
    # Truncated outer JSON — raw_decode at position 0 fails. The inner
    # {"name":"encode"} is valid JSON but carries no "items" key. The adapter
    # should prefer the decode failure (explaining WHY the real payload could not
    # parse) over the generic "JSON missing items array".
    truncated = '{"items":[{"name":"encode"}],"extra":'

    with pytest.raises(AdapterError, match=r"^Failed to parse JSON:"):
        _FIRST.parse(truncated)


def test_parse_when_pathological_nesting_does_raise_adapter_error_not_recursion_error():
    # A bare run of ``{`` is refused at the first key without descending, so the
    # depth comes from arrays under one key; one ``{`` keeps it to one attempt.
    stdout = '{"a":' + "[" * 500_000

    with pytest.raises(AdapterError, match=r"^Failed to parse JSON: Exceeded maximum recursion"):
        _FIRST.parse(stdout)


def test_parse_when_several_candidates_fail_does_report_the_first_failure():
    long_bad = '{"padding":"' + ("x" * 100) + '","bad":@}'
    stdout = f"{long_bad} noise {{!}}"
    with pytest.raises(json.JSONDecodeError) as decode_info:
        json.JSONDecoder().raw_decode(long_bad)

    with pytest.raises(AdapterError) as exc_info:
        _FIRST.parse(stdout)

    assert str(decode_info.value) in str(exc_info.value)


# ---------------------------------------------------------------------------
# no usable payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        pytest.param("not valid json at all", r"^No JSON object found", id="no-json-object"),
        pytest.param(json.dumps({"something": "else"}), r"missing items array", id="key-missing"),
        pytest.param(_payload(), r"^items array is empty", id="array-empty"),
        pytest.param(_payload("not-an-object"), r"^No items yielded$", id="no-entry-yields"),
    ],
)
def test_parse_when_stdout_has_no_usable_payload_does_raise_adapter_error(
    stdout: str, message: str
):
    with pytest.raises(AdapterError, match=message):
        _FIRST.parse(stdout)


# ---------------------------------------------------------------------------
# accepts_metric_prefix
# ---------------------------------------------------------------------------


def test_accepts_metric_prefix_when_prefix_is_clean_does_accept_without_warning():
    warnings: list[str] = []

    accepted = accepts_metric_prefix("a/b", warnings.append, _WORDING)

    assert (accepted, warnings) == (True, [])


def test_accepts_metric_prefix_when_prefix_contains_hash_does_raise_on_one_line():
    # The line terminator proves the '#' check runs first and escapes the prefix.
    prefix = "a#b\u2028c"
    warnings: list[str] = []

    with pytest.raises(AdapterError) as exc_info:
        accepts_metric_prefix(prefix, warnings.append, _WORDING)

    message = str(exc_info.value)
    assert message.startswith(f"{_WORDING.label} {json.dumps(prefix)}")
    assert message.endswith(_WORDING.hash_note)


@pytest.mark.parametrize(
    ("prefix", "named"),
    [
        pytest.param("a\u2028b", '"origin" (terminator note)', id="line-terminator"),
        pytest.param("a//b", '"a//b"', id="empty-path-segment"),
    ],
)
def test_accepts_metric_prefix_when_prefix_unusable_does_reject_with_one_warning(
    prefix: str, named: str
):
    warnings: list[str] = []

    accepted = accepts_metric_prefix(prefix, warnings.append, _WORDING)

    assert (accepted, len(warnings)) == (False, 1)
    assert named in warnings[0]
