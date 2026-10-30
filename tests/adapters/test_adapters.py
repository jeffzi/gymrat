import pytest

from gymrat.adapters.contract import (
    MEMORY_SUFFIX,
    AdapterError,
    MetricDefaults,
    defaults_from_suffixes,
)
from gymrat.adapters.metric_lines import metric_lines_adapter
from gymrat.adapters.registry import get_adapter
from gymrat.errors import GymratError
from gymrat.metric_name import EmptyPathSegmentError

# ---------------------------------------------------------------------------
# get_adapter — unknown name
# ---------------------------------------------------------------------------


def test_get_adapter_when_name_unknown_does_raise_gymrat_error_describing_valid_adapters():
    with pytest.raises(GymratError) as excinfo:
        get_adapter("unknown")

    error = excinfo.value
    # An adapter error would gain a class-name prefix on the CLI; this one names no adapter fault.
    assert not isinstance(error, AdapterError)
    assert '"unknown"' in str(error)
    assert all(name in (error.hint or "") for name in ("metric-lines", "mitata", "tinybench"))


# ---------------------------------------------------------------------------
# defaults_from_suffixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        pytest.param(
            f"bench{MEMORY_SUFFIX}",
            MetricDefaults(
                direction="lower", unit="bytes", kind="memory", short_name="bench", resolution=1.0
            ),
            id="memory-suffix",
        ),
        pytest.param("foo", MetricDefaults(direction="lower"), id="no-suffix"),
    ],
)
def test_defaults_from_suffixes_when_kind_suffix_varies_does_derive_defaults_from_it(
    metric_name: str, expected: MetricDefaults
):
    assert defaults_from_suffixes(metric_name) == expected


# ---------------------------------------------------------------------------
# basic parsing
# ---------------------------------------------------------------------------


def test_parse_when_single_metric_line_does_return_named_value():
    assert metric_lines_adapter.parse("  METRIC foo=42  ") == {"foo": 42.0}


# ---------------------------------------------------------------------------
# line terminators
# ---------------------------------------------------------------------------


def test_parse_when_lines_end_on_bare_carriage_return_does_split_into_metrics():
    assert metric_lines_adapter.parse("METRIC foo=42\rMETRIC bar=1") == {"foo": 42.0, "bar": 1.0}


# ---------------------------------------------------------------------------
# split on the last equals sign
# ---------------------------------------------------------------------------


def test_parse_when_name_contains_equals_does_split_at_last_equals():
    assert metric_lines_adapter.parse("METRIC k=v=3.14") == {"k=v": 3.14}


# ---------------------------------------------------------------------------
# number grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        pytest.param("METRIC val=0x10", 16.0, id="hex"),
        pytest.param("METRIC val=0o17", 15.0, id="octal"),
        pytest.param("METRIC val=0b101", 5.0, id="binary"),
    ],
)
def test_parse_when_value_matches_js_number_grammar_does_convert(stdout: str, expected: float):
    assert metric_lines_adapter.parse(stdout) == {"val": expected}


# ---------------------------------------------------------------------------
# ignore non-matching lines (silent)
# ---------------------------------------------------------------------------


def test_parse_when_non_metric_lines_present_does_ignore_them_silently():
    warnings: list[str] = []

    result = metric_lines_adapter.parse(
        "some other output\nMETRIC valid=1\nother log line", warnings.append
    )

    assert result == {"valid": 1.0}
    assert warnings == []


# ---------------------------------------------------------------------------
# malformed or near-miss METRIC warnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offending",
    [
        pytest.param("METRICfoo=42", id="no-space"),
        pytest.param("METRIC 42", id="no-value"),
        pytest.param("METRIC =5", id="empty-name"),
        pytest.param("METRIC foo=bar", id="non-numeric"),
        pytest.param("METRIC foo=Infinity", id="infinity"),
        pytest.param("METRIC u=1_0", id="underscore-separator"),
        pytest.param("METRIC h=0x10z", id="radix-trailing-junk"),
        pytest.param("METRIC h=-0x10", id="signed-radix"),
        # JavaScript Number() reads only ASCII digits. Full-width digits are the case
        # because Unicode normalization (NFKC) folds them to ASCII, which would let
        # them through.
        pytest.param("METRIC foo=\uff11\uff12", id="full-width-digits"),
        # A radix literal matches the number grammar but names an integer too large
        # to convert to a float; the conversion overflows and the line is skipped.
        pytest.param(f"METRIC big=0x{'f' * 300}", id="radix-overflows-float"),
        # One non-LF/CR terminator proves the adapter keeps the line whole and rejects
        # the name; which characters count as terminators is pinned in
        # tests/utils/test_utils.py (LINE_TERMINATORS).
        pytest.param("METRIC na\u2028me=42", id="name-holds-line-separator"),
    ],
)
def test_parse_when_metric_line_malformed_does_skip_line_with_warning(
    offending: str, capsys: pytest.CaptureFixture[str]
):
    warnings: list[str] = []

    result = metric_lines_adapter.parse(f"{offending}\nMETRIC valid=1", warnings.append)

    assert (result, len(warnings), capsys.readouterr().err) == ({"valid": 1.0}, 1, "")
    assert offending in warnings[0]


# ---------------------------------------------------------------------------
# multi-'#' in name is a hard error
# ---------------------------------------------------------------------------


def test_parse_when_name_contains_multiple_hashes_does_raise_adapter_error():
    with pytest.raises(AdapterError, match="bad#name#extra"):
        metric_lines_adapter.parse("METRIC bad#name#extra=42")


# ---------------------------------------------------------------------------
# empty path segment or empty kind in name
# ---------------------------------------------------------------------------


# Which name shape breaks which grammar rule is pinned in tests/test_metric_name.py;
# here one name proves the adapter inserts the rule's flaw into the warning.
def test_parse_when_name_has_empty_part_does_skip_line_with_warning():
    name = "a//b"
    warnings: list[str] = []

    result = metric_lines_adapter.parse(f"METRIC {name}=42\nMETRIC valid=1", warnings.append)

    assert result == {"valid": 1.0}
    assert len(warnings) == 1
    assert EmptyPathSegmentError.flaw in warnings[0]
    assert f'"{name}"' in warnings[0]


# ---------------------------------------------------------------------------
# repeated metric name → median
# ---------------------------------------------------------------------------


def test_parse_when_name_repeats_does_return_median_per_name():
    result = metric_lines_adapter.parse("METRIC x=1\nMETRIC x=3")

    assert result == {"x": 2.0}


# ---------------------------------------------------------------------------
# zero metrics → AdapterError
# ---------------------------------------------------------------------------


def test_parse_when_no_valid_metrics_does_raise_adapter_error():
    warnings: list[str] = []

    with pytest.raises(AdapterError):
        metric_lines_adapter.parse("some output\nwith no metrics", warnings.append)


# ---------------------------------------------------------------------------
# embedded METRIC token warning
# ---------------------------------------------------------------------------


def test_parse_when_name_embeds_metric_token_does_record_metric_with_warning():
    warnings: list[str] = []

    result = metric_lines_adapter.parse("METRIC METRIC foo=42", warnings.append)

    assert result == {"METRIC foo": 42.0}
    assert len(warnings) == 1
    assert '"METRIC foo"' in warnings[0]
