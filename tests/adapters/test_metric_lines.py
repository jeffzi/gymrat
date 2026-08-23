import pytest

from gymrat_py.adapters.metric_lines import metric_lines_adapter
from gymrat_py.adapters.types import Adapter, AdapterError, MetricDefaults

# ---------------------------------------------------------------------------
# adapter shape
# ---------------------------------------------------------------------------


def test_metric_lines_adapter_when_inspected_does_expose_name():
    assert metric_lines_adapter.name == "metric-lines"


def test_metric_lines_adapter_when_checked_does_satisfy_adapter_protocol():
    assert isinstance(metric_lines_adapter, Adapter)


# ---------------------------------------------------------------------------
# basic parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        pytest.param("METRIC foo=42", {"foo": 42.0}, id="integer"),
        pytest.param("METRIC bar=3.14", {"bar": 3.14}, id="decimal"),
        pytest.param("  METRIC foo=42", {"foo": 42.0}, id="leading-whitespace"),
        pytest.param("METRIC foo=42  ", {"foo": 42.0}, id="trailing-whitespace"),
        pytest.param("  METRIC foo=42  ", {"foo": 42.0}, id="both-whitespace"),
    ],
)
def test_parse_when_single_metric_line_does_return_named_value(
    stdout: str, expected: dict[str, float]
):
    assert metric_lines_adapter.parse(stdout) == expected


# ---------------------------------------------------------------------------
# line terminators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("METRIC foo=42\nMETRIC bar=1", id="line-feed"),
        pytest.param("METRIC foo=42\r\nMETRIC bar=1", id="crlf"),
        pytest.param("METRIC foo=42\rMETRIC bar=1", id="bare-cr"),
        pytest.param("METRIC foo=42\r\nMETRIC bar=1\r", id="mixed"),
    ],
)
def test_parse_when_lines_end_on_terminator_does_split_into_metrics(stdout: str):
    assert metric_lines_adapter.parse(stdout) == {"foo": 42.0, "bar": 1.0}


def test_parse_when_metric_follows_progress_carriage_return_does_find_it():
    assert metric_lines_adapter.parse("50%\rMETRIC foo=42\n") == {"foo": 42.0}


def test_parse_when_name_contains_bare_carriage_return_does_skip_and_warn():
    warnings: list[str] = []

    result = metric_lines_adapter.parse("METRIC fo\ro=42\nMETRIC valid=1", warnings.append)

    assert result == {"valid": 1.0}
    assert warnings == ["Failed to parse METRIC line: METRIC fo"]


# ---------------------------------------------------------------------------
# split on the last equals sign
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        pytest.param("METRIC k=v=3.14", {"k=v": 3.14}, id="two-equals"),
        pytest.param("METRIC a=b=c=d=5", {"a=b=c=d": 5.0}, id="multiple-equals"),
    ],
)
def test_parse_when_value_contains_equals_does_split_at_last_equals(
    stdout: str, expected: dict[str, float]
):
    assert metric_lines_adapter.parse(stdout) == expected


# ---------------------------------------------------------------------------
# number grammar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        pytest.param("METRIC val=-12", -12.0, id="negative-integer"),
        pytest.param("METRIC val=-3.14", -3.14, id="negative-decimal"),
        pytest.param("METRIC val=+5", 5.0, id="positive-sign"),
        pytest.param("METRIC val=+5.0", 5.0, id="positive-decimal-sign"),
        pytest.param("METRIC val=0.001", 0.001, id="small-decimal"),
        pytest.param("METRIC val=1e-9", 1e-9, id="sci-negative-exp"),
        pytest.param("METRIC val=1e9", 1e9, id="sci-positive-exp"),
        pytest.param("METRIC val=1E-9", 1e-9, id="sci-uppercase-e"),
        pytest.param("METRIC h=0x10", 16.0, id="hex"),
        pytest.param("METRIC o=0o17", 15.0, id="octal"),
        pytest.param("METRIC b=0b101", 5.0, id="binary"),
    ],
)
def test_parse_when_value_matches_js_number_grammar_does_convert(stdout: str, expected: float):
    key = stdout.split("=", maxsplit=1)[0].removeprefix("METRIC ")
    assert metric_lines_adapter.parse(stdout) == {key: expected}


# ---------------------------------------------------------------------------
# ignore non-matching lines (silent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("some other output\nMETRIC valid=1\nother log line", id="surrounding-logs"),
        pytest.param("metric foo=42\nMetric bar=3.14\nMETRIC valid=1", id="case-sensitive"),
    ],
)
def test_parse_when_non_metric_lines_present_does_ignore_them(stdout: str):
    assert metric_lines_adapter.parse(stdout) == {"valid": 1.0}


@pytest.mark.parametrize(
    "unrelated",
    [
        pytest.param("some other output", id="log-line"),
        pytest.param("Starting benchmark...", id="startup-line"),
        pytest.param("metric foo=42", id="lowercase-metric"),
    ],
)
def test_parse_when_line_does_not_start_with_metric_does_stay_silent(unrelated: str):
    warnings: list[str] = []

    metric_lines_adapter.parse(f"{unrelated}\nMETRIC valid=1", warnings.append)

    assert warnings == []


# ---------------------------------------------------------------------------
# near-miss METRIC prefix warnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offending",
    [
        pytest.param("METRICfoo=42", id="no-space"),
        pytest.param("METRICS foo=1", id="longer-word"),
        pytest.param("METRIC_foo=1", id="underscore-separator"),
    ],
)
def test_parse_when_near_miss_prefix_does_warn_and_record_nothing(offending: str):
    warnings: list[str] = []

    result = metric_lines_adapter.parse(f"{offending}\nMETRIC valid=1", warnings.append)

    assert result == {"valid": 1.0}
    assert warnings == [f"Failed to parse METRIC line: {offending}"]


# ---------------------------------------------------------------------------
# malformed METRIC warnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offending",
    [
        pytest.param("METRIC foo", id="no-value"),
        pytest.param("METRIC =5", id="empty-name"),
        pytest.param("METRIC foo=bar", id="non-numeric"),
        pytest.param("METRIC foo=", id="empty-value"),
        pytest.param("METRIC foo=NaN", id="nan"),
        pytest.param("METRIC foo=Infinity", id="infinity"),
        pytest.param("METRIC foo=-Infinity", id="negative-infinity"),
        pytest.param("METRIC u=1_0", id="underscore-separator"),
    ],
)
def test_parse_when_metric_value_malformed_does_warn_and_skip(offending: str):
    warnings: list[str] = []

    result = metric_lines_adapter.parse(f"{offending}\nMETRIC valid=1", warnings.append)

    assert result == {"valid": 1.0}
    assert f"Failed to parse METRIC line: {offending}" in warnings


def test_parse_when_radix_value_overflows_float_does_warn_and_skip():
    # A radix literal matches the number grammar but names an integer too large
    # to convert to a float; the conversion overflows and the line is skipped.
    offending = f"METRIC big=0x{'f' * 300}"
    warnings: list[str] = []

    result = metric_lines_adapter.parse(f"{offending}\nMETRIC valid=1", warnings.append)

    assert result == {"valid": 1.0}
    assert f"Failed to parse METRIC line: {offending}" in warnings


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("METRIC x=1\nMETRIC x=\nMETRIC x=3", id="empty-value"),
        pytest.param("METRIC x=1\nMETRIC x=   \nMETRIC x=3", id="whitespace-value"),
    ],
)
def test_parse_when_value_empty_does_exclude_sample_rather_than_read_zero(stdout: str):
    warnings: list[str] = []

    result = metric_lines_adapter.parse(stdout, warnings.append)

    assert result == {"x": 2.0}


def test_parse_when_malformed_line_precedes_valid_does_continue_parsing():
    warnings: list[str] = []

    result = metric_lines_adapter.parse("METRIC foo=bar\nMETRIC valid=42", warnings.append)

    assert result == {"valid": 42.0}


# ---------------------------------------------------------------------------
# names carrying a JSON-illegal line separator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code_point",
    [
        pytest.param(0x2028, id="line-separator-u2028"),
        pytest.param(0x2029, id="paragraph-separator-u2029"),
    ],
)
def test_parse_when_name_holds_line_separator_does_warn_and_skip(code_point: int):
    offending = f"METRIC na{chr(code_point)}me=42"
    warnings: list[str] = []

    result = metric_lines_adapter.parse(f"{offending}\nMETRIC valid=1", warnings.append)

    assert result == {"valid": 1.0}
    assert f"Failed to parse METRIC line: {offending}" in warnings


# ---------------------------------------------------------------------------
# warn routing
# ---------------------------------------------------------------------------


def test_parse_when_sink_injected_does_route_warning_and_leave_stderr_empty(
    capsys: pytest.CaptureFixture[str],
):
    warnings: list[str] = []

    metric_lines_adapter.parse("METRIC foo=bar\nMETRIC valid=1", warnings.append)

    assert warnings == ["Failed to parse METRIC line: METRIC foo=bar"]
    assert capsys.readouterr().err == ""


def test_parse_when_no_sink_given_does_warn_to_stderr(capsys: pytest.CaptureFixture[str]):
    metric_lines_adapter.parse("METRIC foo=bar\nMETRIC valid=1")

    assert "Failed to parse METRIC line: METRIC foo=bar" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# repeated metric name → median
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        pytest.param("METRIC x=1\nMETRIC x=3\nMETRIC x=2", {"x": 2.0}, id="odd-count"),
        pytest.param("METRIC x=1\nMETRIC x=2\nMETRIC x=3\nMETRIC x=4", {"x": 2.5}, id="even-count"),
        pytest.param(
            "METRIC x=1\nMETRIC x=3\nMETRIC y=10\nMETRIC y=20\nMETRIC y=30",
            {"x": 2.0, "y": 20.0},
            id="per-name",
        ),
    ],
)
def test_parse_when_name_repeats_does_return_median_per_name(
    stdout: str, expected: dict[str, float]
):
    assert metric_lines_adapter.parse(stdout) == expected


# ---------------------------------------------------------------------------
# ordinary names / no prototype pollution
# ---------------------------------------------------------------------------


def test_parse_when_never_emitted_does_leave_inherited_name_absent():
    result = metric_lines_adapter.parse("METRIC foo=42")

    assert result["foo"] == 42.0
    assert "toString" not in result


def test_parse_when_name_is_dunder_proto_does_treat_as_ordinary_metric():
    result = metric_lines_adapter.parse("METRIC __proto__=1\nMETRIC __proto__=3")

    assert result == {"__proto__": 2.0}


# ---------------------------------------------------------------------------
# zero metrics → AdapterError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("some output\nwith no metrics", id="no-metric-lines"),
        pytest.param("", id="empty-string"),
        pytest.param("METRIC foo\nMETRIC bar=baz", id="only-malformed"),
    ],
)
def test_parse_when_no_valid_metrics_does_raise_adapter_error(stdout: str):
    warnings: list[str] = []

    with pytest.raises(AdapterError, match=r"^No valid METRIC lines found$"):
        metric_lines_adapter.parse(stdout, warnings.append)


# ---------------------------------------------------------------------------
# embedded METRIC token warning
# ---------------------------------------------------------------------------


def test_parse_when_name_embeds_metric_token_does_warn_but_record():
    warnings: list[str] = []

    result = metric_lines_adapter.parse("METRIC METRIC foo=42", warnings.append)

    assert result == {"METRIC foo": 42.0}
    assert len(warnings) == 1
    assert "METRIC " in warnings[0]


# ---------------------------------------------------------------------------
# defaults() delegation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        pytest.param("foo", MetricDefaults(direction="lower"), id="no-suffix"),
        pytest.param(
            "bench/time",
            MetricDefaults(direction="lower", unit="ns", kind="time", short_name="bench"),
            id="time-suffix",
        ),
        pytest.param(
            "bench/heap",
            MetricDefaults(direction="lower", unit="bytes", kind="memory", short_name="bench"),
            id="heap-suffix",
        ),
    ],
)
def test_defaults_when_given_metric_name_does_delegate_to_suffix_resolver(
    metric_name: str, expected: MetricDefaults
):
    assert metric_lines_adapter.defaults(metric_name) == expected
