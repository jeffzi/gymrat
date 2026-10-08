import pytest

from gymrat.adapters import (
    Adapter,
    AdapterError,
    MetricDefaults,
    defaults_from_suffixes,
    get_adapter,
    metric_lines_adapter,
    mitata_adapter,
)
from gymrat.errors import GymratError
from tests.adapters._inputs import LINE_BREAKS, VALID_ADAPTERS_HINT, build_stdout

# ---------------------------------------------------------------------------
# get_adapter — registered names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "singleton"),
    [
        pytest.param("metric-lines", metric_lines_adapter, id="metric-lines"),
        pytest.param("mitata", mitata_adapter, id="mitata"),
    ],
)
def test_get_adapter_when_name_registered_does_return_matching_singleton(
    name: str, singleton: Adapter
):
    adapter = get_adapter(name)

    assert adapter is singleton
    assert adapter.name == name


# ---------------------------------------------------------------------------
# get_adapter — unknown name
# ---------------------------------------------------------------------------


def test_get_adapter_when_name_unknown_does_raise_gymrat_error_describing_valid_adapters():
    with pytest.raises(GymratError, match=r"Unknown adapter") as excinfo:
        get_adapter("unknown")

    error = excinfo.value
    assert type(error) is GymratError
    assert str(error) == 'Unknown adapter: "unknown".'
    assert error.hint == VALID_ADAPTERS_HINT


# ---------------------------------------------------------------------------
# AdapterError
# ---------------------------------------------------------------------------


def test_adapter_error_when_raised_does_subclass_gymrat_error():
    error = AdapterError("unparseable output")

    with pytest.raises(GymratError):
        raise error


# ---------------------------------------------------------------------------
# defaults_from_suffixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        pytest.param(
            "bench#time",
            MetricDefaults(direction="lower", unit="ns", kind="time", short_name="bench"),
            id="time-suffix",
        ),
        pytest.param(
            "a/b#time",
            MetricDefaults(direction="lower", unit="ns", kind="time", short_name="a/b"),
            id="time-suffix-keeps-path",
        ),
        pytest.param(
            "bench#heap",
            MetricDefaults(direction="lower", unit="bytes", kind="memory", short_name="bench"),
            id="heap-suffix",
        ),
        pytest.param("foo", MetricDefaults(direction="lower"), id="no-suffix"),
    ],
)
def test_defaults_from_suffixes_when_given_metric_name_does_return_expected_defaults(
    metric_name: str,
    expected: MetricDefaults,
):
    assert defaults_from_suffixes(metric_name) == expected


# ---------------------------------------------------------------------------
# basic parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        pytest.param("METRIC foo=42", {"foo": 42.0}, id="integer"),
        pytest.param("METRIC bar=3.14", {"bar": 3.14}, id="decimal"),
        pytest.param("  METRIC foo=42  ", {"foo": 42.0}, id="both-whitespace"),
        pytest.param("METRIC bench#time=42", {"bench#time": 42.0}, id="single-hash"),
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
def test_parse_when_name_contains_equals_does_split_at_last_equals(
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


@pytest.mark.parametrize(
    "stdout",
    [
        pytest.param("some other output\nMETRIC valid=1\nother log line", id="surrounding-logs"),
        pytest.param("metric foo=42\nMetric bar=3.14\nMETRIC valid=1", id="case-sensitive"),
        pytest.param("50%\rMETRIC valid=1", id="progress-carriage-return"),
    ],
)
def test_parse_when_non_metric_lines_present_does_ignore_them_silently(stdout: str):
    warnings: list[str] = []

    result = metric_lines_adapter.parse(stdout, warnings.append)

    assert result == {"valid": 1.0}
    assert warnings == []


# ---------------------------------------------------------------------------
# malformed or near-miss METRIC warnings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offending",
    [
        pytest.param("METRICfoo=42", id="no-space"),
        pytest.param("METRICS foo=1", id="longer-word"),
        pytest.param("METRIC_foo=1", id="prefix-underscore"),
        pytest.param("METRIC foo", id="no-value"),
        pytest.param("METRIC =5", id="empty-name"),
        pytest.param("METRIC foo=bar", id="non-numeric"),
        pytest.param("METRIC foo=", id="empty-value"),
        pytest.param("METRIC foo=NaN", id="nan"),
        pytest.param("METRIC foo=Infinity", id="infinity"),
        pytest.param("METRIC foo=-Infinity", id="negative-infinity"),
        pytest.param("METRIC u=1_0", id="underscore-separator"),
        pytest.param("METRIC h=0x10z", id="radix-trailing-junk"),
        pytest.param("METRIC h=-0x10", id="signed-radix"),
        # A radix literal matches the number grammar but names an integer too large
        # to convert to a float; the conversion overflows and the line is skipped.
        pytest.param(f"METRIC big=0x{'f' * 300}", id="radix-overflows-float"),
        # LF and CR are excluded: the adapter splits on them before it checks a name.
        *(
            pytest.param(f"METRIC na{line_break.char}me=42", id=f"name-holds-{line_break.name}")
            for line_break in LINE_BREAKS
            if line_break.char not in "\n\r"
        ),
    ],
)
def test_parse_when_metric_line_malformed_does_warn_and_skip(offending: str):
    warnings: list[str] = []

    result = metric_lines_adapter.parse(f"{offending}\nMETRIC valid=1", warnings.append)

    assert result == {"valid": 1.0}
    assert warnings == [f"Failed to parse METRIC line: {offending}"]


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
# here one name per rule proves the adapter words the warning from that rule.
@pytest.mark.parametrize(
    ("name", "problem"),
    [
        pytest.param("a//b", "an empty path segment", id="empty-path-segment"),
        pytest.param("foo#", "an empty kind", id="empty-kind"),
    ],
)
def test_parse_when_name_has_empty_part_does_warn_and_skip(name: str, problem: str):
    warnings: list[str] = []

    result = metric_lines_adapter.parse(f"METRIC {name}=42\nMETRIC valid=1", warnings.append)

    assert result == {"valid": 1.0}
    assert warnings == [f'Skipping METRIC line with {problem} in its metric name: "{name}"']


# ---------------------------------------------------------------------------
# warn routing
# ---------------------------------------------------------------------------


_ROUTED_WARNINGS = [
    pytest.param(
        metric_lines_adapter,
        "METRIC foo=bar\nMETRIC valid=1",
        "Failed to parse METRIC line: METRIC foo=bar",
        id="metric-lines",
    ),
    pytest.param(
        mitata_adapter,
        build_stdout([None, {"alias": "valid", "runs": [{"args": {}, "stats": {"p50": 1}}]}]),
        "Skipping benchmark: expected an object, got null",
        id="mitata",
    ),
]


@pytest.mark.parametrize(("adapter", "stdout", "warning"), _ROUTED_WARNINGS)
def test_parse_when_sink_injected_does_route_warning_and_leave_stderr_empty(
    adapter: Adapter, stdout: str, warning: str, capsys: pytest.CaptureFixture[str]
):
    warnings: list[str] = []

    adapter.parse(stdout, warnings.append)

    assert warnings == [warning]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(("adapter", "stdout", "warning"), _ROUTED_WARNINGS)
def test_parse_when_no_sink_given_does_warn_to_stderr(
    adapter: Adapter, stdout: str, warning: str, capsys: pytest.CaptureFixture[str]
):
    adapter.parse(stdout)

    assert capsys.readouterr().err == f"{warning}\n"


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
    assert warnings == [
        (
            'Parsed metric name "METRIC foo" embeds the METRIC token '
            "— the line may carry a duplicate METRIC prefix"
        )
    ]
