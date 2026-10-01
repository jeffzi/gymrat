"""Tests for the CLI flag parsers and option declarations.

These cover the positional grammar, the integer and decimal flag coercers, and
the ``--fail-on`` condition parser, both called directly and through the
commands that declare them.
"""

from pathlib import Path

import pytest
import typer

from gymrat.cli.app import app
from gymrat.cli.options import (
    parse_fail_on,
    parse_max_minutes,
    parse_positional,
    parse_positive_number,
)
from gymrat.config import MAX_SAFE_INTEGER, MAX_TIMEOUT_SECONDS
from gymrat.report.types import GeomeanFailOn, RegressedFailOn
from gymrat.sampling import TargetSpec
from tests._rich import unwrap_panel
from tests.cli._help import help_output
from tests.cli._session import last_command_record, runner, stub_measure
from tests.session.records._fixtures import session_record, write_session_log

_EMPTY_TARGET_MESSAGE = 'the target is empty; write the positional as "[label=]<ref|dir>".'
_POSITIVE_NUMBER_MESSAGE = "must be a positive number."

# ---------------------------------------------------------------------------
# parse_positional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("positional", "expected"),
    [
        pytest.param("main=HEAD", TargetSpec(label="main", target="HEAD"), id="label-and-target"),
        pytest.param("a=b=c", TargetSpec(label="a", target="b=c"), id="first-equals-splits"),
        pytest.param("HEAD", TargetSpec(label=None, target="HEAD"), id="no-equals"),
    ],
)
def test_parse_positional_when_called_does_split_on_the_first_equals(
    positional: str, expected: TargetSpec
):
    assert parse_positional(positional) == expected


def test_parse_positional_when_target_empty_does_raise_dedicated_message():
    with pytest.raises(typer.BadParameter) as exc:
        parse_positional("")

    assert exc.value.message == _EMPTY_TARGET_MESSAGE


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        pytest.param(
            ["measure", "=HEAD"],
            'the label before "=" is empty; write the positional as "label=<ref|dir>" or drop the "=".',
            id="measure-empty-label",
        ),
        pytest.param(
            ["compare", "main", "HEAD="], _EMPTY_TARGET_MESSAGE, id="compare-empty-target"
        ),
    ],
)
def test_positional_argument_when_invalid_does_exit_two_with_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], message: str
):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, argv)
    flat = unwrap_panel(result.stderr)

    assert result.exit_code == 2
    assert message in flat


# ---------------------------------------------------------------------------
# --samples / --timeout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        pytest.param("--samples", "5", 5, id="samples-plain"),
        pytest.param("--samples", str(MAX_SAFE_INTEGER), MAX_SAFE_INTEGER, id="samples-ceiling"),
        pytest.param("--timeout", "42", 42, id="timeout-plain"),
        pytest.param(
            "--timeout", str(MAX_TIMEOUT_SECONDS), MAX_TIMEOUT_SECONDS, id="timeout-ceiling"
        ),
    ],
)
def test_integer_option_when_bare_digits_within_ceiling_does_accept(
    monkeypatch: pytest.MonkeyPatch, repo: str, flag: str, value: str, expected: int
):
    write_session_log(repo, session_record())
    stub_measure(monkeypatch)

    result = runner.invoke(app, ["measure", "main", "--bench", "sh bench.sh", flag, value])

    assert result.exit_code == 0
    assert last_command_record(repo).args[flag.removeprefix("--")] == expected


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        pytest.param("--samples", "+5", id="samples"),
        pytest.param("--timeout", "0", id="timeout"),
    ],
)
def test_integer_option_when_not_bare_positive_digits_does_exit_two_with_positive_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str, value: str
):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh", flag, value])

    assert result.exit_code == 2
    assert "must be a positive integer." in unwrap_panel(result.stderr)


@pytest.mark.parametrize(
    ("flag", "value", "ceiling"),
    [
        pytest.param("--samples", str(MAX_SAFE_INTEGER + 1), MAX_SAFE_INTEGER, id="samples"),
        pytest.param("--timeout", str(MAX_TIMEOUT_SECONDS + 1), MAX_TIMEOUT_SECONDS, id="timeout"),
        pytest.param("--samples", "1" * 4301, MAX_SAFE_INTEGER, id="past-int-conversion-limit"),
    ],
)
def test_integer_option_when_above_ceiling_does_exit_two_naming_the_ceiling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flag: str, value: str, ceiling: int
):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["measure", "--bench", "sh bench.sh", flag, value])

    assert result.exit_code == 2
    assert f"must be at most {ceiling}." in unwrap_panel(result.stderr)


@pytest.mark.parametrize("flag", ["--samples", "--timeout"])
def test_integer_option_when_help_does_show_int_metavar(flag: str):
    out = help_output("measure")

    flag_line = next(line for line in out.splitlines() if flag in line)
    assert "<int>" in flag_line


# ---------------------------------------------------------------------------
# parse_positive_number / parse_max_minutes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [("1", 1.0), ("2.5", 2.5)])
def test_parse_positive_number_when_positive_decimal_does_accept(value: str, expected: float):
    assert parse_positive_number(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-1",
        "abc",
        "",
        "1.",
        pytest.param("1" + "0" * 310, id="overflows-to-infinity"),
    ],
)
def test_parse_positive_number_when_non_positive_or_malformed_does_reject(value: str):
    with pytest.raises(typer.BadParameter) as exc:
        parse_positive_number(value)

    assert exc.value.message == _POSITIVE_NUMBER_MESSAGE


def test_parse_max_minutes_when_within_ceiling_does_accept():
    assert parse_max_minutes("10") == 10.0


def test_parse_max_minutes_when_over_ceiling_does_reject():
    with pytest.raises(typer.BadParameter) as exc:
        parse_max_minutes("35792")

    assert exc.value.message == "must be at most 35791 minutes."


def test_parse_max_minutes_when_non_positive_does_reject_before_bounding():
    with pytest.raises(typer.BadParameter) as exc:
        parse_max_minutes("0")

    assert exc.value.message == _POSITIVE_NUMBER_MESSAGE


# ---------------------------------------------------------------------------
# parse_fail_on
# ---------------------------------------------------------------------------


def test_parse_fail_on_when_regressed_does_accept():
    assert parse_fail_on("regressed") == RegressedFailOn()


@pytest.mark.parametrize(
    ("value", "expected_pct"),
    [
        pytest.param("geomean:2", 2.0, id="integer"),
        pytest.param("geomean:-1.5", -1.5, id="negative-decimal"),
    ],
)
def test_parse_fail_on_when_geomean_percentage_does_accept(value: str, expected_pct: float):
    condition = parse_fail_on(value)

    assert condition == GeomeanFailOn(pct=expected_pct)


@pytest.mark.parametrize(
    "value",
    ["geomean:", "geomean:0x10", "unknown", "", " geomean:2"],
)
def test_parse_fail_on_when_not_regressed_or_geomean_does_reject(value: str):
    with pytest.raises(typer.BadParameter) as exc:
        parse_fail_on(value)

    assert (
        exc.value.message
        == 'allowed values are "regressed" or "geomean:<number>" (e.g. geomean:2).'
    )
