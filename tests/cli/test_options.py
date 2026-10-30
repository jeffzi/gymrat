"""Tests for the CLI flag parsers and option declarations.

These cover the positional grammar, the integer and decimal flag coercers, and
the ``--fail-on`` condition parser, both called directly and through the
commands that declare them.
"""

import pytest
import typer

from gymrat.cli.app import app
from gymrat.cli.commands.supervise import parse_max_minutes, parse_positive_number
from gymrat.cli.options import parse_fail_on, parse_positional
from gymrat.config import MAX_SAFE_INTEGER
from gymrat.report.types import GeomeanFailOn, RegressedFailOn
from gymrat.sampling import TargetSpec
from tests._rich import unwrap_panel
from tests.cli._runner import (
    runner,
)

# ---------------------------------------------------------------------------
# parse_positional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("positional", "expected"),
    [
        pytest.param("a=b=c", TargetSpec(label="a", target="b=c"), id="first-equals-splits"),
        pytest.param("HEAD", TargetSpec(label=None, target="HEAD"), id="no-equals"),
    ],
)
def test_parse_positional_when_equals_present_or_absent_does_split_on_the_first_equals(
    positional: str, expected: TargetSpec
):
    assert parse_positional(positional) == expected


@pytest.mark.parametrize(
    "positional",
    [pytest.param("=HEAD", id="empty-label"), pytest.param("HEAD=", id="empty-target")],
)
def test_parse_positional_when_a_side_of_the_equals_is_empty_does_reject(positional: str):
    with pytest.raises(typer.BadParameter):
        parse_positional(positional)


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
        "",
        "1.",
        pytest.param("1e-9", id="scientific"),
        pytest.param("1" + "0" * 310, id="overflows-to-infinity"),
        pytest.param("\N{ARABIC-INDIC DIGIT THREE}", id="arabic-indic-digit"),
        pytest.param("1.\N{DEVANAGARI DIGIT THREE}", id="non-ascii-fraction-digit"),
    ],
)
def test_parse_positive_number_when_non_positive_or_malformed_does_reject(value: str):
    with pytest.raises(typer.BadParameter):
        parse_positive_number(value)


def test_parse_max_minutes_when_within_ceiling_does_accept():
    assert parse_max_minutes("10") == 10.0


# ---------------------------------------------------------------------------
# parse_fail_on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("regressed", RegressedFailOn(), id="regressed"),
        pytest.param("geomean:2", GeomeanFailOn(pct=2.0), id="geomean-integer"),
        pytest.param("geomean:-1.5", GeomeanFailOn(pct=-1.5), id="geomean-negative-decimal"),
    ],
)
def test_parse_fail_on_when_regressed_or_geomean_percentage_does_accept(
    value: str, expected: RegressedFailOn | GeomeanFailOn
):
    assert parse_fail_on(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "geomean:",
        " geomean:2",
    ],
)
def test_parse_fail_on_when_not_regressed_or_geomean_does_reject(value: str):
    with pytest.raises(typer.BadParameter):
        parse_fail_on(value)


# ---------------------------------------------------------------------------
# invalid flags through the CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "named"),
    [
        pytest.param(
            ["measure", "--bench", "sh bench.sh", "--samples", str(MAX_SAFE_INTEGER + 1)],
            str(MAX_SAFE_INTEGER),
            id="samples-above-ceiling",
        ),
        pytest.param(
            ["supervise", "optimize", "--max-minutes", "35792"], "35791", id="max-minutes"
        ),
    ],
)
@pytest.mark.usefixtures("_in_non_repo")
def test_cli_option_when_invalid_does_exit_two_naming_the_option_or_bound(
    argv: list[str], named: str
):
    result = runner.invoke(app, argv)

    assert result.exit_code == 2
    assert named in unwrap_panel(result.stderr)
