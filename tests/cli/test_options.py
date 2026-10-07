"""Tests for the CLI flag parsers and option declarations.

These cover the positional grammar, the integer and decimal flag coercers, and
the ``--fail-on`` condition parser, both called directly and through the
commands that declare them.
"""

from pathlib import Path

import pytest
import typer

from gymrat.cli.app import app
from gymrat.cli.commands.supervise import parse_max_minutes, parse_positive_number
from gymrat.cli.options import parse_fail_on, parse_positional
from gymrat.config import MAX_SAFE_INTEGER, MAX_TIMEOUT_SECONDS
from gymrat.report.types import GeomeanFailOn, RegressedFailOn
from gymrat.sampling import TargetSpec
from tests._rich import unwrap_panel
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


# ---------------------------------------------------------------------------
# --samples / --timeout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        pytest.param("--samples", str(MAX_SAFE_INTEGER), MAX_SAFE_INTEGER, id="samples-ceiling"),
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
        pytest.param("0x10", id="hex"),
        pytest.param("1e-9", id="scientific"),
        pytest.param("1" + "0" * 310, id="overflows-to-infinity"),
        pytest.param("\N{ARABIC-INDIC DIGIT THREE}", id="arabic-indic-digit"),
        pytest.param("\N{DEVANAGARI DIGIT THREE}.5", id="devanagari-digit"),
        pytest.param("1.\N{DEVANAGARI DIGIT THREE}", id="non-ascii-fraction-digit"),
    ],
)
def test_parse_positive_number_when_non_positive_or_malformed_does_reject(value: str):
    with pytest.raises(typer.BadParameter) as exc:
        parse_positive_number(value)

    assert exc.value.message == _POSITIVE_NUMBER_MESSAGE


def test_parse_max_minutes_when_within_ceiling_does_accept():
    assert parse_max_minutes("10") == 10.0


def test_parse_max_minutes_when_non_positive_does_reject_before_bounding():
    with pytest.raises(typer.BadParameter) as exc:
        parse_max_minutes("0")

    assert exc.value.message == _POSITIVE_NUMBER_MESSAGE


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
        "geomean:0x10",
        "unknown",
        "",
        " geomean:2",
        pytest.param("geomean:\N{ARABIC-INDIC DIGIT THREE}", id="arabic-indic-digit"),
        pytest.param("geomean:-1.\N{DEVANAGARI DIGIT THREE}", id="devanagari-fraction-digit"),
    ],
)
def test_parse_fail_on_when_not_regressed_or_geomean_does_reject(value: str):
    with pytest.raises(typer.BadParameter) as exc:
        parse_fail_on(value)

    assert (
        exc.value.message
        == 'allowed values are "regressed" or "geomean:<number>" (e.g. geomean:2).'
    )


# ---------------------------------------------------------------------------
# invalid positionals and flags through the CLI
# ---------------------------------------------------------------------------


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
        pytest.param(
            ["measure", "--bench", "sh bench.sh", "--samples", "+5"],
            "must be a positive integer.",
            id="samples-signed",
        ),
        pytest.param(
            ["measure", "--bench", "sh bench.sh", "--samples", "0"],
            "must be a positive integer.",
            id="samples-zero",
        ),
        pytest.param(
            ["measure", "--bench", "sh bench.sh", "--samples", str(MAX_SAFE_INTEGER + 1)],
            f"must be at most {MAX_SAFE_INTEGER}.",
            id="samples-above-ceiling",
        ),
        pytest.param(
            ["measure", "--bench", "sh bench.sh", "--timeout", str(MAX_TIMEOUT_SECONDS + 1)],
            f"must be at most {MAX_TIMEOUT_SECONDS}.",
            id="timeout-above-ceiling",
        ),
        pytest.param(
            ["measure", "--bench", "sh bench.sh", "--samples", "1" * 4301],
            f"must be at most {MAX_SAFE_INTEGER}.",
            id="samples-past-int-conversion-limit",
        ),
        pytest.param(
            ["compare", "main", "cand", "--bench", "sh bench.sh", "--fail-on", "banana"],
            'allowed values are "regressed" or "geomean:<number>" (e.g. geomean:2).',
            id="fail-on",
        ),
        pytest.param(
            ["supervise", "optimize", "--max-minutes", "35792"],
            "must be at most 35791 minutes.",
            id="max-minutes",
        ),
        pytest.param(
            ["supervise", "optimize", "--max-minutes", "1", "--max-usd", "0"],
            _POSITIVE_NUMBER_MESSAGE,
            id="max-usd",
        ),
    ],
)
def test_cli_option_when_invalid_does_exit_two_with_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], message: str
):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, argv)

    assert result.exit_code == 2
    assert message in unwrap_panel(result.stderr)
