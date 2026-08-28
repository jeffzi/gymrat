"""Tests for the root ``gymrat`` typer app: version, help, debug, dispatch.

These drive the assembled CLI through :class:`typer.testing.CliRunner`, so the
root callback, the ``--version`` eager option, the shared ``--debug`` flag in
both positions, the root epilogue, and unknown-command routing are exercised the
way a shell would invoke them.
"""

import importlib.metadata
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat_py.cli.app import app
from gymrat_py.cli.shared import BUGS_URL
from tests._ansi import SGR_RE, strip_ansi
from tests.report._inputs import create_measurement_result

runner = CliRunner()

DOCS_URL = "https://github.com/jeffzi/gymrat#readme"
"""The documentation link the root epilogue points at."""


def _normalize(text: str) -> str:
    """Strip ANSI codes then collapse whitespace, so a reflowed help block matches."""
    return " ".join(strip_ansi(text).split())


def _sgr_params(text: str) -> str:
    """The SGR parameter list of the last ANSI escape in ``text`` (e.g. ``"2;34"``).

    Asserts which attributes a styled span carries without pinning the exact
    escape bytes rich emits.
    """
    match = list(SGR_RE.finditer(text))[-1]
    return match.group(1)


@pytest.fixture
def _patched_measure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run measure commands lock-free with a fake bench, from a non-repo cwd."""
    monkeypatch.chdir(tmp_path)

    async def fake_measure(_options: object):
        return create_measurement_result()

    monkeypatch.setattr("gymrat_py.measure.measure", fake_measure)


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def test_app_when_version_flag_does_print_package_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert importlib.metadata.version("gymrat-py") in result.stdout


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_app_when_help_does_show_description():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Performance comparison tool for benchmarks" in result.stdout


def test_app_when_help_does_show_root_epilogue_examples_and_links():
    result = runner.invoke(app, ["--help"])
    normalized = _normalize(result.stdout)

    assert 'gymrat compare main my-branch --bench "npm run bench"' in normalized
    assert (
        'gymrat compare old=main new=perf/decode --bench "npm run bench" --fail-on regressed'
        in normalized
    )
    assert 'gymrat measure --bench "npm run bench"' in normalized
    assert f"Docs: {DOCS_URL}" in normalized
    assert f"Bugs: {BUGS_URL}" in normalized


def test_app_when_help_colored_does_render_the_docs_link_as_a_dim_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    result = runner.invoke(app, ["--help"], color=True)

    docs_line = next(line for line in result.stdout.splitlines() if "Docs:" in strip_ansi(line))
    assert docs_line.lstrip().startswith("\x1b[2m")  # cspell:disable-line
    assert "34" in _sgr_params(docs_line[: docs_line.index(DOCS_URL)])


# ---------------------------------------------------------------------------
# --debug in both positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--debug", "measure", "--bench", "sh bench.sh"], id="before-subcommand"),
        pytest.param(["measure", "--bench", "sh bench.sh", "--debug"], id="after-subcommand"),
    ],
)
@pytest.mark.usefixtures("_patched_measure")
def test_app_when_debug_flag_in_either_position_does_not_error(argv: Sequence[str]):
    result = runner.invoke(app, list(argv))

    assert result.exit_code == 0


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(
            ["--debug", "measure", "--bench", "sh bench.sh"],
            id="before-subcommand",
        ),
        pytest.param(["measure", "--bench", "sh bench.sh", "--debug"], id="after-subcommand"),
    ],
)
def test_app_when_debug_flag_does_show_traceback_on_error(
    argv: Sequence[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)

    async def exploding_measure(_options: object):
        msg = "deliberate boom"
        raise RuntimeError(msg)

    monkeypatch.setattr("gymrat_py.measure.measure", exploding_measure)

    result = runner.invoke(app, list(argv))

    assert result.exit_code == 2
    assert "Traceback" in result.output


# ---------------------------------------------------------------------------
# unknown command
# ---------------------------------------------------------------------------


def test_app_when_unknown_command_does_exit_two():
    result = runner.invoke(app, ["banana"])

    assert result.exit_code == 2


# The registered command set the app exposes.
def test_app_registers_compare_and_measure():
    result = runner.invoke(app, ["--help"])

    assert "compare" in result.stdout
    assert "measure" in result.stdout
