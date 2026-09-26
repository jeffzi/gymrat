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

from gymrat.cli.app import app
from gymrat.cli.shared import BUGS_URL
from tests._ansi import SGR_RE, strip_ansi
from tests.cli._help import help_output
from tests.cli._session import write_config
from tests.report._inputs import create_measurement_result
from tests.session.records._fixtures import (
    committed_keep,
    iteration_record,
    session_record,
    write_session_log,
)

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

    monkeypatch.setattr("gymrat.measure.measure", fake_measure)


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def test_app_when_version_flag_does_print_package_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert importlib.metadata.version("gymrat") in result.stdout


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_app_when_help_does_show_description():
    out = help_output()

    assert "Performance comparison tool for benchmarks" in out
    assert "compare" in out
    assert "measure" in out
    assert "probe" in out
    assert "stop" in out


def test_app_when_help_does_show_root_epilogue_examples_and_links():
    normalized = _normalize(help_output())

    assert 'gymrat compare main my-branch --bench "npm run bench"' in normalized
    assert (
        'gymrat compare old=main new=perf/decode --bench "npm run bench" --fail-on regressed'
        in normalized
    )
    assert 'gymrat measure --bench "npm run bench"' in normalized
    assert f"Docs: {DOCS_URL}" in normalized
    assert f"Bugs: {BUGS_URL}" in normalized


def test_app_when_help_does_show_manual_loop_examples_after_supervise():
    out = help_output()
    lines = out.splitlines()

    supervise_idx = next(i for i, line in enumerate(lines) if "gymrat supervise" in line)
    after_supervise = "\n".join(lines[supervise_idx + 1 :])

    loop_examples = [
        "gymrat start --baseline main",
        "gymrat iterate",
        "gymrat keep -m",
        "gymrat finalize",
    ]
    prev_pos = -1
    for example in loop_examples:
        pos = after_supervise.find(example)
        assert pos > prev_pos, f"{example!r} not found after supervise line (or out of order)"
        prev_pos = pos


def test_app_when_help_colored_does_render_the_docs_link_as_a_dim_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr("typer.rich_utils.FORCE_TERMINAL", True)

    result = runner.invoke(app, ["--help"], color=True, env={"COLUMNS": "200"})

    docs_line = next(line for line in result.stdout.splitlines() if "Docs:" in strip_ansi(line))
    assert docs_line.lstrip().startswith("\x1b[2m")  # cspell:disable-line
    assert "34" in _sgr_params(docs_line[: docs_line.index(DOCS_URL)])


# ---------------------------------------------------------------------------
# --debug in both positions
# ---------------------------------------------------------------------------

DEBUG_FLAG_POSITIONS = [
    pytest.param(["--debug", "measure", "--bench", "sh bench.sh"], id="before-subcommand"),
    pytest.param(["measure", "--bench", "sh bench.sh", "--debug"], id="after-subcommand"),
]


@pytest.mark.parametrize("argv", DEBUG_FLAG_POSITIONS)
@pytest.mark.usefixtures("_patched_measure")
def test_app_when_debug_flag_in_either_position_does_not_error(argv: Sequence[str]):
    result = runner.invoke(app, list(argv))

    assert result.exit_code == 0


@pytest.mark.parametrize("argv", DEBUG_FLAG_POSITIONS)
def test_app_when_debug_flag_does_show_traceback_on_error(
    argv: Sequence[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)

    async def exploding_measure(_options: object):
        msg = "deliberate boom"
        raise RuntimeError(msg)

    monkeypatch.setattr("gymrat.measure.measure", exploding_measure)

    result = runner.invoke(app, list(argv))

    assert result.exit_code == 2
    assert "Traceback" in result.output


# ---------------------------------------------------------------------------
# unknown command
# ---------------------------------------------------------------------------


def test_app_when_unknown_command_does_exit_two():
    result = runner.invoke(app, ["banana"])

    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# root --color / --no-color
# ---------------------------------------------------------------------------


def test_app_when_help_does_list_color_and_no_color_root_options():
    out = help_output()

    tokens = out.split()
    assert "--color" in tokens
    assert "--no-color" in tokens


def test_app_when_root_no_color_does_strip_ansi_from_status_stdout(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    result = runner.invoke(app, ["--no-color", "status"])

    assert result.exit_code == 0
    assert not SGR_RE.search(result.stdout)


def test_app_when_root_color_does_force_ansi_on_status_stdout(repo: str):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    result = runner.invoke(app, ["--color", "status"])

    assert result.exit_code == 0
    assert SGR_RE.search(result.stdout)


def test_app_when_local_no_color_beats_root_color_does_produce_plain_output(repo: str):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    result = runner.invoke(app, ["--color", "status", "--no-color"])

    assert result.exit_code == 0
    assert not SGR_RE.search(result.stdout)


def test_app_when_subcommand_passes_none_does_not_erase_root_no_color(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_config(repo)

    result = runner.invoke(app, ["--no-color", "status"])

    assert result.exit_code == 0
    assert not SGR_RE.search(result.stdout)


# ---------------------------------------------------------------------------
# no <parse leak in any command's --help
# ---------------------------------------------------------------------------

_ALL_COMMANDS = [
    "init",
    "compare",
    "measure",
    "probe",
    "doctor",
    "start",
    "iterate",
    "keep",
    "discard",
    "finalize",
    "stop",
    "status",
    "sync",
    "supervise",
    "export",
]


@pytest.mark.parametrize("command", _ALL_COMMANDS)
def test_app_when_help_does_not_leak_parse_function_repr(command: str):
    out = help_output(command)

    assert "<parse" not in out
