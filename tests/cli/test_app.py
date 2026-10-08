"""Tests for the root ``gymrat`` typer app: version, help, debug, dispatch.

These drive the assembled CLI through :class:`typer.testing.CliRunner`, so the
root callback, the ``--version`` eager option, the shared ``--debug`` flag in
both positions, the root and local color flags, every command's help content,
and the exit-2 error every locking command prints when the repository root
cannot be resolved are exercised the way a shell would invoke them.

``python -m gymrat`` runs in a child process and must behave like
``python -m gymrat.cli.app``, so the two subprocesses are compared directly.
"""

import errno
import importlib.metadata
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.cli.console import is_debug_mode
from gymrat.cli.exit import BUGS_URL
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from tests._ansi import SGR_RE, normalize, sgr_params, strip_ansi
from tests._cli import no_color_env
from tests._rich import unwrap_panel
from tests.cli._session import (
    FailingStdoutRunner,
    closed_stdout_error,
    disk_full_error,
    runner,
    write_bench_config,
)
from tests.session.records._fixtures import (
    committed_keep,
    iteration_record,
    session_record,
    write_session_log,
)

DOCS_URL = "https://github.com/jeffzi/gymrat#readme"
"""The documentation link the root epilogue points at."""


def _help_output(*command: str) -> str:
    """Help text of ``gymrat *command`` rendered wide and ANSI-stripped.

    Ambient color splits a token across escape sequences and a narrow terminal
    wraps it across lines; both break a plain substring match. No arguments
    captures the root ``gymrat --help``.
    """
    result = runner.invoke(app, [*command, "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    return strip_ansi(result.stdout)


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m <module> <args>`` in a child process with color forced off."""
    return subprocess.run(  # noqa: S603 -- fixed interpreter plus test-chosen args
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        check=False,
        env=no_color_env(),
    )


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def test_app_when_version_flag_does_print_package_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert importlib.metadata.version("gymrat") in result.stdout


def test_app_when_version_and_stdout_closed_does_exit_zero_without_stderr():
    result = FailingStdoutRunner(closed_stdout_error()).invoke(app, ["--version"])

    assert (result.exit_code, result.stderr) == (0, "")


def test_app_when_version_and_stdout_write_fails_otherwise_does_exit_two_with_error_on_stderr():
    result = FailingStdoutRunner(disk_full_error()).invoke(app, ["--version"])

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert os.strerror(errno.ENOSPC) in unwrap_panel(result.stderr)


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_app_when_root_help_does_document_the_cli():
    out = _help_output()

    normalized = normalize(out)
    assert "Performance comparison tool for benchmarks" in out
    assert 'gymrat compare main my-branch --bench "npm run bench"' in normalized
    assert (
        'gymrat compare old=main new=perf/decode --bench "npm run bench" --fail-on regressed'
        in normalized
    )
    assert 'gymrat measure --bench "npm run bench"' in normalized
    assert f"Docs: {DOCS_URL}" in normalized
    assert f"Bugs: {BUGS_URL}" in normalized
    assert re.search(
        r"gymrat supervise.*?gymrat start --baseline main.*?gymrat iterate"
        r".*?gymrat keep -m.*?gymrat finalize",
        out,
        re.DOTALL,
    )
    assert re.findall(r"^│ ([a-z][\w-]*)\s{2,}\S", out, re.MULTILINE) == _ALL_COMMANDS
    assert re.search(_COLOR_PAIR, out)


def test_app_when_help_colored_does_render_the_docs_link_as_a_dim_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setattr("typer.rich_utils.FORCE_TERMINAL", True)

    result = runner.invoke(app, ["--help"], color=True, env={"COLUMNS": "200"})

    docs_line = next(line for line in result.stdout.splitlines() if "Docs:" in strip_ansi(line))
    assert docs_line.lstrip().startswith("\x1b[2m")  # cspell:disable-line
    assert sgr_params(docs_line[: docs_line.index(DOCS_URL)]).split(";") == ["2", "34"]


# ---------------------------------------------------------------------------
# --debug in both positions
# ---------------------------------------------------------------------------

DEBUG_FLAG_POSITIONS = [
    pytest.param(["--debug", "measure", "--bench", "sh bench.sh"], id="before-subcommand"),
    pytest.param(["measure", "--bench", "sh bench.sh", "--debug"], id="after-subcommand"),
]


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


#: One runnable argv per command; the ids are the command names.
COMMAND_ARGV = [
    pytest.param(["init"], id="init"),
    pytest.param(["compare", "main", "cand", "--bench", "sh bench.sh"], id="compare"),
    pytest.param(["measure", "--bench", "sh bench.sh"], id="measure"),
    pytest.param(["probe"], id="probe"),
    pytest.param(["doctor"], id="doctor"),
    pytest.param(["start"], id="start"),
    pytest.param(["iterate", "--bench", "sh bench.sh"], id="iterate"),
    pytest.param(["keep"], id="keep"),
    pytest.param(["discard"], id="discard"),
    pytest.param(["finalize"], id="finalize"),
    pytest.param(["stop", "--message", "done"], id="stop"),
    pytest.param(["status"], id="status"),
    pytest.param(["sync"], id="sync"),
    pytest.param(["supervise", "optimize", "--max-minutes", "1"], id="supervise"),
    pytest.param(["export"], id="export"),
]

#: Every registered command name, in registration order.
_ALL_COMMANDS = [param.id for param in COMMAND_ARGV]


@pytest.mark.parametrize("argv", COMMAND_ARGV)
def test_app_when_command_debug_flag_does_turn_debug_mode_on(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, [*argv, "--debug"])

    assert is_debug_mode()


# ---------------------------------------------------------------------------
# repository discovery failure
# ---------------------------------------------------------------------------


#: The commands that resolve the repository root themselves, before taking its lock.
SESSION_COMMANDS = [
    pytest.param(["probe"], id="probe"),
    pytest.param(["start"], id="start"),
    pytest.param(["iterate"], id="iterate"),
    pytest.param(["keep"], id="keep"),
    pytest.param(["discard"], id="discard"),
    pytest.param(["finalize"], id="finalize"),
    pytest.param(["stop", "--message", "done"], id="stop"),
    pytest.param(["status"], id="status"),
    pytest.param(["sync"], id="sync"),
    pytest.param(["supervise", "optimize it", "--max-minutes", "10"], id="supervise"),
]

#: The commands that still run outside a git repository, skipping the lock instead of failing.
LOCK_FREE_COMMANDS = [
    pytest.param(["compare", "main", "main", "--bench", "sh bench.sh"], id="compare"),
    pytest.param(["measure", "--bench", "sh bench.sh"], id="measure"),
]

REPOSITORY_COMMANDS = [*LOCK_FREE_COMMANDS, *SESSION_COMMANDS]


@pytest.mark.parametrize("argv", SESSION_COMMANDS)
@pytest.mark.usefixtures("_in_non_repo")
def test_app_when_run_outside_a_repository_does_exit_two_naming_the_directory(argv: list[str]):
    cwd = os.getcwd()  # noqa: PTH109 -- the message quotes the str cwd repo discovery saw

    result = runner.invoke(app, argv)

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert (
        strip_ansi(result.stderr).split()
        == (f"Error: Not a git repository: {cwd} Run gymrat from inside a git repository.").split()
    )
    assert result.stdout == ""


@pytest.mark.parametrize("argv", REPOSITORY_COMMANDS)
@pytest.mark.usefixtures("repo")
def test_app_when_repository_root_cannot_be_resolved_does_exit_two_with_git_diagnostics(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
):
    def broken_git(*_args: object, **_kwargs: object) -> str:
        raise subprocess.CalledProcessError(
            128, ["git"], stderr="fatal: detected dubious ownership\n"
        )

    monkeypatch.setattr("gymrat.session.paths.run_git", broken_git)
    cwd = os.getcwd()  # noqa: PTH109 -- the message quotes the str cwd repo discovery saw

    result = runner.invoke(app, argv)

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert (
        strip_ansi(result.stderr).split()
        == (
            f"Error: Cannot determine the git repository at {cwd}: fatal: detected dubious ownership"
        ).split()
    )
    assert result.stdout == ""


@pytest.mark.parametrize("argv", LOCK_FREE_COMMANDS)
@pytest.mark.usefixtures("repo")
def test_app_when_repository_discovery_error_carries_a_hint_does_print_message_and_hint(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
):
    def broken_discovery(*_args: object, **_kwargs: object) -> str:
        message = "detected dubious ownership"
        raise GymratError(message, hint="Mark the repository as safe.")

    monkeypatch.setattr("gymrat.session.paths.run_git", broken_discovery)

    result = runner.invoke(app, argv)

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert result.stderr == "Error: detected dubious ownership\nMark the repository as safe.\n"
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# root and local --color / --no-color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expect_sgr"),
    [
        pytest.param(["--color", "status"], True, id="root-color"),
        pytest.param(
            ["--color", "status", "--no-color"], False, id="local-no-color-beats-root-color"
        ),
    ],
)
def test_app_when_color_flags_given_does_style_status_stdout_accordingly(
    repo: str, argv: list[str], expect_sgr: bool
):
    write_session_log(repo, session_record(), (iteration_record(seq=1), committed_keep(1)))
    write_bench_config(repo)

    result = runner.invoke(app, argv)

    assert result.exit_code == 0
    assert bool(SGR_RE.search(result.stdout)) is expect_sgr


# ---------------------------------------------------------------------------
# command help content
# ---------------------------------------------------------------------------

_COLOR_PAIR = r"(?<!\S)--color\s+--no-color(?!\S)"

_SAMPLES = r"--samples\s+-s\s+<int>\s+paired samples per target"
_TIMEOUT = r"--timeout\s+-t\s+<int>\s+timeout in seconds"


def _supervise_option(name: str, metavar: str, description: str) -> str:
    """Pattern matching one ``supervise --help`` row: name, metavar, description."""
    return rf"{re.escape(name)}\s+{re.escape(metavar)}\s*{re.escape(description)}"


#: The option rows each command's help must document, beyond the color pair.
_HELP_OPTIONS: dict[str, tuple[str, ...]] = {
    "init": (r"gymrat\.toml",),
    "compare": (
        _SAMPLES,
        _TIMEOUT,
        r"--fail-on\s+<condition>\s+exit 1 when",
        r"--verbose\s+-v\s+name the",
    ),
    "measure": (_SAMPLES, _TIMEOUT),
    "probe": (_SAMPLES, r"\[NAMES\]\.\.\.\s+<str>\s+metric names to narrow the bench to"),
    "doctor": (_SAMPLES, _TIMEOUT),
    "start": (
        _SAMPLES,
        _TIMEOUT,
        (
            r"--baseline\s+<ref>\s+git ref that pins a freshly opened session; "
            r"defaults to HEAD and is ignored when a session is resumed"
        ),
    ),
    "iterate": (_SAMPLES, _TIMEOUT),
    "keep": (
        _TIMEOUT,
        r"--allow-unimproved\s+keep the edit even when the iteration was not improved",
        r"--message\s+-m\s+<str>\s+commit message for the kept edit",
    ),
    "discard": (r"--force\s+-f\s+skip the confirmation prompt",),
    "finalize": (
        r"--message\s+-m\s+<str>\s+message for the squash commit",
        r"--branch\s+<str>\s+branch to point at the squash commit \(default: <branch>-final\)",
    ),
    "stop": (r"--message\s+-m\s+<str>\s+why the session is being stopped",),
    "status": (),
    "sync": (),
    "supervise": (
        _supervise_option("[PROMPT]", "<str>", "optimization prompt for the agent"),
        _supervise_option(
            "--max-minutes",
            "<float>",
            "wall-clock cap in minutes, counted from when the baseline is recorded",
        ),
        _supervise_option("--max-usd", "<float>", "spend cap in USD"),
        _supervise_option("--log", "<str>", "path for the JSONL event log"),
        _supervise_option("--model", "<str>", "model to use for the agent session"),
        _supervise_option("--effort", "<level>", "effort level"),
        _supervise_option("--allow-dirty", "", "allow launching with uncommitted changes"),
        _supervise_option(
            "--force",
            "",
            "launch even when the cap cannot fit one iteration or a stop condition is already met",
        ),
        _supervise_option("--no-finalize", "", "leave the session open instead of finalizing it"),
    ),
    "export": (
        r"\[SESSION_LOG\]\s+<str>\s+path to session\.jsonl",
        r"--endpoint\s+<str>\s+.*\[env var: OTEL_EXPORTER_OTLP_ENDPOINT\]",
    ),
}


@pytest.mark.parametrize("command", _ALL_COMMANDS)
def test_app_when_command_help_does_document_its_options(command: str):
    out = _help_output(command)

    missing = [
        pattern for pattern in (_COLOR_PAIR, *_HELP_OPTIONS[command]) if not re.search(pattern, out)
    ]
    assert missing == []
    assert "<parse" not in out


def test_app_when_commands_registered_does_match_the_tested_command_list():
    assert [command.name for command in app.registered_commands] == _ALL_COMMANDS
    assert list(_HELP_OPTIONS) == _ALL_COMMANDS


# ---------------------------------------------------------------------------
# python -m gymrat module entry
# ---------------------------------------------------------------------------


def test_main_module_when_help_does_show_same_description_and_epilogue_as_cli_app():
    module_result = _run_module("gymrat", "--help")
    app_result = _run_module("gymrat.cli.app", "--help")

    assert module_result.returncode == 0, module_result.stderr
    module_text = normalize(module_result.stdout)
    app_text = normalize(app_result.stdout)
    assert "Usage: gymrat [" in module_text
    assert module_text == app_text


@pytest.mark.parametrize("module", ["gymrat", "gymrat.cli.app"])
def test_module_entry_when_usage_error_does_print_usage_with_gymrat_program_name(module: str):
    result = _run_module(module, "compare", "main")

    assert result.returncode == 2
    assert "Usage: gymrat compare [" in normalize(result.stderr)
    assert result.stderr.count("Usage:") == 1
