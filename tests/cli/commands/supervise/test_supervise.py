"""Command-level tests for the ``gymrat supervise`` wiring.

These drive the command through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
Git, ``repo_root``, the supervise lock, and the git-exclude write stay real;
the seams the command composes over — the doctor gate, config resolution,
kickoff, the Claude driver, the supervisor run, the run-end exit sequence, the
progress reporter, and the signal cleanup — are replaced at the names
``commands.supervise`` imports them under.

The dirty-tree guards belong to the pre-flight and are tested in
:mod:`tests.cli.supervise.test_preflight`.
"""

from collections.abc import Callable
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from gymrat.config import Effort, StopConfig, SuperviseConfig
from gymrat.errors import GymratError
from gymrat.session.budget import write_budget
from gymrat.session.paths import SESSION_DIR_NAME, budget_path, lockfile_path, session_dir
from gymrat.supervisor.hooks import supervise_hooks_factory
from gymrat.supervisor.tools import gymrat_tools_factory
from gymrat.telemetry.run_spans import setup_tracing
from gymrat.utils import abbreviate_home
from tests._ansi import strip_ansi
from tests._cli import stderr_text
from tests._git import git_exclude_path
from tests._process_helpers import track_cleanups
from tests.cli._command_stubs import (
    force_render_mode,
)
from tests.cli.commands.supervise._seams import (
    CAP_MS,
    command_config,
    install_seams,
    run,
)

_BUDGET_FAILURE = "budget write failed"
_TRACING_FAILURE = "tracing exporter unreachable"


# ---------------------------------------------------------------------------
# the session supervise is handed
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_hand_supervise_its_capped_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run("--max-usd", "2.0")

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert ctx.lock_path == lockfile_path(repo)
    prompt = seams.supervise_calls[0]["prompt"]
    assert (prompt.command_timeout_ms, prompt.max_budget_usd) == (CAP_MS, 2.0)


# ---------------------------------------------------------------------------
# default log path
# ---------------------------------------------------------------------------


def test_supervise_when_no_log_given_does_log_under_the_session_dir(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run()

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert Path(ctx.log_path).parent == Path(session_dir(repo))


def test_supervise_when_run_does_name_the_log_path_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run()

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert abbreviate_home(ctx.log_path) in result.stderr


@pytest.mark.parametrize(
    ("log_args", "excluded"),
    [
        pytest.param([], True, id="default-log-under-the-session-dir"),
        pytest.param(["--log", "custom.jsonl"], False, id="caller-placed-log"),
    ],
)
def test_supervise_when_log_path_resolved_does_git_exclude_the_session_dir_only_for_the_default(
    repo: str, monkeypatch: pytest.MonkeyPatch, log_args: list[str], excluded: bool
):
    install_seams(monkeypatch)
    exclude_file = git_exclude_path(repo)

    result = run(*log_args)

    assert result.exit_code == 0
    exclude_lines = exclude_file.read_text(encoding="utf-8").splitlines()
    assert (f"{SESSION_DIR_NAME}/" in exclude_lines) is excluded


# ---------------------------------------------------------------------------
# driver, kickoff, and reporter wiring
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_compose_kickoff_with_the_prompt_given(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run(prompt="optimize the decoder")

    assert result.exit_code == 0
    assert seams.compose_calls[0] == "optimize the decoder"


def test_supervise_when_run_does_hand_the_reporter_the_session_branch(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch, branch="banana")

    result = run()

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["branch"] == "banana"


def test_supervise_when_color_flag_given_does_forward_it_to_doctor_gate(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run("--no-color")

    assert result.exit_code == 0
    call_kwargs = seams.doctor_gate.call_args.kwargs
    assert call_kwargs.get("color") is False


@pytest.mark.parametrize(
    ("target", "real", "message"),
    [
        pytest.param(
            "gymrat.session.budget.write_budget",
            write_budget,
            _BUDGET_FAILURE,
            id="budget-init",
        ),
        pytest.param(
            "gymrat.telemetry.run_spans.setup_tracing",
            setup_tracing,
            _TRACING_FAILURE,
            id="tracing-setup",
        ),
    ],
)
def test_supervise_when_session_setup_raises_does_tear_down_everything_it_armed(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    real: Callable[..., object],
    message: str,
):
    install_seams(monkeypatch, real_reporter=True)
    force_render_mode(monkeypatch, "supervise", "live")
    registry = track_cleanups(monkeypatch, "gymrat.cli.commands.supervise")
    display_cleanups = track_cleanups(monkeypatch, "gymrat.cli.live_display")
    monkeypatch.setattr(target, create_autospec(real, side_effect=GymratError(message)))

    result = run()

    assert message in stderr_text(result)
    assert (registry.live(), display_cleanups.live()) == ([], [])
    assert not Path(budget_path(repo)).exists()


@pytest.mark.usefixtures("repo")
def test_supervise_when_run_ends_does_uninstall_its_cleanup_once(monkeypatch: pytest.MonkeyPatch):
    install_seams(monkeypatch)
    registry = track_cleanups(monkeypatch, "gymrat.cli.commands.supervise")

    run()

    assert registry.uninstall_counts() == [1]


# ---------------------------------------------------------------------------
# config propagation
# ---------------------------------------------------------------------------


def test_supervise_when_config_sets_max_iterations_does_hand_it_to_the_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    cfg = command_config(stop=StopConfig(max_iterations=9))
    seams = install_seams(monkeypatch, config=cfg)

    result = run()

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["max_iterations"] == 9


# ---------------------------------------------------------------------------
# model and effort resolution
# ---------------------------------------------------------------------------


def _summary_value(stdout: str, label: str) -> str:
    """The value the printed summary shows on the row labelled *label*."""
    rows = [line.split() for line in strip_ansi(stdout).splitlines()]
    (value,) = [row[1] for row in rows if len(row) == 2 and row[0] == label]
    return value


@pytest.mark.parametrize(
    ("flag_args", "configured", "expected"),
    [
        pytest.param(("--model", "sonnet"), "opus", "sonnet", id="flag-overrides-config"),
        pytest.param((), "opus", "opus", id="no-flag-uses-config"),
    ],
)
def test_supervise_when_run_does_resolve_the_model_from_flag_or_config(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    flag_args: tuple[str, ...],
    configured: str,
    expected: str,
):
    config = command_config(supervise=SuperviseConfig(model=configured))
    install_seams(monkeypatch, config=config)

    result = run(*flag_args)

    assert result.exit_code == 0
    assert _summary_value(result.stdout, "model") == expected


@pytest.mark.parametrize(
    ("flag_args", "configured", "expected"),
    [
        pytest.param(("--effort", "max"), "low", "max", id="flag-overrides-config"),
        pytest.param((), "high", "high", id="no-flag-uses-config"),
    ],
)
def test_supervise_when_run_does_resolve_the_effort_from_flag_or_config(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    flag_args: tuple[str, ...],
    configured: Effort,
    expected: str,
):
    config = command_config(supervise=SuperviseConfig(effort=configured))
    install_seams(monkeypatch, config=config)

    result = run(*flag_args)

    assert result.exit_code == 0
    assert _summary_value(result.stdout, "effort") == expected


# ---------------------------------------------------------------------------
# tools and hooks factory wiring
# ---------------------------------------------------------------------------


def test_supervise_when_run_from_subdirectory_does_pass_factories_for_repo_root(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    tools, hooks = object(), object()
    tools_factory = create_autospec(gymrat_tools_factory, return_value=tools)
    hooks_factory = create_autospec(supervise_hooks_factory, return_value=hooks)
    monkeypatch.setattr("gymrat.cli.commands.supervise.gymrat_tools_factory", tools_factory)
    monkeypatch.setattr("gymrat.cli.commands.supervise.supervise_hooks_factory", hooks_factory)
    (Path(repo) / "docs").mkdir()
    monkeypatch.chdir(Path(repo) / "docs")

    result = run()

    assert result.exit_code == 0
    roots = [
        Path(factory.call_args.args[0]).resolve() for factory in (tools_factory, hooks_factory)
    ]
    assert roots == [Path(repo).resolve(), Path(repo).resolve()]
    call_kwargs = seams.create_driver.call_args.kwargs
    assert (call_kwargs.get("tools"), call_kwargs.get("hooks")) == (tools, hooks)
