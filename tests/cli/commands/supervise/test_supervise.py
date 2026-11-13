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

from gymrat.cli.supervise.preflight import run_preflight
from gymrat.config import Effort, StopConfig, SuperviseConfig
from gymrat.errors import GymratError
from gymrat.loop.stop_condition import LoopStopError, stop_condition
from gymrat.session.budget import write_budget
from gymrat.session.paths import SESSION_DIR_NAME, budget_path, lockfile_path, session_dir
from gymrat.supervisor.hooks import supervise_hooks_factory
from gymrat.supervisor.tools import gymrat_tools_factory
from gymrat.telemetry.run_spans import setup_tracing
from gymrat.utils import abbreviate_home
from tests._ansi import strip_ansi
from tests._cli import stderr_text
from tests._config import resolved_config
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
from tests.session.records._fixtures import session_state

_BUDGET_FAILURE = "budget write failed"
_TRACING_FAILURE = "tracing exporter unreachable"


# ---------------------------------------------------------------------------
# the session supervise is handed
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_hand_supervise_its_capped_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run()

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert ctx.lock_path == lockfile_path(repo)
    assert seams.supervise_calls[0]["prompt"].command_timeout_ms == CAP_MS


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
# pre-flight refusals
# ---------------------------------------------------------------------------


def _met_stop_error() -> LoopStopError:
    """The stop error a session that already met its iteration cap raises."""
    error = stop_condition(resolved_config(stop=StopConfig(max_iterations=0)), session_state())
    assert error is not None
    return error


def _one_line(text: str) -> str:
    """*text* with every run of whitespace, line wraps included, collapsed to one space."""
    return " ".join(text.split())


def _refuse_preflight(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Make the faked pre-flight raise *error*."""
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.run_preflight",
        create_autospec(run_preflight, side_effect=error),
    )


@pytest.mark.usefixtures("session_repo")
def test_supervise_when_preflight_meets_a_stop_condition_does_exit_one(
    monkeypatch: pytest.MonkeyPatch,
):
    install_seams(monkeypatch)
    _refuse_preflight(monkeypatch, _met_stop_error())

    result = run()

    assert result.exit_code == 1


@pytest.mark.usefixtures("session_repo")
def test_supervise_when_preflight_meets_a_stop_condition_does_not_show_the_agent_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    install_seams(monkeypatch)
    error = _met_stop_error()
    _refuse_preflight(monkeypatch, error)

    result = run()

    shown = _one_line(stderr_text(result))
    assert str(error) in shown
    assert error.hint is not None
    assert _one_line(error.hint) not in shown


@pytest.mark.usefixtures("session_repo")
def test_supervise_when_preflight_raises_another_error_does_exit_two(
    monkeypatch: pytest.MonkeyPatch,
):
    install_seams(monkeypatch)
    _refuse_preflight(monkeypatch, GymratError("banana"))

    result = run()

    assert result.exit_code == 2


@pytest.mark.usefixtures("session_repo")
def test_supervise_when_supervise_run_raises_a_stop_error_does_keep_its_own_hint(
    monkeypatch: pytest.MonkeyPatch,
):
    error = _met_stop_error()
    install_seams(monkeypatch, raises=error)

    result = run()

    assert error.hint is not None
    assert _one_line(error.hint) in _one_line(stderr_text(result))


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
    install_seams(monkeypatch)
    tools_factory = create_autospec(gymrat_tools_factory, return_value=object())
    hooks_factory = create_autospec(supervise_hooks_factory, return_value=object())
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
