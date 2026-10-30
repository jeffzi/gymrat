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
:mod:`tests.cli.supervise.test_preflight`; here only the ``--allow-dirty``
flag's forwarding to the pre-flight is pinned.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from gymrat.config import Effort, MetricEntry, StopConfig, SuperviseConfig
from gymrat.errors import GymratError
from gymrat.session.budget import write_budget
from gymrat.session.paths import SESSION_DIR_NAME, budget_path, lockfile_path, session_dir
from gymrat.supervisor.driver import SessionPrompt
from gymrat.supervisor.hooks import HooksFactory, supervise_hooks_factory
from gymrat.supervisor.supervise import SupervisedSession
from gymrat.supervisor.tools import ToolsFactory, gymrat_tools_factory
from gymrat.telemetry.run_spans import setup_tracing
from gymrat.utils import abbreviate_home
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
# flag parsing
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_hand_supervise_its_capped_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run("--max-usd", "2.0")

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert ctx.root == repo
    assert ctx.lock_path == lockfile_path(repo)
    assert ctx.max_usd == 2.0
    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert (prompt.command_timeout_ms, prompt.max_budget_usd) == (CAP_MS, 2.0)


# ---------------------------------------------------------------------------
# default log path
# ---------------------------------------------------------------------------


def test_supervise_when_no_log_given_does_log_under_the_session_dir_and_name_it_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run()

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert Path(ctx.log_path).parent == Path(session_dir(repo))
    assert abbreviate_home(ctx.log_path) in result.stderr


def test_supervise_when_log_given_does_use_that_path_as_is(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seams = install_seams(monkeypatch)
    custom = str(tmp_path / "custom.jsonl")

    result = run("--log", custom)

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert ctx.log_path == custom


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
# closing summary
# ---------------------------------------------------------------------------


def test_supervise_when_session_ends_with_final_text_does_show_the_agent_row(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch, final_text="all done here")

    result = run()

    assert result.exit_code == 0
    assert "all done here" in result.stdout


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


def test_supervise_when_run_does_hand_the_reporter_the_cap_unaltered(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run(max_minutes="5.5")

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["max_minutes"] == 5.5


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

    assert result.exit_code == 2
    assert message in stderr_text(result)
    assert (registry.live(), display_cleanups.live()) == ([], [])
    assert not Path(budget_path(repo)).exists()


@pytest.mark.parametrize(
    "raises",
    [
        pytest.param(None, id="run-completes"),
        pytest.param(GymratError("boom"), id="supervise-raises"),
    ],
)
def test_supervise_when_run_ends_does_uninstall_its_cleanup_once_and_clear_the_budget(
    repo: str, monkeypatch: pytest.MonkeyPatch, raises: Exception | None
):
    install_seams(monkeypatch, raises=raises)
    registry = track_cleanups(monkeypatch, "gymrat.cli.commands.supervise")

    run()

    assert registry.uninstall_counts() == [1]
    assert not Path(budget_path(repo)).exists()


# ---------------------------------------------------------------------------
# preflight kwargs from flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_minutes", "extra_args", "expected"),
    [
        pytest.param(
            "10",
            ("--baseline", "feature-branch"),
            {"baseline_ref": "feature-branch"},
            id="baseline-given",
        ),
        pytest.param(
            "42", ("--force",), {"max_minutes": 42.0, "force": True}, id="force-and-max-minutes"
        ),
        pytest.param("10", ("--allow-dirty",), {"allow_dirty": True}, id="allow-dirty"),
    ],
)
def test_supervise_when_run_does_pass_flags_to_preflight(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    max_minutes: str,
    extra_args: tuple[str, ...],
    expected: dict[str, object],
):
    seams = install_seams(monkeypatch)

    result = run(*extra_args, max_minutes=max_minutes)

    assert result.exit_code == 0
    call = seams.preflight_calls[0]
    assert call["root"] == repo
    assert {key: call[key] for key in expected} == expected


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
    seams = install_seams(monkeypatch, config=config)

    result = run(*flag_args)

    assert result.exit_code == 0
    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert prompt.model == expected


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
    seams = install_seams(monkeypatch, config=config)

    result = run(*flag_args)

    assert result.exit_code == 0
    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert prompt.effort == expected


# ---------------------------------------------------------------------------
# tools and hooks factory wiring
# ---------------------------------------------------------------------------


def test_supervise_when_run_from_subdirectory_does_pass_factories_for_repo_root(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    built_tools: dict[Path, ToolsFactory] = {}
    built_hooks: dict[Path, HooksFactory] = {}

    def _record_tools(root: str) -> ToolsFactory:
        built_tools[Path(root).resolve()] = gymrat_tools_factory(root)
        return built_tools[Path(root).resolve()]

    def _record_hooks(root: Path) -> HooksFactory:
        built_hooks[root.resolve()] = supervise_hooks_factory(root)
        return built_hooks[root.resolve()]

    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.gymrat_tools_factory",
        create_autospec(gymrat_tools_factory, side_effect=_record_tools),
    )
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.supervise_hooks_factory",
        create_autospec(supervise_hooks_factory, side_effect=_record_hooks),
    )
    (Path(repo) / "docs").mkdir()
    monkeypatch.chdir(Path(repo) / "docs")

    result = run()

    assert result.exit_code == 0
    assert list(built_tools) == [Path(repo).resolve()]
    assert list(built_hooks) == [Path(repo).resolve()]
    call_kwargs = seams.create_driver.call_args.kwargs
    assert call_kwargs.get("tools") is built_tools[Path(repo).resolve()]
    assert call_kwargs.get("hooks") is built_hooks[Path(repo).resolve()]


# ---------------------------------------------------------------------------
# reporter direction
# ---------------------------------------------------------------------------


def test_supervise_when_primary_is_higher_is_better_does_build_reporter_with_the_primary_direction(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    config = replace(
        command_config(),
        primary="throughput",
        metrics={"throughput": MetricEntry(direction="higher")},
    )
    seams = install_seams(monkeypatch, config=config)

    result = run()

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["primary_direction"] == "higher"
