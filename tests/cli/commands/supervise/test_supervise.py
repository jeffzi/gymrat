"""Command-level tests for the ``gymrat supervise`` wiring.

These drive the command through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
Git, ``repo_root``, and the supervise lock stay real; the seams the command
composes over — config resolution, kickoff, the Claude driver, the supervisor
run, the run-end exit sequence, the progress reporter, the git-exclude write,
and the signal cleanup — are replaced at the names ``commands.supervise``
imports them under, mirroring the upstream test harness.

The dirty-tree guards run inside the pre-flight: a dirty working tree, or an
experiment worktree holding unmeasured or unsettled edits, stops the run before
the agent starts, and ``--allow-dirty`` lets only the working-tree case
through, with a warning. Those tests run the real pre-flight with only the
baseline bench replaced.
"""

import asyncio
import os
import re
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import typer

from gymrat.adapters import MetricDefaults
from gymrat.cli.app import app
from gymrat.cli.commands import supervise as supervise_cmd
from gymrat.cli.supervise.preflight import run_preflight
from gymrat.cli.supervise.progress import create_supervise_reporter
from gymrat.cli.supervise.types import ReadSessionResult
from gymrat.config import (
    MetricEntry,
    ResolvedConfig,
    StopConfig,
    SuperviseConfig,
)
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.exec import kill_live_process_groups
from gymrat.loop.start import StartResult
from gymrat.session.paths import (
    budget_path,
    experiment_worktree_dir,
    lockfile_path,
    supervise_lockfile_path,
)
from gymrat.supervisor.driver import SessionPrompt
from gymrat.supervisor.hooks import HooksFactory, supervise_hooks_factory
from gymrat.supervisor.supervise import SupervisedSession, SupervisionResult
from gymrat.supervisor.tools import ToolsFactory, gymrat_tools_factory
from tests._ansi import strip_ansi
from tests._lock import FIXED_HOLDER_AT, hold_lock
from tests._rich import (
    track,
    unwrap_panel,
)
from tests.cli._session import (
    FailingStdoutRunner,
    closed_stdout_error,
    make_discard_repo,
)
from tests.cli.commands.supervise._seams import (
    CAP_MINUTES,
    CAP_MS,
    TRACING_FAILURE,
    Seams,
    command_config,
    err_text,
    exploding_setup_tracing,
    install_seams,
    run,
    track_cleanups,
)
from tests.cli.supervise._fixtures import (
    install_baseline_seam,
    launch_event,
    make_supervision_result,
    render_frame,
    session_state_three_iterations,
    start_open_session,
)
from tests.sampling._adapters import make_adapter
from tests.supervisor._mock_driver import CostStep, create_mock_driver

# The message the budget-initialization seam fails with when a test makes it explode.
_BUDGET_FAILURE = "budget write failed"


# ---------------------------------------------------------------------------
# flag parsing
# ---------------------------------------------------------------------------


def test_supervise_when_max_minutes_missing_does_exit_two_naming_the_flag(repo: str):
    result = run("my prompt")

    assert result.exit_code == 2
    assert "--max-minutes" in err_text(result)


def test_supervise_when_run_does_carry_the_caps_into_the_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    before_ms = time.time() * 1000

    result = run("optimize it", "--max-minutes", str(CAP_MINUTES), "--max-usd", "2.0")

    after_ms = time.time() * 1000
    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert ctx.root == repo
    assert re.search(r"\.gymrat[/\\]supervisor-\d+\.jsonl", ctx.log_path)
    assert ctx.lock_path == lockfile_path(repo)
    assert isinstance(ctx.config, ResolvedConfig)
    assert before_ms + CAP_MS <= ctx.deadline_ms <= after_ms + CAP_MS
    assert ctx.max_minutes == CAP_MINUTES
    assert ctx.max_usd == 2.0
    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert (prompt.command_timeout_ms, prompt.max_budget_usd) == (CAP_MS, 2.0)


# ---------------------------------------------------------------------------
# default log path
# ---------------------------------------------------------------------------


def test_supervise_when_no_log_given_does_report_the_session_dir_log_in_a_plain_summary(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert re.search(r"\.gymrat[/\\]supervisor-\d+\.jsonl", result.stderr)
    seams.ensure_git_exclude.assert_called_once_with(repo)
    assert result.stdout.splitlines()[0] == "✓ completed · 1m 0s · $0.05"
    assert result.stdout.count(".jsonl") == 1
    assert "\x1b[" not in result.stdout


def test_supervise_when_log_given_does_use_that_path_as_is(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seams = install_seams(monkeypatch)
    custom = str(tmp_path / "custom.jsonl")

    result = run("optimize it", "--max-minutes", "10", "--log", custom)

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert ctx.log_path == custom
    seams.ensure_git_exclude.assert_not_called()


# ---------------------------------------------------------------------------
# launch line — log path abbreviation
# ---------------------------------------------------------------------------


def test_supervise_when_log_under_home_does_abbreviate_path_in_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    install_seams(monkeypatch)
    log_path = str(Path.home() / ".gymrat" / "supervisor-1.jsonl")

    result = run("optimize it", "--max-minutes", "10", "--log", log_path)

    assert result.exit_code == 0
    assert "~/.gymrat/supervisor-1.jsonl" in result.stderr


# ---------------------------------------------------------------------------
# supervise lock
# ---------------------------------------------------------------------------


def test_supervise_when_lock_held_by_live_process_does_exit_two_naming_another_run(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch)
    lock_path = supervise_lockfile_path(repo)
    blocker = hold_lock(
        lock_path,
        holder={"pid": os.getpid(), "command": "supervise", "at": FIXED_HOLDER_AT},
    )

    try:
        result = run("optimize it", "--max-minutes", "10")

        assert result.exit_code == 2
        assert re.search(r"another gymrat", result.stderr, re.IGNORECASE)
    finally:
        blocker.release()


# ---------------------------------------------------------------------------
# closing summary
# ---------------------------------------------------------------------------


def test_supervise_when_session_ends_with_final_text_does_show_the_agent_row(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch, final_text="all done here")

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert "all done here" in result.stdout


def test_supervise_when_session_has_iterations_does_show_them_in_the_summary_loop_row(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    state = session_state_three_iterations(-4.2, "improved", seq=3)
    install_seams(monkeypatch, session_result=ReadSessionResult(state=state, has_baseline=True))

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert "  loop    3 iterations · 2 kept · 1 discarded · last -4.2% improved" in result.stdout


def test_supervise_when_log_path_is_long_does_print_it_unwrapped(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    install_seams(monkeypatch)
    nested = tmp_path / ("supervise-log-directory-" * 3)
    nested.mkdir()
    custom = str(nested / "supervisor-1.jsonl")

    result = run("optimize it", "--max-minutes", "10", "--log", custom)

    assert result.exit_code == 0
    # On Windows CI tmp_path lives under $HOME, so the display path is ~/…
    # abbreviated.  Check the row is a single unwrapped line.
    assert any(
        line.startswith("  log     ") and "supervisor-1.jsonl" in line
        for line in result.stdout.splitlines()
    )


def test_supervise_when_stdout_reader_closed_does_exit_zero_without_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch, config=replace(command_config(), checks="npm test"))
    monkeypatch.setattr("gymrat.cli.commands.supervise.resolve_render_mode", lambda: "live")

    result = FailingStdoutRunner(closed_stdout_error()).invoke(
        app, ["supervise", "optimize it", "--max-minutes", "10"]
    )

    assert (result.exit_code, result.stderr) == (0, "")


def test_supervise_when_stdout_reader_closed_and_preflight_fails_does_exit_two(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch, config=replace(command_config(), checks="npm test"))
    monkeypatch.setattr("gymrat.cli.commands.supervise.run_preflight", run_preflight)
    monkeypatch.setattr("gymrat.cli.commands.supervise.resolve_render_mode", lambda: "live")

    result = FailingStdoutRunner(closed_stdout_error()).invoke(
        app, ["supervise", "optimize it", "--max-minutes", "10"]
    )

    assert result.exit_code == 2
    assert "Error:" in result.stderr


# ---------------------------------------------------------------------------
# driver, kickoff, and reporter wiring
# ---------------------------------------------------------------------------


def test_supervise_when_driver_session_completes_does_run_the_real_supervisor(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    commands_supervise = supervise_cmd.supervise
    seams = install_seams(monkeypatch)
    seams.create_driver.return_value = create_mock_driver([CostStep(cost_usd=0.01)])
    monkeypatch.setattr("gymrat.cli.commands.supervise.supervise", commands_supervise)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.exit_calls[0]["ended_by"] == "session"


@pytest.mark.parametrize(
    ("prompt_args", "prompt"),
    [
        pytest.param(["optimize the decoder"], "optimize the decoder", id="prompt-given"),
        pytest.param([], None, id="no-prompt"),
    ],
)
def test_supervise_when_run_does_compose_kickoff_with_the_prompt_given(
    repo: str, monkeypatch: pytest.MonkeyPatch, prompt_args: list[str], prompt: str | None
):
    seams = install_seams(monkeypatch)

    result = run(*prompt_args, "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.compose_calls[0][1] == prompt


def test_supervise_when_max_minutes_fractional_does_forward_it_without_flooring_to_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run("optimize it", "--max-minutes", "5.5")

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["max_minutes"] == 5.5


def _dashboard_title(reporter_kwargs: Mapping[str, Any]) -> str:
    """Build the real dashboard from the arguments the command passed and return its title line."""
    reporter = track(create_supervise_reporter(**reporter_kwargs))
    reporter.observer(launch_event(1000))
    frame = render_frame(reporter)
    return next(line for line in frame.splitlines() if line.startswith("╭"))


@pytest.mark.parametrize(
    ("branch", "title_parts"),
    [
        pytest.param("banana", ["· branch banana"], id="branch-shown"),
        pytest.param("", [], id="no-branch-omitted"),
    ],
)
def test_supervise_when_live_does_show_the_session_branch_in_the_dashboard_title_if_any(
    repo: str, monkeypatch: pytest.MonkeyPatch, branch: str, title_parts: list[str]
):
    seams = install_seams(monkeypatch, branch=branch)
    monkeypatch.setattr("gymrat.cli.commands.supervise.resolve_render_mode", lambda: "live")

    result = run("optimize it", "--max-minutes", "10")
    title = _dashboard_title(seams.reporter_calls[0])

    assert result.exit_code == 0
    assert re.findall(r"· branch \w+", title) == title_parts


def _plain_writes(reporter_kwargs: Mapping[str, Any]) -> list[str]:
    """Build the real plain reporter from the command's arguments and return what launch prints."""
    writes: list[str] = []
    reporter = track(create_supervise_reporter(**reporter_kwargs, plain_write=writes.append))
    reporter.observer(launch_event(1000))
    reporter.stop()
    return writes


def test_supervise_when_plain_and_session_has_branch_does_print_no_title(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch, branch="banana")
    monkeypatch.setattr("gymrat.cli.commands.supervise.resolve_render_mode", lambda: "plain")

    result = run("optimize it", "--max-minutes", "10")
    writes = _plain_writes(seams.reporter_calls[0])

    assert result.exit_code == 0
    assert writes
    assert not any("banana" in line for line in writes)
    assert "banana" not in strip_ansi(result.stderr)


@pytest.mark.parametrize(
    ("flag_args", "expected_color"),
    [
        pytest.param(("--no-color",), False, id="no-color"),
        pytest.param(("--color",), True, id="color"),
        pytest.param((), None, id="default"),
    ],
)
def test_supervise_when_color_flag_given_does_forward_it_to_doctor_gate(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    flag_args: tuple[str, ...],
    expected_color: bool | None,
):
    seams = install_seams(monkeypatch)

    result = run("optimize it", "--max-minutes", "10", *flag_args)

    assert result.exit_code == 0
    call_kwargs = seams.doctor_gate.call_args.kwargs
    assert call_kwargs.get("color") is expected_color


def test_supervise_when_doctor_gate_refuses_does_exit_two_before_the_session_starts(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    seams.doctor_gate.side_effect = typer.Exit(TOOL_FAILURE_EXIT_CODE)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert seams.supervise_calls == []


def test_supervise_when_run_starts_does_build_the_reporter_before_installing_any_cleanup(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    # The dashboard installs its own erase cleanup while it is built, so building
    # it first makes the erase run before the budget release and the process kill.
    install_seams(monkeypatch)
    registry = track_cleanups(monkeypatch)
    armed_at_build: list[list[Callable[[], None]]] = []
    build_reporter: Callable[..., object] = supervise_cmd.create_supervise_reporter

    def probing_reporter(**kwargs: object) -> object:
        armed_at_build.append(registry.live())
        return build_reporter(**kwargs)

    monkeypatch.setattr("gymrat.cli.commands.supervise.create_supervise_reporter", probing_reporter)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert armed_at_build == [[]]


def test_supervise_when_session_runs_does_arm_the_kill_cleanup_only_for_its_duration(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    registry = track_cleanups(monkeypatch)
    armed_during_run: list[bool] = []

    async def probing_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        seams.record_supervise_call(args, kwargs)
        armed_during_run.append(any(live is kill_live_process_groups for live in registry.live()))
        return make_supervision_result()

    monkeypatch.setattr("gymrat.cli.commands.supervise.supervise", probing_supervise)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert armed_during_run == [True]
    assert registry.live() == []


def _exploding_write_budget(*_args: object) -> None:
    """Stand in for the budget write, failing the way a read-only session dir does."""
    raise GymratError(_BUDGET_FAILURE)


@pytest.mark.parametrize(
    ("target", "replacement", "message"),
    [
        pytest.param(
            "gymrat.cli.commands.supervise.write_budget",
            _exploding_write_budget,
            _BUDGET_FAILURE,
            id="budget-init",
        ),
        pytest.param(
            "gymrat.telemetry.run_spans.setup_tracing",
            exploding_setup_tracing,
            TRACING_FAILURE,
            id="tracing-setup",
        ),
    ],
)
def test_supervise_when_session_setup_raises_does_tear_down_everything_it_armed(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    replacement: Callable[..., object],
    message: str,
):
    seams = install_seams(monkeypatch)
    registry = track_cleanups(monkeypatch)
    monkeypatch.setattr(target, replacement)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert message in err_text(result)
    seams.reporter_stop.assert_called()
    assert registry.live() == []
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
            "10", (), {"baseline_ref": None, "allow_dirty": False}, id="no-optional-flags"
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

    result = run("optimize it", "--max-minutes", max_minutes, *extra_args)

    assert result.exit_code == 0
    call = seams.preflight_calls[0]
    assert call["root"] == repo
    assert {key: call[key] for key in expected} == expected


# ---------------------------------------------------------------------------
# preflight failure and config propagation
# ---------------------------------------------------------------------------


def test_supervise_when_preflight_raises_does_exit_two_with_message(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch)

    msg = "cap too small"

    def exploding_preflight(**_kwargs: object) -> StartResult:
        raise GymratError(msg)

    monkeypatch.setattr("gymrat.cli.commands.supervise.run_preflight", exploding_preflight)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "cap too small" in result.stderr


def test_supervise_when_config_resolved_does_reach_every_consumer(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    cfg = command_config(stop=StopConfig(max_iterations=9))
    seams = install_seams(monkeypatch, config=cfg)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    passed_config = seams.compose_calls[0][0]
    assert isinstance(passed_config, ResolvedConfig)
    assert passed_config.stop is not None
    assert passed_config.stop.max_iterations == 9

    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert ctx.config.stop is not None
    assert ctx.config.stop.max_iterations == 9

    assert seams.reporter_calls[0]["max_iterations"] == 9


# ---------------------------------------------------------------------------
# --effort flag parsing and resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param("banana", id="unknown-word"),
        pytest.param("HIGH", id="wrong-case"),
        pytest.param("", id="empty-string"),
    ],
)
def test_supervise_when_effort_invalid_does_exit_two_with_choice_message(repo: str, bad_value: str):
    result = run("optimize it", "--max-minutes", "10", "--effort", bad_value)
    flat = unwrap_panel(err_text(result))

    assert result.exit_code == 2
    assert f"{bad_value!r} is not one of 'low', 'medium', 'high', 'xhigh', 'max'." in flat


@pytest.mark.parametrize(
    ("flag_args", "supervise_config", "expected"),
    [
        pytest.param(
            ("--effort", "max"),
            SuperviseConfig(effort="low"),
            (None, "max"),
            id="effort-flag-overrides-config",
        ),
        pytest.param(
            (),
            SuperviseConfig(effort="high"),
            (None, "high"),
            id="no-effort-flag-uses-config",
        ),
        pytest.param(
            (),
            SuperviseConfig(model="opus"),
            ("opus", None),
            id="no-model-flag-uses-config",
        ),
        pytest.param((), None, (None, None), id="no-flags-no-section-leaves-both-unset"),
    ],
)
def test_supervise_when_run_does_resolve_model_and_effort_from_flag_or_config(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    flag_args: tuple[str, ...],
    supervise_config: SuperviseConfig | None,
    expected: tuple[str | None, str | None],
):
    seams = install_seams(monkeypatch, config=command_config(supervise=supervise_config))

    result = run("optimize it", "--max-minutes", "10", *flag_args)

    assert result.exit_code == 0
    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    expected_model, expected_effort = expected
    assert prompt.model == expected_model
    assert prompt.effort == expected_effort


# ---------------------------------------------------------------------------
# tools and hooks factory wiring
# ---------------------------------------------------------------------------


def test_supervise_when_run_from_subdirectory_does_pass_factories_for_repo_root(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)
    tools_factory_roots: list[str] = []
    built_hooks: dict[Path, HooksFactory] = {}
    real_tools_factory = gymrat_tools_factory
    real_hooks_factory = supervise_hooks_factory

    def _recording_tools_factory(root: str) -> ToolsFactory:
        tools_factory_roots.append(root)
        return real_tools_factory(root)

    def _record_hooks(root: Path) -> HooksFactory:
        built_hooks[root.resolve()] = real_hooks_factory(root)
        return built_hooks[root.resolve()]

    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.gymrat_tools_factory", _recording_tools_factory
    )
    monkeypatch.setattr("gymrat.cli.commands.supervise.supervise_hooks_factory", _record_hooks)
    (Path(repo) / "docs").mkdir()
    monkeypatch.chdir(Path(repo) / "docs")

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert [Path(root).resolve() for root in tools_factory_roots] == [Path(repo).resolve()]
    assert list(built_hooks) == [Path(repo).resolve()]
    call_kwargs = seams.create_driver.call_args.kwargs
    tools = call_kwargs.get("tools")
    assert callable(tools), "create_claude_driver must receive a tools callable"
    server_config: dict[str, Any] = tools(asyncio.Event(), {})  # type: ignore[assignment]  # McpSdkServerConfig is a TypedDict
    assert server_config["name"] == "gymrat"
    assert call_kwargs.get("hooks") is built_hooks[Path(repo).resolve()]


# ---------------------------------------------------------------------------
# reporter direction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("primary", "metrics", "expected"),
    [
        pytest.param("geomean", None, "lower", id="geomean-ignores-the-adapter"),
        pytest.param(
            "throughput",
            {"throughput": MetricEntry(direction="lower")},
            "lower",
            id="metric-entry-direction",
        ),
        pytest.param("throughput", None, "higher", id="adapter-default"),
        pytest.param(
            "throughput",
            {"throughput": MetricEntry(gating=False)},
            "higher",
            id="metric-entry-without-direction",
        ),
    ],
)
def test_supervise_when_adapter_defaults_to_higher_does_build_reporter_with_the_primary_direction(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    primary: str,
    metrics: dict[str, MetricEntry] | None,
    expected: str,
):
    config = replace(command_config(), primary=primary, metrics=metrics)
    adapters = {config.adapter: make_adapter(lambda _name: MetricDefaults(direction="higher"))}
    monkeypatch.setattr("gymrat.cli.commands.supervise.get_adapter", adapters.__getitem__)
    seams = install_seams(monkeypatch, config=config)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["primary_direction"] == expected


def _install_seams_with_real_preflight(monkeypatch: pytest.MonkeyPatch) -> Seams:
    """Install the command seams, keeping the real pre-flight with a stand-in baseline bench."""
    seams = install_seams(monkeypatch)
    monkeypatch.setattr("gymrat.cli.commands.supervise.run_preflight", run_preflight)
    install_baseline_seam(monkeypatch)
    return seams


# ---------------------------------------------------------------------------
# dirty-tree guard
# ---------------------------------------------------------------------------


_DIRTY = r"dirty|uncommitted|untracked"


@pytest.mark.parametrize(
    ("files", "args", "exit_code", "stderr_has"),
    [
        pytest.param(
            ["uncommitted.txt"],
            [],
            2,
            {_DIRTY: True, r"commit|stash": True, r"--allow-dirty": True},
            id="dirty-refused-with-guidance",
        ),
        pytest.param(
            ["uncommitted.txt"], ["--allow-dirty"], 0, {_DIRTY: True}, id="dirty-allowed-warns"
        ),
        pytest.param([], [], 0, {_DIRTY: False}, id="clean-stays-silent"),
        pytest.param(
            ["new-dir/a.txt", "new-dir/b.txt", "new-dir/c.txt"],
            [],
            2,
            {r"Working tree has 3 uncommitted files\.": True},
            id="untracked-directory-counts-its-files",
        ),
    ],
)
def test_supervise_when_tree_state_varies_does_guard_the_launch_accordingly(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    files: list[str],
    args: list[str],
    exit_code: int,
    stderr_has: dict[str, bool],
):
    _install_seams_with_real_preflight(monkeypatch)
    for name in files:
        path = Path(repo) / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("dirty", encoding="utf-8")

    result = run("optimize it", "--max-minutes", "10", *args)

    assert result.exit_code == exit_code
    stderr = unwrap_panel(result.stderr)
    assert {
        pattern: bool(re.search(pattern, stderr, re.IGNORECASE)) for pattern in stderr_has
    } == stderr_has


# ---------------------------------------------------------------------------
# dirty experiment-worktree guard
# ---------------------------------------------------------------------------


def _dirty_experiment_worktree(repo: str, *names: str) -> None:
    worktree = Path(experiment_worktree_dir(repo))
    for name in names:
        (worktree / name).write_text("dirty\n", encoding="utf-8")


def _setup_open_session_missing_worktree(repo: str) -> None:
    """An open session whose experiment worktree directory no longer exists on disk."""
    start_open_session(repo)
    shutil.rmtree(experiment_worktree_dir(repo))


@pytest.mark.parametrize(
    "extra_args",
    [
        pytest.param((), id="default"),
        pytest.param(("--allow-dirty",), id="allow-dirty"),
    ],
)
def test_supervise_when_experiment_worktree_dirty_with_unsettled_does_exit_two_with_settle_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch, extra_args: tuple[str, ...]
):
    _install_seams_with_real_preflight(monkeypatch)
    make_discard_repo(repo)
    _dirty_experiment_worktree(repo, "scratch.txt")

    result = run("optimize it", "--max-minutes", "10", *extra_args)

    assert result.exit_code == 2
    text = err_text(result)
    assert re.search(r"unsettled", text, re.IGNORECASE)
    assert "gymrat keep" in text
    assert "gymrat discard" in text


def test_supervise_when_experiment_worktree_dirty_without_unsettled_does_exit_two_with_iterate_and_discard_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams_with_real_preflight(monkeypatch)
    start_open_session(repo)
    _dirty_experiment_worktree(repo, "a.txt", "b.txt")

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    text = err_text(result)
    assert "2 unmeasured edit" in text
    assert "gymrat iterate" in text
    assert "gymrat discard" in text


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(_setup_open_session_missing_worktree, id="missing-worktree"),
        # An open session whose experiment worktree has no uncommitted changes.
        pytest.param(start_open_session, id="clean-worktree"),
    ],
)
def test_supervise_when_experiment_worktree_guard_finds_no_issue_does_proceed(
    repo: str, monkeypatch: pytest.MonkeyPatch, setup: Callable[[str], None]
):
    _install_seams_with_real_preflight(monkeypatch)
    setup(repo)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
