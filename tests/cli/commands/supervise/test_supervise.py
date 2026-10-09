"""Command-level tests for the ``gymrat supervise`` wiring.

These drive the command through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
Git, ``repo_root``, and the supervise lock stay real; the seams the command
composes over — config resolution, kickoff, the Claude driver, the supervisor
run, the run-end exit sequence, the progress reporter, the git-exclude write,
and the signal cleanup — are replaced at the names ``commands.supervise``
imports them under, mirroring the upstream test harness.

The dirty-tree guards belong to the pre-flight and are tested in
:mod:`tests.cli.supervise.test_preflight`; here only the ``--allow-dirty``
flag's forwarding to the pre-flight is pinned.
"""

import os
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import create_autospec

import pytest
import typer

from gymrat.adapters import MetricDefaults, get_adapter
from gymrat.cli.app import app
from gymrat.cli.commands import supervise as supervise_cmd
from gymrat.cli.supervise.preflight import run_preflight
from gymrat.cli.supervise.progress import create_supervise_reporter
from gymrat.config import (
    MetricEntry,
    ResolvedConfig,
    StopConfig,
    SuperviseConfig,
)
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.session.budget import write_budget
from gymrat.session.paths import (
    SESSION_DIR_NAME,
    budget_path,
    lockfile_path,
    supervise_lockfile_path,
)
from gymrat.supervisor.driver import SessionPrompt
from gymrat.supervisor.hooks import HooksFactory, supervise_hooks_factory
from gymrat.supervisor.supervise import SupervisedSession
from gymrat.supervisor.tools import ToolsFactory, gymrat_tools_factory
from gymrat.telemetry.run_spans import setup_tracing
from gymrat.utils import abbreviate_home
from tests._ansi import strip_ansi
from tests._git import git_exclude_path
from tests._lock import FIXED_HOLDER_AT, hold_lock
from tests._process_helpers import track_cleanups
from tests.cli._session import (
    FailingStdoutRunner,
    closed_stdout_error,
    force_render_mode,
)
from tests.cli.commands.supervise._seams import (
    CAP_MINUTES,
    CAP_MS,
    command_config,
    err_text,
    install_seams,
    run,
)
from tests.sampling._adapters import make_adapter

# The messages the budget-initialization and tracing-setup seams fail with when a
# test makes them explode.
_BUDGET_FAILURE = "budget write failed"
_TRACING_FAILURE = "tracing exporter unreachable"


# ---------------------------------------------------------------------------
# flag parsing
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_hand_supervise_its_capped_session(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = install_seams(monkeypatch)

    result = run("optimize it", "--max-minutes", str(CAP_MINUTES), "--max-usd", "2.0")

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert ctx.root == repo
    assert re.search(r"\.gymrat[/\\]supervisor-\d+\.jsonl", ctx.log_path)
    assert ctx.lock_path == lockfile_path(repo)
    assert isinstance(ctx.config, ResolvedConfig)
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
    install_seams(monkeypatch)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert re.search(r"\.gymrat[/\\]supervisor-\d+\.jsonl", result.stderr)
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

    result = run("optimize it", "--max-minutes", "10", *log_args)

    assert result.exit_code == 0
    exclude_lines = exclude_file.read_text(encoding="utf-8").splitlines()
    assert (f"{SESSION_DIR_NAME}/" in exclude_lines) is excluded


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


def test_supervise_when_stdout_reader_closed_and_preflight_fails_does_exit_two(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    install_seams(monkeypatch, config=replace(command_config(), checks="npm test"))
    monkeypatch.setattr("gymrat.cli.commands.supervise.run_preflight", run_preflight)
    force_render_mode(monkeypatch, "supervise", "live")

    result = FailingStdoutRunner(closed_stdout_error()).invoke(
        app, ["supervise", "optimize it", "--max-minutes", "10"]
    )

    assert result.exit_code == 2
    assert "Error:" in result.stderr


# ---------------------------------------------------------------------------
# driver, kickoff, and reporter wiring
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    ("branch", "max_minutes", "reporter_key", "expected"),
    [
        pytest.param(None, "5.5", "max_minutes", 5.5, id="fractional-max-minutes"),
        pytest.param("banana", "10", "branch", "banana", id="session-branch"),
    ],
)
def test_supervise_when_run_does_hand_the_reporter_the_cap_and_branch_unaltered(
    *,
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    branch: str | None,
    max_minutes: str,
    reporter_key: str,
    expected: object,
):
    seams = install_seams(monkeypatch, branch=branch)

    result = run("optimize it", "--max-minutes", max_minutes)

    assert result.exit_code == 0
    assert seams.reporter_calls[0][reporter_key] == expected


def test_supervise_when_plain_and_session_has_branch_does_print_no_title(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    build_reporter = supervise_cmd.create_supervise_reporter
    seams = install_seams(monkeypatch, branch="banana")
    monkeypatch.setattr("gymrat.cli.commands.supervise.create_supervise_reporter", build_reporter)
    force_render_mode(monkeypatch, "supervise", "plain")
    seams.supervise_hook = lambda call: call["observer"](call["launch"])

    result = run("optimize it", "--max-minutes", "10")

    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert result.exit_code == 0
    assert strip_ansi(result.stderr).splitlines() == [
        f"log: {abbreviate_home(ctx.log_path)}",
        "caps 10m",
    ]


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
    registry = track_cleanups(monkeypatch, "gymrat.cli.commands.supervise")
    armed_at_build: list[list[Callable[[], None]]] = []
    build_reporter: Callable[..., object] = supervise_cmd.create_supervise_reporter

    def probing_reporter(**kwargs: object) -> object:
        armed_at_build.append(registry.live())
        return build_reporter(**kwargs)

    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.create_supervise_reporter",
        create_autospec(create_supervise_reporter, side_effect=probing_reporter),
    )

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert armed_at_build == [[]]


@pytest.mark.parametrize(
    ("target", "real", "message"),
    [
        pytest.param(
            "gymrat.cli.commands.supervise.write_budget",
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
    build_reporter = supervise_cmd.create_supervise_reporter
    install_seams(monkeypatch)
    monkeypatch.setattr("gymrat.cli.commands.supervise.create_supervise_reporter", build_reporter)
    force_render_mode(monkeypatch, "supervise", "live")
    registry = track_cleanups(monkeypatch, "gymrat.cli.commands.supervise")
    display_cleanups = track_cleanups(monkeypatch, "gymrat.cli.live_display")
    monkeypatch.setattr(target, create_autospec(real, side_effect=GymratError(message)))

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert message in err_text(result)
    assert (registry.live(), display_cleanups.live()) == ([], [])
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
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.run_preflight",
        create_autospec(run_preflight, side_effect=GymratError(msg)),
    )

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
# model and effort resolution
# ---------------------------------------------------------------------------


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
            ("--model", "sonnet"),
            SuperviseConfig(model="opus"),
            ("sonnet", None),
            id="model-flag-overrides-config",
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

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert list(built_tools) == [Path(repo).resolve()]
    assert list(built_hooks) == [Path(repo).resolve()]
    call_kwargs = seams.create_driver.call_args.kwargs
    assert call_kwargs.get("tools") is built_tools[Path(repo).resolve()]
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
    monkeypatch.setattr(
        "gymrat.cli.commands.supervise.get_adapter",
        create_autospec(get_adapter, side_effect=adapters.__getitem__),
    )
    seams = install_seams(monkeypatch, config=config)

    result = run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["primary_direction"] == expected
