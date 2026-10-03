"""Command-level tests for the ``gymrat supervise`` wiring.

These drive the command through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
Git, ``repo_root``, and the supervise lock stay real; the seams the command
composes over — config resolution, kickoff, the Claude driver, the supervisor
run, the run-end exit sequence, the progress reporter, the git-exclude write,
and the signal cleanup — are
replaced at the names ``supervise.cmd`` imports them under, mirroring the
upstream test harness.
"""

import asyncio
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, get_args
from unittest.mock import Mock, create_autospec

import pytest
import typer
from typer.testing import CliRunner, Result

from gymrat.cli.app import app
from gymrat.cli.exit import write_stdout
from gymrat.cli.supervise import cmd as supervise_cmd
from gymrat.cli.supervise.preflight import run_preflight
from gymrat.cli.supervise.progress import SuperviseReporter, create_supervise_reporter
from gymrat.cli.supervise.types import ReadSessionResult
from gymrat.config.types import Effort, ResolvedConfig, StopConfig, SuperviseConfig
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.exec import kill_live_process_groups
from gymrat.loop.start import StartResult
from gymrat.session.paths import (
    lockfile_path,
    supervise_lockfile_path,
)
from gymrat.session.workspace import Worktrees, ensure_git_exclude
from gymrat.signals import install_termination_cleanup
from gymrat.supervisor.claude import create_claude_driver
from gymrat.supervisor.context import SupervisedSession
from gymrat.supervisor.driver import SessionPrompt
from gymrat.supervisor.exit_sequence import ExitPhase, ExitReport, ExitStep
from gymrat.supervisor.hooks import HooksFactory, supervise_hooks_factory
from gymrat.supervisor.supervise import SupervisionResult
from gymrat.supervisor.tools import ToolsFactory, gymrat_tools_factory
from tests._ansi import strip_ansi
from tests._rich import CleanupRegistry, unwrap_panel
from tests.cli._help import help_output
from tests.cli._session import FailingStdoutRunner, closed_stdout_error
from tests.cli.supervise._fixtures import (
    fire_launch,
    make_supervision_result,
    render_frame,
    session_state_three_iterations,
)
from tests.conftest import hold_lock
from tests.session.records._fixtures import (
    empty_session_state,
    session_record,
)
from tests.supervisor._mock_driver import CostStep, create_mock_driver
from tests.telemetry._fixtures import (
    isolate_tracing_provider as _isolate_tracing_provider,  # noqa: F401 -- registers the autouse fixture
)

runner = CliRunner()

# The fixed ISO-8601 stamp a lockfile fixture carries; its exact value is
# immaterial to the tests, which only care that a live holder is named.
_LOCK_AT = "2026-01-01T00:00:00.000Z"

# The wall-clock cap the ``--max-minutes``-driven tests below assert against.
_CAP_MINUTES = 10
_CAP_MS = _CAP_MINUTES * 60_000

# The message the budget-initialization seam fails with when a test makes it explode.
_BUDGET_FAILURE = "budget write failed"

# The message the tracing-setup seam fails with when a test makes it explode.
_TRACING_FAILURE = "tracing exporter unreachable"


# ---------------------------------------------------------------------------
# seam installation
# ---------------------------------------------------------------------------


def _real_reporter() -> SuperviseReporter:
    """The production reporter, built only as an autospec source for ``stop``.

    ``SuperviseReporter.stop`` is a nested closure with no importable name, so it
    can't be targeted directly by ``create_autospec``. Building a real
    (side-effect-free, plain-mode) reporter and taking its attribute gives
    ``create_autospec`` the actual production callable to bind against.
    """
    return create_supervise_reporter(root="/tmp/repo", max_minutes=1.0, mode="plain")


class _Seams:
    """The recorders and doubles a single command run wires up.

    ``supervise_calls`` / ``reporter_calls`` / ``exit_calls`` capture the keyword
    payloads their seams received; ``compose_calls`` records ``(config, prompt)``
    per call. The ``reporter_stop``, ``ensure_git_exclude``, ``install_cleanup``,
    and ``create_driver`` mocks stand in for the side-effecting seams so a test can
    assert they fired. The reporter's observer, exit-phase, and warn sinks append
    to ``observed_events``, ``exit_phases``, and ``warnings``. The fake exit
    sequence returns ``exit_report`` after calling ``exit_hook`` (when set) with
    its recorded call, so a test can act from inside the sequence.
    ``session_result`` is what the reporter's session reader returns right now:
    the reporter shows it at construction and again only after
    ``refresh_session``, so a change made later reaches the summary only
    through a refresh.
    """

    def __init__(self) -> None:
        self.driver = object()
        self.observed_events: list[object] = []
        self.observer: Callable[[object], None] = self.observed_events.append
        self.exit_phases: list[ExitPhase] = []
        self.warnings: list[str] = []
        self.exit_report = ExitReport(steps=(ExitStep(kind="nothing", text="nothing to settle"),))
        self.exit_hook: Callable[[dict[str, Any]], None] | None = None
        self.exit_calls: list[dict[str, Any]] = []
        self.session_result: ReadSessionResult | None = None
        self.final_text: str | None = None
        real_reporter = _real_reporter()
        self.reporter_stop = create_autospec(real_reporter.stop, name="reporter.stop")
        self.ensure_git_exclude = create_autospec(ensure_git_exclude, name="ensure_git_exclude")
        self.create_driver = create_autospec(
            create_claude_driver, name="create_claude_driver", return_value=self.driver
        )
        self.install_cleanup = create_autospec(
            install_termination_cleanup, name="install_termination_cleanup", return_value=Mock()
        )
        self.doctor_gate = Mock()
        self.preflight_calls: list[dict[str, object]] = []
        self.supervise_calls: list[dict[str, object]] = []
        self.reporter_calls: list[dict[str, object]] = []
        self.compose_calls: list[tuple[object, object]] = []

    def record_supervise_call(self, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        call = {**kwargs, **dict(zip(("driver", "prompt"), args, strict=False))}
        self.supervise_calls.append(call)

    def installed_cleanups(self) -> list[Callable[[], None]]:
        """Return every termination cleanup the run installed, in install order."""
        return [call.args[0] for call in self.install_cleanup.call_args_list]


def _config(
    *,
    stop: StopConfig | None = None,
    runbook: str | None = "runbook.md",
    supervise: SuperviseConfig | None = None,
) -> ResolvedConfig:
    """A resolved config the pre-flight returns and the kickoff/reporter read fields off of."""
    return ResolvedConfig(
        bench="npm run bench",
        adapter="mitata",
        samples=1,
        timeout_seconds=60,
        unstable_noise_pct=5.0,
        primary="geomean",
        runbook=runbook,
        stop=stop,
        supervise=supervise,
    )


def _make_start_result(root: str = "/repo", branch: str | None = None) -> StartResult:
    """Build a ``StartResult`` carrying sensible defaults, its session on ``branch`` when given."""
    overrides = {} if branch is None else {"branch": branch}
    rec = session_record(
        worktrees=Worktrees(
            experiment=f"{root}/.gymrat/worktrees/experiment",
            baseline=f"{root}/.gymrat/worktrees/baseline",
        ),
        **overrides,
    )
    return StartResult(
        session=rec,
        state=empty_session_state(),
        resumed=False,
    )


def _install_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: ResolvedConfig | None = None,
    result: SupervisionResult | None = None,
    session_result: ReadSessionResult | None = None,
    final_text: str | None = None,
    raises: Exception | None = None,
    branch: str | None = None,
) -> _Seams:
    """Replace every seam ``supervise.cmd`` composes over, returning the recorders."""
    seams = _Seams()
    seams.session_result = session_result
    seams.final_text = final_text
    resolved = config if config is not None else _config()
    handed_back = result if result is not None else make_supervision_result()

    def fake_preflight(
        *,
        root: str,
        config: object,
        baseline_ref: object = None,
        max_minutes: float,
        force: bool,
    ) -> StartResult:
        seams.preflight_calls.append({
            "root": root,
            "config": config,
            "baseline_ref": baseline_ref,
            "max_minutes": max_minutes,
            "force": force,
        })
        return _make_start_result(root, branch)

    def fake_compose(
        cfg: object,
        prompt: object = None,
        *,
        experiment_worktree: object = None,
    ) -> SimpleNamespace:
        seams.compose_calls.append((cfg, prompt))
        return SimpleNamespace(kickoff="begin optimization", system_prompt_append="system prompt")

    async def fake_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        seams.record_supervise_call(args, kwargs)
        if raises is not None:
            raise raises
        return handed_back

    async def fake_run_exit_sequence(context: object, **kwargs: object) -> ExitReport:
        call = {"context": context, **kwargs}
        seams.exit_calls.append(call)
        if seams.exit_hook is not None:
            seams.exit_hook(call)
        return seams.exit_report

    def fake_reporter(**kwargs: object) -> SimpleNamespace:
        seams.reporter_calls.append(kwargs)
        shown = SimpleNamespace(session=seams.session_result)

        def refresh_session() -> None:
            shown.session = seams.session_result

        return SimpleNamespace(
            observer=seams.observer,
            stop=seams.reporter_stop,
            exit_phase=seams.exit_phases.append,
            warn=seams.warnings.append,
            refresh_session=refresh_session,
            session_result=lambda: shown.session,
            final_text=lambda: seams.final_text,
        )

    def fake_resolve(_flags: object, _base_dir: object = None) -> ResolvedConfig:
        return resolved

    monkeypatch.setattr("gymrat.cli.supervise.cmd.doctor_gate", seams.doctor_gate)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.resolve_config", fake_resolve)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.run_preflight", fake_preflight)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.compose_kickoff", fake_compose)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.create_claude_driver", seams.create_driver)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.supervise", fake_supervise)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.run_exit_sequence", fake_run_exit_sequence)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.create_supervise_reporter", fake_reporter)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.ensure_git_exclude", seams.ensure_git_exclude)
    monkeypatch.setattr(
        "gymrat.cli.supervise.cmd.install_termination_cleanup", seams.install_cleanup
    )
    return seams


def _track_cleanups(monkeypatch: pytest.MonkeyPatch) -> CleanupRegistry:
    """Swap the command's cleanup installer for a registry a test can read at any point."""
    registry = CleanupRegistry()
    monkeypatch.setattr("gymrat.cli.supervise.cmd.install_termination_cleanup", registry.install)
    return registry


def _record_stdout_writes(monkeypatch: pytest.MonkeyPatch, order: list[str], label: str) -> None:
    """Append ``label`` to ``order`` at every stdout write the command makes, then forward it."""

    def tracking_write(data: str) -> None:
        order.append(label)
        write_stdout(data)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_stdout", tracking_write)


def _run(*args: str) -> Result:
    return runner.invoke(app, ["supervise", *args])


def _err_text(result: Result) -> str:
    """The combined stdout+stderr of a run, for flag-name and message probes."""
    return strip_ansi((result.stdout or "") + (result.stderr or ""))


# ---------------------------------------------------------------------------
# flag parsing
# ---------------------------------------------------------------------------


def test_supervise_when_max_minutes_missing_does_exit_two_naming_the_flag(repo: str):
    result = _run("my prompt")

    assert result.exit_code == 2
    assert "--max-minutes" in _err_text(result)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("abc", id="non-numeric"),
        pytest.param("0", id="zero"),
        pytest.param("-5", id="negative"),
        pytest.param("0x10", id="hex"),
        pytest.param("1e-9", id="scientific"),
        pytest.param("35792", id="over-ceiling"),
    ],
)
def test_supervise_when_max_minutes_invalid_does_exit_two_naming_the_flag(repo: str, value: str):
    result = _run("my prompt", "--max-minutes", value)

    assert result.exit_code == 2
    assert "--max-minutes" in _err_text(result)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("abc", id="non-numeric"),
        pytest.param("0", id="zero"),
        pytest.param("-5", id="negative"),
        pytest.param("0x10", id="hex"),
        pytest.param("1e-9", id="scientific"),
    ],
)
def test_supervise_when_max_usd_invalid_does_exit_two_naming_the_flag(repo: str, value: str):
    result = _run("my prompt", "--max-minutes", "10", "--max-usd", value)

    assert result.exit_code == 2
    assert "--max-usd" in _err_text(result)


def test_supervise_when_run_does_build_context_with_all_fields(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    before_ms = time.time() * 1000

    result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES), "--max-usd", "2.0")

    after_ms = time.time() * 1000
    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert ctx.root == repo
    assert re.search(r"\.gymrat[/\\]supervisor-\d+\.jsonl", ctx.log_path)
    assert ctx.lock_path == lockfile_path(repo)
    assert isinstance(ctx.config, ResolvedConfig)
    assert before_ms + _CAP_MS <= ctx.deadline_ms <= after_ms + _CAP_MS
    assert ctx.max_minutes == _CAP_MINUTES
    assert ctx.max_usd == 2.0


# ---------------------------------------------------------------------------
# default log path
# ---------------------------------------------------------------------------


def test_supervise_when_no_log_given_does_default_under_the_session_dir(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert re.search(r"\.gymrat[/\\]supervisor-\d+\.jsonl", result.stderr)


def test_supervise_when_no_log_given_does_ensure_git_exclude_with_root(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    seams.ensure_git_exclude.assert_called_once_with(repo)


def test_supervise_when_log_given_does_use_it_verbatim_and_skip_git_exclude(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    seams = _install_seams(monkeypatch)
    custom = str(tmp_path / "custom.jsonl")

    result = _run("optimize it", "--max-minutes", "10", "--log", custom)

    assert result.exit_code == 0
    assert Path(custom).name in result.stderr
    seams.ensure_git_exclude.assert_not_called()


# ---------------------------------------------------------------------------
# launch line — log path abbreviation
# ---------------------------------------------------------------------------


def test_supervise_when_log_under_home_does_abbreviate_path_in_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    log_path = str(Path.home() / ".gymrat" / "supervisor-1.jsonl")

    result = _run("optimize it", "--max-minutes", "10", "--log", log_path)

    assert result.exit_code == 0
    assert "~/.gymrat/supervisor-1.jsonl" in result.stderr


# ---------------------------------------------------------------------------
# supervise lock
# ---------------------------------------------------------------------------


def test_supervise_when_lock_held_by_live_process_does_exit_two_naming_another_run(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    lock_path = supervise_lockfile_path(repo)
    blocker = hold_lock(
        lock_path,
        holder={"pid": os.getpid(), "command": "supervise", "at": _LOCK_AT},
    )

    try:
        result = _run("optimize it", "--max-minutes", "10")

        assert result.exit_code == 2
        assert re.search(r"another gymrat", result.stderr, re.IGNORECASE)
    finally:
        blocker.release()


# ---------------------------------------------------------------------------
# closing summary
# ---------------------------------------------------------------------------


def test_supervise_when_run_completes_does_print_summary_on_stdout_and_log_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert result.stdout.splitlines()[0] == "✓ completed · 1m 0s · $0.05"
    assert result.stdout.count(".jsonl") == 1
    assert ".jsonl" in result.stderr


def test_supervise_when_session_ends_with_final_text_does_show_the_agent_row(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch, final_text="all done here")

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert "all done here" in result.stdout


def test_supervise_when_session_has_iterations_does_show_them_in_the_summary_loop_row(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    state = session_state_three_iterations(-4.2, "improved", seq=3)
    _install_seams(monkeypatch, session_result=ReadSessionResult(state=state, has_baseline=True))

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert "  loop    3 iterations · 2 kept · 1 discarded · last -4.2% improved" in result.stdout


def test_supervise_when_log_path_is_long_does_print_it_unwrapped(
    repo: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _install_seams(monkeypatch)
    nested = tmp_path / ("supervise-log-directory-" * 3)
    nested.mkdir()
    custom = str(nested / "supervisor-1.jsonl")

    result = _run("optimize it", "--max-minutes", "10", "--log", custom)

    assert result.exit_code == 0
    # On Windows CI tmp_path lives under $HOME, so the display path is ~/…
    # abbreviated.  Check the row is a single unwrapped line.
    assert any(
        line.startswith("  log     ") and "supervisor-1.jsonl" in line
        for line in result.stdout.splitlines()
    )


def test_supervise_when_stdout_is_not_a_tty_does_print_the_summary_without_ansi_codes(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert "\x1b[" not in result.stdout


def _live_mode() -> Literal["live", "plain"]:
    """Stand in for ``resolve_render_mode`` so the log path goes to the dashboard, not stderr."""
    return "live"


def test_supervise_when_stdout_reader_closed_does_exit_zero_without_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch, config=replace(_config(), checks="npm test"))
    monkeypatch.setattr("gymrat.cli.supervise.cmd.resolve_render_mode", _live_mode)

    result = FailingStdoutRunner(closed_stdout_error()).invoke(
        app, ["supervise", "optimize it", "--max-minutes", "10"]
    )

    assert (result.exit_code, result.stderr) == (0, "")


def test_supervise_when_stdout_reader_closed_and_preflight_fails_does_exit_two(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch, config=replace(_config(), checks="npm test"))
    monkeypatch.setattr("gymrat.cli.supervise.cmd.run_preflight", run_preflight)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.resolve_render_mode", _live_mode)

    result = FailingStdoutRunner(closed_stdout_error()).invoke(
        app, ["supervise", "optimize it", "--max-minutes", "10"]
    )

    assert result.exit_code == 2
    assert "Error:" in result.stderr


# ---------------------------------------------------------------------------
# driver, kickoff, and reporter wiring
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_pass_the_claude_driver_to_supervise(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    seams.create_driver.assert_called_once()
    assert seams.supervise_calls[0]["driver"] is seams.driver


def test_supervise_when_driver_session_completes_does_run_the_real_supervisor(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    commands_supervise = supervise_cmd.supervise
    seams = _install_seams(monkeypatch)
    seams.create_driver.return_value = create_mock_driver([CostStep(cost_usd=0.01)])
    monkeypatch.setattr("gymrat.cli.supervise.cmd.supervise", commands_supervise)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.exit_calls[0]["ended_by"] == "session"


def test_supervise_when_prompt_given_does_compose_kickoff_with_it(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize the decoder", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.compose_calls[0][1] == "optimize the decoder"


def test_supervise_when_no_prompt_given_does_compose_kickoff_without_one(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.compose_calls[0][1] is None


def test_supervise_when_run_does_pass_the_reporter_observer_to_supervise(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.supervise_calls[0]["observer"] is seams.observer


def test_supervise_when_config_sets_max_iterations_does_build_reporter_with_it(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch, config=_config(stop=StopConfig(max_iterations=7)))

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["max_iterations"] == 7


def test_supervise_when_max_minutes_fractional_does_forward_it_without_flooring_to_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "5.5")

    assert result.exit_code == 0
    assert seams.reporter_calls[0]["max_minutes"] == 5.5


def _dashboard_title(reporter_kwargs: Mapping[str, Any]) -> str:
    """Build the real dashboard from the arguments the command passed and return its title line."""
    reporter = create_supervise_reporter(**reporter_kwargs)
    try:
        fire_launch(reporter.observer, 1000)
        frame = render_frame(reporter)
        return next(line for line in frame.splitlines() if line.startswith("╭"))
    finally:
        reporter.stop()


def test_supervise_when_live_and_session_has_branch_does_show_it_in_the_dashboard_title(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch, branch="banana")
    monkeypatch.setattr("gymrat.cli.supervise.cmd.resolve_render_mode", _live_mode)

    result = _run("optimize it", "--max-minutes", "10")
    title = _dashboard_title(seams.reporter_calls[0])

    assert result.exit_code == 0
    assert "· branch banana" in title


def test_supervise_when_live_and_session_has_no_branch_does_omit_it_from_the_dashboard_title(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch, branch="")
    monkeypatch.setattr("gymrat.cli.supervise.cmd.resolve_render_mode", _live_mode)

    result = _run("optimize it", "--max-minutes", "10")
    title = _dashboard_title(seams.reporter_calls[0])

    assert result.exit_code == 0
    assert "branch" not in title


def _plain_mode() -> Literal["live", "plain"]:
    """Stand in for ``resolve_render_mode`` so the run reports in plain mode."""
    return "plain"


def _plain_writes(reporter_kwargs: Mapping[str, Any]) -> list[str]:
    """Build the real plain reporter from the command's arguments and return what launch prints."""
    writes: list[str] = []
    reporter = create_supervise_reporter(**reporter_kwargs, plain_write=writes.append)
    try:
        fire_launch(reporter.observer, 1000)
    finally:
        reporter.stop()
    return writes


def test_supervise_when_plain_and_session_has_branch_does_print_no_title(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch, branch="banana")
    monkeypatch.setattr("gymrat.cli.supervise.cmd.resolve_render_mode", _plain_mode)

    result = _run("optimize it", "--max-minutes", "10")
    writes = _plain_writes(seams.reporter_calls[0])

    assert result.exit_code == 0
    assert writes
    assert not any("banana" in line for line in writes)
    assert "banana" not in strip_ansi(result.stderr)


def test_supervise_when_no_color_passed_does_still_run_supervise(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10", "--no-color")

    assert result.exit_code == 0
    assert seams.supervise_calls


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
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10", *flag_args)

    assert result.exit_code == 0
    call_kwargs = seams.doctor_gate.call_args.kwargs
    assert call_kwargs.get("color") is expected_color


def test_supervise_when_doctor_gate_refuses_does_exit_two_before_the_session_starts(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    seams.doctor_gate.side_effect = typer.Exit(TOOL_FAILURE_EXIT_CODE)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == TOOL_FAILURE_EXIT_CODE
    assert seams.supervise_calls == []


def test_supervise_when_supervise_raises_does_still_stop_the_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch, raises=GymratError("boom"))

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    seams.reporter_stop.assert_called()


def test_supervise_when_run_starts_does_build_the_reporter_before_installing_any_cleanup(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    # The dashboard installs its own erase cleanup while it is built, so building
    # it first makes the erase run before the budget release and the process kill.
    _install_seams(monkeypatch)
    registry = _track_cleanups(monkeypatch)
    armed_at_build: list[list[Callable[[], None]]] = []
    build_reporter: Callable[..., object] = supervise_cmd.create_supervise_reporter

    def probing_reporter(**kwargs: object) -> object:
        armed_at_build.append(registry.live())
        return build_reporter(**kwargs)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.create_supervise_reporter", probing_reporter)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert armed_at_build == [[]]


def test_supervise_when_session_runs_does_arm_the_kill_cleanup_only_for_its_duration(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
    registry = _track_cleanups(monkeypatch)
    armed_during_run: list[bool] = []

    async def probing_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        seams.record_supervise_call(args, kwargs)
        armed_during_run.append(any(live is kill_live_process_groups for live in registry.live()))
        return make_supervision_result()

    monkeypatch.setattr("gymrat.cli.supervise.cmd.supervise", probing_supervise)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert armed_during_run == [True]
    assert registry.live() == []


def _exploding_write_budget(*_args: object) -> None:
    """Stand in for the budget write, failing the way a read-only session dir does."""
    raise GymratError(_BUDGET_FAILURE)


def _exploding_setup_tracing(*_args: object, **_kwargs: object) -> tuple[object, ...]:
    """Stand in for the tracing setup, failing the way a bad exporter does."""
    raise GymratError(_TRACING_FAILURE)


@pytest.mark.parametrize(
    ("target", "replacement", "message"),
    [
        pytest.param(
            "gymrat.cli.supervise.cmd.write_budget",
            _exploding_write_budget,
            _BUDGET_FAILURE,
            id="budget-init",
        ),
        pytest.param(
            "gymrat.telemetry.run_spans.setup_tracing",
            _exploding_setup_tracing,
            _TRACING_FAILURE,
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
    seams = _install_seams(monkeypatch)
    registry = _track_cleanups(monkeypatch)
    monkeypatch.setattr(target, replacement)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert message in _err_text(result)
    seams.reporter_stop.assert_called()
    assert registry.live() == []


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
        pytest.param("10", (), {"baseline_ref": None}, id="no-baseline-given"),
        pytest.param(
            "42", ("--force",), {"max_minutes": 42.0, "force": True}, id="force-and-max-minutes"
        ),
    ],
)
def test_supervise_when_run_does_pass_flags_to_preflight(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    max_minutes: str,
    extra_args: tuple[str, ...],
    expected: dict[str, object],
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", max_minutes, *extra_args)

    assert result.exit_code == 0
    call = seams.preflight_calls[0]
    assert call["root"] == repo
    for key, value in expected.items():
        assert call[key] == value


def test_supervise_when_help_does_describe_flags(repo: str):
    result = _run("--help")
    text = _err_text(result)
    flat = unwrap_panel(text)

    assert "--baseline" in text
    assert re.search(r"pin.*freshly opened", flat, re.IGNORECASE)
    assert re.search(r"default.*HEAD", flat, re.IGNORECASE)
    assert re.search(r"ignored.*resumed", flat, re.IGNORECASE)
    assert "--force" in text
    assert re.search(r"cap.*cannot fit.*iteration", flat, re.IGNORECASE)
    assert re.search(r"stop condition.*already met", flat, re.IGNORECASE)
    assert "--max-minutes" in text
    assert re.search(r"counted.*baseline.*recorded", flat, re.IGNORECASE)


@pytest.mark.parametrize(
    ("name", "metavar", "description"),
    [
        pytest.param("[PROMPT]", "<str>", "optimization prompt for the agent", id="prompt"),
        pytest.param("--max-minutes", "<float>", "wall-clock cap in minutes", id="max-minutes"),
        pytest.param("--max-usd", "<float>", "spend cap in USD", id="max-usd"),
        pytest.param("--log", "<str>", "path for the JSONL event log", id="log"),
        pytest.param("--model", "<str>", "model to use for the agent session", id="model"),
        pytest.param("--effort", "<level>", "effort level", id="effort"),
        pytest.param("--allow-dirty", "", "allow launching with uncommitted changes", id="dirty"),
        pytest.param(
            "--no-finalize", "", "leave the session open instead of finalizing it", id="finalize"
        ),
    ],
)
def test_supervise_when_help_does_list_each_flag_with_its_metavar_and_text(
    name: str, metavar: str, description: str
):
    text = help_output("supervise")

    assert re.search(rf"{re.escape(name)}\s+{re.escape(metavar)}\s*{re.escape(description)}", text)


# ---------------------------------------------------------------------------
# step ordering
# ---------------------------------------------------------------------------


def test_supervise_when_preflight_raises_does_exit_two_with_message(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)

    msg = "cap too small"

    def exploding_preflight(**_kwargs: object) -> StartResult:
        raise GymratError(msg)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.run_preflight", exploding_preflight)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "cap too small" in result.stderr


def test_supervise_when_run_does_propagate_resolved_config_to_kickoff_context_and_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    cfg = _config(stop=StopConfig(max_iterations=9))
    seams = _install_seams(monkeypatch, config=cfg)

    result = _run("optimize it", "--max-minutes", "10")

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
        pytest.param("extreme", id="plausible-but-wrong"),
        pytest.param("HIGH", id="wrong-case"),
        pytest.param("", id="empty-string"),
    ],
)
def test_supervise_when_effort_invalid_does_exit_two_with_choice_message(repo: str, bad_value: str):
    result = _run("optimize it", "--max-minutes", "10", "--effort", bad_value)
    flat = unwrap_panel(_err_text(result))

    assert result.exit_code == 2
    assert f"{bad_value!r} is not one of 'low', 'medium', 'high', 'xhigh', 'max'." in flat


@pytest.mark.parametrize("level", get_args(Effort))
def test_supervise_when_effort_level_given_does_pass_it_unchanged_to_the_prompt(
    repo: str, monkeypatch: pytest.MonkeyPatch, level: str
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10", "--effort", level)

    assert result.exit_code == 0
    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert prompt.effort == level


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
    ],
)
def test_supervise_when_run_does_resolve_model_and_effort_from_flag_or_config(
    repo: str,
    monkeypatch: pytest.MonkeyPatch,
    flag_args: tuple[str, ...],
    supervise_config: SuperviseConfig,
    expected: tuple[str | None, str | None],
):
    seams = _install_seams(monkeypatch, config=_config(supervise=supervise_config))

    result = _run("optimize it", "--max-minutes", "10", *flag_args)

    assert result.exit_code == 0
    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    expected_model, expected_effort = expected
    assert prompt.model == expected_model
    assert prompt.effort == expected_effort


# ---------------------------------------------------------------------------
# shell-command ceiling
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_set_command_timeout_to_wall_clock_cap(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    call = seams.supervise_calls[0]
    prompt = call["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert prompt.command_timeout_ms == _CAP_MS


# ---------------------------------------------------------------------------
# session prompt — max_budget_usd
# ---------------------------------------------------------------------------


def test_supervise_when_max_usd_given_does_pass_it_as_max_budget_usd_on_prompt(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10", "--max-usd", "5.0")

    assert result.exit_code == 0
    prompt = seams.supervise_calls[0]["prompt"]
    assert isinstance(prompt, SessionPrompt)
    assert prompt.max_budget_usd == 5.0


# ---------------------------------------------------------------------------
# tools and hooks factory wiring
# ---------------------------------------------------------------------------


def test_supervise_when_run_from_subdirectory_does_pass_factories_for_repo_root(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)
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

    monkeypatch.setattr("gymrat.cli.supervise.cmd.gymrat_tools_factory", _recording_tools_factory)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.supervise_hooks_factory", _record_hooks)
    (Path(repo) / "docs").mkdir()
    monkeypatch.chdir(Path(repo) / "docs")

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert [Path(root).resolve() for root in tools_factory_roots] == [Path(repo).resolve()]
    assert list(built_hooks) == [Path(repo).resolve()]
    call_kwargs = seams.create_driver.call_args.kwargs
    tools = call_kwargs.get("tools")
    assert callable(tools), "create_claude_driver must receive a tools callable"
    server_config: dict[str, Any] = tools(asyncio.Event(), {})  # type: ignore[assignment]  # McpSdkServerConfig is a TypedDict
    assert server_config["name"] == "gymrat"
    assert call_kwargs.get("hooks") is built_hooks[Path(repo).resolve()]
