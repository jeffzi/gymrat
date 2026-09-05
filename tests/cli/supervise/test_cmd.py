"""Command-level tests for the ``gymrat supervise`` wiring.

These drive the command through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
Git, ``repo_root``, and the supervise lock stay real; the seams the command
composes over — config resolution, kickoff, the Claude driver, the supervisor
run, the progress reporter, the git-exclude write, and the signal cleanup — are
replaced at the names ``supervise.cmd`` imports them under, mirroring the
upstream test harness.
"""

import os
import re
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec

import pytest
from typer.testing import CliRunner, Result

from gymrat.cli.app import app
from gymrat.cli.shared import write_and_flush
from gymrat.cli.supervise.progress import ReadSessionResult, create_supervise_reporter
from gymrat.config import ResolvedConfig, StopConfig, SuperviseConfig
from gymrat.errors import GymratError
from gymrat.loop.start import StartResult
from gymrat.session import Worktrees, append_record
from gymrat.session.budget import Budget, read_budget, write_budget
from gymrat.session.clock import now_ms
from gymrat.session.paths import (
    budget_path,
    experiment_worktree_dir,
    lockfile_path,
    session_jsonl_path,
    supervise_lockfile_path,
)
from gymrat.session.workspace import ensure_git_exclude
from gymrat.signals import install_termination_cleanup
from gymrat.supervisor import SessionPrompt, SupervisionResult, create_claude_driver
from gymrat.supervisor.context import SupervisedSession
from tests._ansi import strip_ansi
from tests.cli._loop_cmds import make_discard_repo
from tests.cli.supervise._fixtures import (
    empty_session_state,
    make_supervision_result,
    session_state_three_iterations,
    start_open_session,
)
from tests.conftest import hold_lock
from tests.session.records._fixtures import (
    committed_keep,
    finalize_record,
    iteration_record,
    session_record,
)

runner = CliRunner()

# The fixed ISO-8601 stamp a lockfile fixture carries; its exact value is
# immaterial to the tests, which only care that a live holder is named.
_LOCK_AT = "2026-01-01T00:00:00.000Z"

# The wall-clock cap the ``--max-minutes``-driven tests below assert against.
_CAP_MINUTES = 10
_CAP_MS = _CAP_MINUTES * 60_000


# ---------------------------------------------------------------------------
# seam installation
# ---------------------------------------------------------------------------


def _real_reporter_stop() -> Callable[[], None]:
    """The production reporter's real ``stop`` closure, used only as an autospec source.

    ``SuperviseReporter.stop`` is a nested closure with no importable name, so it
    can't be targeted directly by ``create_autospec``. Building a real (side-effect-free,
    plain-mode) reporter and taking its ``stop`` attribute gives ``create_autospec``
    the actual production callable to bind against.
    """
    return create_supervise_reporter(root="/tmp/repo", max_minutes=1.0, mode="plain").stop


class _Seams:
    """The recorders and doubles a single command run wires up.

    ``supervise_calls`` / ``reporter_calls`` capture the keyword payloads their
    seams received; ``compose_calls`` records ``(config, prompt)`` per call. The
    ``reporter_stop``, ``ensure_git_exclude``, ``install_cleanup``, and
    ``create_driver`` mocks stand in for the side-effecting seams so a test can
    assert they fired.
    """

    def __init__(self) -> None:
        self.driver = object()
        self.observer: Callable[[object], None] = lambda _event: None
        self.session_result: ReadSessionResult | None = None
        self.final_text: str | None = None
        self.reporter_stop = create_autospec(_real_reporter_stop(), name="reporter.stop")
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


def _make_start_result(root: str = "/repo") -> StartResult:
    """Build a ``StartResult`` carrying sensible defaults for the test harness."""
    rec = session_record(
        worktrees=Worktrees(
            experiment=f"{root}/.gymrat/worktrees/experiment",
            baseline=f"{root}/.gymrat/worktrees/baseline",
        ),
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
        seams.preflight_calls.append(
            {
                "root": root,
                "config": config,
                "baseline_ref": baseline_ref,
                "max_minutes": max_minutes,
                "force": force,
            }
        )
        return _make_start_result(root)

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

    def fake_reporter(**kwargs: object) -> SimpleNamespace:
        seams.reporter_calls.append(kwargs)
        return SimpleNamespace(
            observer=seams.observer,
            stop=seams.reporter_stop,
            session_result=lambda: seams.session_result,
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
    monkeypatch.setattr("gymrat.cli.supervise.cmd.create_supervise_reporter", fake_reporter)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.ensure_git_exclude", seams.ensure_git_exclude)
    monkeypatch.setattr(
        "gymrat.cli.supervise.cmd.install_termination_cleanup", seams.install_cleanup
    )
    return seams


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
# dirty-tree guard
# ---------------------------------------------------------------------------


def test_supervise_when_tree_dirty_and_not_allowed_does_exit_two_with_guidance(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    (Path(repo) / "uncommitted.txt").write_text("dirty", encoding="utf-8")

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert re.search(r"dirty|uncommitted|untracked", result.stderr, re.IGNORECASE)
    assert re.search(r"commit|stash", result.stderr, re.IGNORECASE)
    assert "--allow-dirty" in result.stderr


def test_supervise_when_tree_dirty_and_allowed_does_warn_and_proceed(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    (Path(repo) / "uncommitted.txt").write_text("dirty", encoding="utf-8")

    result = _run("optimize it", "--max-minutes", "10", "--allow-dirty")

    assert result.exit_code == 0
    assert re.search(r"dirty|uncommitted|untracked", result.stderr, re.IGNORECASE)


def test_supervise_when_tree_clean_does_not_warn(repo: str, monkeypatch: pytest.MonkeyPatch):
    _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert not re.search(r"dirty|uncommitted|untracked", result.stderr, re.IGNORECASE)


def test_supervise_when_untracked_directory_dirty_does_count_its_files(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    nested = Path(repo) / "new-dir"
    nested.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (nested / name).write_text(name, encoding="utf-8")

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "3" in result.stderr


# ---------------------------------------------------------------------------
# dirty experiment-worktree guard
# ---------------------------------------------------------------------------


def _dirty_experiment_worktree(repo: str, *names: str) -> None:
    worktree = Path(experiment_worktree_dir(repo))
    for name in names:
        (worktree / name).write_text("dirty\n", encoding="utf-8")


def _setup_finalized_with_dirty_worktree(repo: str) -> None:
    """A finalized session whose experiment worktree still has uncommitted files."""
    start_open_session(repo)
    log = session_jsonl_path(repo)
    append_record(log, iteration_record(seq=1))
    append_record(log, committed_keep(seq=1))
    append_record(log, finalize_record())
    _dirty_experiment_worktree(repo, "stale.txt")


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
    _install_seams(monkeypatch)
    make_discard_repo(repo)
    _dirty_experiment_worktree(repo, "scratch.txt")

    result = _run("optimize it", "--max-minutes", "10", *extra_args)

    assert result.exit_code == 2
    text = _err_text(result)
    assert re.search(r"unsettled", text, re.IGNORECASE)
    assert "gymrat keep" in text
    assert "gymrat discard" in text


def test_supervise_when_experiment_worktree_dirty_without_unsettled_does_exit_two_with_iterate_and_discard_hint(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    start_open_session(repo)
    _dirty_experiment_worktree(repo, "a.txt", "b.txt")

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    text = _err_text(result)
    assert "2 unmeasured edit" in text
    assert "gymrat iterate" in text
    assert "gymrat discard" in text


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(_setup_finalized_with_dirty_worktree, id="finalized-session"),
        pytest.param(_setup_open_session_missing_worktree, id="missing-worktree"),
        # An open session whose experiment worktree has no uncommitted changes.
        pytest.param(start_open_session, id="clean-worktree"),
    ],
)
def test_supervise_when_experiment_worktree_guard_finds_no_issue_does_proceed(
    repo: str, monkeypatch: pytest.MonkeyPatch, setup: Callable[[str], None]
):
    _install_seams(monkeypatch)
    setup(repo)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0


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
    """A wrapped path breaks copy-paste, so the log row is never re-flowed."""
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


# ---------------------------------------------------------------------------
# exit codes
# ---------------------------------------------------------------------------


def test_supervise_when_a_cap_ended_the_session_does_exit_one_naming_the_cap(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(
        monkeypatch,
        result=make_supervision_result(
            reason="interrupted", ended_by="wall-clock", duration_ms=_CAP_MS, cost_usd=1.0
        ),
    )

    result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 1
    assert result.stdout.splitlines()[0] == "! interrupted by wall-clock cap · 10m 0s · $1.00"


def test_supervise_when_supervise_raises_does_exit_two_with_message_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch, raises=GymratError("config broken"))

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert "config broken" in result.stderr


def test_supervise_when_outcome_error_with_message_does_exit_two_and_surface_it(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(
        monkeypatch,
        result=make_supervision_result(
            reason="error", duration_ms=5_000, cost_usd=0.03, message="SDK connection lost"
        ),
    )

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert result.stdout.splitlines()[0] == "✗ error · 5s · $0.03"
    assert "SDK connection lost" in result.stderr


@pytest.mark.parametrize(
    "message",
    [pytest.param(None, id="omitted"), pytest.param("", id="empty-string")],
)
def test_supervise_when_outcome_error_without_message_does_exit_two_quietly(
    repo: str, monkeypatch: pytest.MonkeyPatch, message: str | None
):
    _install_seams(
        monkeypatch,
        result=make_supervision_result(
            reason="error", duration_ms=5_000, cost_usd=0.01, message=message
        ),
    )

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert not re.search(r"\berror\b", result.stderr, re.IGNORECASE)


# ---------------------------------------------------------------------------
# driver, kickoff, and reporter wiring
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_pass_the_claude_driver_to_supervise(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    seams.create_driver.assert_called_once_with()
    assert seams.supervise_calls[0]["driver"] is seams.driver


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


def test_supervise_when_run_completes_does_stop_the_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.reporter_stop.called


def test_supervise_when_run_completes_does_stop_reporter_before_printing_summary(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """The reporter must be stopped before the summary is printed.

    Without this ordering, the summary text appends to the still-open status
    row, corrupting the output.
    """
    order: list[str] = []
    seams = _install_seams(monkeypatch)
    seams.reporter_stop.side_effect = lambda: order.append("stop")

    original_waf = write_and_flush

    def tracking_waf(stream: Any, data: str) -> None:
        if stream is sys.stdout:
            order.append("write")
        original_waf(stream, data)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_and_flush", tracking_waf)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert "stop" in order, "reporter.stop() was never called"
    assert "write" in order, "summary was never written to stdout"
    stop_idx = order.index("stop")
    write_idx = order.index("write")
    assert stop_idx < write_idx, (
        f"reporter.stop() at index {stop_idx} must precede summary write at {write_idx}; "
        f"order was {order}"
    )


def test_supervise_when_supervise_raises_does_still_stop_the_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch, raises=GymratError("boom"))

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert seams.reporter_stop.called


def test_supervise_when_run_does_register_a_termination_cleanup(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    _run("optimize it", "--max-minutes", "10")

    (registered,) = seams.install_cleanup.call_args_list[0].args
    registered()

    assert seams.reporter_stop.called


# ---------------------------------------------------------------------------
# budget lifecycle
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_write_budget_before_supervise(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """Budget file must exist and be live before the agent's first turn."""
    seams = _install_seams(monkeypatch)
    seen_budgets: list[Budget | None] = []

    async def probing_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        seams.record_supervise_call(args, kwargs)
        seen_budgets.append(read_budget(repo, now_ms=now_ms()))
        return make_supervision_result()

    monkeypatch.setattr("gymrat.cli.supervise.cmd.supervise", probing_supervise)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert len(seen_budgets) == 1
    assert seen_budgets[0] is not None


def test_supervise_when_run_does_write_budget_with_correct_deadline(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    captured_budgets: list[Budget] = []

    def capturing_write(root: str, budget: Budget) -> None:
        captured_budgets.append(budget)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_budget", capturing_write)

    result = _run("optimize it", "--max-minutes", str(_CAP_MINUTES))

    assert result.exit_code == 0
    assert len(captured_budgets) == 1
    budget = captured_budgets[0]
    assert budget.max_minutes == _CAP_MINUTES
    expected_deadline = budget.started_at_ms + _CAP_MS
    assert budget.deadline_ms == expected_deadline


def test_supervise_when_run_completes_does_clear_budget(repo: str, monkeypatch: pytest.MonkeyPatch):
    _install_seams(monkeypatch)
    cleared: list[str] = []
    monkeypatch.setattr("gymrat.cli.supervise.cmd.clear_budget", cleared.append)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert len(cleared) == 1


def test_supervise_when_supervise_raises_does_still_clear_budget(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch, raises=GymratError("boom"))
    cleared: list[str] = []
    monkeypatch.setattr("gymrat.cli.supervise.cmd.clear_budget", cleared.append)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert len(cleared) >= 1


def test_supervise_when_run_does_clear_budget_before_stopping_reporter(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """Budget must be cleared before the reporter stops."""
    seams = _install_seams(monkeypatch)
    budget_gone_at_stop: list[bool] = []

    def probing_stop() -> None:
        budget_gone_at_stop.append(not Path(budget_path(repo)).exists())

    seams.reporter_stop.side_effect = probing_stop

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert budget_gone_at_stop == [True]


def test_supervise_when_run_does_register_budget_termination_cleanup(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """A termination signal (SIGTERM, SIGINT) must clear the budget file."""
    seams = _install_seams(monkeypatch)

    _run("optimize it", "--max-minutes", "10")
    (registered,) = seams.install_cleanup.call_args_list[1].args
    write_budget(repo, Budget(started_at_ms=0.0, max_minutes=10, deadline_ms=600_000.0))
    registered()

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
    flat = re.sub(r"[│╭╮╰╯─\s]+", " ", text)
    # --baseline
    assert "--baseline" in text
    assert re.search(r"pin.*freshly opened", flat, re.IGNORECASE)
    assert re.search(r"default.*HEAD", flat, re.IGNORECASE)
    assert re.search(r"ignored.*resumed", flat, re.IGNORECASE)
    # --force
    assert "--force" in text
    assert re.search(r"cap.*cannot fit.*iteration", flat, re.IGNORECASE)
    assert re.search(r"stop condition.*already met", flat, re.IGNORECASE)
    # --max-minutes
    assert "--max-minutes" in text
    assert re.search(r"counted.*baseline.*recorded", flat, re.IGNORECASE)


# ---------------------------------------------------------------------------
# step ordering
# ---------------------------------------------------------------------------


def test_supervise_when_preflight_raises_does_exit_two_with_message(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """A pre-flight error (feasibility, stop condition, doctor) surfaces on stderr."""
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
    """The same resolved config reaches kickoff composition, the session context, and the reporter's ``max_iterations`` — sourced from ``config.stop``, not a separate field."""
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
def test_supervise_when_effort_invalid_does_exit_two_with_expected_message(
    repo: str, bad_value: str
):
    result = _run("optimize it", "--max-minutes", "10", "--effort", bad_value)

    assert result.exit_code == 2
    text = _err_text(result)
    assert '"low", "medium", "high", "xhigh" or "max"' in text


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
