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
from gymrat.config import BenchlessConfig, StopConfig
from gymrat.errors import GymratError
from gymrat.loop.start import start_session
from gymrat.session import BaselineRecord, append_record
from gymrat.session.budget import Budget, write_budget
from gymrat.session.paths import (
    experiment_worktree_dir,
    lockfile_path,
    session_jsonl_path,
    supervise_lockfile_path,
)
from gymrat.session.workspace import ensure_git_exclude
from gymrat.signals import install_termination_cleanup
from gymrat.supervisor import SupervisionResult, create_claude_driver
from gymrat.supervisor.context import SupervisedSession
from tests._ansi import strip_ansi
from tests.cli._loop_cmds import make_discard_repo
from tests.cli.supervise._fixtures import (
    make_supervision_result,
    session_state_three_iterations,
)
from tests.conftest import hold_lock
from tests.loop.iterate._fixtures import resolved_config
from tests.session.records._fixtures import (
    committed_keep,
    finalize_record,
    iteration_record,
)

runner = CliRunner()

# The fixed ISO-8601 stamp a lockfile fixture carries; its exact value is
# immaterial to the tests, which only care that a live holder is named.
_LOCK_AT = "2026-01-01T00:00:00.000Z"


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
        self.supervise_calls: list[dict[str, object]] = []
        self.reporter_calls: list[dict[str, object]] = []
        self.compose_calls: list[tuple[object, object]] = []


def _config(
    *, stop: StopConfig | None = None, runbook: str | None = "runbook.md"
) -> BenchlessConfig:
    """A benchless config the mocked kickoff and reporter read fields off of."""
    return BenchlessConfig(
        adapter="mitata",
        samples=1,
        timeout_seconds=60,
        unstable_noise_pct=5.0,
        primary="geomean",
        runbook=runbook,
        stop=stop,
    )


def _install_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: BenchlessConfig | None = None,
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

    def fake_resolve(_flags: object, _base_dir: object = None) -> BenchlessConfig:
        return resolved

    def fake_compose(cfg: object, prompt: object = None) -> SimpleNamespace:
        seams.compose_calls.append((cfg, prompt))
        return SimpleNamespace(kickoff="begin optimization", system_prompt_append="system prompt")

    async def fake_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        call = {**kwargs, **dict(zip(("driver", "prompt"), args, strict=False))}
        seams.supervise_calls.append(call)
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

    monkeypatch.setattr("gymrat.cli.supervise.cmd.resolve_benchless_config", fake_resolve)
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
    """Invoke the assembled app's ``supervise`` command with ``args``."""
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


def test_supervise_when_max_minutes_valid_does_pass_it_through_in_context(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("my prompt", "--max-minutes", "30")

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert ctx.max_minutes == 30.0


def test_supervise_when_max_usd_valid_does_pass_it_through_in_context(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("my prompt", "--max-minutes", "30", "--max-usd", "5.50")

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert ctx.max_usd == 5.5


def test_supervise_when_run_does_build_context_with_all_fields(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10", "--max-usd", "2.0")

    assert result.exit_code == 0
    ctx = seams.supervise_calls[0]["context"]
    assert isinstance(ctx, SupervisedSession)
    assert ctx.root == repo
    assert ctx.log_path
    assert ctx.lock_path == lockfile_path(repo)
    assert isinstance(ctx.config, BenchlessConfig)
    assert ctx.deadline_ms > 0
    assert ctx.max_minutes == 10.0
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


def _start_open_session(repo: str) -> None:
    """Start a gymrat session so the experiment worktree and session log exist."""
    start_session(repo, "main", resolved_config())


def _dirty_experiment_worktree(repo: str, *names: str) -> None:
    """Write each of ``names`` as an uncommitted file into the experiment worktree."""
    worktree = Path(experiment_worktree_dir(repo))
    for name in names:
        (worktree / name).write_text("dirty\n", encoding="utf-8")


def _setup_finalized_with_dirty_worktree(repo: str) -> None:
    """A finalized session whose experiment worktree still has uncommitted files."""
    _start_open_session(repo)
    log = session_jsonl_path(repo)
    append_record(log, iteration_record(seq=1))
    append_record(log, committed_keep(seq=1))
    append_record(log, finalize_record())
    _dirty_experiment_worktree(repo, "stale.txt")


def _setup_open_session_missing_worktree(repo: str) -> None:
    """An open session whose experiment worktree directory no longer exists on disk."""
    _start_open_session(repo)
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
    _start_open_session(repo)
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
        pytest.param(_start_open_session, id="clean-worktree"),
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
    assert "  loop   3 iterations · 2 kept · 1 discarded · last -4.2% improved" in result.stdout


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
        line.startswith("  log    ") and "supervisor-1.jsonl" in line
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
            reason="interrupted", ended_by="wall-clock", duration_ms=600_000, cost_usd=1.0
        ),
    )

    result = _run("optimize it", "--max-minutes", "10")

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

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.install_cleanup.call_count >= 1
    (registered,) = seams.install_cleanup.call_args_list[0].args
    assert callable(registered)


# ---------------------------------------------------------------------------
# budget lifecycle
# ---------------------------------------------------------------------------


def test_supervise_when_run_does_write_budget_before_supervise(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """Budget file must exist before the agent's first turn."""
    order: list[str] = []
    seams = _install_seams(monkeypatch)

    original_supervise = seams.supervise_calls

    async def tracking_supervise(*args: object, **kwargs: object) -> SupervisionResult:
        order.append("supervise")
        call = {**kwargs, **dict(zip(("driver", "prompt"), args, strict=False))}
        original_supervise.append(call)
        return make_supervision_result()

    monkeypatch.setattr("gymrat.cli.supervise.cmd.supervise", tracking_supervise)

    original_write = write_budget

    def tracking_write(root: str, budget: object) -> None:
        order.append("write_budget")
        original_write(root, budget)  # type: ignore[arg-type]

    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_budget", tracking_write)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert "write_budget" in order
    assert "supervise" in order
    assert order.index("write_budget") < order.index("supervise")


def test_supervise_when_run_does_write_budget_with_correct_deadline(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    captured_budgets: list[Budget] = []

    def capturing_write(root: str, budget: Budget) -> None:
        captured_budgets.append(budget)

    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_budget", capturing_write)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert len(captured_budgets) == 1
    budget = captured_budgets[0]
    assert budget.max_minutes == 10
    expected_deadline = budget.started_at_ms + 10 * 60_000
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
    order: list[str] = []
    seams = _install_seams(monkeypatch)
    seams.reporter_stop.side_effect = lambda: order.append("reporter_stop")

    def _track_clear(_root: str) -> None:
        order.append("clear_budget")

    monkeypatch.setattr("gymrat.cli.supervise.cmd.clear_budget", _track_clear)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert "clear_budget" in order
    assert "reporter_stop" in order
    assert order.index("clear_budget") < order.index("reporter_stop")


def test_supervise_when_run_does_register_budget_termination_cleanup(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    """A termination signal (SIGTERM, SIGINT) must clear the budget file."""
    seams = _install_seams(monkeypatch)

    def _noop_write(_root: str, _budget: Budget) -> None:
        pass

    def _noop_clear(_root: str) -> None:
        pass

    monkeypatch.setattr("gymrat.cli.supervise.cmd.write_budget", _noop_write)
    monkeypatch.setattr("gymrat.cli.supervise.cmd.clear_budget", _noop_clear)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert seams.install_cleanup.call_count >= 2


# ---------------------------------------------------------------------------
# feasibility check
# ---------------------------------------------------------------------------


def _baseline_record(duration_ms: float | None = None) -> BaselineRecord:
    """A baseline record with an optional wall-clock duration."""
    return BaselineRecord(
        type="baseline",
        at="2026-08-08T14:15:30.000Z",
        label="main",
        samples=({"total_ms": 15200},),
        duration_ms=duration_ms,
    )


def _seed_session_with_baseline(repo: str, *, baseline_duration_ms: float) -> None:
    """Start a session and write a baseline record with the given duration."""
    _start_open_session(repo)
    log = session_jsonl_path(repo)
    append_record(log, _baseline_record(duration_ms=baseline_duration_ms))


def _seed_session_with_iteration(
    repo: str, *, iteration_duration_ms: float, include_baseline: bool = True
) -> None:
    """Start a session and write an iteration record with the given duration.

    When ``include_baseline`` is True (the default), a baseline record without
    a duration is also appended so the session reflects normal usage.
    """
    _start_open_session(repo)
    log = session_jsonl_path(repo)
    if include_baseline:
        append_record(log, _baseline_record())
    append_record(log, iteration_record(duration_ms=iteration_duration_ms))


def test_supervise_when_cap_cannot_fit_one_iterate_does_exit_two_with_arithmetic(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    _seed_session_with_baseline(repo, baseline_duration_ms=1_440_000)

    result = _run("optimize it", "--max-minutes", "30")

    assert result.exit_code == 2
    text = _err_text(result)
    assert "24m" in text
    assert "48m" in text
    assert "30m" in text


def test_supervise_when_cap_cannot_fit_one_iterate_does_hint_at_raising_cap_or_force(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    _seed_session_with_baseline(repo, baseline_duration_ms=1_440_000)

    result = _run("optimize it", "--max-minutes", "30")

    assert result.exit_code == 2
    text = _err_text(result)
    assert "--max-minutes" in text
    assert "--force" in text


def test_supervise_when_session_has_baseline_does_need_one_iterate(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    _seed_session_with_iteration(repo, iteration_duration_ms=2_880_000, include_baseline=True)

    result = _run("optimize it", "--max-minutes", "47")

    assert result.exit_code == 2
    text = _err_text(result)
    assert "48m" in text


def test_supervise_when_session_lacks_baseline_does_need_one_iterate_plus_one_side(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    _seed_session_with_iteration(repo, iteration_duration_ms=2_880_000, include_baseline=False)

    result = _run("optimize it", "--max-minutes", "60")

    assert result.exit_code == 2
    text = _err_text(result)
    assert "72m" in text


def test_supervise_when_force_passed_does_bypass_feasibility_check(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    _seed_session_with_baseline(repo, baseline_duration_ms=1_440_000)

    result = _run("optimize it", "--max-minutes", "30", "--force")

    assert result.exit_code == 0


def test_supervise_when_force_help_does_mention_cap_bypass(repo: str):
    result = _run("--help")

    text = _err_text(result)
    assert "--force" in text
    assert re.search(r"cap.*cannot fit", text, re.IGNORECASE)


def test_supervise_when_no_estimate_available_does_print_iterate_cost_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    text = _err_text(result)
    assert re.search(r"one iterate.*pass", text, re.IGNORECASE)
