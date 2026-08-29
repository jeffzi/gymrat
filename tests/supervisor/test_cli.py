"""Command-level tests for the ``gymrat supervise`` wiring.

These drive the command through :class:`typer.testing.CliRunner` against a
throwaway repository from the shared ``create_scratch_repo`` factory, so the
suite is order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.
Git, ``repo_root``, and the supervise lock stay real; the seams the command
composes over — config resolution, kickoff, the Claude driver, the supervisor
run, the progress reporter, the git-exclude write, and the signal cleanup — are
replaced at the names ``supervise_cmd`` imports them under, mirroring the
upstream test harness.
"""

import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import Mock, create_autospec

import pytest
from typer.testing import CliRunner, Result

from gymrat.cli.app import app
from gymrat.cli.shared import write_and_flush
from gymrat.cli.supervise_progress import create_supervise_reporter
from gymrat.config import BenchlessConfig, StopConfig
from gymrat.errors import GymratError
from gymrat.session.paths import supervise_lockfile_path
from gymrat.session.workspace import ensure_git_exclude
from gymrat.signals import install_termination_cleanup
from gymrat.supervisor import SessionOutcome, SupervisionResult, create_claude_driver

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


def _result(
    *,
    reason: Literal["completed", "error", "interrupted"] = "completed",
    ended_by: Literal["session", "spend-cap", "wall-clock"] = "session",
    duration_ms: int = 60_000,
    cost_usd: float = 0.05,
    message: str | None = None,
) -> SupervisionResult:
    """A ``SupervisionResult`` the mocked ``supervise`` hands back."""
    outcome = SessionOutcome(reason=reason, cost_usd=cost_usd, message=message)
    return SupervisionResult(
        outcome=outcome, ended_by=ended_by, duration_ms=duration_ms, cost_usd=cost_usd
    )


def _install_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: BenchlessConfig | None = None,
    result: SupervisionResult | None = None,
    raises: Exception | None = None,
) -> _Seams:
    """Replace every seam ``supervise_cmd`` composes over, returning the recorders."""
    seams = _Seams()
    resolved = config if config is not None else _config()
    handed_back = result if result is not None else _result()

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
        return SimpleNamespace(observer=seams.observer, stop=seams.reporter_stop)

    monkeypatch.setattr("gymrat.cli.supervise_cmd.resolve_benchless_config", fake_resolve)
    monkeypatch.setattr("gymrat.cli.supervise_cmd.compose_kickoff", fake_compose)
    monkeypatch.setattr("gymrat.cli.supervise_cmd.create_claude_driver", seams.create_driver)
    monkeypatch.setattr("gymrat.cli.supervise_cmd.supervise", fake_supervise)
    monkeypatch.setattr("gymrat.cli.supervise_cmd.create_supervise_reporter", fake_reporter)
    monkeypatch.setattr("gymrat.cli.supervise_cmd.ensure_git_exclude", seams.ensure_git_exclude)
    monkeypatch.setattr(
        "gymrat.cli.supervise_cmd.install_termination_cleanup", seams.install_cleanup
    )
    return seams


@pytest.fixture
def repo(create_scratch_repo: Callable[[], str], monkeypatch: pytest.MonkeyPatch) -> str:
    """A fresh scratch repository, chdir'd into so the command runs there."""
    root = create_scratch_repo()
    monkeypatch.chdir(root)
    return root


def _run(*args: str) -> Result:
    """Invoke the assembled app's ``supervise`` command with ``args``."""
    return runner.invoke(app, ["supervise", *args])


def _err_text(result: Result) -> str:
    """The combined stdout+stderr of a run, for flag-name and message probes."""
    return (result.stdout or "") + (result.stderr or "")


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


def test_supervise_when_max_minutes_valid_does_pass_it_through_as_a_number(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("my prompt", "--max-minutes", "30")

    assert result.exit_code == 0
    assert seams.supervise_calls[0]["max_minutes"] == 30.0


def test_supervise_when_max_usd_valid_does_pass_it_through_as_a_number(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    seams = _install_seams(monkeypatch)

    result = _run("my prompt", "--max-minutes", "30", "--max-usd", "5.50")

    assert result.exit_code == 0
    assert seams.supervise_calls[0]["max_usd"] == 5.5


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
    assert custom in result.stderr
    seams.ensure_git_exclude.assert_not_called()


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
# supervise lock
# ---------------------------------------------------------------------------


def test_supervise_when_lock_held_by_live_process_does_exit_two_naming_another_run(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)
    lock_path = Path(supervise_lockfile_path(repo))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "command": "supervise", "at": _LOCK_AT}),
        encoding="utf-8",
    )

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
    assert re.search(r"another gymrat", result.stderr, re.IGNORECASE)


# ---------------------------------------------------------------------------
# closing summary
# ---------------------------------------------------------------------------


def test_supervise_when_run_completes_does_print_summary_on_stdout_and_log_on_stderr(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert re.search(r"session", result.stdout, re.IGNORECASE)
    assert re.search(r"duration|time", result.stdout, re.IGNORECASE)
    assert re.search(r"\$?\d+\.\d+|cost", result.stdout, re.IGNORECASE)
    assert ".jsonl" in result.stdout
    assert ".jsonl" in result.stderr


def test_supervise_when_outcome_interrupted_does_name_it_in_the_summary(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch, result=_result(reason="interrupted", cost_usd=0.02))

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0
    assert re.search(r"interrupted", _err_text(result), re.IGNORECASE)


# ---------------------------------------------------------------------------
# exit codes
# ---------------------------------------------------------------------------


def test_supervise_when_session_completes_does_exit_zero(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(monkeypatch)

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 0


def test_supervise_when_a_cap_ended_the_session_does_exit_one(
    repo: str, monkeypatch: pytest.MonkeyPatch
):
    _install_seams(
        monkeypatch,
        result=_result(ended_by="wall-clock", duration_ms=600_000, cost_usd=1.0),
    )

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 1


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
        result=_result(
            reason="error", duration_ms=5_000, cost_usd=0.03, message="SDK connection lost"
        ),
    )

    result = _run("optimize it", "--max-minutes", "10")

    assert result.exit_code == 2
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
        result=_result(reason="error", duration_ms=5_000, cost_usd=0.01, message=message),
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

    monkeypatch.setattr("gymrat.cli.supervise_cmd.write_and_flush", tracking_waf)

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
    assert seams.install_cleanup.call_count == 1
    (registered,) = seams.install_cleanup.call_args.args
    assert callable(registered)
