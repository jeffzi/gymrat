"""Tests for the ``gymrat probe`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` against a real
scratch repository, a real session log, and the real ``probe_session`` engine.
The one seam replaced is the measurement engine — it shells out to the
consumer's bench script, which no test here can run — so the option surface, the
lock, the progress reporter, the budget trailer, the JSON document, and every
engine refusal are exercised end to end.

The signal test is the exception: it runs the CLI out of process against a real
shell bench, because a process group can only be killed by a real signal.
"""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gymrat.cli.app import app
from gymrat.loop.probe import PROBE_DEFAULT_SAMPLES
from gymrat.progress_events import PrepareFinished, PrepareStarted
from gymrat.session import append_record, experiment_worktree_dir, session_jsonl_path
from gymrat.session.paths import lockfile_path, repo_root
from tests._ansi import strip_ansi
from tests._cli import ENTRY, no_color_env
from tests._git import git
from tests._process_helpers import is_alive
from tests.cli._budget import install_budget, install_tight_budget
from tests.cli._loop_cmds import last_command_record, plain_lines, write_config
from tests.conftest import hold_lock
from tests.loop._probe import MeasureRecorder, baseline_of, install_measure, measurement, only_call
from tests.loop.settle._fixtures import start_with
from tests.session.records._fixtures import finalize_record, iteration_record

runner = CliRunner()

FILTER = "sh bench.sh --filter {names}"
"""The rerun template a scoped probe interpolates its metric names into."""

PREPARE_MILESTONES = (
    PrepareStarted(label="experiment", at_ms=0),
    PrepareFinished(label="experiment", at_ms=1_000),
)
"""The one prepare step the stubbed engine reports through the command's progress callback."""


@pytest.fixture
def measure(monkeypatch: pytest.MonkeyPatch) -> MeasureRecorder:
    """The stubbed measurement engine, reporting one prepare milestone then a canned result."""
    return install_measure(
        monkeypatch, measurement(adapter="metric-lines"), progress=PREPARE_MILESTONES
    )


@pytest.fixture
def probe_repo(repo: str) -> str:
    """A scratch repo with an open session, a recorded baseline, and a filter template."""
    start_with(repo, (baseline_of(),))
    write_config(repo, filter=FILTER)
    return repo


# ---------------------------------------------------------------------------
# the option surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "option",
    [
        pytest.param(["--bench", "sh bench.sh"], id="bench"),
        pytest.param(["--prepare", "make"], id="prepare"),
        pytest.param(["--adapter", "mitata"], id="adapter"),
        pytest.param(["--timeout", "30"], id="timeout"),
        pytest.param(["--bogus"], id="unknown"),
    ],
)
@pytest.mark.usefixtures("probe_repo")
def test_probe_command_when_unsupported_option_given_does_exit_two_as_a_usage_error(
    option: list[str],
):
    result = runner.invoke(app, ["probe", *option])

    assert result.exit_code == 2
    assert "No such option" in result.stderr


@pytest.mark.parametrize(
    "option",
    [
        pytest.param(["--debug"], id="debug"),
        pytest.param(["--format", "text"], id="format"),
        pytest.param(["--color"], id="color"),
        pytest.param(["--no-color"], id="no-color"),
        pytest.param(["--samples", "3"], id="samples"),
        pytest.param(["-c", "gymrat.toml"], id="config-short"),
        pytest.param(["--config", "gymrat.toml"], id="config-long"),
    ],
)
@pytest.mark.usefixtures("probe_repo", "measure")
def test_probe_command_when_supported_option_given_does_complete(option: list[str]):

    result = runner.invoke(app, ["probe", *option])

    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# what gets benched
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("probe_repo")
def test_probe_command_when_no_names_given_does_bench_the_whole_bench_at_the_probe_default(
    measure: MeasureRecorder,
):

    result = runner.invoke(app, ["probe"])

    assert result.exit_code == 0
    assert only_call(measure).bench == "npm run bench"
    assert only_call(measure).samples == PROBE_DEFAULT_SAMPLES


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX quoting only")
@pytest.mark.usefixtures("probe_repo")
def test_probe_command_when_names_and_samples_given_does_scope_the_bench_to_them(
    measure: MeasureRecorder,
):

    result = runner.invoke(app, ["probe", "total_ms", "decode large payload", "--samples", "3"])

    assert result.exit_code == 0
    assert only_call(measure).bench == "sh bench.sh --filter total_ms 'decode large payload'"
    assert only_call(measure).samples == 3


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("probe_repo", "measure")
def test_probe_command_when_run_does_render_the_report_to_stdout():

    result = runner.invoke(app, ["probe"])

    assert result.exit_code == 0
    output = strip_ansi(result.stdout)
    assert "gymrat probe · experiment" in output
    assert "total_ms" in output


@pytest.mark.usefixtures("probe_repo", "measure")
def test_probe_command_when_run_does_route_progress_milestones_to_stderr():

    result = runner.invoke(app, ["probe"])

    assert result.exit_code == 0
    assert "prepared experiment" in strip_ansi(result.stderr)
    assert "prepared experiment" not in strip_ansi(result.stdout)


@pytest.mark.parametrize(
    "install",
    [
        pytest.param(install_budget, id="budget-active"),
        pytest.param(None, id="no-budget"),
    ],
)
@pytest.mark.usefixtures("measure")
def test_probe_command_when_budget_state_varies_does_render_time_left_line_accordingly(
    probe_repo: str, monkeypatch: pytest.MonkeyPatch, install: Callable[..., None] | None
):
    if install is not None:
        install(probe_repo, monkeypatch)

    result = runner.invoke(app, ["probe"])

    assert result.exit_code == 0
    if install is None:
        assert "left of" not in result.stdout
    else:
        lines = plain_lines(result.stdout)
        assert "left of 30m" in lines[-1]


# ---------------------------------------------------------------------------
# --format json
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("probe_repo", "measure")
def test_probe_command_when_format_json_does_emit_the_result_as_a_json_document():

    result = runner.invoke(app, ["probe", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["scoped"] is False
    assert doc["names"] == []
    assert doc["samples"] == PROBE_DEFAULT_SAMPLES
    assert doc["metrics"]["total_ms"] == {
        "median": 90.0,
        "spread": 2.0,
        "reference_median": 100.0,
        "delta_pct": pytest.approx(-10.0),
    }


@pytest.mark.usefixtures("probe_repo", "measure")
def test_probe_command_when_format_json_and_scoped_does_name_the_probed_metrics():

    result = runner.invoke(app, ["probe", "total_ms", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert doc["scoped"] is True
    assert doc["names"] == ["total_ms"]


@pytest.mark.parametrize(
    "install",
    [
        pytest.param(install_budget, id="budget-active"),
        pytest.param(None, id="no-budget"),
    ],
)
@pytest.mark.usefixtures("measure")
def test_probe_command_when_format_json_and_budget_state_varies_does_reflect_budget_key(
    probe_repo: str, monkeypatch: pytest.MonkeyPatch, install: Callable[..., None] | None
):
    if install is not None:
        install(probe_repo, monkeypatch)

    result = runner.invoke(app, ["probe", "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    if install is None:
        assert "budget" not in doc
    else:
        assert doc["budget"]["cap_minutes"] == 30
        assert isinstance(doc["budget"]["remaining_seconds"], int)


# ---------------------------------------------------------------------------
# duration warnings — the per-side estimate, taken before the lock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("duration_ms", "warns"),
    [
        pytest.param(720_000, True, id="half-the-estimate-outlasts-the-budget"),
        pytest.param(400_000, False, id="half-the-estimate-still-fits"),
    ],
)
@pytest.mark.usefixtures("measure")
def test_probe_command_when_budget_tight_does_warn_on_the_halved_estimate(
    probe_repo: str, monkeypatch: pytest.MonkeyPatch, duration_ms: int, warns: bool
):
    install_tight_budget(probe_repo, monkeypatch)
    append_record(session_jsonl_path(probe_repo), iteration_record(duration_ms=duration_ms))

    result = runner.invoke(app, ["probe"])

    assert result.exit_code == 0
    assert ("warning" in strip_ansi(result.stderr).lower()) is warns


# ---------------------------------------------------------------------------
# the repository lock
# ---------------------------------------------------------------------------


def test_probe_command_when_rival_lock_held_does_exit_two_without_benching(
    probe_repo: str, measure: MeasureRecorder
):
    blocker = hold_lock(
        lockfile_path(repo_root(probe_repo)),
        holder={"pid": os.getpid(), "command": "iterate", "at": "2026-01-01T00:00:00.000Z"},
    )

    try:
        result = runner.invoke(app, ["probe"])
    finally:
        blocker.release()

    assert result.exit_code == 2
    assert f"PID {os.getpid()}" in strip_ansi(result.stderr)
    assert measure.calls == []


# ---------------------------------------------------------------------------
# refusals — exit code, message, and the recorded reason
# ---------------------------------------------------------------------------


def _no_session(repo: str) -> None:
    """A configured repository where no session was ever opened."""
    write_config(repo, filter=FILTER)


def _finalized_session(repo: str) -> None:
    """A configured repository whose session was closed by a finalize record."""
    start_with(repo, (baseline_of(), finalize_record()))
    write_config(repo, filter=FILTER)


def _no_filter(repo: str) -> None:
    """An open session with a baseline but no filter template to scope a probe."""
    start_with(repo, (baseline_of(),))
    write_config(repo)


def _no_baseline(repo: str) -> None:
    """An open session with a filter template but nothing recorded to compare against."""
    start_with(repo)
    write_config(repo, filter=FILTER)


REFUSALS = [
    pytest.param(_no_session, ["probe"], "gymrat start", None, id="no-session"),
    pytest.param(_finalized_session, ["probe"], "was finalized onto", "finalized", id="finalized"),
    pytest.param(
        _no_filter,
        ["probe", "total_ms"],
        "filter is not configured",
        "no-filter",
        id="no-filter",
    ),
    pytest.param(
        _no_baseline,
        ["probe"],
        "gymrat measure --record",
        "no-baseline",
        id="no-baseline",
    ),
]
"""Every way a probe refuses: the setup, the argv, a message fragment, and the recorded
reason — ``None`` for the one refusal (``no-session``) that writes no command record."""


@pytest.mark.parametrize(("setup", "argv", "fragment", "reason"), REFUSALS)
def test_probe_command_when_the_session_cannot_be_probed_does_exit_two_with_the_engine_message(
    *,
    repo: str,
    measure: MeasureRecorder,
    setup: Callable[[str], None],
    argv: list[str],
    fragment: str,
    reason: str | None,
):
    setup(repo)

    result = runner.invoke(app, argv)

    assert result.exit_code == 2
    assert fragment in strip_ansi(result.stderr)
    assert result.stdout == ""
    assert measure.calls == []
    if reason is not None:
        cmd = last_command_record(repo)
        assert cmd.name == "probe"
        assert cmd.exit_code == 2
        assert cmd.reason == reason


@pytest.mark.parametrize(
    ("argv", "names", "samples"),
    [
        pytest.param([], [], None, id="defaults"),
        pytest.param(["total_ms", "--samples", "3"], ["total_ms"], 3, id="names-and-samples"),
    ],
)
@pytest.mark.usefixtures("measure")
def test_probe_command_when_run_completes_does_record_the_trace_with_names_and_samples(
    probe_repo: str,
    argv: list[str],
    names: list[str],
    samples: int | None,
):

    result = runner.invoke(app, ["probe", *argv])

    assert result.exit_code == 0
    cmd = last_command_record(probe_repo)
    assert cmd.name == "probe"
    assert cmd.args["names"] == names
    assert cmd.args["samples"] == samples
    assert cmd.exit_code == 0
    assert cmd.reason is None


# ---------------------------------------------------------------------------
# signal handling
# ---------------------------------------------------------------------------


_TRACKED_BENCH = """#!/bin/sh
echo $$ > bench.pid
echo 'METRIC total_ms=1'
sleep 120
"""
"""A bench that records its own pid, emits one metric, then blocks past any test."""


def _wait_for_pid(path: Path, timeout_s: float = 60.0) -> int:
    """Poll ``path`` until the bench has written a complete pid into it."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            raw = path.read_text().strip()
        except FileNotFoundError:
            raw = ""
        if raw.isdigit():
            return int(raw)
        if time.monotonic() > deadline:
            message = f"bench pid never appeared at {path}"
            raise AssertionError(message)
        time.sleep(0.05)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")
@pytest.mark.parametrize(
    ("signal_number", "expected_code"),
    [
        pytest.param(signal.SIGINT, 130, id="sigint"),
        pytest.param(signal.SIGTERM, 143, id="sigterm"),
    ],
)
def test_probe_command_when_signalled_mid_bench_does_kill_the_bench_and_exit_128_plus_signal(
    repo: str, signal_number: int, expected_code: int
):
    Path(repo, "bench.sh").write_text(_TRACKED_BENCH, encoding="utf-8")
    write_config(repo, bench="sh bench.sh", adapter="metric-lines", timeout_seconds=300)
    git(repo, "add", "bench.sh", "gymrat.toml")
    git(repo, "commit", "-m", "bench harness")
    subprocess.run(  # noqa: S603
        [*ENTRY, "start", "main"], cwd=repo, env=no_color_env(), capture_output=True, check=True
    )
    append_record(session_jsonl_path(repo), baseline_of())

    proc = subprocess.Popen(  # noqa: S603
        [*ENTRY, "probe"],
        cwd=repo,
        env=no_color_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        bench_pid = _wait_for_pid(Path(experiment_worktree_dir(repo), "bench.pid"))
        proc.send_signal(signal_number)
        proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == expected_code
    deadline = time.monotonic() + 30
    while is_alive(bench_pid):
        assert time.monotonic() < deadline, f"bench {bench_pid} outlived the CLI"
        time.sleep(0.05)
