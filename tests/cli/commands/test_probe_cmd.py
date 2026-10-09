"""Tests for the ``gymrat probe`` command wiring.

These drive the command through :class:`typer.testing.CliRunner` against a real
scratch repository, a real session log, and the real ``probe_session`` engine.
The one seam replaced is the measurement engine — it shells out to the
consumer's bench script, which no test here can run — so the option surface, the
lock, the progress reporter, the JSON document, and every engine refusal are
exercised end to end; the budget trailer, the tight-budget warning on the halved
estimate, and the finalized refusal are pinned with every other command's in
``test_session_cmds``.

The signal test is the exception: it runs the CLI out of process against a real
shell bench, because a process group can only be killed by a real signal.
"""

import json
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.cli.app import app
from gymrat.loop.probe import PROBE_DEFAULT_SAMPLES
from gymrat.progress_events import PrepareFinished, PrepareStarted
from gymrat.session.paths import (
    experiment_worktree_dir,
    lockfile_path,
    repo_root,
    session_jsonl_path,
)
from tests._ansi import (
    strip_ansi,
    stripped_lines,
)
from tests._cli import run_cli
from tests._git import run_git
from tests._lock import FIXED_HOLDER_AT, hold_lock
from tests._process_helpers import (
    wait_for_pid_file_blocking,
    wait_until_dead_blocking,
)
from tests.cli._budget import SUPERVISED_HINT
from tests.cli._session import (
    last_command_record,
    open_probe_session,
    runner,
    stub_probe_measure,
    write_bench_config,
)
from tests.cli._signalled_cli import pid_recording_script, spawned_gymrat, stop_by_signal
from tests.loop._probe import (
    BASELINE_SAMPLES,
    MeasureRecorder,
    only_call,
)
from tests.loop._settle import start_with
from tests.session._budget import install_tight_budget
from tests.session.records._fixtures import (
    append_records,
    baseline_record,
    iteration_record,
)

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
    return stub_probe_measure(monkeypatch, PREPARE_MILESTONES)


@pytest.fixture
def probe_repo(repo: str) -> str:
    """A scratch repo with an open session, a recorded baseline, and a filter template."""
    open_probe_session(repo, filter=FILTER)
    return repo


# ---------------------------------------------------------------------------
# the option surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "option",
    [
        pytest.param(["--format", "text"], id="format"),
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


@pytest.mark.parametrize(
    ("args", "bench", "samples", "traced"),
    [
        pytest.param(
            [],
            "npm run bench",
            PROBE_DEFAULT_SAMPLES,
            {"names": [], "samples": None},
            id="whole-bench-at-probe-default",
        ),
        pytest.param(
            ["total_ms", "decode large payload", "--samples", "3"],
            "sh bench.sh --filter total_ms 'decode large payload'",
            3,
            {"names": ["total_ms", "decode large payload"], "samples": 3},
            id="names-and-samples-scope-the-bench",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX quoting only"),
        ),
    ],
)
def test_probe_command_when_run_does_bench_and_trace_the_scope_and_samples_asked_for(
    *,
    probe_repo: str,
    args: list[str],
    bench: str,
    samples: int,
    traced: dict[str, object],
    measure: MeasureRecorder,
):
    result = runner.invoke(app, ["probe", *args])

    assert result.exit_code == 0
    run = only_call(measure).run.sampling
    assert (run.bench, run.samples) == (bench, samples)
    cmd = last_command_record(probe_repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("probe", 0, None)
    assert {key: cmd.args[key] for key in traced} == traced


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("probe_repo", "measure")
def test_probe_command_when_run_does_keep_the_report_on_stdout_apart_from_progress():
    result = runner.invoke(app, ["probe"])

    assert result.exit_code == 0
    output = strip_ansi(result.stdout)
    assert "gymrat probe · experiment" in output
    assert "total_ms" in output
    assert "prepared experiment" not in output
    assert "prepared experiment" in strip_ansi(result.stderr)


# ---------------------------------------------------------------------------
# --format json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("names", "scoped"),
    [
        pytest.param([], False, id="whole-bench"),
        pytest.param(["total_ms"], True, id="scoped"),
    ],
)
@pytest.mark.usefixtures("probe_repo", "measure")
def test_probe_command_when_format_json_does_emit_the_result_as_a_json_document(
    names: list[str], scoped: bool
):
    result = runner.invoke(app, ["probe", *names, "--format", "json"])

    assert result.exit_code == 0
    doc = json.loads(result.stdout)
    assert (doc["scoped"], doc["names"], doc["samples"]) == (scoped, names, PROBE_DEFAULT_SAMPLES)
    assert doc["metrics"]["total_ms"] == {
        "median": 90.0,
        "spread": 2.0,
        "reference_median": 100.0,
        "delta_pct": pytest.approx(-10.0),
    }


# ---------------------------------------------------------------------------
# refused while a supervised run is live
# ---------------------------------------------------------------------------


@pytest.fixture
def supervised_probe_repo(probe_repo: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """A probe-ready repo under a tight live budget its last iteration outlasts, run from the shell."""
    install_tight_budget(probe_repo, monkeypatch)
    append_records(probe_repo, iteration_record(duration_ms=720_000))
    return probe_repo


def test_probe_command_when_supervised_run_live_does_refuse_before_benching_or_warning(
    supervised_probe_repo: str, measure: MeasureRecorder
):
    result = runner.invoke(app, ["probe"])

    lines = [line for line in stripped_lines(result.stderr, keep_blank=False) if line.strip()]
    assert len(lines) == 2
    assert "a supervised run is live; use the probe tool" in lines[0]
    assert SUPERVISED_HINT in lines[1]
    assert measure.calls == []


# ---------------------------------------------------------------------------
# the repository lock
# ---------------------------------------------------------------------------


def test_probe_command_when_rival_lock_held_does_exit_two_without_benching(
    probe_repo: str, measure: MeasureRecorder
):
    blocker = hold_lock(
        lockfile_path(repo_root(probe_repo)),
        holder={"pid": os.getpid(), "command": "iterate", "at": FIXED_HOLDER_AT},
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
    write_bench_config(repo, filter=FILTER)


def _no_filter(repo: str) -> None:
    """An open session with a baseline but no filter template to scope a probe."""
    open_probe_session(repo)


def _no_baseline(repo: str) -> None:
    """An open session with a filter template but nothing recorded to compare against."""
    start_with(repo)
    write_bench_config(repo, filter=FILTER)


REFUSALS = [
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
"""Every probe-only refusal of an opened session: the setup, the argv, a message fragment,
and the recorded reason. The finalized refusal every session command shares is pinned in
``test_session_cmds``."""


def test_probe_command_when_no_session_was_opened_does_refuse_without_recording_a_command(
    repo: str, measure: MeasureRecorder
):
    _no_session(repo)

    result = runner.invoke(app, ["probe"])

    assert result.exit_code == 2
    assert "gymrat start" in strip_ansi(result.stderr)
    assert result.stdout == ""
    assert measure.calls == []
    assert not Path(session_jsonl_path(repo)).exists()


@pytest.mark.parametrize(("setup", "argv", "fragment", "reason"), REFUSALS)
def test_probe_command_when_the_session_cannot_be_probed_does_exit_two_with_the_engine_message(
    *,
    repo: str,
    measure: MeasureRecorder,
    setup: Callable[[str], None],
    argv: list[str],
    fragment: str,
    reason: str,
):
    setup(repo)

    result = runner.invoke(app, argv)

    assert result.exit_code == 2
    assert fragment in strip_ansi(result.stderr)
    assert result.stdout == ""
    assert measure.calls == []
    cmd = last_command_record(repo)
    assert (cmd.name, cmd.exit_code, cmd.reason) == ("probe", 2, reason)


# ---------------------------------------------------------------------------
# signal handling
# ---------------------------------------------------------------------------


#: A bench that records its own pid, emits one metric, then blocks past any test.
_TRACKED_BENCH = pid_recording_script("bench.pid", "echo 'METRIC total_ms=1'\nsleep 120\n")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only shell and signals")
def test_probe_command_when_signalled_mid_bench_does_exit_on_the_signal_code_leaving_no_bench(
    repo: str, reap_groups: list[int]
):
    Path(repo, "bench.sh").write_text(_TRACKED_BENCH, encoding="utf-8")
    write_bench_config(repo, bench="sh bench.sh", adapter="metric-lines", timeout_seconds=300)
    run_git(["add", "bench.sh", "gymrat.toml"], repo)
    run_git(["commit", "-m", "bench harness"], repo)
    run_cli(
        ["start", "--baseline", "main"],
        repo,
        timeout=60,
    )
    append_records(repo, baseline_record(samples=BASELINE_SAMPLES))

    with spawned_gymrat(["probe"], repo) as proc:
        bench_pid = wait_for_pid_file_blocking(
            Path(experiment_worktree_dir(repo), "bench.pid"), timeout_s=60.0
        )
        reap_groups.append(os.getpgid(bench_pid))
        stop_by_signal(proc, signal.SIGINT, timeout_s=60)

    assert proc.returncode == 128 + signal.SIGINT
    wait_until_dead_blocking(bench_pid, timeout_s=30.0)
