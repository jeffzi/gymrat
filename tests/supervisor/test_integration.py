"""Integration tests that drive a real gymrat session under the supervisor.

Two flows are covered:

- A mock agent that runs the real CLI out-of-process — ``start``, ``iterate``,
  ``keep``, ``finalize`` — one command per driver action step, so the supervisor
  sees a whole optimization session complete through the shipped binary rather
  than a stubbed driver. A trailing cost step gives the run a non-zero spend.
- A wall-clock cap firing before a long-delayed step can settle, asserted for a
  single ``cap`` event on both the observer and the JSONL log.

POSIX-only: the first flow leans on real git worktrees and bench subprocesses,
matching the other subprocess integration suites.
"""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat_py.session.paths import experiment_worktree_dir, session_jsonl_path
from gymrat_py.supervisor.supervise import supervise
from tests.loop._bench import BASELINE_LATENCY, TUNING_FILE, commit_project
from tests.supervisor._fixtures import (
    collecting_observer,
    make_launch,
    make_prompt,
    read_log_lines,
)
from tests.supervisor._mock_driver import ActionStep, CostStep, create_mock_driver

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only worktrees and gating")

_ENTRY = [sys.executable, "-m", "gymrat_py.cli.app"]

#: Generous budget: every action creates real worktrees and spawns real benches.
LONG_RUN_TIMEOUT = 180

#: The latency the edit tunes to — an improvement over the untuned baseline.
TUNED_LATENCY = 90


def _run_gymrat(args: list[str], cwd: str) -> None:
    """Run one gymrat CLI command in ``cwd``, blocking until it finishes.

    Blocking is what makes each driver action mirror a real agent: the command
    runs to completion before the next step. A non-zero exit is re-raised with
    the child's stderr attached so the mock driver's error outcome carries a
    debuggable message.
    """
    try:
        subprocess.run(  # noqa: S603
            [*_ENTRY, *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=LONG_RUN_TIMEOUT,
        )
    except subprocess.CalledProcessError as error:
        detail = f"gymrat {' '.join(args)} failed (exit {error.returncode}): {error.stderr}"
        raise AssertionError(detail) from error


def _tune_experiment(repo: str, latency: int) -> None:
    """Tune the experiment worktree to ``latency``, the edit an agent would make."""
    (Path(experiment_worktree_dir(repo)) / TUNING_FILE).write_text(f"{latency}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# a complete session driven through the real CLI
# ---------------------------------------------------------------------------


async def test_supervise_when_mock_agent_drives_real_cli_does_complete_the_session(
    create_scratch_repo: Callable[[], str],
):
    repo = create_scratch_repo()
    commit_project(repo, samples=5)
    log_path = Path(repo) / "supervisor-events.jsonl"

    async def start() -> None:
        _run_gymrat(["start", "main"], repo)

    async def iterate() -> None:
        _tune_experiment(repo, TUNED_LATENCY)
        _run_gymrat(["iterate"], repo)

    async def keep() -> None:
        _run_gymrat(["keep", "-m", "tune latency to 90"], repo)

    async def finalize() -> None:
        _run_gymrat(["finalize"], repo)

    # Sanity-check the tuned value is a real improvement over the untuned bench,
    # so the iterate is a step forward, not a no-op.
    assert TUNED_LATENCY < BASELINE_LATENCY

    driver = create_mock_driver(
        [
            ActionStep(action=start),
            ActionStep(action=iterate),
            ActionStep(action=keep),
            ActionStep(action=finalize),
            CostStep(cost_usd=0.42),
        ]
    )

    result = await supervise(
        driver=driver,
        prompt=make_prompt(cwd=repo),
        max_minutes=30,
        log_path=log_path,
        launch=make_launch(),
    )

    # A failed CLI command surfaces here as an error outcome; show its message.
    if result.outcome.reason == "error":
        pytest.fail(result.outcome.message or "session ended with an unreported error")

    assert result.ended_by == "session"
    assert result.outcome.reason == "completed"
    assert result.cost_usd == 0.42

    # The event log opens with the launch event, then carries the cost step's
    # usage update from the running session.
    log_lines = read_log_lines(log_path)
    assert log_lines[0]["type"] == "launch"
    assert any(line["type"] == "usage_update" for line in log_lines[1:])

    # The session log the CLI left on disk holds the whole run, open to close.
    session_records = read_log_lines(session_jsonl_path(repo))
    record_types = {record["type"] for record in session_records}
    assert "session" in record_types
    assert "finalize" in record_types


# ---------------------------------------------------------------------------
# the wall-clock cap fires before the session finishes
# ---------------------------------------------------------------------------


async def test_supervise_when_wall_clock_caps_a_long_session_does_report_wall_clock(
    tmp_path: Path,
):
    probe = collecting_observer()
    # A single step delayed far past the cap, so the wall-clock cap always wins.
    driver = create_mock_driver([CostStep(cost_usd=0.01, delay_ms=60_000)])
    log_path = tmp_path / "supervisor-events.jsonl"

    result = await supervise(
        driver=driver,
        prompt=make_prompt(cwd=str(tmp_path)),
        max_minutes=0.001,
        log_path=log_path,
        launch=make_launch(max_minutes=0.001),
        observer=probe.observer,
        grace_ms=50,
    )

    assert result.ended_by == "wall-clock"
    assert result.outcome.reason == "interrupted"

    cap_events = [event for event in probe.events if event.type == "cap"]
    assert len(cap_events) == 1
    assert cap_events[0].cap == "wall-clock"  # type: ignore[attr-defined]

    cap_lines = [line for line in read_log_lines(log_path) if line["type"] == "cap"]
    assert len(cap_lines) == 1
    assert cap_lines[0]["cap"] == "wall-clock"
