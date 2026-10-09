"""Integration tests that drive a real gymrat session under the supervisor.

A mock agent runs the real CLI out-of-process — ``start``, ``iterate``,
``keep``, ``finalize`` — one command per driver action step, so the supervisor
sees a whole optimization session complete through the shipped binary rather
than a stubbed driver. Before ``finalize``, the agent ends a turn while another
holder has the repository lock, so the supervisor's probe of the repository's
real lock file must see it held and wait it out before replying. A trailing cost
step gives the run a non-zero spend, and a closing agent turn end makes the
supervisor read the session log the CLI left in the repository, so the run ends
on the finalized session.

POSIX-only: the flow leans on real git worktrees and bench subprocesses,
matching the other subprocess integration suites.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from gymrat.session.paths import lockfile_path, session_jsonl_path
from gymrat.supervisor.events import FollowUpEvent, SessionEvent
from gymrat.supervisor.supervise import supervise
from tests._cli import run_cli
from tests._lock import hold_lock
from tests._platform import needs_posix_worktrees
from tests.loop._bench import LONG_RUN_TIMEOUT, commit_project, tune_experiment
from tests.supervisor._fixtures import (
    make_context,
    make_launch,
    make_prompt,
    read_log_lines,
)
from tests.supervisor._mock_driver import ActionStep, CostStep, TurnEndStep, create_mock_driver

if TYPE_CHECKING:
    from filelock import FileLock

pytestmark = needs_posix_worktrees

#: The latency the edit tunes to — an improvement over the untuned baseline.
TUNED_LATENCY = 90


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
        run_cli(["start", "--baseline", "main"], repo, timeout=LONG_RUN_TIMEOUT)

    async def iterate() -> None:
        tune_experiment(repo, TUNED_LATENCY)
        run_cli(["iterate"], repo, timeout=LONG_RUN_TIMEOUT)

    async def keep() -> None:
        run_cli(["keep", "-m", "tune latency to 90"], repo, timeout=LONG_RUN_TIMEOUT)

    async def finalize() -> None:
        run_cli(["finalize"], repo, timeout=LONG_RUN_TIMEOUT)

    other_holder: list[FileLock] = []

    async def hold_repo_lock() -> None:
        other_holder.append(hold_lock(lockfile_path(repo)))

    # The other holder finishes once the supervisor responds to the turn end, so
    # the agent's next command can take the lock again.
    def release_on_follow_up(event: SessionEvent) -> None:
        if isinstance(event, FollowUpEvent):
            for lock in other_holder:
                lock.release()

    driver = create_mock_driver([
        ActionStep(action=start),
        ActionStep(action=iterate),
        ActionStep(action=keep),
        ActionStep(action=hold_repo_lock),
        TurnEndStep(),
        ActionStep(action=finalize),
        CostStep(cost_usd=0.42),
        TurnEndStep(),
    ])

    result = await supervise(
        driver=driver,
        prompt=make_prompt(cwd=repo),
        context=make_context(
            root=repo, lock_path=lockfile_path(repo), max_minutes=30, log_path=str(log_path)
        ),
        launch=make_launch(),
        observer=release_on_follow_up,
    )

    # A failed CLI command surfaces here as an error outcome; show its message.
    assert result.outcome.reason == "completed", result.outcome.message
    assert result.ended_by == "session"
    assert result.outcome.cost_usd == 0.42
    log_lines = read_log_lines(log_path)
    assert log_lines[0]["type"] == "launch"
    assert any(line["type"] == "usage_update" for line in log_lines[1:])
    # The supervisor waited out the repository lock before replying, then ended
    # the run on the finalized session it read from the repo.
    assert [
        (line["action"], line.get("reason")) for line in log_lines if line["type"] == "follow_up"
    ] == [("waiting", None), ("replied", None), ("ended", "finished")]
    # The session log the CLI left on disk holds the whole run, open to close.
    session_records = read_log_lines(session_jsonl_path(repo))
    record_types = {record["type"] for record in session_records}
    assert "session" in record_types
    assert "finalize" in record_types
