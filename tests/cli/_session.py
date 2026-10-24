"""Session-log builders shared by the CLI command test files.

Each opens or writes the session a command runs against, optionally with a
bench config and a live budget, and most double as rows in an arrange table.
This is test-support code, not a test module: it carries no test functions.
"""

from pathlib import Path

import pytest

from gymrat.loop.finalize import finalize_session
from gymrat.session.records import SessionLogRecord, SessionRecord
from tests._config import resolved_config
from tests.cli._command_stubs import stub_probe_measure
from tests.config._toml import write_config
from tests.loop._probe import BASELINE_SAMPLES
from tests.loop._settle import (
    CHECKS,
    commit_and_keep,
    edit_experiment,
    keep_iteration,
    start_with,
)
from tests.loop.iterate._fixtures import write_iterate_session, write_outlasted_session
from tests.session._budget import install_budget
from tests.session.records._fixtures import (
    baseline_record,
    committed_keep,
    iteration_record,
    session_header_of,
    session_record,
    settled_history,
    write_session_log,
)


def leave_as_is(_repo: str, _monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange nothing: the no-op row of an ``(repo, monkeypatch)`` arrange table."""


def make_discard_repo(repo: str) -> str:
    """Set up ``repo`` with an open session and one unsettled iteration to discard."""
    start_with(repo, (iteration_record(seq=1),), config=resolved_config())
    return repo


def open_stop_ready_session(repo: str) -> None:
    """Open a configured session with one settled iteration, ready for the stop command."""
    start_with(repo)
    keep_iteration(repo, 1)
    write_bench_config(repo)


def write_settled_session(repo: str, *trailing_records: SessionLogRecord) -> None:
    """Log a configured session with one kept iteration, followed by ``trailing_records``."""
    write_session_log(
        repo, session_record(), (iteration_record(seq=1), committed_keep(1), *trailing_records)
    )
    write_bench_config(repo)


def open_probe_session(repo: str, **config: object) -> None:
    """Open a session on a recorded baseline and write the bench config, ready for probe.

    Args:
        repo: The repository the session opens in.
        **config: Extra ``gymrat.toml`` keys written beside the bench command.
    """
    start_with(repo, (baseline_record(samples=BASELINE_SAMPLES),))
    write_bench_config(repo, **config)


def open_stubbed_probe_session(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a probe-ready session and stub the measurement engine: an arrange-table row."""
    open_probe_session(repo)
    stub_probe_measure(monkeypatch)


def start_edited_session(
    root: str,
    history: tuple[SessionLogRecord, ...] = (iteration_record(seq=1),),
    **config: object,
) -> None:
    """Open a session on ``history``, edit the experiment, and write the config.

    Args:
        root: The repository root.
        history: The records logged after the session header; one unsettled
            iteration by default.
        **config: Extra ``gymrat.toml`` entries beside the bench.
    """
    start_with(root, history)
    edit_experiment(root)
    write_bench_config(root, **config)


def open_unedited_session(root: str) -> None:
    """Open a session with one unsettled iteration and configured checks, and edit nothing."""
    start_with(root, (iteration_record(seq=1),))
    write_bench_config(root, checks=CHECKS)


def open_session(repo: str) -> None:
    """Open a session in ``repo`` so a command has a session log to write to."""
    write_session_log(repo, session_record())


def open_session_with_one_keep(root: str) -> SessionRecord:
    """Open a session, commit and log one kept iteration, and return the session header."""
    start_with(root, config=resolved_config())
    commit_and_keep(root, 1, "cache the regex")
    return session_header_of(root)


def close_session_with_one_keep(root: str) -> str:
    """Open a session with one kept commit, finalize it, and return its closed id."""
    header = open_session_with_one_keep(root)
    finalize_session(root)
    return header.session_id


def write_bench_config(root: str, **extra: object) -> None:
    """Write the implicit ``gymrat.toml`` at the repository root, naming a bench command."""
    write_config(Path(root), {"bench": "npm run bench", **extra})


def write_capped_session(repo: str) -> None:
    """Write a settled session that already reached its configured cap of one iteration."""
    write_iterate_session(repo, settled_history())
    write_bench_config(repo, stop={"max_iterations": 1})


def open_unsettled_session_under_budget(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session whose last iteration was never settled, under a live budget."""
    write_iterate_session(repo, (iteration_record(seq=1),))
    write_bench_config(repo)
    install_budget(repo, monkeypatch)


def open_capped_session_under_budget(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a session at its iteration cap, under a live budget."""
    write_capped_session(repo)
    install_budget(repo, monkeypatch)


def open_outlasted_session(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a settled session whose last iteration outlasts the 5 minutes the budget has left."""
    write_outlasted_session(repo, monkeypatch)
    write_bench_config(repo)


def open_hooked_iterate_session(repo: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Open a session whose before hook touches a marker file, under a live budget.

    The hook runs in the experiment worktree, so the worktree is created for it to fire.

    Args:
        repo: The repository the session opens in.
        monkeypatch: The fixture that releases the supervise lock at teardown.

    Returns:
        The marker file the before hook creates when it runs.
    """
    header = write_iterate_session(repo)
    Path(header.worktrees.experiment).mkdir()
    marker = Path(repo, "hook-ran")
    write_bench_config(repo, hooks={"before": f"touch '{marker}'"})
    install_budget(repo, monkeypatch)
    return marker
