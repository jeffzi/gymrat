"""Shared fixtures and stubs for the settle (keep / discard) tests.

The one boundary these tests mock is the checks command: it is the consumer's
own test suite, which no test here can run. Every git operation is real, driven
against a throwaway repository from the ``create_scratch_repo`` factory, so the
suite stays order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The ``_`` prefix marks a shared helper rather than a test module; it is
imported as ``tests.loop._settle``.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gymrat.config import ResolvedConfig
from gymrat.errors import GymratError
from gymrat.loop.start import start_session
from gymrat.session.paths import experiment_worktree_dir, session_jsonl_path
from gymrat.session.records import (
    BaselineRecord,
    CommandRecord,
    Confirm,
    DiscardRecord,
    IterationPrimary,
    IterationRecord,
    KeepRecord,
    PairedSamples,
    SessionLogRecord,
)
from gymrat.session.schema import Outcome
from tests._config import resolved_config
from tests._exec_fixtures import (
    ExecRecorder,
    expected_result,
    install_exec,
)
from tests._git import commit_all
from tests.session.records._fixtures import (
    append_records,
    committed_keep,
    discard_record,
    gate_block,
    iteration_record,
    log_records,
    metric_verdict,
    settled_history,
)

CHECKS = "npm test"
CHECKS_STDOUT = "3 tests failed"
CHECKS_STDERR = "AssertionError: expected 2 to be 3"


def checks_config(**overrides: Any) -> ResolvedConfig:
    """Build a resolved config defaulted to the checks command every settle test exercises.

    ``timeout_seconds`` stays at 1800 so the run timeout the settle passes to
    ``exec`` is 1_800_000 ms, the value the tests assert on.

    Args:
        **overrides: Config fields to set over the defaults; ``checks=None``
            models a run with the gate switched off.

    Returns:
        The resolved config.
    """
    defaults: dict[str, Any] = {"bench": "sh bench.sh", "unstable_noise_pct": 2.0, "checks": CHECKS}
    return resolved_config(**(defaults | overrides))


def start_with(
    repo_dir: str,
    history: tuple[SessionLogRecord, ...] = (),
    *,
    config: ResolvedConfig | None = None,
) -> None:
    """Open a session on ``main`` in the scratch repo and leave ``history`` behind its header.

    Args:
        repo_dir: The repository the session opens in.
        history: The records logged after the session header.
        config: The configuration the session opens with; ``None`` means :func:`checks_config`.
    """
    start_session(repo_dir, "main", checks_config() if config is None else config)
    append_records(repo_dir, *history)


def keep_iteration(
    root: str, seq: int, *, commit: str | None = None, message: str | None = None
) -> None:
    """Log iteration ``seq`` and the committed keep that settled it.

    Args:
        root: The repository whose session log is appended to.
        seq: The iteration number both records carry.
        commit: The commit the keep names; ``None`` keeps the builder's default.
        message: The keep's message; ``None`` keeps the builder's default.
    """
    overrides = {
        name: value
        for name, value in (("commit", commit), ("message", message))
        if value is not None
    }
    append_records(root, iteration_record(seq=seq), committed_keep(seq, **overrides))


def commit_iteration(root: str, seq: int, message: str) -> str:
    """Commit one edit in the experiment worktree and log the iteration behind it.

    The experiment worktree is checked out on the session branch, so each call
    moves that branch forward exactly as a real ``gymrat keep`` would. The keep
    record is left to the caller.

    Args:
        root: The repository whose session the edit belongs to.
        seq: The iteration number the logged iteration carries.
        message: The commit message.

    Returns:
        The SHA of the new commit.
    """
    commit = commit_all(experiment_worktree_dir(root), message, file=f"step-{seq}.txt")
    append_records(root, iteration_record(seq=seq))
    return commit


def commit_and_keep(root: str, seq: int, message: str) -> str:
    """Commit one edit, log its iteration, and log the keep that settled it.

    Args:
        root: The repository whose session is kept.
        seq: The iteration number the logged iteration and keep carry.
        message: The commit message, also recorded as the keep's message.

    Returns:
        The SHA of the commit the keep names.
    """
    commit = commit_iteration(root, seq, message)
    append_records(root, committed_keep(seq, commit=commit, message=message))
    return commit


def edit_experiment(repo_dir: str) -> None:
    """Leave a tracked edit and an untracked file in the experiment worktree."""
    worktree = Path(experiment_worktree_dir(repo_dir))
    (worktree / "README.md").write_text("# edited by the agent\n", encoding="utf-8")
    (worktree / "scratch.txt").write_text("notes\n", encoding="utf-8")


def commit_experiment_directly(repo: str) -> str:
    """Commit the experiment worktree outside a keep, returning the standing commit."""
    return commit_all(experiment_worktree_dir(repo), "committed outside the keep")


#: The ``exec`` the keep module calls, which the checks stand-ins replace.
KEEP_EXEC = "gymrat.loop.keep.exec"


def checks_pass(monkeypatch: pytest.MonkeyPatch) -> ExecRecorder:
    """Answer the checks command with a clean run."""
    return install_exec(monkeypatch, KEEP_EXEC, expected_result("10 passed"))


def checks_fail(monkeypatch: pytest.MonkeyPatch) -> ExecRecorder:
    """Answer the checks command with a failing run that wrote to both streams."""
    return install_exec(
        monkeypatch, KEEP_EXEC, expected_result(CHECKS_STDOUT, CHECKS_STDERR, exit_code=1)
    )


#: A result the no-checks path must never reach; installed only to prove exec stayed unused.
UNUSED_EXEC = expected_result()


def settling_record_of(root: str) -> SessionLogRecord:
    """Find the last record in a session log that settles state.

    Command traces and the baseline record a committed ``keep`` appends are
    bookkeeping that trails the settlement, so both are skipped: a caller asking
    for the settling record wants the keep, discard, stop, or finalize itself.

    Args:
        root: The repository whose session log is read.

    Returns:
        The last keep, discard, stop, finalize, iteration, or session record.

    Raises:
        AssertionError: The log holds no settling record.
    """
    records = log_records(root)
    for record in reversed(records):
        if not isinstance(record, CommandRecord | BaselineRecord):
            return record
    msg = f"expected a settling record in {session_jsonl_path(root)}"
    raise AssertionError(msg)


def capture_error(action: Callable[[], object]) -> GymratError:
    """Run ``action`` expecting a :class:`GymratError`, returning the raised error."""
    with pytest.raises(GymratError) as excinfo:
        action()
    return excinfo.value


def assert_settling_record(
    actual: KeepRecord | DiscardRecord, expected: KeepRecord | DiscardRecord
) -> None:
    """Assert a settling record equals the expected one once its stamped ``at`` is normalized.

    The settle stamps a real nanosecond timestamp the fixtures cannot predict, so
    the ``at`` is checked as a positive integer and then aligned before the
    structural compare.

    Args:
        actual: The record the settle wrote.
        expected: The record it should equal, apart from ``at``.
    """
    assert isinstance(actual.at, int)
    assert actual.at > 0
    assert actual.model_copy(update={"at": expected.at}) == expected


# ---------------------------------------------------------------------------
# Iteration record builders
# ---------------------------------------------------------------------------

#: The paired rerun samples a filtered bench reports when it only emits ``total_ms``.
RERUN_SAMPLES = PairedSamples(experiment=({"total_ms": 14_120},), baseline=({"total_ms": 15_170},))

#: Three experiment rounds over two metrics whose medians are neither the first
#: nor the last round: ``total_ms`` reads 14_200 and ``alloc_bytes`` reads 2_048.
KEPT_ROUNDS: tuple[dict[str, float], ...] = (
    {"total_ms": 14_300, "alloc_bytes": 2_100},
    {"total_ms": 14_100, "alloc_bytes": 2_048},
    {"total_ms": 14_200, "alloc_bytes": 2_000},
)


def measured_rounds(seq: int) -> IterationRecord:
    """An improved iteration numbered ``seq`` measured over ``KEPT_ROUNDS``."""
    return iteration_record(
        seq=seq,
        samples=PairedSamples(
            experiment=KEPT_ROUNDS,
            baseline=({"total_ms": 15_200, "alloc_bytes": 2_400},),
        ),
    )


def undefined_delta(seq: int) -> IterationRecord:
    """An iteration numbered ``seq`` whose deltas a zero baseline median left undefined."""
    return iteration_record(
        seq=seq,
        metrics={"total_ms": metric_verdict(delta_pct=None, verdict="no-signal")},
        primary=IterationPrimary(kind="geomean", delta_pct=None),
        outcome="no-signal",
    )


def confirmed_regression(seq: int) -> IterationRecord:
    """An iteration whose gating metric regressed and stayed regressed on the rerun."""
    return iteration_record(
        seq=seq,
        metrics={"total_ms": metric_verdict(delta_pct=9.4, verdict="regressed", confirmed=True)},
        primary=IterationPrimary(kind="geomean", delta_pct=9.4),
        outcome="regressed",
    )


def gating_blocked_history() -> tuple[SessionLogRecord, ...]:
    """A history whose iteration 1 regressed and a gating-regression block settled it."""
    return (confirmed_regression(1), gate_block(1, "gating-regression"))


#: Histories that leave nothing measured to settle, one ``pytest.param`` each.
NOTHING_MEASURED_HISTORIES = [
    pytest.param((), id="no-iteration-ever-recorded"),
    pytest.param(settled_history(), id="last-iteration-already-kept"),
    pytest.param(
        (*gating_blocked_history(), discard_record(2)), id="gating-block-already-discarded"
    ),
]


def exact_regression(seq: int) -> IterationRecord:
    """An iteration whose exact gating metric regressed, which no rerun is run to confirm."""
    return iteration_record(
        seq=seq,
        metrics={"total_ms": metric_verdict(delta_pct=9.4, verdict="regressed", method="exact")},
        primary=IterationPrimary(kind="geomean", delta_pct=9.4),
        outcome="regressed",
    )


def unimproved(seq: int, outcome: Outcome) -> IterationRecord:
    """Build an iteration that read as ``no-signal`` or ``regressed``.

    The regressed shape carries an unconfirmed permutation verdict, so the gating
    gate lets it through and the outcome gate is the only one that can refuse it.

    Args:
        seq: The iteration number.
        outcome: ``no-signal`` or ``regressed``.

    Returns:
        The iteration record.
    """
    no_signal = outcome == "no-signal"
    delta_pct = 0.1 if no_signal else 9.4
    return iteration_record(
        seq=seq,
        metrics={"total_ms": metric_verdict(delta_pct=delta_pct, verdict=outcome)},
        primary=IterationPrimary(kind="geomean", delta_pct=delta_pct),
        outcome=outcome,
    )


def unmeasured_regression(seq: int) -> IterationRecord:
    """An iteration whose gating ``alloc_bytes`` regressed then went missing from the rerun."""
    return iteration_record(
        seq=seq,
        metrics={
            "total_ms": metric_verdict(),
            "alloc_bytes": metric_verdict(delta_pct=9.4, verdict="regressed"),
        },
        primary=IterationPrimary(kind="geomean", delta_pct=9.4),
        outcome="regressed",
        confirm=Confirm(
            ran=True,
            filtered=("total_ms", "alloc_bytes"),
            absent=("alloc_bytes",),
            samples=RERUN_SAMPLES,
        ),
    )
