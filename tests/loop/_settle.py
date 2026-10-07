"""Shared fixtures and stubs for the settle (keep / discard) tests.

The one boundary these tests mock is the checks command: it is the consumer's
own test suite, which no test here can run. Every git operation is real, driven
against a throwaway repository from the ``create_scratch_repo`` factory, so the
suite stays order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.loop._settle``.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gymrat.config import ResolvedConfig
from gymrat.errors import GymratError
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError
from gymrat.loop.start import start_session
from gymrat.session.paths import experiment_worktree_dir, session_jsonl_path
from gymrat.session.records import (
    BaselineRecord,
    CommandRecord,
    Confirm,
    DiscardRecord,
    IterationPrimary,
    IterationRecord,
    KeepChecks,
    KeepRecord,
    PairedSamples,
    SessionLogRecord,
)
from gymrat.session.schema import Outcome
from tests._config import resolved_config
from tests._exec_fixtures import expected_result
from tests._git import head_of, run_git
from tests.session.records._fixtures import (
    append_records,
    iteration_record,
    log_records,
    metric_verdict,
)

CHECKS = "npm test"
CHECKS_STDOUT = "3 tests failed"
CHECKS_STDERR = "AssertionError: expected 2 to be 3"


def checks_config(**overrides: Any) -> ResolvedConfig:
    """A resolved config defaulted to the checks command every settle test exercises.

    ``timeout_seconds`` stays at 1800 so the run timeout the settle passes to
    ``exec`` is 1_800_000 ms, the value the tests assert on. Pass ``checks=None``
    to model a run with the gate switched off.
    """
    defaults: dict[str, Any] = {"bench": "sh bench.sh", "unstable_noise_pct": 2.0, "checks": CHECKS}
    return resolved_config(**(defaults | overrides))


def status_of(worktree: str) -> str:
    """The porcelain status of ``worktree`` — empty when nothing is uncommitted."""
    return run_git(["status", "--porcelain"], worktree)


def start_with(repo_dir: str, history: tuple[SessionLogRecord, ...] = ()) -> None:
    """Open a session in the scratch repo and leave ``history`` behind its header."""
    start_session(repo_dir, "main", checks_config())
    append_records(repo_dir, *history)


def edit_experiment(repo_dir: str) -> None:
    """Leave a tracked edit and an untracked file in the experiment worktree."""
    worktree = Path(experiment_worktree_dir(repo_dir))
    (worktree / "README.md").write_text("# edited by the agent\n", encoding="utf-8")
    (worktree / "scratch.txt").write_text("notes\n", encoding="utf-8")


class ExecRecorder:
    """A stand-in for ``exec`` that records its calls and answers with a fixed result.

    Tests reach into ``calls`` to assert the checks command ran (or never did) and
    to read the working directory and timeout it was handed.
    """

    def __init__(self, result: ExecResult | ExecTimeoutError) -> None:
        self.result = result
        self.calls: list[tuple[str, ExecOptions]] = []

    async def __call__(self, command: str, options: ExecOptions) -> ExecResult | ExecTimeoutError:
        self.calls.append((command, options))
        return self.result


def install_exec(
    monkeypatch: pytest.MonkeyPatch, result: ExecResult | ExecTimeoutError
) -> ExecRecorder:
    """Replace the keep module's ``exec`` with a recorder answering ``result``."""
    recorder = ExecRecorder(result)
    monkeypatch.setattr("gymrat.loop.keep.exec", recorder)
    return recorder


def checks_pass(monkeypatch: pytest.MonkeyPatch) -> ExecRecorder:
    """Answer the checks command with a clean run."""
    return install_exec(monkeypatch, expected_result("10 passed"))


def checks_fail(monkeypatch: pytest.MonkeyPatch) -> ExecRecorder:
    """Answer the checks command with a failing run that wrote to both streams."""
    return install_exec(monkeypatch, expected_result(CHECKS_STDOUT, CHECKS_STDERR, exit_code=1))


def settling_record_of(root: str) -> SessionLogRecord:
    """The last record in ``root``'s log that settles state, failing when the log holds none.

    Command traces and the baseline record a committed ``keep`` appends are
    bookkeeping that trails the settlement, so both are skipped: a caller asking
    for the settling record wants the keep, discard, stop, or finalize itself.
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


# ---------------------------------------------------------------------------
# Record builders — the iteration and keep shapes the engine produces
# ---------------------------------------------------------------------------

#: The run timeout from ``checks_config().timeout_seconds``, in milliseconds.
TIMEOUT_MS = 1_800_000

#: A result the no-checks path must never reach; installed only to prove exec stayed unused.
UNUSED_EXEC = expected_result()

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only .git pointer sabotage")

#: The paired rerun samples a filtered bench reports when it only emits ``total_ms``.
RERUN_SAMPLES = PairedSamples(experiment=({"total_ms": 14_120},), baseline=({"total_ms": 15_170},))

#: Three experiment rounds over two metrics whose medians are neither the first
#: nor the last round: ``total_ms`` reads 14_200 and ``alloc_bytes`` reads 2_048.
KEPT_ROUNDS: tuple[dict[str, float], ...] = (
    {"total_ms": 14_300, "alloc_bytes": 2_100},
    {"total_ms": 14_100, "alloc_bytes": 2_048},
    {"total_ms": 14_200, "alloc_bytes": 2_000},
)

#: The per-metric medians of ``KEPT_ROUNDS`` as ``status`` renders them, in the
#: order the rounds report the metrics.
KEPT_MEDIANS_LINE = "total_ms 14200 · alloc_bytes 2048"


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


def unimproved(seq: int, outcome: Outcome) -> IterationRecord:
    """An iteration numbered ``seq`` that read as ``no-signal`` or ``regressed``.

    The regressed shape carries an unconfirmed permutation verdict, so the gating
    gate lets it through and the outcome gate is the only one that can refuse it.
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


def failed_checks(stdout: str, stderr: str) -> KeepChecks:
    """The ``checks`` a blocked keep records for a failing run that printed both streams."""
    return KeepChecks(
        configured=True,
        passed=False,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
    )


def long_output(prefix: str) -> str:
    """200 lines of exactly 100 bytes each, every one numbered behind ``prefix``.

    The uniform line width puts the relay's byte budget on a line a test can name:
    81 lines are 8100 bytes and fit the 8192-byte budget the hook relay uses, an
    82nd would take it to 8200 and overrun it.
    """
    return "".join(f"{prefix}-{index:03d}".ljust(99, ".") + "\n" for index in range(200))


LONG_STDOUT = long_output("out")
LONG_STDERR = long_output("err")


def assert_settling_record(
    actual: KeepRecord | DiscardRecord, expected: KeepRecord | DiscardRecord
) -> None:
    """Assert ``actual`` equals ``expected`` once its stamped ``at`` is normalized.

    The settle stamps a real nanosecond timestamp the fixtures cannot predict, so
    the ``at`` is checked as an integer and then aligned before the structural compare.
    """
    assert isinstance(actual.at, int)
    assert actual.at > 0
    assert actual.model_copy(update={"at": expected.at}) == expected


def commit_experiment_directly(repo: str) -> str:
    """Commit the experiment worktree outside a keep, returning the standing commit."""
    worktree = experiment_worktree_dir(repo)
    run_git(["add", "-A"], worktree)
    run_git(["commit", "-m", "committed outside the keep"], worktree)
    return head_of(worktree)
