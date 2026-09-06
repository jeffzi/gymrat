"""Shared fixtures and stubs for the settle (keep / discard) tests.

The one boundary these tests mock is the checks command: it is the consumer's
own test suite, which no test here can run. Every git operation is real, driven
against a throwaway repository from the ``create_scratch_repo`` factory, so the
suite stays order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.loop.settle._fixtures``.
"""

import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from gymrat.config import (
    HooksConfig,
    KindEntry,
    MetricEntry,
    ResolvedConfig,
    StopConfig,
)
from gymrat.errors import GymratError
from gymrat.exec import ExecOptions, ExecResult, ExecTimeoutError
from gymrat.loop.start import start_session
from gymrat.session import (
    Confirm,
    DiscardRecord,
    IterationPrimary,
    IterationRecord,
    KeepChecks,
    KeepRecord,
    MetricVerdict,
    PairedSamples,
    SessionLogRecord,
    append_record,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests.session.records._fixtures import blocked_keep, iteration_record

CHECKS = "npm test"
CHECKS_STDOUT = "3 tests failed"
CHECKS_STDERR = "AssertionError: expected 2 to be 3"


def checks_config(
    *,
    bench: str = "sh bench.sh",
    prepare: str | None = None,
    adapter: str = "metric-lines",
    samples: int = 10,
    timeout_seconds: int = 1800,
    unstable_noise_pct: float = 2.0,
    primary: str = "geomean",
    checks: str | None = CHECKS,
    metrics: dict[str, MetricEntry] | None = None,
    kinds: dict[str, KindEntry] | None = None,
    runbook: str | None = None,
    filter: str | None = None,  # noqa: A002
    stop: StopConfig | None = None,
    hooks: HooksConfig | None = None,
) -> ResolvedConfig:
    """A resolved config defaulted to the checks command every settle test exercises.

    ``timeout_seconds`` is 1800 so the run timeout the settle passes to ``exec``
    is 1_800_000 ms, the value the tests assert on. Pass ``checks=None`` to model
    a run with the gate switched off.
    """
    return ResolvedConfig(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout_seconds=timeout_seconds,
        unstable_noise_pct=unstable_noise_pct,
        primary=primary,
        checks=checks,
        metrics=metrics,
        kinds=kinds,
        runbook=runbook,
        filter=filter,
        stop=stop,
        hooks=hooks,
    )


def git(args: list[str], cwd: str) -> str:
    """Run git in ``cwd`` and return its stripped stdout, failing loudly on error."""
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def head_of(worktree: str) -> str:
    """The commit ``worktree`` currently has checked out."""
    return git(["rev-parse", "HEAD"], worktree)


def status_of(worktree: str) -> str:
    """The porcelain status of ``worktree`` — empty when nothing is uncommitted."""
    return git(["status", "--porcelain"], worktree)


def iteration(seq: int, **overrides: object) -> IterationRecord:
    """A measured iteration numbered ``seq``, improved unless a test says otherwise."""
    return iteration_record(seq=seq, **overrides)


def start_with(repo_dir: str, history: tuple[SessionLogRecord, ...] = ()) -> None:
    """Open a session in the scratch repo and leave ``history`` behind its header."""
    start_session(repo_dir, "main", checks_config())
    jsonl_path = session_jsonl_path(repo_dir)
    for record in history:
        append_record(jsonl_path, record)


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
    """Replace the settle_checks module's ``exec`` with a recorder answering ``result``."""
    recorder = ExecRecorder(result)
    monkeypatch.setattr("gymrat.loop.settle.checks.exec", recorder)
    return recorder


def checks_pass(monkeypatch: pytest.MonkeyPatch) -> ExecRecorder:
    """Answer the checks command with a clean run."""
    return install_exec(
        monkeypatch,
        ExecResult(
            stdout="10 passed",
            stderr="",
            exit_code=0,
            stdout_bytes=len(b"10 passed"),
            stderr_bytes=0,
        ),
    )


def checks_fail(monkeypatch: pytest.MonkeyPatch) -> ExecRecorder:
    """Answer the checks command with a failing run that wrote to both streams."""
    return install_exec(
        monkeypatch,
        ExecResult(
            stdout=CHECKS_STDOUT,
            stderr=CHECKS_STDERR,
            exit_code=1,
            stdout_bytes=len(CHECKS_STDOUT.encode()),
            stderr_bytes=len(CHECKS_STDERR.encode()),
        ),
    )


def last_record_of(root: str) -> SessionLogRecord:
    """The record ``root``'s log ends on, failing the test when the log is empty."""
    records = read_records(session_jsonl_path(root))
    if not records:
        msg = f"expected a record in {session_jsonl_path(root)}"
        raise AssertionError(msg)
    return records[-1]


def capture_error(action: Callable[[], object]) -> GymratError:
    """Run ``action`` expecting a :class:`GymratError`, returning the raised error."""
    with pytest.raises(GymratError) as excinfo:
        action()
    return excinfo.value


# ---------------------------------------------------------------------------
# Record builders — the iteration and keep shapes the engine produces
# ---------------------------------------------------------------------------

ISO_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

#: The run timeout from ``checks_config().timeout_seconds``, in milliseconds.
TIMEOUT_MS = 1_800_000

#: A result the no-checks path must never reach; installed only to prove exec stayed unused.
UNUSED_EXEC = ExecResult(stdout="", stderr="", exit_code=0, stdout_bytes=0, stderr_bytes=0)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only .git pointer sabotage")

#: The paired rerun samples a filtered bench reports when it only emits ``total_ms``.
RERUN_SAMPLES = PairedSamples(experiment=({"total_ms": 14_120},), baseline=({"total_ms": 15_170},))


def metric(**overrides: object) -> MetricVerdict:
    """A metric verdict the engine produces, improved and gating unless overridden."""
    defaults: dict[str, object] = {
        "delta_pct": -7.2,
        "verdict": "improved",
        "method": "permutation",
        "p": 0.002,
        "noise_pct": 1.4,
        "gating": True,
        "confirmed": False,
    }
    defaults.update(overrides)
    return MetricVerdict(**defaults)  # type: ignore[arg-type]


def undefined_delta(seq: int) -> IterationRecord:
    """An iteration numbered ``seq`` whose deltas a zero baseline median left undefined."""
    return iteration(
        seq,
        metrics={"total_ms": metric(delta_pct=None, verdict="no-signal")},
        primary=IterationPrimary(kind="geomean", delta_pct=None),
        outcome="no-signal",
    )


def confirmed_regression(seq: int) -> IterationRecord:
    """An iteration whose gating metric regressed and stayed regressed on the rerun."""
    return iteration(
        seq,
        metrics={"total_ms": metric(delta_pct=9.4, verdict="regressed", confirmed=True)},
        primary=IterationPrimary(kind="geomean", delta_pct=9.4),
        outcome="regressed",
    )


def unmeasured_regression(seq: int) -> IterationRecord:
    """An iteration whose gating ``alloc_bytes`` regressed then went missing from the rerun."""
    return iteration(
        seq,
        metrics={
            "total_ms": metric(),
            "alloc_bytes": metric(delta_pct=9.4, verdict="regressed"),
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


def gating_block(seq: int) -> KeepRecord:
    """The keep a gating regression refused, numbered with the iteration it refused."""
    return blocked_keep(seq, reason="gating-regression", checks=KeepChecks(configured=True))


def nothing_to_commit_block(seq: int) -> KeepRecord:
    """The keep blocked because the experiment worktree had nothing to commit."""
    return blocked_keep(seq, reason="nothing-to-commit", checks=KeepChecks(configured=True))


def nothing_measured_block(seq: int) -> KeepRecord:
    """The keep refusing because nothing has been measured since the last settle."""
    return blocked_keep(seq, reason="nothing-measured", checks=KeepChecks(configured=True))


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

    The settle stamps a real timestamp the fixtures cannot predict, so the ``at``
    is checked against the ISO shape and then aligned before the structural compare.
    """
    assert ISO_PATTERN.match(actual.at)
    assert actual.model_copy(update={"at": expected.at}) == expected


def commit_experiment_directly(repo: str) -> str:
    """Commit the experiment worktree outside a keep, returning the standing commit."""
    worktree = experiment_worktree_dir(repo)
    git(["add", "-A"], worktree)
    git(["commit", "-m", "committed outside the keep"], worktree)
    return head_of(worktree)
