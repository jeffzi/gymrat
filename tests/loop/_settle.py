"""Shared fixtures and stubs for the settle (keep / discard) tests.

The one boundary these tests mock is the checks command: it is the consumer's
own test suite, which no test here can run. Every git operation is real, driven
against a throwaway repository from the ``create_scratch_repo`` factory, so the
suite stays order-independent and safe under ``pytest-xdist`` / ``pytest-randomly``.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.loop._settle``.
"""

import subprocess
from pathlib import Path

import pytest

from gymrat_py.config import (
    HooksConfig,
    KindEntry,
    MetricEntry,
    ResolvedConfig,
    StopConfig,
)
from gymrat_py.exec import ExecOptions, ExecResult, ExecTimeoutError
from gymrat_py.loop.start import start_session
from gymrat_py.session import (
    IterationRecord,
    SessionLogRecord,
    append_record,
    experiment_worktree_dir,
    read_records,
    session_jsonl_path,
)
from tests.session._records import iteration_record

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
    """Replace the settle module's ``exec`` with a recorder answering ``result``."""
    recorder = ExecRecorder(result)
    monkeypatch.setattr("gymrat_py.loop.settle.exec", recorder)
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
