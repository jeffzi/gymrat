"""Sampling stubs and record helpers shared by the iterate tests.

The one boundary these fixtures mock is sampling: it shells out to the
consumer's bench script, which no test can run. Everything downstream of it —
verdicts, aggregation, the record, the report — runs for real, so the stubs key
their answers on the worktree *directory* each context names rather than on call
order. A side that landed in the wrong half of the record could then only have
come from the wrong worktree, which is what lets the assertions downstream read
as evidence.

The module is name-prefixed with ``_`` so pytest never collects it: it is a
helper imported as ``tests.loop.iterate._fixtures``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gymrat.config import ResolvedConfig
from gymrat.errors import GymratError
from gymrat.sampling import SamplingOptions, TargetContext, TargetSamples
from gymrat.session import (
    IterationRecord,
    SessionLogRecord,
    SessionRecord,
    Worktrees,
    read_records,
    record_to_wire,
    session_jsonl_path,
)
from tests.session.records._fixtures import SESSION_ID
from tests.session.records._fixtures import iteration_record as _iteration_record
from tests.session.records._fixtures import session_record as _session_record_defaults

if TYPE_CHECKING:
    import pytest

#: Ten rounds of a bench that stayed near 100.
BASELINE_MS: list[float] = [100, 101, 99, 100, 102, 98, 100, 101, 99, 100]

#: The same ten rounds, an order of magnitude larger, so a second metric reads differently.
BASELINE_BYTES: list[float] = [value * 10 for value in BASELINE_MS]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def scaled(values: list[float], factor: float) -> list[float]:
    """Scale every round by ``factor``, moving the median by exactly that much.

    A constant factor leaves every pairwise difference the same sign, which is
    what makes the permutation test call the move rather than shrug at it.
    """
    return [value * factor for value in values]


def rounds(total_ms: list[float], alloc_bytes: list[float]) -> list[dict[str, float]]:
    """One round per entry, pairing each metric with the value it reported that round."""
    return [
        {"total_ms": total, "alloc_bytes": alloc}
        for total, alloc in zip(total_ms, alloc_bytes, strict=False)
    ]


def baseline_rounds() -> list[dict[str, float]]:
    """The ten rounds the baseline worktree reports in every test here."""
    return rounds(BASELINE_MS, BASELINE_BYTES)


def improved_rounds() -> list[dict[str, float]]:
    """Ten rounds 10% faster and 20% leaner than the baseline's."""
    return rounds(scaled(BASELINE_MS, 0.9), scaled(BASELINE_BYTES, 0.8))


def session_record(root: str) -> SessionRecord:
    """A session header whose worktrees sit beside the default paths.

    Placing the worktrees on ``side-experiment`` / ``side-baseline`` rather than
    on the defaults means a run that recomputed the paths instead of reading them
    off the record would bench directories no test ever filled.
    """
    return _session_record_defaults(
        session_id=SESSION_ID,
        worktrees=Worktrees(
            experiment=str(Path(root) / "side-experiment"),
            baseline=str(Path(root) / "side-baseline"),
        ),
    )


def iteration(seq: int) -> IterationRecord:
    """A measured iteration numbered ``seq``, settled by nobody."""
    return _iteration_record(seq=seq)


def resolved_config(**overrides: Any) -> ResolvedConfig:
    """A settled run configuration, geomean-led unless a test names its own primary."""
    default = ResolvedConfig(
        bench="npm run bench",
        adapter="metric-lines",
        samples=10,
        timeout_seconds=1800,
        unstable_noise_pct=200.0,
        primary="geomean",
    )
    return replace(default, **overrides) if overrides else default


@dataclass(frozen=True, slots=True)
class PairedRun:
    """One paired run's answer to a sampling call: the rounds each worktree reports."""

    experiment: list[dict[str, float]]
    baseline: list[dict[str, float]]


@dataclass(frozen=True, slots=True)
class _RecordedCall:
    """The positional arguments one ``collect_samples`` call was handed."""

    adapter: object
    targets: list[TargetContext]
    options: SamplingOptions
    abort: object


@dataclass(frozen=True, slots=True)
class SamplingCall:
    """The targets and bench command of one recorded sampling call."""

    targets: list[TargetContext]
    bench: str


class CollectSamplesRecorder:
    """A stand-in for ``collect_samples`` that records every call it answers.

    The recorder is installed once by :func:`install_collect_samples`; a test
    then configures how it answers with :func:`stub_samples` or
    :func:`stub_runs`. Every call is stored so :func:`sampling_call` can read the
    targets and bench a call was handed.
    """

    def __init__(self) -> None:
        self.calls: list[_RecordedCall] = []
        self._answer: Any = None

    async def __call__(
        self,
        adapter: object,
        targets: Any,
        options: SamplingOptions,
        abort: object,
    ) -> list[TargetSamples]:
        target_list = list(targets)
        self.calls.append(_RecordedCall(adapter, target_list, options, abort))
        if self._answer is None:
            message = "collect_samples was called before a stub was installed"
            raise AssertionError(message)
        return self._answer(target_list)

    @property
    def call_count(self) -> int:
        return len(self.calls)


def install_collect_samples(monkeypatch: pytest.MonkeyPatch) -> CollectSamplesRecorder:
    """Replace ``gymrat.loop.iterate.bench.collect_samples`` with a fresh recorder."""
    recorder = CollectSamplesRecorder()
    monkeypatch.setattr("gymrat.loop.iterate.bench.collect_samples", recorder)
    return recorder


def stub_samples(
    mock: CollectSamplesRecorder,
    root: str,
    experiment: list[dict[str, float]],
    baseline: list[dict[str, float]],
) -> None:
    """Answer every sampling call with ``experiment`` and ``baseline`` keyed on worktree dir."""
    worktrees = session_record(root).worktrees
    by_dir = {worktrees.experiment: experiment, worktrees.baseline: baseline}

    def answer(targets: list[TargetContext]) -> list[TargetSamples]:
        collected: list[TargetSamples] = []
        for ctx in targets:
            if ctx.dir not in by_dir:
                message = f"stub_samples: unrecognized worktree dir {ctx.dir}"
                raise AssertionError(message)
            collected.append(TargetSamples(ctx=ctx, samples=by_dir[ctx.dir]))
        return collected

    mock._answer = answer


def stub_runs(
    mock: CollectSamplesRecorder,
    root: str,
    runs: list[PairedRun | GymratError],
) -> None:
    """Answer the nth sampling call with the nth entry of ``runs``, keyed on worktree dir.

    A :class:`GymratError` entry rejects that call, standing in for a bench that
    failed mid-run. A call past the end of ``runs`` rejects too, so an unexpected
    extra rerun surfaces as a failure rather than as silently reused samples.
    """
    worktrees = session_record(root).worktrees
    state = {"index": 0}

    def answer(targets: list[TargetContext]) -> list[TargetSamples]:
        index = state["index"]
        state["index"] = index + 1
        if index >= len(runs):
            message = f"unexpected sampling call {index + 1}"
            raise AssertionError(message)
        run = runs[index]
        if isinstance(run, GymratError):
            raise run
        by_dir = {worktrees.experiment: run.experiment, worktrees.baseline: run.baseline}
        collected: list[TargetSamples] = []
        for ctx in targets:
            if ctx.dir not in by_dir:
                message = f"stub_runs: unrecognized worktree dir {ctx.dir}"
                raise AssertionError(message)
            collected.append(TargetSamples(ctx=ctx, samples=by_dir[ctx.dir]))
        return collected

    mock._answer = answer


def sampling_call(mock: CollectSamplesRecorder, index: int) -> SamplingCall:
    """The targets and bench command of sampling call ``index``, failing when there was none."""
    if index >= len(mock.calls):
        message = f"expected collect_samples to have been called {index + 1} time(s)"
        raise AssertionError(message)
    call = mock.calls[index]
    return SamplingCall(targets=call.targets, bench=call.options.bench)


def trimmed_report_lines(report: str) -> list[str]:
    """The report's lines, stripped of color and of the indentation a grouped metric carries."""
    return [_ANSI_RE.sub("", line).strip() for line in report.split("\n")]


def as_logged(value: SessionLogRecord) -> object:
    """``value`` after the round trip through the wire the session log puts it through.

    A record read back off the log is a fresh dataclass built from JSON, so the
    two sides have to meet on the logged shape to compare field by field.
    """
    return json.loads(json.dumps(record_to_wire(value)))


def last_iteration_of(root: str) -> IterationRecord:
    """The iteration record ``root``'s log ends on, failing when it ends on something else."""
    records = read_records(session_jsonl_path(root))
    last = records[-1] if records else None
    assert isinstance(last, IterationRecord), (
        f"expected an iteration record at the end of {session_jsonl_path(root)}"
    )
    return last
