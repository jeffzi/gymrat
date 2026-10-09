"""Sampling stubs and record helpers shared by the iterate tests.

The one boundary these fixtures mock is sampling: it shells out to the
consumer's bench script, which no test can run. Everything downstream of it —
verdicts, aggregation, the record, the report — runs for real, so the stubs key
their answers on the worktree *directory* each context names rather than on call
order. A side that landed in the wrong half of the record could then only have
come from the wrong worktree, which is what lets the assertions downstream read
as evidence.

The ``_`` prefix marks a shared helper rather than a test module; it is
imported as ``tests.loop.iterate._fixtures``.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from gymrat.errors import GymratError
from gymrat.progress_events import PassFinished, PassStarted
from gymrat.sampling import SamplingOptions, TargetContext, TargetSamples
from gymrat.session.paths import session_jsonl_path
from gymrat.session.records import (
    IterationRecord,
    MetricVerdict,
    SessionLogRecord,
    SessionRecord,
    record_to_wire,
)
from gymrat.session.workspace import Worktrees
from tests._ansi import stripped_lines
from tests._exec_fixtures import expected_result
from tests.session.records._fixtures import (
    SESSION_ID,
    log_records,
    session_record,
    write_session_log,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from gymrat.exec import ExecOptions, ExecResult

#: Where ``iterate_session`` looks up ``collect_samples``, the one sampling seam the stubs replace.
COLLECT_SAMPLES_TARGET = "gymrat.loop.iterate.confirm.collect_samples"

#: Ten rounds of a bench that stayed near 100.
BASELINE_MS: list[float] = [100, 101, 99, 100, 102, 98, 100, 101, 99, 100]

#: The same ten rounds, an order of magnitude larger, so a second metric reads differently.
BASELINE_BYTES: list[float] = [value * 10 for value in BASELINE_MS]


def scaled(values: list[float], factor: float) -> list[float]:
    """Scale every round by the same factor, moving the median by exactly that much.

    A constant factor leaves every pairwise difference the same sign, which is
    what makes the permutation test call the move rather than shrug at it.

    Args:
        values: The rounds to scale.
        factor: The multiplier applied to every round.

    Returns:
        The scaled rounds, in the same order.
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


#: The filter template a bench that can run a subset of its metrics is configured with.
FILTER = "npm run bench -- --filter {names}"


def regressed_rounds() -> list[dict[str, float]]:
    """Ten rounds 10% slower and 10% fatter than the baseline's."""
    return rounds(scaled(BASELINE_MS, 1.1), scaled(BASELINE_BYTES, 1.1))


def improved_rounds() -> list[dict[str, float]]:
    """Ten rounds 10% faster and 20% leaner than the baseline's."""
    return rounds(scaled(BASELINE_MS, 0.9), scaled(BASELINE_BYTES, 0.8))


def iterate_session_header(root: str, *, experiment: str | None = None) -> SessionRecord:
    """A session header whose worktrees sit beside the default paths.

    Placing the worktrees on ``side-experiment`` / ``side-baseline`` rather than
    on the defaults means a run that recomputed the paths instead of reading them
    off the record would bench directories no test ever filled.

    Args:
        root: The repository the worktrees sit under.
        experiment: The experiment worktree to name instead of
            ``<root>/side-experiment``; None keeps the side path.

    Returns:
        The session header naming the side worktrees.
    """
    return session_record(
        session_id=SESSION_ID,
        worktrees=Worktrees(
            experiment=str(Path(root) / "side-experiment") if experiment is None else experiment,
            baseline=str(Path(root) / "side-baseline"),
        ),
    )


#: Fourteen minutes: an iteration longer than what the budget-refusal tests leave on the clock.
OUTLASTING_ITERATION_MS = 840_000


def write_iterate_session(root: str, history: tuple[SessionLogRecord, ...] = ()) -> SessionRecord:
    """Write an open iterate session log: the side-worktree header, then ``history``.

    Args:
        root: The repository whose session log is written.
        history: The records that follow the header; empty leaves a fresh session.

    Returns:
        The session header written.
    """
    header = iterate_session_header(root)
    write_session_log(root, header, history)
    return header


@dataclass(frozen=True, slots=True)
class PairedRun:
    """One paired run's answer to a sampling call: the rounds each worktree reports."""

    experiment: list[dict[str, float]]
    baseline: list[dict[str, float]]


@dataclass(frozen=True, slots=True)
class _RecordedCall:
    """The positional arguments one ``collect_samples`` call was handed."""

    targets: list[TargetContext]
    options: SamplingOptions


class CollectSamplesRecorder:
    """A stand-in for ``collect_samples`` that records every call it answers.

    The recorder is installed once by :func:`install_collect_samples`; a test
    then configures how it answers with :func:`stub_samples` or
    :func:`stub_runs`. Every call is stored in ``calls``, so a test can read the
    targets and options a call was handed.
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
        self.calls.append(_RecordedCall(target_list, options))
        if self._answer is None:
            message = "collect_samples was called before a stub was installed"
            raise AssertionError(message)
        return self._answer(target_list)

    @property
    def call_count(self) -> int:
        return len(self.calls)


def regressed_run() -> PairedRun:
    """A paired run whose both metrics read 10% worse than the baseline's."""
    return PairedRun(regressed_rounds(), baseline_rounds())


def assert_permutation(
    metric: MetricVerdict, *, delta: float, verdict: str, confirmed: bool
) -> None:
    """Assert ``metric`` is a gating permutation verdict with the given delta and outcome.

    Args:
        metric: The verdict an iteration recorded for one metric.
        delta: The expected percentage change.
        verdict: The expected verdict word.
        confirmed: Whether the confirm rerun is expected to have settled it.
    """
    assert metric.delta_pct == pytest.approx(delta, abs=1e-6)
    assert metric.verdict == verdict
    assert metric.method == "permutation"
    assert metric.p is not None
    assert metric.noise_pct is not None
    assert metric.gating is True
    assert metric.confirmed is confirmed


def install_collect_samples(monkeypatch: pytest.MonkeyPatch) -> CollectSamplesRecorder:
    """Replace ``gymrat.loop.iterate.confirm.collect_samples`` with a fresh recorder."""
    recorder = CollectSamplesRecorder()
    monkeypatch.setattr(COLLECT_SAMPLES_TARGET, recorder)
    return recorder


def run_before_each_call(
    monkeypatch: pytest.MonkeyPatch,
    recorder: CollectSamplesRecorder,
    before: Callable[[SamplingOptions], Awaitable[None]],
) -> None:
    """Wrap ``recorder`` so every sampling call first runs ``before``, then is answered.

    Args:
        monkeypatch: The patcher that installs the wrapper over ``collect_samples``.
        recorder: The installed recorder that still answers each call.
        before: The work a call does before its samples come back, such as
            advancing a fake clock or writing into the worktree. It is handed
            the options the call was given.
    """

    async def sample_after_before(
        adapter: object,
        targets: Sequence[TargetContext],
        options: SamplingOptions,
        abort: object,
    ) -> list[TargetSamples]:
        await before(options)
        return await recorder(adapter, targets, options, abort)

    monkeypatch.setattr(COLLECT_SAMPLES_TARGET, sample_after_before)


def report_a_pass_per_call(
    monkeypatch: pytest.MonkeyPatch, recorder: CollectSamplesRecorder
) -> None:
    """Wrap ``recorder`` so every sampling call reports one pass the way sampling does.

    The pass is reported through the ``on_progress`` the call was handed, so a
    run that stopped forwarding sampling progress shows no pass at all.

    Args:
        monkeypatch: The patcher that installs the wrapper over ``collect_samples``.
        recorder: The installed recorder that still answers each call.
    """

    async def report_a_pass(options: SamplingOptions) -> None:
        if options.on_progress is not None:
            for event_type in (PassStarted, PassFinished):
                options.on_progress(
                    event_type(round=1, total_rounds=1, target_count=1, label="x", at_ms=0)
                )

    run_before_each_call(monkeypatch, recorder, report_a_pass)


def _samples_by_dir(
    targets: list[TargetContext], by_dir: dict[str, list[dict[str, float]]], stub: str
) -> list[TargetSamples]:
    """Answer each target with the rounds ``by_dir`` holds for its worktree directory."""
    collected: list[TargetSamples] = []
    for ctx in targets:
        try:
            samples = by_dir[ctx.dir]
        except KeyError as error:
            message = f"{stub}: unrecognized worktree dir {ctx.dir}"
            raise AssertionError(message) from error
        collected.append(TargetSamples(ctx=ctx, samples=samples))
    return collected


def stub_samples(
    mock: CollectSamplesRecorder,
    root: str,
    experiment: list[dict[str, float]],
    baseline: list[dict[str, float]],
) -> None:
    """Answer every sampling call with ``experiment`` and ``baseline`` keyed on worktree dir."""
    worktrees = iterate_session_header(root).worktrees
    by_dir = {worktrees.experiment: experiment, worktrees.baseline: baseline}

    def answer(targets: list[TargetContext]) -> list[TargetSamples]:
        return _samples_by_dir(targets, by_dir, "stub_samples")

    mock._answer = answer


def stub_improved_samples(mock: CollectSamplesRecorder, root: str) -> None:
    """Answer every sampling call with an experiment 10% faster and 20% leaner than the baseline."""
    stub_samples(mock, root, improved_rounds(), baseline_rounds())


def stub_runs(
    mock: CollectSamplesRecorder,
    root: str,
    runs: list[PairedRun | GymratError],
) -> None:
    """Answer each sampling call with the next paired run, keyed on worktree dir.

    A call past the last run rejects, so an unexpected extra rerun surfaces as a
    failure rather than as silently reused samples.

    Args:
        mock: The sampling recorder whose answers to set.
        root: The repository whose session header names the worktrees.
        runs: One entry per expected sampling call, in order. A
            :class:`GymratError` entry rejects that call, standing in for a
            bench that failed mid-run.
    """
    worktrees = iterate_session_header(root).worktrees
    state = {"index": 0}

    def answer(targets: list[TargetContext]) -> list[TargetSamples]:
        index = state["index"]
        state["index"] = index + 1
        try:
            run = runs[index]
        except IndexError as error:
            message = f"unexpected sampling call {index + 1}"
            raise AssertionError(message) from error
        if isinstance(run, GymratError):
            raise run
        by_dir = {worktrees.experiment: run.experiment, worktrees.baseline: run.baseline}
        return _samples_by_dir(targets, by_dir, "stub_runs")

    mock._answer = answer


def trimmed_report_lines(report: str) -> list[str]:
    """The report's lines, stripped of color and of the indentation a grouped metric carries."""
    return stripped_lines(report, keep_blank=True)


def as_logged(value: SessionLogRecord) -> object:
    """A record after the round trip through the wire the session log puts it through.

    A record read back off the log is a fresh dataclass built from JSON, so the
    two sides have to meet on the logged shape to compare field by field.

    Args:
        value: The record to send through the wire.

    Returns:
        The decoded JSON object the session log would hold for ``value``.
    """
    return json.loads(json.dumps(record_to_wire(value)))


def last_iteration_of(root: str) -> IterationRecord:
    """The iteration record ``root``'s log ends on, failing when it ends on something else."""
    records = log_records(root)
    last = records[-1] if records else None
    assert isinstance(last, IterationRecord), (
        f"expected an iteration record at the end of {session_jsonl_path(root)}"
    )
    return last


#: What the metric-lines adapter says about the malformed line :func:`bench_malformed_once` prints.
MALFORMED_LINE_WARNING = "Failed to parse METRIC line: METRIC foo=bar"


def bench_malformed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the bench process: every run reports, the first one a malformed line too."""
    run_indexes = itertools.count()

    async def fake_exec(_command: str, _options: ExecOptions) -> ExecResult:
        index = next(run_indexes)
        stdout = f"METRIC total_ms={BASELINE_MS[index % len(BASELINE_MS)]}"
        if index == 0:
            stdout += "\nMETRIC foo=bar"
        return expected_result(stdout)

    monkeypatch.setattr("gymrat.sampling.exec", fake_exec)
