"""Confirmation reruns for gating regressions found by the first bench pass.

A confirmation re-measures only the gating metrics the first run called
regressed.  ``exact`` metrics never take part: one differing sample is already
their whole signal, so a rerun could only add noise to a decision that has none.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gymrat.config import FILTER_PLACEHOLDER
from gymrat.loop.iterate.bench import IterationContext, bench_and_judge
from gymrat.progress_events import (
    ConfirmFinished,
    ConfirmStarted,
    PassFinished,
    PassStarted,
    ProgressEvent,
    default_clock,
    emit_progress,
)

if TYPE_CHECKING:
    from gymrat.model import MetricVerdict, ResolvedMetricMeta
    from gymrat.session import PairedSamples


@dataclass(frozen=True, slots=True)
class Confirmation:
    """What a confirmation rerun measured, and which regressions it stood behind.

    Attributes:
        filtered: The metrics the rerun re-measured, in the order the run
            reported them.
        samples: The rerun's own rounds, kept raw so a later statistics change
            can re-read them.
        confirmed: The subset of ``filtered`` the rerun also gated as regressed.
        absent: The subset of ``filtered`` the rerun produced no verdict for at
            all — disjoint from ``confirmed``, and not the complement of it.
    """

    filtered: tuple[str, ...]
    samples: PairedSamples
    confirmed: frozenset[str]
    absent: frozenset[str]


def with_confirm_phase(ctx: IterationContext) -> IterationContext:
    """Return a context whose ``on_progress`` tags pass events as confirmation runs."""
    original = ctx.options.on_progress
    if original is None:
        return ctx

    def wrapper(event: ProgressEvent) -> None:
        if isinstance(event, PassStarted | PassFinished):
            original(replace(event, phase="confirm"))
        else:
            original(event)

    return replace(ctx, options=replace(ctx.options, on_progress=wrapper))


def _needs_confirmation(meta: ResolvedMetricMeta, verdict: MetricVerdict | None) -> bool:
    """Whether ``meta``'s metric is a gating, non-exact regression the first run found."""
    return meta.gating and not meta.exact and verdict is not None and verdict.verdict == "regressed"


async def confirm_regressions(
    ctx: IterationContext,
    verdicts: dict[str, MetricVerdict],
    metric_meta: dict[str, ResolvedMetricMeta],
) -> Confirmation | None:
    """Re-measure the gating metrics the first run called regressed, once.

    ``exact`` metrics never take part: one differing sample is already their whole
    signal, so a rerun could only add noise to a decision that has none. A
    ``filter`` template benches just the named metrics; without one the whole
    bench re-runs and the same metrics are read out of it.

    Returns what the rerun found, or ``None`` when nothing called for one.

    Raises:
        GymratError: When the rerun's bench command fails — an iteration nobody
            could confirm is not recorded.
    """
    filtered = [
        name for name, meta in metric_meta.items() if _needs_confirmation(meta, verdicts.get(name))
    ]
    if not filtered:
        return None

    filtered_tuple = tuple(filtered)
    names = " ".join(_shell_quote(name) for name in filtered)
    bench = (
        ctx.config.bench
        if ctx.config.filter is None
        else ctx.config.filter.replace(FILTER_PLACEHOLDER, names)
    )
    emit_progress(
        ctx.options.on_progress,
        ConfirmStarted(
            filtered_metrics=None if ctx.config.filter is None else filtered_tuple,
            at_ms=default_clock(),
        ),
    )

    confirm_ctx = with_confirm_phase(ctx)
    rerun = await bench_and_judge(confirm_ctx, bench, metric_meta)
    confirmed = frozenset(
        name
        for name in filtered
        if (verdict := rerun.verdicts.get(name)) is not None and verdict.verdict == "regressed"
    )
    absent = frozenset(name for name in filtered if rerun.verdicts.get(name) is None)

    reproduced = len(confirmed) > 0
    emit_progress(
        ctx.options.on_progress, ConfirmFinished(reproduced=reproduced, at_ms=default_clock())
    )

    return Confirmation(
        filtered=filtered_tuple,
        samples=rerun.samples,
        confirmed=confirmed,
        absent=absent,
    )


def apply_confirmation(
    verdicts: dict[str, MetricVerdict],
    confirmation: Confirmation | None,
) -> dict[str, MetricVerdict]:
    """The verdicts as finally read, with every regression the rerun disowned demoted.

    A metric the rerun never reported is left regressed — the rerun's job is to
    disprove a regression, and silence disproves nothing. Only the verdict word
    moves; the delta, noise, and p-value stay the first run's.
    """
    if confirmation is None:
        return verdicts

    settled: dict[str, MetricVerdict] = {}
    for name, verdict in verdicts.items():
        disagreed = (
            name in confirmation.filtered
            and name not in confirmation.confirmed
            and name not in confirmation.absent
        )
        settled[name] = replace(verdict, verdict="no-signal") if disagreed else verdict
    return settled


def _shell_quote(value: str) -> str:
    """``value`` as a single shell-safe word, platform-aware.

    Metric names are the bench's to choose, and mitata's ``sort(n=1000)/time``
    alias shape is an ordinary one: spliced into the filter template raw, the
    shell either splits the name across arguments or refuses the command as a
    syntax error — and a rerun that cannot run demotes a real regression to no
    signal.

    On POSIX, ``shlex.quote`` handles safe-word detection and single-quote
    escaping. On win32, ``cmd.exe`` uses double quotes, and
    ``subprocess.list2cmdline`` produces the correct escaping.
    """
    if sys.platform == "win32":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)
