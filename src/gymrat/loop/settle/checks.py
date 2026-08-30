"""Run the configured checks command and evaluate gating predicates.

Shapes the checks output for the keep gate and decides whether a keep may
proceed based on gating regressions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.markup import escape

from gymrat.eta import MS_PER_SECOND
from gymrat.exec import (
    ExecOptions,
    ExecTimeoutError,
    exec,  # noqa: A004 -- names the subprocess executor `exec`
)
from gymrat.loop.output_limit import limit_output
from gymrat.report.style import RENDER_WIDTH, color_from_env, format_hint, render_lines

if TYPE_CHECKING:
    from gymrat.config import BenchlessConfig
    from gymrat.session.records import IterationRecord, MetricVerdict


@dataclass(frozen=True, slots=True)
class ChecksRun:
    """What the checks command answered, once it has run."""

    passed: bool
    output: str
    stdout_bytes: int
    stderr_bytes: int


def _stderr_color() -> bool:
    """Whether the warning gymrat writes to stderr carries color.

    :func:`color_from_env` owns the ``FORCE_COLOR`` / ``NO_COLOR`` precedence
    every color surface shares; with neither declared, stderr's own TTY state
    decides, so a warning piped into a file stays plain.
    """
    declared = color_from_env()
    return declared if declared is not None else sys.stderr.isatty()


async def run_checks(config: BenchlessConfig, experiment_dir: str) -> ChecksRun | None:
    """Run the configured checks in the experiment worktree.

    A timeout counts as a failure with whatever the command managed to write: the
    gate asks whether the tree is provably good, and a run that never finished has
    not answered. Each stream is cut to the relay limit on its own, so a suite that
    writes its failures to stderr is as readable as one that writes them to stdout.

    Returns:
        What the command answered, or ``None`` when no checks are configured — in
        which case the missing gate is warned about instead.
    """
    command = config.checks
    if command is None:
        hint = render_lines(
            format_hint(
                "set `checks` in `gymrat.toml` to the command that must pass before an "
                "edit is kept."
            ),
            color=_stderr_color(),
            width=RENDER_WIDTH,
        )
        sys.stderr.write(
            "Warning: no checks command is configured, so gymrat keep is committing "
            f"with the gate off.\n{hint}\n"
        )
        return None

    result = await exec(
        command,
        ExecOptions(cwd=experiment_dir, timeout_ms=config.timeout_seconds * MS_PER_SECOND),
    )

    if isinstance(result, ExecTimeoutError):
        passed = False
        lead = [f"{command} timed out after {result.timeout_ms}ms"]
    else:
        passed = result.exit_code == 0
        lead: list[str] = []

    output = "\n".join(
        part
        for part in (*lead, limit_output(result.stdout), limit_output(result.stderr))
        if part.strip() != ""
    )

    return ChecksRun(
        passed=passed,
        output=output,
        stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes,
    )


def has_standing_gating_regression(iteration: IterationRecord) -> bool:
    """Whether the iteration carries a regression the loop refuses to commit over.

    Both halves are required: the outcome is what the agent was shown, and a gating
    metric standing behind the regression is what makes it real. A noisy metric
    earns that standing from the confirmation rerun — a regression the rerun would
    not repeat leaves the iteration keepable. An exact metric is deterministic, so
    the rerun skips it and its ``confirmed`` stays ``False``; gating on ``confirmed``
    alone would let every exact regression through.

    Silence earns the same standing as disagreement: a metric the rerun was asked
    about and never reported back on lands in ``confirm.absent``, its ``confirmed``
    still ``False`` because nothing re-measured it. The gate fails closed on those —
    a rerun that cannot see the metric is not evidence the regression went away.
    """
    if iteration.outcome != "regressed":
        return False
    if _unmeasured_gating_regressions(iteration):
        return True
    return any(
        _is_gating_regression(metric) and (metric.confirmed or metric.method == "exact")
        for metric in iteration.metrics.values()
    )


def _unmeasured_gating_regressions(iteration: IterationRecord) -> list[str]:
    """The gating metrics that regressed and the confirmation rerun never reported on."""
    confirm = iteration.confirm
    absent = set(confirm.absent) if confirm is not None and confirm.absent is not None else set()
    return [
        name
        for name, metric in iteration.metrics.items()
        if _is_gating_regression(metric) and name in absent
    ]


def _is_gating_regression(metric: MetricVerdict) -> bool:
    """Whether a metric is a gating metric that the checks called a regression."""
    return metric.gating and metric.verdict == "regressed"


def gating_refusal(iteration: IterationRecord) -> str:
    """How the refusal reads to the agent that has to act on it, as markup.

    A regression the rerun stood behind needs no explaining beyond the number the
    iteration already reported. One the rerun never re-measured does: the agent is
    looking at a metric its own report called regressed and unconfirmed, and without
    the missing measurement named, the block reads as gymrat contradicting itself.
    The extra hint points at the likeliest cause — a filter template that narrows
    the rerun to a subset the bench does not answer with.
    """
    refusal = f"Keep refused: iteration {iteration.seq} regressed a gating metric."
    settle_hint = "fix the regression and run `gymrat iterate` again, or run `gymrat discard`"

    unmeasured = _unmeasured_gating_regressions(iteration)
    if not unmeasured:
        return f"{refusal}\n{format_hint(f'{settle_hint}.')}"

    named = "\n".join(
        f"  {escape(name)}: not measured on the confirmation rerun, so the regression stands"
        for name in unmeasured
    )
    reported = ", ".join(unmeasured)
    return f"{refusal}\n{named}\n" + format_hint(
        f"check that the filter template (or the bench itself) reports {reported}, "
        f"then {settle_hint}."
    )
