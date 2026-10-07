"""Pre-flight checks for ``gymrat supervise``.

Owns everything between the doctor gate and the budget file: the ``checks``
warning, the dirty-tree guards, the session open/resume under the repository
lock, the stop-condition refusal, the baseline measurement, and the
feasibility check.
Everything after the warning runs as the ``supervise`` command's preflight
stage, which appends its own ``command`` record to the session log.
The module raises :class:`GymratError` for refusals and lets the command's
boundary route them to exit 2, except the doctor gate, which renders its own
report to stderr and leaves with code 2 itself.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import typer

from gymrat.cli.console import resolve_stream_color
from gymrat.cli.exit import write_and_flush, write_stdout
from gymrat.cli.run_setup import SharedFlags, begin_run
from gymrat.command_run import CommandTrace, with_repo_lock
from gymrat.config import CliFlags, ResolvedConfig
from gymrat.doctor import build_doctor_report, render_doctor_report
from gymrat.errors import TOOL_FAILURE_EXIT_CODE, GymratError
from gymrat.loop.baseline import measure_baseline
from gymrat.loop.iterate.run import stop_condition
from gymrat.loop.start import start_session
from gymrat.report.loop import format_start_summary
from gymrat.sampling import RunOptions, TargetSpec
from gymrat.session.budget import (
    estimate_iterate_duration,
    minutes_to_ms,
    ms_to_minutes,
)
from gymrat.session.paths import (
    baseline_worktree_dir,
    experiment_worktree_dir,
    session_jsonl_path,
)
from gymrat.session.store import (
    append_record,
    fold_session,
    last_kept_position,
    latest_baseline,
    read_records,
)
from gymrat.session.workspace import changed_file_count, dirty_file_count
from gymrat.utils import pluralize, warn_to_stderr

if TYPE_CHECKING:
    from gymrat.loop.start import StartResult
    from gymrat.session.records import SessionLogRecord
    from gymrat.session.store import SessionState

_BASELINE_LABEL = ".gymrat/worktrees/baseline"


def _read_records(root: str) -> list[SessionLogRecord]:
    return read_records(session_jsonl_path(root))


@dataclass(frozen=True, slots=True)
class PreflightFlags:
    """The ``supervise`` flags the pre-flight reads.

    Attributes:
        baseline_ref: The git ref to measure as baseline, or ``None`` to
            reuse the existing baseline.
        max_minutes: The session's wall-clock cap; the feasibility check
            refuses to launch when one estimated iterate cannot fit inside it.
        force: Whether to launch despite a met stop condition or a failed
            feasibility check.
        allow_dirty: Whether to launch from a main working tree with
            uncommitted changes, warning instead of refusing.
    """

    baseline_ref: str | None
    max_minutes: float
    force: bool
    allow_dirty: bool


def run_preflight(*, root: str, config: ResolvedConfig, flags: PreflightFlags) -> StartResult:
    """Run every judgment-free setup step before the agent's first turn.

    Order: checks warning, then under the repository lock the working-tree
    guard, experiment-worktree guard, session, stop condition, baseline
    measurement, and feasibility check. This function does not run
    ``doctor_gate``; call it first.

    Args:
        root: The repository root path.
        config: The resolved configuration for the session.
        flags: The ``supervise`` flags the pre-flight reads.

    Returns:
        The started or resumed session: its header, folded state, and whether
        it was resumed.

    Raises:
        GymratError: When the main working tree is dirty (without
            ``allow_dirty``), the experiment worktree holds unsettled or
            unmeasured changes, a stop condition is met (without ``force``),
            the feasibility check refuses, or another process holds the
            repository lock.
    """
    if config.checks is None:
        warn_to_stderr("warning: checks is not configured — keep will commit with the gate off")

    async def body(_trace: CommandTrace) -> StartResult:
        _working_tree_gate(root, allow_dirty=flags.allow_dirty)
        validate_experiment_worktree(root)
        result = _session_step(root, config, flags.baseline_ref)
        _stop_condition_gate(config, result.state, force=flags.force)
        await _baseline_step(root, config)
        _check_feasibility(root, max_minutes=flags.max_minutes, force=flags.force)
        return result

    return asyncio.run(with_repo_lock("supervise", body, args={"stage": "preflight"}, root=root))


def doctor_gate(root: str, *, color: bool | None = None) -> None:
    """Run the four doctor sections and refuse to launch if any check fails.

    On a failure the rendered doctor report goes to stderr before anything else
    runs.

    Args:
        root: The repository root path.
        color: The explicit color choice for the report, or ``None`` to defer to
            the environment and TTY detection.

    Raises:
        typer.Exit: With the tool-failure code when a doctor check fails.
    """
    report = build_doctor_report(CliFlags(), root)
    if not report.has_failures:
        return
    resolved_color = resolve_stream_color(color, sys.stderr)
    rendered = render_doctor_report(report, color=resolved_color)
    write_and_flush(sys.stderr, rendered + "\n")
    raise typer.Exit(TOOL_FAILURE_EXIT_CODE)


def _session_step(
    root: str,
    config: ResolvedConfig,
    baseline_ref: str | None,
) -> StartResult:
    """Open, resume, or archive-and-reopen the session.

    The caller holds the repository lock for the full session-through-feasibility span.

    Args:
        root: The repository root path.
        config: The resolved configuration for the session.
        baseline_ref: The git ref to measure as baseline, or ``None`` to reuse the existing
            baseline.

    Returns:
        The session start result.
    """
    result = start_session(root, baseline_ref, config)

    summary = format_start_summary(result, config.runbook)
    write_stdout(summary + "\n")

    if result.resumed and baseline_ref is not None:
        warn_to_stderr(
            f"warning: --baseline {baseline_ref} ignored because the session was resumed"
        )
    return result


def _stop_condition_gate(
    config: ResolvedConfig,
    state: SessionState,
    *,
    force: bool,
) -> None:
    """Refuse when a stop condition is already met, unless ``force``."""
    error = stop_condition(config, state)
    if error is None:
        return

    message = str(error)
    hint = "Start a new session, or raise the limit in gymrat.toml."
    if force:
        warn_to_stderr(f"warning: {message}")
        return
    raise GymratError(message, hint=hint)


async def _baseline_step(
    root: str,
    config: ResolvedConfig,
) -> None:
    """Measure the baseline when the log holds no baseline record.

    The caller holds the repository lock for the full session-through-feasibility span.

    Args:
        root: The repository root.
        config: The resolved configuration the baseline is measured with.
    """
    if latest_baseline(_read_records(root)) is not None:
        return

    target = TargetSpec(label=_BASELINE_LABEL, target=baseline_worktree_dir(root))
    progress = begin_run(SharedFlags(), 1, command="supervise")
    try:
        run_options = RunOptions.from_config(
            config, on_progress=progress.report, warn=progress.warn
        )
        _result, record = await measure_baseline(target, run_options)
        append_record(session_jsonl_path(root), record)
    finally:
        progress.stop()


def _check_feasibility(root: str, *, max_minutes: float, force: bool) -> None:
    """Refuse to launch when the cap cannot fit one iterate, unless ``force``."""
    records = _read_records(root)
    estimate = estimate_iterate_duration(records)
    if estimate is None:
        write_and_flush(
            sys.stderr,
            "one iterate runs one baseline pass and one experiment pass\n",
        )
        return

    needed_ms = estimate.duration_ms
    cap_ms = minutes_to_ms(max_minutes)
    if needed_ms <= cap_ms or force:
        return

    source_minutes = round(ms_to_minutes(estimate.source_duration_ms))
    needed_minutes = round(ms_to_minutes(needed_ms))
    cap_minutes = round(ms_to_minutes(cap_ms))
    message = (
        f"the {estimate.source} took {source_minutes}m; "
        f"one iterate needs about {needed_minutes}m; "
        f"the {cap_minutes}m cap cannot fit one."
    )
    hint = "Raise --max-minutes, or pass --force to launch anyway."
    raise GymratError(message, hint=hint)


def _working_tree_gate(root: str, *, allow_dirty: bool) -> None:
    """Refuse a dirty main working tree unless ``allow_dirty`` was set, warning when it was."""
    count = dirty_file_count(root)
    if count == 0:
        return

    if not allow_dirty:
        message = f"Working tree has {pluralize(count, 'uncommitted file')}."
        hint = "Commit or stash your changes, or pass --allow-dirty to proceed anyway."
        raise GymratError(message, hint=hint)

    warn_to_stderr(
        f"warning: working tree has {pluralize(count, 'dirty file')} — "
        "proceeding because --allow-dirty was set"
    )


def validate_experiment_worktree(root: str) -> None:
    """Refuse to launch when the experiment worktree has unmeasured changes.

    An unsettled iteration needs settling first; unmeasured edits — committed or
    still uncommitted — need measuring or reverting. The check runs regardless of
    ``--allow-dirty``, which covers only the main working tree.

    Args:
        root: The repository root path.

    Raises:
        GymratError: When the experiment worktree has unsettled or unmeasured
            changes.
    """
    state = fold_session(_read_records(root))
    if state.finalized is not None or state.session is None:
        return

    worktree = experiment_worktree_dir(root)
    target = last_kept_position(state, state.session.baseline.sha)
    count = changed_file_count(worktree, target)
    if count == 0:
        return

    if state.unsettled:
        message = "The experiment worktree has an unsettled iteration with uncommitted changes."
        hint = "Run gymrat keep or gymrat discard first."
    elif state.ends_on_gating_block:
        message = "The last keep was refused for a gating regression."
        hint = "Run gymrat discard to revert it."
    else:
        message = f"The experiment worktree has {pluralize(count, 'unmeasured edit')}."
        hint = "Measure them with gymrat iterate or revert them with gymrat discard."
    raise GymratError(message, hint=hint)
