"""Pre-flight checks for ``gymrat supervise``.

Owns everything between the doctor gate and the budget file: the ``checks``
warning, the session open/resume under the repository lock, the
stop-condition refusal, the baseline measurement, and the feasibility check.
The module raises :class:`GymratError` for refusals and lets the command's
boundary route them to exit 2, except the doctor gate, which renders its own
report to stderr and exits with code 2 directly.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from gymrat.cli.shared import (
    SharedFlags,
    begin_run,
    resolve_stream_color,
    run_options_of,
    write_and_flush,
)
from gymrat.config import CliFlags, ResolvedConfig
from gymrat.doctor.render import render_doctor_report
from gymrat.doctor.report import build_doctor_report
from gymrat.errors import GymratError
from gymrat.loop.baseline import measure_baseline
from gymrat.loop.iterate.run import stop_condition
from gymrat.loop.start import start_session
from gymrat.report.loop import format_start_summary
from gymrat.sampling import TargetSpec
from gymrat.session import (
    BaselineRecord,
    SessionLogRecord,
    append_record,
    baseline_worktree_dir,
    read_records,
    session_jsonl_path,
)
from gymrat.session.budget import (
    estimate_iterate_duration,
    minutes_to_ms,
    ms_to_minutes,
)
from gymrat.session.lock import acquire_lock
from gymrat.session.paths import lockfile_path

if TYPE_CHECKING:
    from gymrat.loop.start import StartResult
    from gymrat.session.store import SessionState

_BASELINE_LABEL = ".gymrat/worktrees/baseline"


def _read_records(root: str) -> list[SessionLogRecord]:
    return read_records(session_jsonl_path(root))


def _has_baseline(records: list[SessionLogRecord]) -> bool:
    return any(isinstance(r, BaselineRecord) for r in records)


def run_preflight(
    *,
    root: str,
    config: ResolvedConfig,
    baseline_ref: str | None,
    max_minutes: float,
    force: bool,
) -> StartResult:
    """Run every judgment-free setup step before the agent's first turn.

    Order: checks warning, session (under repo lock), stop condition,
    baseline measurement, feasibility check. The doctor gate runs before
    this function — the command calls it earlier.

    Raises:
        GymratError: When a stop condition is met (without ``force``) or the
            feasibility check refuses.
    """
    _checks_warning(config)
    result = _session_step(root, config, baseline_ref)
    _stop_condition_gate(config, result.state, force=force)
    _baseline_step(root, config)
    _check_feasibility(root, max_minutes=max_minutes, force=force)
    return result


def doctor_gate(root: str, *, color: bool | None = None) -> None:
    """Run the four doctor sections and refuse if any check fails."""
    report = build_doctor_report(CliFlags(), root)
    if not report.has_failures:
        return
    resolved_color = resolve_stream_color(color, sys.stderr)
    rendered = render_doctor_report(report, color=resolved_color)
    write_and_flush(sys.stderr, rendered + "\n")
    sys.exit(2)


def _checks_warning(config: ResolvedConfig) -> None:
    if config.checks is None:
        write_and_flush(
            sys.stderr,
            "warning: checks is not configured — keep will commit with the gate off\n",
        )


def _session_step(
    root: str,
    config: ResolvedConfig,
    baseline_ref: str | None,
) -> StartResult:
    """Open, resume, or archive-and-reopen the session under the repository lock."""
    release = acquire_lock(lockfile_path(root), "supervise")
    try:
        result = start_session(root, baseline_ref, config)
    finally:
        release()

    summary = format_start_summary(result, config.runbook)
    write_and_flush(sys.stdout, summary + "\n")

    if result.resumed and baseline_ref is not None:
        write_and_flush(
            sys.stderr,
            f"warning: --baseline {baseline_ref} ignored because the session was resumed\n",
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
        write_and_flush(sys.stderr, f"warning: {message}\n")
        return
    raise GymratError(message, hint=hint)


def _baseline_step(
    root: str,
    config: ResolvedConfig,
) -> None:
    """Measure the baseline when the log holds no baseline record."""
    if _has_baseline(_read_records(root)):
        return

    worktree_dir = baseline_worktree_dir(root)
    target = TargetSpec(label=_BASELINE_LABEL, target=str(worktree_dir))
    progress = begin_run(SharedFlags(), 1, command="supervise")
    try:
        run_options = run_options_of(config, progress)
        release = acquire_lock(lockfile_path(root), "supervise")
        try:
            _result, record = asyncio.run(measure_baseline(target, run_options))
            append_record(session_jsonl_path(root), record)
        finally:
            release()
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
