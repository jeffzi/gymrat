"""The optimization-loop subcommands: start, iterate, keep, discard, finalize, status.

Each command resolves its configuration at the repository root — so a run from a
subdirectory still finds the implicit ``gymrat.toml`` — and, where it mutates the
session, holds the repository's single-flight lock for the length of the work.
Two commands break that lock pattern deliberately:

- ``discard`` runs its confirmation prompt *before* taking the lock: prompting a
  human can block indefinitely, and the repository must not be held hostage to a
  reader who never answers. The session id read at prompt time is carried into
  the locked revert as a guard, so a session that turned over while the prompt
  waited is refused rather than silently discarded.
- ``status`` takes no lock at all: it only reads the log, and a read never races
  a writer into corruption the way two writers would.

The interrupt wiring for ``iterate`` routes a ``SIGINT`` / ``SIGTERM`` into an
abort event the in-flight bench watches, so an interrupted iteration abandons the
current sample rather than the process being torn down mid-measurement.

The loop engines are imported at module load rather than lazily: unlike the
measurement stack these are light, and every loop command reaches one.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import typer

from gymrat_py.cli.iterate_progress import create_fan_out, create_iterate_renderer
from gymrat_py.cli.shared import (
    GATE_EXIT_CODE,
    AdapterOption,
    BenchOption,
    BranchOption,
    ConfigOption,
    DebugOption,
    ForceOption,
    MessageOption,
    NoColorOption,
    PrepareOption,
    SamplesOption,
    TimeoutOption,
    VerboseOption,
    color_override_of,
    exit_with_error,
    is_tty,
    resolve_render_mode,
    resolve_stream_color,
    run_cli,
    run_with_signal_abort,
    set_debug_mode,
    with_repo_lock,
    write_and_flush,
)
from gymrat_py.config import CliFlags, resolve_benchless_config, resolve_config
from gymrat_py.confirm import confirm_action
from gymrat_py.loop.finalize import FinalizeOptions, FinalizeResult, finalize_session
from gymrat_py.loop.iterate import IterateOptions, IterateResult, LoopStopError, iterate_session
from gymrat_py.loop.settle import (
    DiscardResult,
    KeepOptions,
    KeepResult,
    discard_session,
    keep_session,
)
from gymrat_py.loop.start import StartResult, start_session
from gymrat_py.loop.status import status_session
from gymrat_py.report.format import pluralize
from gymrat_py.report.loop import format_baseline_ref
from gymrat_py.session.paths import repo_root
from gymrat_py.session.progress_file import clear_progress, create_sidecar_writer
from gymrat_py.session.store import require_open_session

_RefArgument = typer.Argument(
    metavar="[REF]", help="ref the baseline is pinned to; defaults to HEAD"
)


@dataclass(frozen=True, slots=True)
class _StartOutcome:
    """What the locked ``start`` produced: the session, and the runbook to point at.

    ``runbook`` is carried out of the lock so the summary can name it after the
    lock is released, the way the report itself is written unlocked.
    """

    result: StartResult
    runbook: str | None


def format_start_summary(result: StartResult, runbook: str | None) -> str:
    """The multi-line summary ``start`` prints: a headline, then the session's rows.

    A fresh session opens on ``Started session <id>``; a resumed one leads with
    its history instead. The rows name the branch, the baseline, and both
    worktrees, their labels padded to a common width. A configured runbook and an
    archived predecessor each add a trailing row when present.
    """
    session = result.session
    state = result.state
    if result.resumed:
        headline = (
            f"Resumed session {session.session_id} — "
            f"{pluralize(state.iteration_count, 'iteration')}, "
            f"{pluralize(state.keep_count, 'keep')}"
        )
    else:
        headline = f"Started session {session.session_id}"

    rows: list[tuple[str, str]] = [
        ("branch", session.branch),
        ("baseline", format_baseline_ref(session.baseline)),
        ("experiment worktree", session.worktrees.experiment),
        ("baseline worktree", session.worktrees.baseline),
    ]
    label_width = max(len(label) for label, _ in rows) + 1

    lines = [headline]
    lines.extend(f"  {f'{label}:':<{label_width}} {value}" for label, value in rows)
    if runbook is not None:
        lines.append(f"  runbook: {runbook} — read it before your first edit")
    if result.archived_path is not None:
        lines.append(
            f"  archived the finalized session {result.archived} to {result.archived_path}"
        )
    return "\n".join(lines)


def start(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    ref: str | None = _RefArgument,
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    debug: DebugOption = False,
) -> None:
    """Create or resume this repository's optimization session."""
    if debug:
        set_debug_mode(True)

    flags = CliFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
    )

    async def run() -> None:
        async def body() -> _StartOutcome:
            root = repo_root()
            resolved = resolve_config(flags, root)
            return _StartOutcome(
                result=start_session(root, ref, resolved), runbook=resolved.runbook
            )

        outcome = await with_repo_lock("start", body)
        write_and_flush(sys.stdout, format_start_summary(outcome.result, outcome.runbook) + "\n")

    run_cli(run)


def iterate(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    no_color: NoColorOption = False,
    verbose: VerboseOption = False,
    debug: DebugOption = False,
) -> None:
    """Measure the session's experiment worktree against its baseline."""
    if debug:
        set_debug_mode(True)

    color_override = color_override_of(not no_color)
    resolved_color = resolve_stream_color(color_override, sys.stdout)
    flags = CliFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
    )

    async def run() -> None:
        async def body() -> IterateResult:
            root = repo_root()
            resolved = resolve_config(flags, root)
            required = require_open_session(root, "iterate")

            from gymrat_py.cli.console import (  # noqa: PLC0415 -- console.py imports from shared.py
                stderr_console,
            )

            mode = resolve_render_mode()
            console = stderr_console(color_flag=not no_color)
            seq = required.state.last_seq + 1
            metric_count = len(resolved.metrics) if resolved.metrics is not None else 0
            renderer = create_iterate_renderer(
                mode,
                console,
                seq,
                required.session.session_id,
                resolved.samples,
                metric_count,
                resolved.primary,
                verbose=verbose,
            )
            sidecar_writer = create_sidecar_writer(root, seq, started_at=time.time())
            fan_out = create_fan_out([renderer.report, sidecar_writer])
            try:
                return await run_with_signal_abort(
                    lambda abort: iterate_session(
                        root,
                        resolved,
                        IterateOptions(abort=abort, on_progress=fan_out),
                        color=resolved_color,
                    )
                )
            finally:
                renderer.stop()
                clear_progress(root)

        try:
            result = await with_repo_lock("iterate", body)
        except LoopStopError as error:
            # A stop condition is a gate trip, not a tool failure: the loop did
            # what it was configured to do and declined another iteration.
            exit_with_error(error, GATE_EXIT_CODE)
        write_and_flush(sys.stdout, result.report + "\n")

    run_cli(run)


def keep(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    message: MessageOption = None,
    debug: DebugOption = False,
) -> None:
    """Commit the session's measured edit once its checks pass."""
    if debug:
        set_debug_mode(True)

    resolved_color = resolve_stream_color(None, sys.stdout)
    flags = CliFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
    )

    async def run() -> None:
        async def body() -> KeepResult:
            root = repo_root()
            return await keep_session(
                root,
                resolve_benchless_config(flags, root),
                KeepOptions(message=message),
                color=resolved_color,
            )

        result = await with_repo_lock("keep", body)
        write_and_flush(sys.stdout, result.report + "\n")
        # A blocked keep reports first, then trips the gate: the agent reads why
        # the edit was refused on stdout before the exit code says it was.
        if result.record.status == "blocked":
            raise typer.Exit(GATE_EXIT_CODE)

    run_cli(run)


def discard(*, force: ForceOption = False, debug: DebugOption = False) -> None:
    """Revert the session's experiment worktree to its last commit."""
    if debug:
        set_debug_mode(True)

    async def run() -> None:
        root = repo_root()
        # The session id read while prompting guards the locked revert: a session
        # that turned over between the prompt and the lock is refused rather than
        # discarded out from under a run that never confirmed this one.
        confirmed_session_id: str | None = None
        if is_tty(sys.stdin) and not force:
            required = require_open_session(root, "discard")
            confirmed_session_id = required.session.session_id
            confirmed = confirm_action(
                "discard will revert uncommitted changes in "
                f"{required.session.worktrees.experiment}.\nProceed?",
                sys.stdin,
            )
            if not confirmed:
                write_and_flush(sys.stderr, "discard cancelled\n")
                raise typer.Exit(GATE_EXIT_CODE)

        async def body() -> DiscardResult:
            return discard_session(root, confirmed_session_id)

        result = await with_repo_lock("discard", body)
        write_and_flush(sys.stdout, result.report + "\n")

    run_cli(run)


def finalize(
    *,
    message: MessageOption = None,
    branch: BranchOption = None,
    debug: DebugOption = False,
) -> None:
    """Collapse the session's kept iterations into one commit and close it."""
    if debug:
        set_debug_mode(True)

    async def run() -> None:
        async def body() -> FinalizeResult:
            return finalize_session(repo_root(), FinalizeOptions(message=message, branch=branch))

        result = await with_repo_lock("finalize", body)
        write_and_flush(sys.stdout, result.report + "\n")

    run_cli(run)


def status(  # noqa: PLR0913 -- one parameter per CLI flag, mirroring the shared option surface
    *,
    bench: BenchOption = None,
    prepare: PrepareOption = None,
    adapter: AdapterOption = None,
    samples: SamplesOption = None,
    timeout: TimeoutOption = None,
    config: ConfigOption = None,
    no_color: NoColorOption = False,
    debug: DebugOption = False,
) -> None:
    """Show this repository's session history, read from its log."""
    if debug:
        set_debug_mode(True)

    color_override = color_override_of(not no_color)
    resolved_color = resolve_stream_color(color_override, sys.stdout)
    flags = CliFlags(
        bench=bench,
        prepare=prepare,
        adapter=adapter,
        samples=samples,
        timeout=timeout,
        config=config,
    )

    try:
        root = repo_root()
        report = status_session(root, resolve_benchless_config(flags, root), color=resolved_color)
    except typer.Exit:
        raise
    except Exception as error:  # noqa: BLE001 -- CLI boundary: route any failure through the formatter
        exit_with_error(error)

    write_and_flush(sys.stdout, report + "\n")
