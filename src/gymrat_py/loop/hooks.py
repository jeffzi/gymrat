"""Run the before/after commands a consumer hangs off each measurement.

Hooks steer the loop; they cannot brick it. Every invocation that reaches
:func:`run_hook` runs its command -- deciding whether a stage has a command at
all is the caller's job. A command that fails, overruns its timeout, or never
starts at all comes back as a report and a record, never as a raised exception:
there is no hook failure worth throwing away a measurement over.

Two shaping choices are worth spelling out:

- The record's byte counts are what the command *wrote*, not what was relayed.
  A figure above the relay limit is how a reader of the log learns the report
  was cut, so the pre-relay totals from ``exec`` are what land in the record.
- A successful hook's stderr is kept out of the report. Commands write progress
  there routinely, and repeating it would drown the measurement the hook was
  annotating. A *failing* hook's stderr is shown -- and held to the same byte
  cap as its stdout, since a build log buries a measurement as easily on one
  channel as on the other.
"""

import asyncio
import json
import time
from dataclasses import dataclass

# Bound at module scope under the builtin's name so a test can substitute the
# subprocess boundary via ``monkeypatch.setattr`` on this module.
from gymrat_py.exec import (
    FAILURE_EXIT_CODE,
    ExecOptions,
    ExecResult,
    ExecTimeoutError,
    exec,  # noqa: A004 -- names the subprocess executor `exec`
)
from gymrat_py.loop.output_limit import limit_output
from gymrat_py.session import HookRecord, IterationRecord, SessionRecord, record_to_wire
from gymrat_py.session.schema import HookStage

#: How long a hook may run before it is killed. Long enough to build, short
#: enough to notice.
HOOK_TIMEOUT_MS = 30_000


@dataclass(frozen=True, slots=True)
class HookInvocation:
    """Which command to run, and everything the payload tells it about the loop so far.

    Args:
        command: The command line the consumer configured for this stage.
        stage: Which side of a measurement the hook runs on.
        seq: The iteration the hook brackets -- about to be measured, or just recorded.
        session: The session header, source of the worktree, baseline, and branch.
        last_iteration: The iteration the hook can read, ``None`` while the
            session has measured nothing.
        iteration_count: How many iterations the log holds as of this invocation.
        timeout_ms: Milliseconds before the command is killed; ``None`` uses
            :data:`HOOK_TIMEOUT_MS`.
        abort: Event whose setting kills the hook's process group; ``None``
            leaves the hook uninterruptible.
    """

    command: str
    stage: HookStage
    seq: int
    session: SessionRecord
    last_iteration: IterationRecord | None
    iteration_count: int
    timeout_ms: int | None = None
    abort: asyncio.Event | None = None


@dataclass(frozen=True, slots=True)
class HookRun:
    """What one fired hook leaves behind: a record for the log, a report for the agent.

    Args:
        record: The record to append to the session log.
        report: The hook's stdout, truncated and labeled with its stage, then a
            note naming the exit code or timeout when the command did not
            succeed, and the truncated stderr under it. Empty when a successful
            hook printed nothing.
    """

    record: HookRecord
    report: str


@dataclass(frozen=True, slots=True)
class _CommandOutcome:
    """What the command itself did, before any of it is shaped for log or report."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    #: Bytes the command wrote on stdout, ahead of exec's cap and the relay's.
    stdout_bytes: int
    #: Bytes the command wrote on stderr, ahead of exec's cap and the relay's.
    stderr_bytes: int


async def run_hook(invocation: HookInvocation) -> HookRun:
    """Run the stage's command, handing it the loop as JSON on stdin.

    The command runs in the experiment worktree with the payload on its stdin.
    Whatever it does -- succeed, fail, time out, or fail to start -- comes back
    as a :class:`HookRun`; nothing here raises. The record is not appended to
    any log: the caller owns that.
    """
    timeout_ms = HOOK_TIMEOUT_MS if invocation.timeout_ms is None else invocation.timeout_ms
    payload = json.dumps(_build_payload(invocation))

    started_at = time.perf_counter()
    result = await exec(
        invocation.command,
        ExecOptions(
            cwd=invocation.session.worktrees.experiment,
            timeout_ms=timeout_ms,
            abort=invocation.abort,
            stdin=f"{payload}\n",
        ),
    )
    duration_ms = (time.perf_counter() - started_at) * 1000
    outcome = _describe_outcome(result)

    record = HookRecord(
        type="hook",
        stage=invocation.stage,
        seq=invocation.seq,
        exit_code=outcome.exit_code,
        duration_ms=duration_ms,
        # What the command wrote, not what was relayed: a figure above the relay
        # limit is how a reader of the log learns the report was cut.
        stdout_bytes=outcome.stdout_bytes,
        stderr_bytes=outcome.stderr_bytes,
        timed_out=outcome.timed_out,
    )
    return HookRun(record=record, report=_format_report(invocation.stage, outcome, timeout_ms))


def _describe_outcome(result: ExecResult | ExecTimeoutError) -> _CommandOutcome:
    """Fold exec's two result shapes into the one the log and report read.

    A timeout carries no exit code of its own -- the process was killed before
    it had one -- so the shared failure code stands in for it.
    """
    if isinstance(result, ExecTimeoutError):
        return _CommandOutcome(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=FAILURE_EXIT_CODE,
            timed_out=True,
            stdout_bytes=result.stdout_bytes,
            stderr_bytes=result.stderr_bytes,
        )
    return _CommandOutcome(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        timed_out=False,
        stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes,
    )


def _build_payload(invocation: HookInvocation) -> dict[str, object]:
    """The loop as the hook reads it: where the edit lives, which iteration, whose session."""
    session = invocation.session
    last_iteration = invocation.last_iteration
    return {
        "stage": invocation.stage,
        "experimentDir": session.worktrees.experiment,
        "seq": invocation.seq,
        "lastIteration": record_to_wire(last_iteration) if last_iteration is not None else None,
        "session": {
            "sessionId": session.session_id,
            "baseline": {"ref": session.baseline.ref, "sha": session.baseline.sha},
            "branch": session.branch,
            "iterationCount": invocation.iteration_count,
        },
    }


def _format_report(stage: HookStage, outcome: _CommandOutcome, timeout_ms: int) -> str:
    """Every stdout line labeled with the stage, then a failing hook's note and stderr under it."""
    lines = _split_lines(limit_output(outcome.stdout))
    note = _failure_note(outcome, timeout_ms)

    if note is not None:
        lines.append(note)
        lines.extend(_split_lines(limit_output(outcome.stderr)))

    return "\n".join(f"[{stage}] {line}" for line in lines)


def _failure_note(outcome: _CommandOutcome, timeout_ms: int) -> str | None:
    """What to tell the reader about a hook that did not succeed, or ``None`` if it did."""
    if outcome.timed_out:
        return f"hook timed out after {timeout_ms}ms"
    if outcome.exit_code != 0:
        return f"hook exited {outcome.exit_code}"
    return None


def _split_lines(text: str) -> list[str]:
    """``text`` as lines, with the trailing newline a command leaves behind dropped."""
    trimmed = text.removesuffix("\n")
    return [] if trimmed == "" else trimmed.split("\n")
