import type { ExecResult, ExecTimeoutError } from "../exec.js";
import { exec, FAILURE_EXIT_CODE } from "../exec.js";
import type { HookRecord, IterationRecord, SessionRecord } from "../session/records.js";
import { limitOutput } from "./output-limit.js";

/** Which side of a measurement a hook command runs on. */
export type HookStage = HookRecord["stage"];

/** Which command to run, and everything the payload tells it about the loop so far. */
export interface HookInvocation {
  /** The command line the consumer configured for this stage. */
  command: string;
  stage: HookStage;
  /** The iteration the hook brackets: the one about to be measured, or the one just recorded. */
  seq: number;
  session: SessionRecord;
  /** The iteration the hook can read, `null` while the session has measured nothing. */
  lastIteration: IterationRecord | null;
  /** How many iterations the log holds as of this invocation. */
  iterationCount: number;
  /** Milliseconds before the command is killed. Defaults to {@link HOOK_TIMEOUT_MS}. */
  timeoutMs?: number;
}

/** What one fired hook leaves behind: a line for the log, and a block for the agent. */
export interface HookRun {
  /** The record to append to the session log. */
  record: HookRecord;
  /**
   * The hook's stdout, truncated and labeled with its stage, followed by a note
   * naming the exit code or the timeout when the command did not succeed and the
   * truncated stderr under it.
   *
   * Empty when a successful hook printed nothing.
   */
  report: string;
}

/** How long a hook may run before it is killed. Long enough to build, short enough to notice. */
const HOOK_TIMEOUT_MS = 30_000;

/** What the hook is told about the loop, as it reads it on stdin. */
interface HookPayload {
  stage: HookStage;
  /** Where the edit under measurement lives, so a hook can inspect or touch it. */
  experimentDir: string;
  seq: number;
  lastIteration: IterationRecord | null;
  session: {
    sessionId: string;
    baseline: SessionRecord["baseline"];
    branch: string;
    iterationCount: number;
  };
}

/** What the command itself did, before any of it is shaped for the log or the report. */
interface CommandOutcome {
  stdout: string;
  stderr: string;
  exitCode: number;
  timedOut: boolean;
}

/**
 * Run the command the consumer configured for this stage, handing it a JSON
 * payload on stdin and the experiment worktree as its working directory.
 *
 * Hooks steer the loop; they cannot brick it. A command that fails, overruns its
 * timeout, or never starts at all comes back as a report and a record rather
 * than as a rejection — there is no hook failure worth throwing away a
 * measurement over. Deciding whether a stage has a command at all is the
 * caller's: every invocation that reaches here runs.
 *
 * @returns What the hook did: the line for the log, and the block for the agent.
 */
export async function runHook(invocation: HookInvocation): Promise<HookRun> {
  const timeoutMs = invocation.timeoutMs ?? HOOK_TIMEOUT_MS;
  const payload = JSON.stringify(buildPayload(invocation));

  const startedAt = performance.now();
  const result = await exec(invocation.command, {
    cwd: invocation.session.worktrees.experiment,
    timeoutMs,
    stdin: `${payload}\n`,
  });
  const durationMs = performance.now() - startedAt;
  const outcome = describeOutcome(result);

  return {
    record: {
      type: "hook",
      stage: invocation.stage,
      seq: invocation.seq,
      exitCode: outcome.exitCode,
      durationMs,
      // What the hook wrote, not what was relayed: a figure above the limit is
      // how a reader of the log learns the report was cut short.
      stdoutBytes: Buffer.byteLength(outcome.stdout, "utf-8"),
      timedOut: outcome.timedOut,
    },
    report: formatReport(invocation.stage, outcome, timeoutMs),
  };
}

/**
 * The run in the one shape the log and the report read, whichever of `exec`'s
 * two results came back.
 *
 * A timeout carries no exit code of its own — the process was killed before it
 * had one — so the shared failure code stands in for it.
 */
function describeOutcome(result: ExecResult | ExecTimeoutError): CommandOutcome {
  if ("kind" in result) {
    return {
      stdout: result.stdout,
      stderr: result.stderr,
      exitCode: FAILURE_EXIT_CODE,
      timedOut: true,
    };
  }
  return { ...result, timedOut: false };
}

/** The loop as the hook reads it: where the edit lives, which iteration, and whose session. */
function buildPayload(invocation: HookInvocation): HookPayload {
  return {
    stage: invocation.stage,
    experimentDir: invocation.session.worktrees.experiment,
    seq: invocation.seq,
    lastIteration: invocation.lastIteration,
    session: {
      sessionId: invocation.session.sessionId,
      baseline: invocation.session.baseline,
      branch: invocation.session.branch,
      iterationCount: invocation.iterationCount,
    },
  };
}

/**
 * The hook's output as the agent reads it: every line labeled with the stage, and
 * a failing hook's exit code and stderr spelled out under its own output.
 *
 * A successful hook's stderr is left out. Commands write progress there routinely,
 * and repeating it would drown the measurement the hook was annotating. A failing
 * hook's stderr is held to the same cap as its stdout — a build log is as easy to
 * bury a measurement under on one channel as on the other.
 */
function formatReport(stage: HookStage, outcome: CommandOutcome, timeoutMs: number): string {
  const lines = splitLines(limitOutput(outcome.stdout));
  const note = failureNote(outcome, timeoutMs);

  if (note !== undefined) {
    lines.push(note, ...splitLines(limitOutput(outcome.stderr)));
  }

  return lines.map((line) => `[${stage}] ${line}`).join("\n");
}

/** What to tell the reader about a hook that did not succeed, or nothing if it did. */
function failureNote(outcome: CommandOutcome, timeoutMs: number): string | undefined {
  if (outcome.timedOut) {
    return `hook timed out after ${timeoutMs}ms`;
  }
  if (outcome.exitCode !== 0) {
    return `hook exited ${outcome.exitCode}`;
  }
  return undefined;
}

/** `text` as lines, with the trailing newline a command leaves behind dropped. */
function splitLines(text: string): string[] {
  const trimmed = text.replace(/\n$/, "");
  return trimmed === "" ? [] : trimmed.split("\n");
}
