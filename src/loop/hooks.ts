import { spawn, type ChildProcessByStdio } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import type { Readable, Writable } from "node:stream";

import { messageOf } from "../errors.js";
import { killTree } from "../exec.js";
import type { HookRecord, IterationRecord, SessionRecord } from "../session/records.js";

/** Which side of a measurement a hook script runs on. */
export type HookStage = HookRecord["stage"];

/** Which script to run, and everything the payload tells it about the loop so far. */
export interface HookInvocation {
  /** Directory holding `before.sh` and `after.sh`, already resolved against the repo root. */
  hooksDir: string;
  stage: HookStage;
  /** The iteration the hook brackets: the one about to be measured, or the one just recorded. */
  seq: number;
  session: SessionRecord;
  /** The iteration the hook can read, `null` while the session has measured nothing. */
  lastIteration: IterationRecord | null;
  /** How many iterations the log holds as of this invocation. */
  iterationCount: number;
  /** Milliseconds before the script is killed. Defaults to {@link HOOK_TIMEOUT_MS}. */
  timeoutMs?: number;
}

/** What one fired hook leaves behind: a line for the log, and a block for the agent. */
export interface HookRun {
  /** The record to append to the session log. */
  record: HookRecord;
  /**
   * The hook's stdout, truncated and labeled with its stage, followed by a note
   * naming the exit code or the timeout when the script did not succeed.
   *
   * Empty when a successful hook printed nothing.
   */
  report: string;
}

/** How long a hook may run before it is killed. Long enough to build, short enough to notice. */
const HOOK_TIMEOUT_MS = 30_000;

/**
 * How much hook stdout reaches the agent driving the loop.
 *
 * A hook that dumps a build log would otherwise bury the measurement it was
 * meant to annotate, and the whole of it is on disk anyway.
 */
const STDOUT_LIMIT_BYTES = 8192;

/** Reported when the script has no exit status of its own: killed, or never started. */
const FAILURE_EXIT_CODE = 1;

const NEWLINE_BYTE = 0x0a;

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

/** What the script itself did, before any of it is shaped for the log or the report. */
interface ScriptOutcome {
  stdout: string;
  stderr: string;
  exitCode: number;
  timedOut: boolean;
}

/**
 * Run the consumer's `<stage>.sh` hook, handing it a JSON payload on stdin.
 *
 * Hooks steer the loop; they cannot brick it. A script that is missing or that
 * the filesystem does not mark executable is skipped without a word, and one
 * that fails or overruns its timeout comes back as a report and a record rather
 * than as a rejection — there is no hook failure worth throwing away a
 * measurement over.
 *
 * The scripts are POSIX shell by convention and are executed directly, so they
 * carry a shebang and an executable bit.
 *
 * @returns What the hook did, or `undefined` when there was no hook to run.
 */
export async function runHook(invocation: HookInvocation): Promise<HookRun | undefined> {
  const scriptPath = path.join(invocation.hooksDir, `${invocation.stage}.sh`);
  if (!isExecutableFile(scriptPath)) {
    return undefined;
  }

  const timeoutMs = invocation.timeoutMs ?? HOOK_TIMEOUT_MS;
  const payload = JSON.stringify(buildPayload(invocation));

  const startedAt = performance.now();
  const outcome = await runScript(scriptPath, payload, timeoutMs);
  const durationMs = performance.now() - startedAt;

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
 * Whether `scriptPath` is a file this process may execute.
 *
 * Both answers are ordinary: a consumer who wants no hook writes no file, and
 * one who is drafting a hook leaves the executable bit off until it is ready.
 */
function isExecutableFile(scriptPath: string): boolean {
  try {
    return fs.statSync(scriptPath).isFile() && accessible(scriptPath);
  } catch {
    return false;
  }
}

function accessible(scriptPath: string): boolean {
  try {
    fs.accessSync(scriptPath, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
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
 * Execute `scriptPath` with `payload` on its stdin, capturing both output streams.
 *
 * The script runs in its own process group so a hook that spawned a helper is
 * killed along with it on timeout; a timeout snapshots the output received so
 * far rather than waiting for pipes a surviving descendant may still hold.
 *
 * The working directory is inherited: the payload names the experiment worktree,
 * so a hook that wants to work somewhere else says so itself.
 */
async function runScript(
  scriptPath: string,
  payload: string,
  timeoutMs: number,
): Promise<ScriptOutcome> {
  return new Promise((resolve) => {
    let child: ChildProcessByStdio<Writable, Readable, Readable>;
    try {
      child = spawn(scriptPath, {
        // POSIX: detach into its own process group so killTree can SIGKILL the
        // whole group via negative PID.
        detached: process.platform !== "win32",
        stdio: ["pipe", "pipe", "pipe"],
      });
    } catch (err) {
      resolve({
        stdout: "",
        stderr: `${messageOf(err)}\n`,
        exitCode: FAILURE_EXIT_CODE,
        timedOut: false,
      });
      return;
    }

    // setEncoding handles multi-byte characters that straddle pipe reads and
    // flushes any partial sequence when the stream ends.
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    let stdout = "";
    let stderr = "";
    let settled = false;

    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    function settle(outcome: ScriptOutcome): void {
      if (settled) return;
      settled = true;
      clearTimeout(timeoutHandle);

      // "close", "error" and the timeout can all fire for one run; `settled`
      // keeps the first of them the only one.
      // oxlint-disable-next-line promise/no-multiple-resolved
      resolve(outcome);
    }

    // A hook is free to ignore its stdin, which closes the pipe under this write
    // and raises EPIPE. Without a listener that would take the whole process
    // down over a hook doing something entirely reasonable.
    child.stdin.on("error", ignoreStdinError);
    child.stdin.end(`${payload}\n`);

    const timeoutHandle = setTimeout(() => {
      /* v8 ignore next 3 -- pid is always set when stdio is "pipe" */
      if (child.pid !== undefined) {
        killTree(child.pid);
      }
      settle({ stdout, stderr, exitCode: FAILURE_EXIT_CODE, timedOut: true });
    }, timeoutMs);

    // "close", not "exit": it waits for the stdio pipes, which the script's own
    // children can still be writing to after the script itself is gone.
    child.on("close", (code) => {
      // null when the script was killed by a signal.
      settle({ stdout, stderr, exitCode: code ?? FAILURE_EXIT_CODE, timedOut: false });
    });

    function onFailure(err: Error): void {
      settle({
        stdout,
        stderr: `${stderr}${err.message}\n`,
        exitCode: FAILURE_EXIT_CODE,
        timedOut: false,
      });
    }

    // A script that never started has nothing to say on its own stderr, so the
    // spawn failure only reaches the caller if it is written there.
    child.on("error", onFailure);

    child.stdout.on("error", onFailure);
    child.stderr.on("error", onFailure);
  });
}

/** The EPIPE a hook that never reads its stdin leaves behind is the hook's choice, not a fault. */
function ignoreStdinError(): void {
  // Deliberately nothing: the exit code and stderr already say how the hook fared.
}

/**
 * The hook's output as the agent reads it: every line labeled with the stage, and
 * a failing hook's exit code and stderr spelled out under its own output.
 *
 * A successful hook's stderr is left out. Scripts write progress there routinely,
 * and repeating it would drown the measurement the hook was annotating.
 */
function formatReport(stage: HookStage, outcome: ScriptOutcome, timeoutMs: number): string {
  const lines = splitLines(truncateStdout(outcome.stdout));

  if (outcome.timedOut) {
    lines.push(`hook timed out after ${timeoutMs}ms`);
  } else if (outcome.exitCode !== 0) {
    lines.push(`hook exited ${outcome.exitCode}`);
  }
  if (outcome.timedOut || outcome.exitCode !== 0) {
    lines.push(...splitLines(outcome.stderr));
  }

  return lines.map((line) => `[${stage}] ${line}`).join("\n");
}

/** `text` as lines, with the trailing newline a command leaves behind dropped. */
function splitLines(text: string): string[] {
  const trimmed = text.replace(/\n$/, "");
  return trimmed === "" ? [] : trimmed.split("\n");
}

/**
 * At most {@link STDOUT_LIMIT_BYTES} of `text`, cut back to the last whole line
 * and, when a single line already overruns the limit, to the last whole
 * character.
 *
 * The limit is counted in bytes because that is what a consumer sizing their
 * hook's output can measure; cutting mid-character would put a replacement
 * character in the agent's transcript instead.
 */
function truncateStdout(text: string): string {
  const encoded = Buffer.from(text, "utf-8");
  if (encoded.byteLength <= STDOUT_LIMIT_BYTES) {
    return text;
  }

  const head = encoded.subarray(0, STDOUT_LIMIT_BYTES);
  const lastNewline = head.lastIndexOf(NEWLINE_BYTE);
  if (lastNewline >= 0) {
    return head.subarray(0, lastNewline).toString("utf-8");
  }

  // A streaming decoder holds back the bytes of a character the cut split
  // instead of emitting U+FFFD for them.
  return new TextDecoder("utf-8").decode(head, { stream: true });
}
