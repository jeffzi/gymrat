import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import type { Readable } from "node:stream";

import { hasErrorCode, messageOf } from "./errors.js";

/** Shape returned when the command runs to completion, whether it exits cleanly or is aborted. */
export interface ExecResult {
  stdout: string;
  stderr: string;
  exitCode: number;
  /** Total bytes received on stdout before any cap truncation. */
  stdoutBytes: number;
  /** Total bytes received on stderr before any cap truncation. */
  stderrBytes: number;
}

/** Shape returned when the command is killed for exceeding `ExecOptions.timeoutMs`. */
export interface ExecTimeoutError {
  kind: "timeout";
  stdout: string;
  stderr: string;
  timeoutMs: number;
  /** Total bytes received on stdout before any cap truncation. */
  stdoutBytes: number;
  /** Total bytes received on stderr before any cap truncation. */
  stderrBytes: number;
}

/** Options controlling where a command runs, what it reads, and how it can be stopped early. */
export interface ExecOptions {
  cwd: string;
  timeoutMs?: number;
  signal?: AbortSignal | undefined;
  /** Text written to the command's stdin, which is then closed. Omitted means no input at all. */
  stdin?: string | undefined;
}

/** Reported when the child has no exit status of its own: killed, or never started. */
export const FAILURE_EXIT_CODE = 1;

/**
 * The result for a run that never produced any output: the signal was already
 * aborted before the command could spawn, or the spawned child came up with no
 * usable stdio streams.
 */
const EMPTY_FAILURE_RESULT: ExecResult = {
  stdout: "",
  stderr: "",
  exitCode: FAILURE_EXIT_CODE,
  stdoutBytes: 0,
  stderrBytes: 0,
};

/**
 * Per-stream accumulation cap. A command that exceeds this resolves normally
 * with truncated capture rather than growing into an OOM or RangeError.
 */
export const OUTPUT_CAP_BYTES = 64 * 1024 * 1024;

/** Mutable buffer accumulating a child process's stdout/stderr as it streams in. */
interface OutputBuffer {
  stdout: string;
  stderr: string;
  stdoutBytes: number;
  stderrBytes: number;
}

/**
 * Accumulate `stdout` and `stderr` into a buffer as data arrives, so callers can
 * snapshot the output received so far at any point — on close, timeout, or abort
 * — without waiting for the streams to end.
 *
 * Each stream is independently capped at `OUTPUT_CAP_BYTES`. Once the cap is
 * reached, subsequent chunks are silently dropped — the stream stays open so the
 * child process is not signalled, but the buffer stops growing.
 */
function attachCapped(
  stream: Readable,
  buffer: OutputBuffer,
  textKey: "stdout" | "stderr",
  bytesKey: "stdoutBytes" | "stderrBytes",
): void {
  stream.on("data", (chunk: string) => {
    const chunkBytes = Buffer.byteLength(chunk, "utf8");
    buffer[bytesKey] += chunkBytes;
    if (buffer[bytesKey] - chunkBytes >= OUTPUT_CAP_BYTES) return;
    buffer[textKey] += chunk;
  });
}

function captureOutput(stdout: Readable, stderr: Readable): OutputBuffer {
  const buffer: OutputBuffer = { stdout: "", stderr: "", stdoutBytes: 0, stderrBytes: 0 };
  attachCapped(stdout, buffer, "stdout", "stdoutBytes");
  attachCapped(stderr, buffer, "stderr", "stderrBytes");
  return buffer;
}

/** Snapshot the four output fields shared by every `ExecResult`/`ExecTimeoutError` variant. */
function snapshotOutput(
  buf: OutputBuffer,
): Pick<ExecResult, "stdout" | "stderr" | "stdoutBytes" | "stderrBytes"> {
  return {
    stdout: buf.stdout,
    stderr: buf.stderr,
    stdoutBytes: buf.stdoutBytes,
    stderrBytes: buf.stderrBytes,
  };
}

/** The status taskkill exits with when no process carries the given pid. */
const TASKKILL_NOT_FOUND_STATUS = 128;

/** Kill the process group led by `pid`, descendants included. Never throws. */
function killTree(pid: number): void {
  try {
    if (process.platform === "win32") {
      // taskkill /T kills the process and all descendants; /F forces it.
      // Without this, cmd.exe dies but sh/sleep survive, holding file locks.
      execFileSync("taskkill", ["/T", "/F", "/PID", String(pid)], { stdio: "ignore" });
    } else {
      process.kill(-pid, "SIGKILL");
    }
  } catch (err) {
    // emitWarning, not throw: callers run from setTimeout/AbortSignal contexts
    // where a throw becomes an uncaught exception.
    //
    // "Process already gone" surfaces differently per platform: POSIX process.kill
    // throws with code "ESRCH", while Windows taskkill exits with
    // TASKKILL_NOT_FOUND_STATUS. Every other taskkill status is a genuine failure
    // — access denied is 5 — and has to reach the caller as a warning.
    const alreadyGone =
      process.platform === "win32"
        ? err instanceof Error && "status" in err && err.status === TASKKILL_NOT_FOUND_STATUS
        : /* istanbul ignore next -- ESRCH and EPERM need a group that dies between the kill and here */
          hasErrorCode(err, "ESRCH");
    if (!alreadyGone) {
      process.emitWarning(err instanceof Error ? err : String(err));
    }
  }
}

/**
 * The EPIPE a command that never reads its stdin leaves behind is its choice, not a fault, and
 * ERR_STREAM_DESTROYED is expected when the child exits before the write completes. Anything else
 * is a genuine failure to deliver `options.stdin` and has to reach the caller as a warning, since
 * `exec` settles from the child's own close/error events and would otherwise resolve as an
 * ordinary result with the payload silently missing.
 */
function ignoreStdinError(err: unknown): void {
  if (hasErrorCode(err, "EPIPE") || hasErrorCode(err, "ERR_STREAM_DESTROYED")) return;
  process.emitWarning(err instanceof Error ? err : String(err));
}

/**
 * Runs the command in a detached process group to enable killing all child processes
 * on timeout. On POSIX systems, sends SIGKILL to the negative PID to terminate the
 * entire process group.
 *
 * Aborting `options.signal` kills that same process group. Unlike a timeout, an abort
 * resolves with an ExecResult holding the output captured so far, so callers can tell a
 * caller-initiated cancellation apart from a timeout. A signal that is already aborted
 * when the call arrives never spawns the shell at all, and resolves with that same
 * cancelled shape.
 *
 * A run that ends on its own is snapshotted when the child's stdio closes rather than when
 * the shell exits, so a line a background job writes after the shell returns is still
 * captured. A descendant that keeps the pipe open therefore holds the call open too:
 * `timeoutMs` is what bounds it. Timeout and abort snapshot the output received so far
 * instead, so neither can be held open by a descendant that survived the group kill.
 *
 * @param command - The shell command to execute
 * @param options - Execution options (working directory, optional timeout in ms, optional
 *                  AbortSignal for cancellation, optional stdin text). The command's stdin is
 *                  closed either way, so a command that reads it sees end-of-input instead of
 *                  blocking, and one that ignores it is not held to account for the broken pipe.
 * @returns A promise resolving to either an ExecResult with the command output and exit code,
 *          or an ExecTimeoutError if the timeout is exceeded
 *
 * @example
 * const result = await exec('echo hello', { cwd: '/tmp' });
 * if ('kind' in result) {
 *   console.log('Timed out after', result.timeoutMs, 'ms');
 * } else {
 *   console.log('Output:', result.stdout);
 * }
 */
export async function exec(
  command: string,
  options: ExecOptions,
): Promise<ExecResult | ExecTimeoutError> {
  return new Promise((resolve) => {
    const spawned = trySpawn(command, options);
    if ("exitCode" in spawned) {
      resolve(spawned);
      return;
    }
    wireChildEvents(spawned, options, resolve);
  });
}

interface SpawnedChild {
  child: ChildProcess;
  stdout: Readable;
  stderr: Readable;
}

function trySpawn(command: string, options: ExecOptions): ExecResult | SpawnedChild {
  // A signal already aborted when the call arrives is a cancellation, not a run.
  if (options.signal?.aborted) {
    return EMPTY_FAILURE_RESULT;
  }

  // Some spawn failures are raised here rather than on the "error" event —
  // ENOTDIR from a cwd that is not a directory is one.
  let child: ChildProcess;
  try {
    child = spawn(command, {
      shell: true,
      cwd: options.cwd,
      // POSIX: detach into its own process group so killTree can SIGKILL
      // the whole group via negative PID. Windows has no process groups,
      // and detached: true there allocates a new console that breaks
      // stdio pipes — stdout/stderr go to the console instead of the pipe.
      detached: process.platform !== "win32",
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (err) {
    const errMsg = `${messageOf(err)}\n`;
    return {
      stdout: "",
      stderr: errMsg,
      exitCode: FAILURE_EXIT_CODE,
      stdoutBytes: 0,
      stderrBytes: Buffer.byteLength(errMsg, "utf8"),
    };
  }

  // fd exhaustion can leave stdio as null even though spawn did not throw.
  if (child.stdout === null || child.stderr === null || child.stdin === null) {
    if (child.pid !== undefined) {
      killTree(child.pid);
    }
    child.on("error", () => {});
    return EMPTY_FAILURE_RESULT;
  }

  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");

  child.stdin.on("error", ignoreStdinError);
  child.stdin.end(options.stdin ?? "");

  return { child, stdout: child.stdout, stderr: child.stderr };
}

function wireChildEvents(
  spawned: SpawnedChild,
  options: ExecOptions,
  resolve: (value: ExecResult | ExecTimeoutError) => void,
): void {
  const { child, stdout: childStdout, stderr: childStderr } = spawned;
  const output = captureOutput(childStdout, childStderr);
  let timeoutHandle: NodeJS.Timeout | undefined;
  let resolved = false;

  function settle(result: ExecResult | ExecTimeoutError): void {
    if (resolved) return;
    resolved = true;
    clearTimeout(timeoutHandle);
    options.signal?.removeEventListener("abort", onAbort);
    childStdout.destroy();
    childStderr.destroy();
    // oxlint-disable-next-line promise/no-multiple-resolved
    resolve(result);
  }

  function killGroup(): void {
    if (!child.pid) return;
    killTree(child.pid);
  }

  function onAbort(): void {
    killGroup();
    settle({ ...snapshotOutput(output), exitCode: FAILURE_EXIT_CODE });
  }

  const { timeoutMs } = options;
  if (timeoutMs) {
    timeoutHandle = setTimeout(() => {
      killGroup();
      settle({ kind: "timeout", ...snapshotOutput(output), timeoutMs });
    }, timeoutMs);
  }

  options.signal?.addEventListener("abort", onAbort, { once: true });

  child.on("close", (code) => {
    settle({ ...snapshotOutput(output), exitCode: code ?? FAILURE_EXIT_CODE });
  });

  function onFailure(err: Error): void {
    killGroup();
    const errSuffix = `${err.message}\n`;
    settle({
      ...snapshotOutput(output),
      stderr: `${output.stderr}${errSuffix}`,
      exitCode: FAILURE_EXIT_CODE,
      stderrBytes: output.stderrBytes + Buffer.byteLength(errSuffix, "utf8"),
    });
  }

  child.on("error", onFailure);
  childStdout.on("error", onFailure);
  childStderr.on("error", onFailure);
}
