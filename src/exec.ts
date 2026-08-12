import { execFileSync, spawn, type ChildProcessByStdio } from "node:child_process";
import type { Readable, Writable } from "node:stream";

import { hasErrorCode, messageOf } from "./errors.js";

/** Shape returned when the command runs to completion, whether it exits cleanly or is aborted. */
export interface ExecResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

/** Shape returned when the command is killed for exceeding `ExecOptions.timeoutMs`. */
export interface ExecTimeoutError {
  kind: "timeout";
  stdout: string;
  stderr: string;
  timeoutMs: number;
}

/** Options controlling where a command runs, what it reads, and how it can be stopped early. */
export interface ExecOptions {
  cwd: string;
  timeoutMs?: number;
  signal?: AbortSignal;
  /** Text written to the command's stdin, which is then closed. Omitted means no input at all. */
  stdin?: string;
}

/** Reported when the child has no exit status of its own: killed, or never started. */
export const FAILURE_EXIT_CODE = 1;

/** Mutable buffer accumulating a child process's stdout/stderr as it streams in. */
interface OutputBuffer {
  stdout: string;
  stderr: string;
}

/**
 * Accumulate `stdout` and `stderr` into a buffer as data arrives, so callers can
 * snapshot the output received so far at any point — on close, timeout, or abort
 * — without waiting for the streams to end.
 */
function captureOutput(stdout: Readable, stderr: Readable): OutputBuffer {
  const buffer: OutputBuffer = { stdout: "", stderr: "" };
  stdout.on("data", (chunk: string) => {
    buffer.stdout += chunk;
  });
  stderr.on("data", (chunk: string) => {
    buffer.stderr += chunk;
  });
  return buffer;
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
        : /* v8 ignore next -- ESRCH and EPERM need a group that dies between the kill and here */
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
    // A signal already aborted when the call arrives is a cancellation, not a run:
    // spawning would execute the command text and only then kill it. Settle with the
    // shape a mid-run abort produces, so callers handle cancellation one way.
    if (options.signal?.aborted) {
      resolve({ stdout: "", stderr: "", exitCode: FAILURE_EXIT_CODE });
      return;
    }

    // Some spawn failures are raised here rather than on the "error" event —
    // ENOTDIR from a cwd that is not a directory is one. Both paths report the
    // same way, so a caller never has to handle a rejection as well as a result.
    let child: ChildProcessByStdio<Writable, Readable, Readable>;
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
      resolve({
        stdout: "",
        stderr: `${messageOf(err)}\n`,
        exitCode: FAILURE_EXIT_CODE,
      });
      return;
    }

    // setEncoding handles multi-byte characters that straddle pipe reads and
    // flushes any partial sequence when the stream ends.
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");

    // A command is free to ignore its stdin, which closes the pipe under this
    // write and raises EPIPE. Without a listener that would take the whole
    // process down over a command doing something entirely reasonable.
    child.stdin.on("error", ignoreStdinError);
    // Always end: a command that reads stdin must see end-of-input rather than
    // block on a pipe no one will ever write to.
    child.stdin.end(options.stdin ?? "");

    const output = captureOutput(child.stdout, child.stderr);
    let timeoutHandle: NodeJS.Timeout | undefined;
    let resolved = false;

    function settle(result: ExecResult | ExecTimeoutError): void {
      if (resolved) return;
      resolved = true;
      clearTimeout(timeoutHandle);
      options.signal?.removeEventListener("abort", onAbort);
      // The snapshot in `result` is final, but a descendant that survived the group
      // kill still holds the write end of these pipes and would keep appending to
      // `output`. Destroying the read end releases the pipe with the call.
      child.stdout.destroy();
      child.stderr.destroy();

      // "close", "error", the timeout and the abort can all fire for one run;
      // `resolved` keeps the first of them the only one.
      // oxlint-disable-next-line promise/no-multiple-resolved
      resolve(result);
    }

    function killGroup(): void {
      // A spawn that failed outright never made a child, so there is no group to
      // kill and nothing to report.
      if (!child.pid) {
        return;
      }
      killTree(child.pid);
    }

    function onAbort(): void {
      killGroup();
      // Snapshot rather than wait for "close": the caller asked to stop now, and a
      // descendant that outlived the group kill would still be holding the pipe.
      settle({ stdout: output.stdout, stderr: output.stderr, exitCode: FAILURE_EXIT_CODE });
    }

    const { timeoutMs } = options;
    if (timeoutMs) {
      timeoutHandle = setTimeout(() => {
        killGroup();
        settle({ kind: "timeout", stdout: output.stdout, stderr: output.stderr, timeoutMs });
      }, timeoutMs);
    }

    // Aborting after this point is the only case left to listen for: a signal already
    // aborted settled the call before the spawn, and nothing between there and here
    // yields to the event loop.
    options.signal?.addEventListener("abort", onAbort, { once: true });

    // "close", not "exit": it waits for the stdio pipes, which the shell's
    // descendants can still be writing to after the shell itself is gone.
    child.on("close", (code) => {
      settle({
        stdout: output.stdout,
        stderr: output.stderr,
        // null when the child was killed by a signal, which is how an abort ends.
        exitCode: code ?? FAILURE_EXIT_CODE,
      });
    });

    function onFailure(err: Error): void {
      // Settling clears the timeout and drops the abort listener, so this is the
      // last point that can reach the group. A stdio failure leaves the child
      // itself running: without this it would outlive the call, still holding its
      // cwd.
      killGroup();
      settle({
        stdout: output.stdout,
        stderr: `${output.stderr}${err.message}\n`,
        exitCode: FAILURE_EXIT_CODE,
      });
    }

    // A child that never started has nothing to say on its own stderr, so the
    // spawn failure only reaches the caller if it is written there. Node emits
    // this before any "close", so the cause survives the single-resolution guard.
    child.on("error", onFailure);

    // A pipe read can fail on its own (EIO on a closing pty, for one). Without a
    // listener that is an unhandled "error" event, which takes the whole process
    // down instead of failing the one call that owns the pipe.
    child.stdout.on("error", onFailure);
    child.stderr.on("error", onFailure);
  });
}
