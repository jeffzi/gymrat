import { execFileSync, spawn, type ChildProcessByStdio } from "node:child_process";
import type { Readable } from "node:stream";

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

/** Options controlling where a command runs and how it can be stopped early. */
export interface ExecOptions {
  cwd: string;
  timeoutMs?: number;
  signal?: AbortSignal;
}

/** Reported when the child has no exit status of its own: killed, or never started. */
const FAILURE_EXIT_CODE = 1;

/** The status taskkill exits with when no process carries the given pid. */
const TASKKILL_NOT_FOUND_STATUS = 128;

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
 * Executes a shell command and captures its output with optional timeout support.
 *
 * Runs the command in a detached process group to enable killing all child processes
 * on timeout. On POSIX systems, sends SIGKILL to the negative PID to terminate the
 * entire process group.
 *
 * Aborting `options.signal` kills that same process group. Unlike a timeout, an abort
 * resolves with an ExecResult holding the output captured so far, so callers can tell a
 * caller-initiated cancellation apart from a timeout.
 *
 * A run that ends on its own is snapshotted when the child's stdio closes rather than when
 * the shell exits, so a line a background job writes after the shell returns is still
 * captured. A descendant that keeps the pipe open therefore holds the call open too:
 * `timeoutMs` is what bounds it. Timeout and abort snapshot the output received so far
 * instead, so neither can be held open by a descendant that survived the group kill.
 *
 * @param command - The shell command to execute
 * @param options - Execution options (working directory, optional timeout in ms, optional
 *                  AbortSignal for cancellation)
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
    // Some spawn failures are raised here rather than on the "error" event —
    // ENOTDIR from a cwd that is not a directory is one. Both paths report the
    // same way, so a caller never has to handle a rejection as well as a result.
    let child: ChildProcessByStdio<null, Readable, Readable>;
    try {
      child = spawn(command, {
        shell: true,
        cwd: options.cwd,
        // POSIX: detach into its own process group so killTree can SIGKILL
        // the whole group via negative PID. Windows has no process groups,
        // and detached: true there allocates a new console that breaks
        // stdio pipes — stdout/stderr go to the console instead of the pipe.
        detached: process.platform !== "win32",
        stdio: ["ignore", "pipe", "pipe"],
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
    let stdout = "";
    let stderr = "";
    let timeoutHandle: NodeJS.Timeout | undefined;
    let resolved = false;

    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });

    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    function settle(result: ExecResult | ExecTimeoutError): void {
      if (resolved) return;
      resolved = true;
      clearTimeout(timeoutHandle);
      options.signal?.removeEventListener("abort", onAbort);

      // "close", "error", the timeout and the abort can all fire for one run;
      // `resolved` keeps the first of them the only one.
      // oxlint-disable-next-line promise/no-multiple-resolved
      resolve(result);
    }

    function killGroup(): void {
      /* v8 ignore if -- pid is always set when stdio is "pipe" */
      if (!child.pid) {
        return;
      }
      killTree(child.pid);
    }

    function onAbort(): void {
      killGroup();
      // Snapshot rather than wait for "close": the caller asked to stop now, and a
      // descendant that outlived the group kill would still be holding the pipe.
      settle({ stdout, stderr, exitCode: FAILURE_EXIT_CODE });
    }

    const { timeoutMs } = options;
    if (timeoutMs) {
      timeoutHandle = setTimeout(() => {
        killGroup();
        settle({ kind: "timeout", stdout, stderr, timeoutMs });
      }, timeoutMs);
    }

    // A signal aborted before this call never dispatches to a listener added after
    // the fact, so an already-aborted signal must clean up directly instead of
    // relying on the "abort" event.
    if (options.signal?.aborted) {
      onAbort();
    } else {
      options.signal?.addEventListener("abort", onAbort, { once: true });
    }

    // "close", not "exit": it waits for the stdio pipes, which the shell's
    // descendants can still be writing to after the shell itself is gone.
    child.on("close", (code) => {
      settle({
        stdout,
        stderr,
        // null when the child was killed by a signal, which is how an abort ends.
        exitCode: code ?? FAILURE_EXIT_CODE,
      });
    });

    // A child that never started has nothing to say on its own stderr, so the
    // spawn failure only reaches the caller if it is written there. Node emits
    // this before any "close", so the cause survives the single-resolution guard.
    child.on("error", (err: Error) => {
      settle({ stdout, stderr: `${stderr}${err.message}\n`, exitCode: FAILURE_EXIT_CODE });
    });

    // A pipe read can fail on its own (EIO on a closing pty, for one). Without a
    // listener that is an unhandled "error" event, which takes the whole process
    // down instead of failing the one call that owns the pipe.
    function onStreamError(err: Error): void {
      settle({ stdout, stderr: `${stderr}${err.message}\n`, exitCode: FAILURE_EXIT_CODE });
    }

    child.stdout.on("error", onStreamError);
    child.stderr.on("error", onStreamError);
  });
}
