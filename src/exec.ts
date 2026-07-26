import { spawn } from "node:child_process";

export interface ExecResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}

export interface ExecTimeoutError {
  kind: "timeout";
  stdout: string;
  stderr: string;
  timeoutMs: number;
}

export interface ExecOptions {
  cwd: string;
  timeoutMs?: number;
}

/**
 * Executes a shell command and captures its output with optional timeout support.
 *
 * Runs the command in a detached process group to enable killing all child processes
 * on timeout. On POSIX systems, sends SIGKILL to the negative PID to terminate the
 * entire process group.
 *
 * @param command - The shell command to execute
 * @param options - Execution options (working directory and optional timeout in ms)
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
    const child = spawn(command, {
      shell: true,
      cwd: options.cwd,
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let timedOut = false;

    child.stdout.on("data", (data: Buffer) => {
      stdout += data.toString();
    });

    child.stderr.on("data", (data: Buffer) => {
      stderr += data.toString();
    });

    const cleanup = () => {
      /* v8 ignore if -- pid is always set when stdio is "pipe" */
      if (!child.pid) {
        return;
      }
      try {
        process.kill(-child.pid, "SIGKILL");
      } catch {
        // Swallowed deliberately: the process may exit between the pid check and
        // the kill, and a cleanup that throws would mask the real result.
      }
    };

    let timeoutHandle: NodeJS.Timeout | undefined;
    let resolved = false;

    const handleCompletion = (exitCode: number | null) => {
      /* v8 ignore if -- double-fire guard; see comment on child.on("exit") */
      if (resolved) return;
      resolved = true;
      clearTimeout(timeoutHandle);

      const result: ExecResult | ExecTimeoutError = timedOut
        ? {
            kind: "timeout",
            stdout,
            stderr,
            timeoutMs: options.timeoutMs!,
          }
        : {
            stdout,
            stderr,
            /* v8 ignore next -- exitCode is always defined on normal exit */
            exitCode: exitCode ?? 1,
          };

      // Both "exit" and "error" can fire on spawn failure; the `resolved` guard above
      // prevents double-resolution at runtime.
      // oxlint-disable-next-line promise/no-multiple-resolved
      resolve(result);
    };

    if (options.timeoutMs) {
      timeoutHandle = setTimeout(() => {
        timedOut = true;
        cleanup();
      }, options.timeoutMs);
    }

    child.on("exit", (code) => {
      handleCompletion(code);
    });
    child.on("error", () => {
      /* v8 ignore next -- only fires when the shell binary itself can't be spawned */
      handleCompletion(1);
    });
  });
}
