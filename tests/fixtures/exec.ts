import type { ExecResult, ExecTimeoutError } from "../../src/exec.js";

/** Build an exec result with computed byte lengths, overridable per field. */
export function createExecResult(overrides: Partial<ExecResult> = {}): ExecResult {
  const base = {
    exitCode: 0,
    stdout: "",
    stderr: "",
    ...overrides,
  };
  return {
    ...base,
    stdoutBytes: base.stdoutBytes ?? Buffer.byteLength(base.stdout, "utf-8"),
    stderrBytes: base.stderrBytes ?? Buffer.byteLength(base.stderr, "utf-8"),
  };
}

/** Build a timeout error with computed byte lengths, overridable per field. */
export function createExecTimeout(overrides: Partial<ExecTimeoutError> = {}): ExecTimeoutError {
  const base = {
    kind: "timeout" as const,
    timeoutMs: 30_000,
    stdout: "",
    stderr: "",
    ...overrides,
  };
  return {
    ...base,
    stdoutBytes: base.stdoutBytes ?? Buffer.byteLength(base.stdout, "utf-8"),
    stderrBytes: base.stderrBytes ?? Buffer.byteLength(base.stderr, "utf-8"),
  };
}
