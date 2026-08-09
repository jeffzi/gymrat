import { Command } from "commander";
import { vi } from "vitest";

import { createProgram } from "../../src/cli.js";

/**
 * Turn `process.exit` into a catchable rejection carrying the intended exit code.
 *
 * Prevents `process.exit` from killing the vitest worker: tests can assert on
 * exit-code behavior via `.rejects.toHaveProperty("exitCode", N)` instead.
 */
export function mockProcessExit(): ReturnType<typeof vi.spyOn> {
  // oxlint-disable-next-line typescript/no-unsafe-type-assertion -- vitest mock requires cast
  return vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
    throw Object.assign(new Error(`process.exit(${String(code)})`), { exitCode: code });
  }) as never);
}

/** The exit code carried by the error a mocked `process.exit` threw. */
export function exitCodeOf(error: unknown): number {
  if (error instanceof Error && "exitCode" in error && typeof error.exitCode === "number") {
    return error.exitCode;
  }
  throw error;
}

/** Collect everything the program writes to stdout. */
export function captureStdout(): () => string {
  let stdout = "";
  vi.spyOn(process.stdout, "write").mockImplementation((chunk) => {
    stdout += String(chunk);
    return true;
  });
  return () => stdout;
}

/** Build a fresh program that throws instead of exiting, ready for a single successful parse. */
export function createRunnableProgram(): Command {
  const program = createProgram();
  program.exitOverride();
  return program;
}
