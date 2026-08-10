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

/**
 * Collect everything the program writes to stdout.
 *
 * `silenceStderr` keeps a command's warnings out of the test runner's own
 * output; `drainOnRead` empties the buffer on every read, so a caller running a
 * sequence of commands reads each one's output on its own.
 */
export function captureStdout(
  options: { silenceStderr?: boolean; drainOnRead?: boolean } = {},
): () => string {
  let stdout = "";
  vi.spyOn(process.stdout, "write").mockImplementation((chunk) => {
    stdout += String(chunk);
    return true;
  });
  if (options.silenceStderr === true) {
    vi.spyOn(process.stderr, "write").mockReturnValue(true);
  }
  return () => {
    const collected = stdout;
    if (options.drainOnRead === true) {
      stdout = "";
    }
    return collected;
  };
}

/** Which commands get the test `exitOverride` instead of the one production installed. */
export type ExitOverrideScope = "root" | "all" | "none";

/**
 * Build a fresh program that throws instead of exiting.
 *
 * `exitOverride` decides how far the test override reaches. `"root"` — the
 * default — leaves gymrat's own override on the subcommands, so a subcommand's
 * usage error still surfaces as exit code 2; `"all"` replaces it everywhere, so
 * Commander's own exit code survives; `"none"` keeps production's throughout.
 *
 * `silent` swallows Commander's stderr, which is what a test asserting on a
 * usage error wants — the error is what it reads, not the help text beside it.
 */
export function createRunnableProgram(
  options: { exitOverride?: ExitOverrideScope; silent?: boolean } = {},
): Command {
  const program = createProgram();
  const scope = options.exitOverride ?? "root";
  const overridden =
    scope === "all" ? [program, ...program.commands] : scope === "root" ? [program] : [];
  for (const command of overridden) {
    command.exitOverride();
  }
  if (options.silent === true) {
    for (const command of [program, ...program.commands]) {
      command.configureOutput({ writeErr: () => {} });
    }
  }
  return program;
}
