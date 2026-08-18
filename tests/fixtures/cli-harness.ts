import { Command } from "commander";
import { type MockInstance, vi } from "vitest";

import { createProgram } from "../../src/cli.js";

/** The completion callback `stream.write(chunk, callback)` hands back once the chunk lands. */
type WriteCompletion = (error?: Error | null) => void;

function isWriteCompletion(value: unknown): value is WriteCompletion {
  return typeof value === "function";
}

/** The completion callback a `write` call passed, when it passed one. */
function writeCompletionOf(args: readonly unknown[]): WriteCompletion | undefined {
  const last = args.at(-1);
  return isWriteCompletion(last) ? last : undefined;
}

/** A stand-in for one of the process write streams. */
export type WriteSpy = MockInstance<NodeJS.WriteStream["write"]>;

/**
 * Stub `stream.write` with a spy that accepts every chunk and completes it.
 *
 * A real stream reports a chunk as landed by invoking the callback it was
 * handed, which is how a caller knows the bytes are out — the `true` return
 * value only says the chunk fit under the high-water mark. A stub that returns
 * `true` and drops the callback would leave any awaited write pending forever,
 * so completion is signalled here on the next tick, as Node does.
 *
 * `onChunk` observes each chunk as the stream takes it, for a test that has to
 * read some state at the moment of the write rather than after it.
 */
export function stubWrite(stream: NodeJS.WriteStream, onChunk?: (chunk: string) => void): WriteSpy {
  return vi.spyOn(stream, "write").mockImplementation((...args: unknown[]): boolean => {
    onChunk?.(String(args[0]));
    const complete = writeCompletionOf(args);
    if (complete) {
      process.nextTick(complete);
    }
    return true;
  });
}

/** A write stream that has accepted chunks without reporting them as landed yet. */
export interface DeferredWriteStub {
  /** The spy standing in for `stream.write`. */
  spy: WriteSpy;
  /** Reports every chunk accepted so far as landed. */
  flush: () => void;
}

/**
 * Stub `stream.write` so accepted chunks stay in flight until the test releases
 * them, modelling a pipe that took the bytes but has not handed them on.
 *
 * The stub returns `true`, so a caller reading only the return value sees a
 * write it believes is complete — which is the window where an early
 * `process.exit` truncates output.
 */
export function stubDeferredWrite(stream: NodeJS.WriteStream): DeferredWriteStub {
  const inFlight: WriteCompletion[] = [];
  const spy = vi.spyOn(stream, "write").mockImplementation((...args: unknown[]): boolean => {
    const complete = writeCompletionOf(args);
    if (complete) {
      inFlight.push(complete);
    }
    return true;
  });
  return {
    spy,
    flush: () => {
      for (const complete of inFlight.splice(0)) {
        complete();
      }
    },
  };
}

/** The chunks a write spy was handed, as strings. */
export function writtenChunks(spy: WriteSpy): string[] {
  return spy.mock.calls.map((call) => String(call[0]));
}

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
  stubWrite(process.stdout, (chunk) => {
    stdout += chunk;
  });
  if (options.silenceStderr === true) {
    stubWrite(process.stderr);
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
  let overridden: Command[];
  if (scope === "all") overridden = [program, ...program.commands];
  else if (scope === "root") overridden = [program];
  else overridden = [];
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
