/* eslint-disable typescript/no-unsafe-type-assertion -- the process.exit stub needs the never cast */
import { vi } from "vitest";

/** The signals gymrat installs termination cleanup for. */
export const TERMINATION_SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;

/** One of the termination signals gymrat installs cleanup for. */
export type SignalName = (typeof TERMINATION_SIGNALS)[number];

/** Thrown by the stubbed `process.exit` so a handler unwinds where it really would. */
export class ProcessExited extends Error {
  constructor(readonly code: number | string | null | undefined) {
    super(`process.exit(${String(code)})`);
    this.name = "ProcessExited";
  }
}

/**
 * Replace `process.exit` with a throw, so a signal handler stops where the real
 * one would instead of taking the test runner down with it.
 *
 * The spy is undone by `vi.restoreAllMocks()`, which every suite using this
 * runs in its own `afterEach`.
 */
export function stubProcessExit(): void {
  vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
    throw new ProcessExited(code ?? 0);
    // process.exit's overloaded signature accepts string | number | null | undefined
  }) as (code?: string | number | null) => never);
}

/**
 * Snapshot every termination signal's current listener list.
 *
 * Taken at module load, this is the set that is definitively not ours —
 * gymrat attaches one handler per signal for the lifetime of the process and
 * reuses it for every run, so a baseline captured after the first run would
 * already contain it.
 */
export function snapshotSignalListeners(): Record<SignalName, readonly unknown[]> {
  return {
    SIGINT: process.listeners("SIGINT").slice(),
    SIGTERM: process.listeners("SIGTERM").slice(),
    SIGHUP: process.listeners("SIGHUP").slice(),
  };
}

/** Undo every listener on every termination signal added since `baseline`. */
export function removeAllLeakedListeners(baseline: Record<SignalName, readonly unknown[]>): void {
  for (const signal of TERMINATION_SIGNALS) {
    removeLeakedListeners(signal, baseline[signal]);
  }
}

function isSignalListener(value: unknown): value is (signal: SignalName) => void {
  return typeof value === "function";
}

/** How many listeners each termination signal currently carries. */
export function signalListenerCounts(): Record<SignalName, number> {
  return {
    SIGINT: process.listeners("SIGINT").length,
    SIGTERM: process.listeners("SIGTERM").length,
    SIGHUP: process.listeners("SIGHUP").length,
  };
}

/**
 * Remove every listener on `signal` that was not already present `before`.
 *
 * Reports whether it found (and removed) one, so a caller can name every signal
 * that still had a handler installed after a run had settled.
 */
export function removeLeakedListeners(signal: SignalName, before: readonly unknown[]): boolean {
  let leaked = false;
  for (const listener of process.listeners(signal)) {
    if (!before.includes(listener) && isSignalListener(listener)) {
      process.removeListener(signal, listener);
      leaked = true;
    }
  }
  return leaked;
}

/**
 * Run the handlers installed for `signal` since `before`, and report the code
 * they exit with.
 *
 * Emitting the signal for real would also trip vitest's own handling and tear
 * the test run down, so only the newly added listeners are invoked. With
 * {@link stubProcessExit} in place, a handler unwinds exactly where the real one
 * would stop.
 */
export function raiseSignal(
  signal: SignalName,
  before: readonly unknown[],
): number | string | null | undefined {
  const installed = process.listeners(signal).filter((listener) => !before.includes(listener));
  if (installed.length === 0) {
    throw new Error(`no new ${signal} handler was installed`);
  }

  try {
    for (const listener of installed) {
      if (isSignalListener(listener)) {
        listener(signal);
      }
    }
  } catch (error) {
    if (error instanceof ProcessExited) {
      return error.code;
    }
    throw error;
  }
  throw new Error(`the ${signal} handler returned instead of exiting`);
}
