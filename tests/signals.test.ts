/* eslint-disable typescript/no-unsafe-type-assertion -- process.exit mock requires never cast */
import os from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { installTerminationCleanup } from "../src/signals.js";

class ProcessExited extends Error {
  constructor(readonly code: number) {
    super(`process.exit(${String(code)})`);
    this.name = "ProcessExited";
  }
}

afterEach(() => {
  vi.restoreAllMocks();
});

function stubProcessExit(): void {
  vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
    throw new ProcessExited(code ?? 0);
    // process.exit's overloaded signature accepts string | number | null | undefined
  }) as (code?: string | number | null) => never);
}

function isSignalListener(value: unknown): value is (signal: NodeJS.Signals) => void {
  return typeof value === "function";
}

function raiseNewListeners(signal: NodeJS.Signals, before: readonly unknown[]): number {
  const installed = process.listeners(signal).filter((l) => !before.includes(l));
  if (installed.length === 0) {
    throw new Error(`no new ${signal} handler installed`);
  }
  for (const listener of installed) {
    try {
      if (isSignalListener(listener)) listener(signal);
    } catch (error) {
      if (error instanceof ProcessExited) return error.code;
      throw error;
    }
  }
  throw new Error(`${signal} handler returned instead of exiting`);
}

function removeNewListeners(signal: NodeJS.Signals, before: readonly unknown[]): void {
  for (const listener of process.listeners(signal)) {
    if (!before.includes(listener)) {
      process.removeListener(signal, listener);
    }
  }
}

describe("installTerminationCleanup", () => {
  describe("before uninstall", () => {
    it("runs cleanup then exits 128+signum on SIGINT", () => {
      // Arrange
      stubProcessExit();
      const cleanup = vi.fn();
      const before = process.listeners("SIGINT").slice();
      const uninstall = installTerminationCleanup(cleanup);

      try {
        // Act
        const code = raiseNewListeners("SIGINT", before);

        // Assert
        expect(cleanup).toHaveBeenCalledOnce();
        expect(code).toBe(128 + os.constants.signals.SIGINT);
      } finally {
        uninstall();
        removeNewListeners("SIGINT", before);
      }
    });
  });

  describe("after uninstall", () => {
    it("a signal still exits with 128+signum (exit-only handler remains)", () => {
      // Arrange
      stubProcessExit();
      const cleanup = vi.fn();
      const before = process.listeners("SIGINT").slice();
      const uninstall = installTerminationCleanup(cleanup);
      uninstall();

      try {
        // Act
        const code = raiseNewListeners("SIGINT", before);

        // Assert
        expect(code).toBe(128 + os.constants.signals.SIGINT);
      } finally {
        removeNewListeners("SIGINT", before);
      }
    });

    it("does not run cleanup on a post-uninstall signal", () => {
      // Arrange
      stubProcessExit();
      const cleanup = vi.fn();
      const before = process.listeners("SIGINT").slice();
      const uninstall = installTerminationCleanup(cleanup);
      uninstall();

      try {
        // Act
        try {
          raiseNewListeners("SIGINT", before);
        } catch {
          // exit throws — ignore
        }

        // Assert
        expect(cleanup).not.toHaveBeenCalled();
      } finally {
        removeNewListeners("SIGINT", before);
      }
    });
  });
});
