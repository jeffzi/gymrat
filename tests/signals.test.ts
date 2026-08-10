import os from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { installTerminationCleanup } from "../src/signals.js";
import {
  raiseSignal,
  removeLeakedListeners,
  type SignalName,
  stubProcessExit,
} from "./fixtures/signal-probe.js";

afterEach(() => {
  vi.restoreAllMocks();
});

/** Remove every listener on `signal` added since `before`. */
function removeNewListeners(signal: SignalName, before: readonly unknown[]): void {
  removeLeakedListeners(signal, before);
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
        const code = raiseSignal("SIGINT", before);

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
        const code = raiseSignal("SIGINT", before);

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
          raiseSignal("SIGINT", before);
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
