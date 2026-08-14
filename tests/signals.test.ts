import os from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { installTerminationCleanup } from "../src/signals.js";
import {
  raiseSignal,
  removeAllLeakedListeners,
  removeLeakedListeners,
  signalListenerCounts,
  snapshotSignalListeners,
  stubProcessExit,
} from "./fixtures/signal-probe.js";

afterEach(() => {
  vi.restoreAllMocks();
});

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
        removeLeakedListeners("SIGINT", before);
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
        removeLeakedListeners("SIGINT", before);
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
        const code = raiseSignal("SIGINT", before);

        // Assert - the code pins that a handler ran at all: swallowing whatever
        // `raiseSignal` threw would let "no handler installed" pass as "cleanup
        // did not run".
        expect.soft(code).toBe(130);
        expect(cleanup).not.toHaveBeenCalled();
      } finally {
        removeLeakedListeners("SIGINT", before);
      }
    });
  });

  describe("across sequential runs", () => {
    it("runs the second run's cleanup on a signal after the first run was uninstalled", () => {
      // Arrange
      stubProcessExit();
      const firstCleanup = vi.fn();
      const secondCleanup = vi.fn();
      const before = process.listeners("SIGINT").slice();
      installTerminationCleanup(firstCleanup)();
      const uninstallSecond = installTerminationCleanup(secondCleanup);

      try {
        // Act
        const code = raiseSignal("SIGINT", before);

        // Assert
        expect.soft(secondCleanup).toHaveBeenCalledOnce();
        expect.soft(firstCleanup).not.toHaveBeenCalled();
        expect(code).toBe(128 + os.constants.signals.SIGINT);
      } finally {
        uninstallSecond();
        removeLeakedListeners("SIGINT", before);
      }
    });

    it("keeps the listener count flat across repeated install/uninstall cycles", () => {
      // Arrange
      const baseline = snapshotSignalListeners();
      installTerminationCleanup(vi.fn())();
      const afterOneCycle = signalListenerCounts();

      try {
        // Act - well past the 10-listener default that triggers a MaxListeners warning
        for (let cycle = 0; cycle < 12; cycle += 1) {
          installTerminationCleanup(vi.fn())();
        }

        // Assert
        expect(signalListenerCounts()).toStrictEqual(afterOneCycle);
      } finally {
        removeAllLeakedListeners(baseline);
      }
    });
  });
});
