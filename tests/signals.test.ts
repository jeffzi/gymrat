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

/** The exit code the handler under test uses for SIGINT: 128 + signum. */
const SIGINT_EXIT_CODE = 128 + os.constants.signals.SIGINT;

/** Stub the exit path, install a cleanup, and capture the listeners already present. */
function installedCleanup(): {
  cleanup: ReturnType<typeof vi.fn>;
  before: readonly unknown[];
  uninstall: () => void;
} {
  stubProcessExit();
  const cleanup = vi.fn();
  const before = process.listeners("SIGINT").slice();
  const uninstall = installTerminationCleanup(cleanup);
  return { cleanup, before, uninstall };
}

describe("installTerminationCleanup", () => {
  describe("before uninstall", () => {
    it("runs cleanup then exits 128+signum on SIGINT", () => {
      const { cleanup, before, uninstall } = installedCleanup();

      try {
        const code = raiseSignal("SIGINT", before);

        expect(cleanup).toHaveBeenCalledOnce();
        expect(code).toBe(SIGINT_EXIT_CODE);
      } finally {
        uninstall();
        removeLeakedListeners("SIGINT", before);
      }
    });
  });

  describe("after uninstall", () => {
    it("a signal still exits with 128+signum (exit-only handler remains)", () => {
      const { before, uninstall } = installedCleanup();
      uninstall();

      try {
        const code = raiseSignal("SIGINT", before);

        expect(code).toBe(SIGINT_EXIT_CODE);
      } finally {
        removeLeakedListeners("SIGINT", before);
      }
    });

    it("does not run cleanup on a post-uninstall signal", () => {
      const { cleanup, before, uninstall } = installedCleanup();
      uninstall();

      try {
        const code = raiseSignal("SIGINT", before);

        // The code pins that a handler ran at all: swallowing whatever
        // `raiseSignal` threw would let "no handler installed" pass as "cleanup
        // did not run".
        expect.soft(code).toBe(SIGINT_EXIT_CODE);
        expect(cleanup).not.toHaveBeenCalled();
      } finally {
        removeLeakedListeners("SIGINT", before);
      }
    });
  });

  describe("when a cleanup throws", () => {
    it("still runs remaining cleanups and exits with 128+signum", () => {
      stubProcessExit();
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      const before = process.listeners("SIGINT").slice();
      const throwingCleanup = vi.fn(() => {
        throw new Error("cleanup boom");
      });
      const survivingCleanup = vi.fn();
      const uninstall1 = installTerminationCleanup(throwingCleanup);
      const uninstall2 = installTerminationCleanup(survivingCleanup);

      try {
        const code = raiseSignal("SIGINT", before);

        expect.soft(throwingCleanup).toHaveBeenCalledOnce();
        expect.soft(survivingCleanup).toHaveBeenCalledOnce();
        expect.soft(warnSpy).toHaveBeenCalled();
        expect(code).toBe(SIGINT_EXIT_CODE);
      } finally {
        uninstall1();
        uninstall2();
        removeLeakedListeners("SIGINT", before);
      }
    });
  });

  describe("across sequential runs", () => {
    it("runs the second run's cleanup on a signal after the first run was uninstalled", () => {
      stubProcessExit();
      const firstCleanup = vi.fn();
      const secondCleanup = vi.fn();
      const before = process.listeners("SIGINT").slice();
      installTerminationCleanup(firstCleanup)();
      const uninstallSecond = installTerminationCleanup(secondCleanup);

      try {
        const code = raiseSignal("SIGINT", before);

        expect.soft(secondCleanup).toHaveBeenCalledOnce();
        expect.soft(firstCleanup).not.toHaveBeenCalled();
        expect(code).toBe(SIGINT_EXIT_CODE);
      } finally {
        uninstallSecond();
        removeLeakedListeners("SIGINT", before);
      }
    });

    it("keeps the listener count flat across repeated install/uninstall cycles", () => {
      const baseline = snapshotSignalListeners();
      installTerminationCleanup(vi.fn())();
      const afterOneCycle = signalListenerCounts();

      try {
        // Well past the 10-listener default that triggers a MaxListeners warning
        for (let cycle = 0; cycle < 12; cycle += 1) {
          installTerminationCleanup(vi.fn())();
        }

        expect(signalListenerCounts()).toStrictEqual(afterOneCycle);
      } finally {
        removeAllLeakedListeners(baseline);
      }
    });
  });
});
