/* oxlint-disable typescript/require-await -- mock action steps are typed () => Promise<void>; async without await satisfies the interface */
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { Driver, SessionPrompt } from "../../src/supervisor/driver.js";
import type { LaunchEvent, SessionEvent, SessionObserver } from "../../src/supervisor/events.js";
import { createMockDriver } from "../../src/supervisor/mock.js";
import type { MockStep } from "../../src/supervisor/mock.js";
import { supervise } from "../../src/supervisor/supervise.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePrompt(overrides: Partial<SessionPrompt> = {}): SessionPrompt {
  return {
    kickoff: "optimize the decoder",
    cwd: "/tmp/test",
    ...overrides,
  };
}

function makeLaunch(overrides: Partial<LaunchEvent> = {}): LaunchEvent {
  return {
    type: "launch",
    timestamp: 1000,
    headSha: "abc123def",
    dirty: false,
    maxMinutes: 5,
    maxUsd: undefined,
    model: undefined,
    runbookPath: "/path/to/runbook.md",
    kickoffSummary: "test kickoff",
    ...overrides,
  };
}

function makeTempLogPath(): string {
  const dir = mkdtempSync(join(tmpdir(), "supervise-"));
  return join(dir, "events.jsonl");
}

function readLogLines(logPath: string): unknown[] {
  return readFileSync(logPath, "utf-8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as unknown);
}

// ---------------------------------------------------------------------------
// supervise
// ---------------------------------------------------------------------------

describe("supervise", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  describe("when the session completes normally", () => {
    it("resolves with endedBy session, the outcome, duration, and final cost", async () => {
      const steps: MockStep[] = [{ costUsd: 0.05 }, { costUsd: 0.12 }];
      const driver = createMockDriver(steps);
      const logPath = makeTempLogPath();

      const result = await supervise({
        driver,
        prompt: makePrompt(),
        maxMinutes: 10,
        logPath,
        launch: makeLaunch(),
      });

      expect(result.endedBy).toBe("session");
      expect(result.outcome).toStrictEqual({ reason: "completed", costUsd: 0.12 });
      expect(result.costUsd).toBe(0.12);
      expect(result.durationMs).toBeGreaterThanOrEqual(0);
    });

    it("writes the launch event as the first log line followed by every session event", async () => {
      const emitted: SessionEvent = {
        type: "text_delta",
        timestamp: 2000,
        chunk: "hello",
      };
      const steps: MockStep[] = [{ emit: emitted }, { costUsd: 0.01 }];
      const driver = createMockDriver(steps);
      const logPath = makeTempLogPath();
      const launch = makeLaunch();

      await supervise({
        driver,
        prompt: makePrompt(),
        maxMinutes: 10,
        logPath,
        launch,
      });

      const lines = readLogLines(logPath);
      expect(lines.length).toBeGreaterThanOrEqual(3);
      expect(lines[0]).toEqual(launch);
      expect(lines[1]).toMatchObject({ type: "text_delta", chunk: "hello" });
      expect(lines[2]).toMatchObject({ type: "usage_update" });
    });

    it("forwards the launch event and session events to the optional observer", async () => {
      const observed: SessionEvent[] = [];
      const observer: SessionObserver = (e) => {
        observed.push(e);
      };
      const steps: MockStep[] = [{ costUsd: 0.03 }];
      const driver = createMockDriver(steps);
      const logPath = makeTempLogPath();
      const launch = makeLaunch();

      await supervise({
        driver,
        prompt: makePrompt(),
        maxMinutes: 10,
        logPath,
        launch,
        observer,
      });

      expect(observed[0]).toStrictEqual(launch);
      expect(observed.some((e) => e.type === "usage_update")).toBe(true);
    });
  });

  describe("wall-clock cap", () => {
    it("interrupts the session and reports endedBy wall-clock when maxMinutes elapses", async () => {
      vi.useFakeTimers();
      const steps: MockStep[] = [{ costUsd: 0.05, delayMs: 300_000 }];
      const driver = createMockDriver(steps);
      const logPath = makeTempLogPath();

      const resultPromise = supervise({
        driver,
        prompt: makePrompt(),
        maxMinutes: 1,
        logPath,
        launch: makeLaunch({ maxMinutes: 1 }),
      });

      await vi.advanceTimersByTimeAsync(60_000);
      const result = await resultPromise;

      expect(result.endedBy).toBe("wall-clock");
      expect(result.outcome.reason).toBe("interrupted");
    });

    it("reports endedBy session when the session ends before maxMinutes", async () => {
      vi.useFakeTimers();
      const steps: MockStep[] = [{ costUsd: 0.01 }];
      const driver = createMockDriver(steps);
      const logPath = makeTempLogPath();

      const resultPromise = supervise({
        driver,
        prompt: makePrompt(),
        maxMinutes: 1,
        logPath,
        launch: makeLaunch({ maxMinutes: 1 }),
      });

      await vi.advanceTimersByTimeAsync(0);
      const result = await resultPromise;

      expect(result.endedBy).toBe("session");
      expect(result.outcome.reason).toBe("completed");
    });
  });

  describe("grace fallback", () => {
    it("fires the abort signal after the grace period when the session has not ended", async () => {
      vi.useFakeTimers();
      let capturedSignal: AbortSignal | undefined;
      const inner = createMockDriver([
        {
          action: async () => {
            await new Promise<void>((r) => setTimeout(r, 300_000));
          },
        },
      ]);
      const driver: Driver = {
        start(prompt, observer, signal) {
          capturedSignal = signal;
          return inner.start(prompt, observer, signal);
        },
      };
      const logPath = makeTempLogPath();

      const resultPromise = supervise({
        driver,
        prompt: makePrompt(),
        maxMinutes: 1,
        logPath,
        launch: makeLaunch({ maxMinutes: 1 }),
      });

      await vi.advanceTimersByTimeAsync(60_000);
      expect(capturedSignal).toBeDefined();
      expect(capturedSignal!.aborted).toBe(false);

      await vi.advanceTimersByTimeAsync(30_000);
      expect(capturedSignal!.aborted).toBe(true);

      await vi.advanceTimersByTimeAsync(300_000);
      const result = await resultPromise;

      expect(result.endedBy).toBe("wall-clock");
    });
  });

  describe("spend cap", () => {
    it("interrupts the session and reports endedBy spend-cap when costUsd reaches maxUsd", async () => {
      const steps: MockStep[] = [
        { costUsd: 0.05 },
        { costUsd: 0.12 },
        {
          action: async () => {
            throw new Error("should not run");
          },
        },
      ];
      const driver = createMockDriver(steps);
      const logPath = makeTempLogPath();

      const result = await supervise({
        driver,
        prompt: makePrompt(),
        maxMinutes: 10,
        maxUsd: 0.1,
        logPath,
        launch: makeLaunch({ maxUsd: 0.1 }),
      });

      expect(result.endedBy).toBe("spend-cap");
      expect(result.outcome.reason).toBe("interrupted");
    });

    it("does not enforce cost when maxUsd is not set", async () => {
      const steps: MockStep[] = [{ costUsd: 5.0 }, { costUsd: 10.0 }];
      const driver = createMockDriver(steps);
      const logPath = makeTempLogPath();

      const result = await supervise({
        driver,
        prompt: makePrompt(),
        maxMinutes: 10,
        logPath,
        launch: makeLaunch(),
      });

      expect(result.endedBy).toBe("session");
      expect(result.outcome.reason).toBe("completed");
      expect(result.costUsd).toBe(10.0);
    });
  });

  describe("cap racing", () => {
    it("reports the first cap that fires and calls interrupt only once", async () => {
      vi.useFakeTimers();
      let interruptCount = 0;
      const inner = createMockDriver([
        { costUsd: 0.15 },
        {
          action: async () => {},
          delayMs: 120_000,
        },
      ]);
      const driver: Driver = {
        start(prompt, observer, signal) {
          const session = inner.start(prompt, observer, signal);
          return {
            inject: (msg: string) => {
              session.inject(msg);
            },
            interrupt: async () => {
              interruptCount++;
              return session.interrupt();
            },
            usage: () => session.usage(),
            outcome: session.outcome,
          };
        },
      };
      const logPath = makeTempLogPath();

      const resultPromise = supervise({
        driver,
        prompt: makePrompt(),
        maxMinutes: 1,
        maxUsd: 0.1,
        logPath,
        launch: makeLaunch({ maxMinutes: 1, maxUsd: 0.1 }),
      });

      await vi.advanceTimersByTimeAsync(0);
      await vi.advanceTimersByTimeAsync(120_000);
      const result = await resultPromise;

      expect(result.endedBy).toBe("spend-cap");
      expect(interruptCount).toBe(1);
    });
  });

  describe("error outcome", () => {
    it("surfaces the error message in the result and logs all events up to the failure", async () => {
      const emitted: SessionEvent = {
        type: "text_delta",
        timestamp: 2000,
        chunk: "partial output",
      };
      const steps: MockStep[] = [
        { emit: emitted },
        { costUsd: 0.02 },
        {
          action: async () => {
            throw new Error("kaboom");
          },
        },
      ];
      const driver = createMockDriver(steps);
      const logPath = makeTempLogPath();
      const launch = makeLaunch();

      const result = await supervise({
        driver,
        prompt: makePrompt(),
        maxMinutes: 10,
        logPath,
        launch,
      });

      expect(result.outcome.reason).toBe("error");
      expect(result.outcome.message).toBe("kaboom");
      expect(result.endedBy).toBe("session");

      const lines = readLogLines(logPath);
      expect(lines[0]).toEqual(launch);
      const lineTypes = lines.map((l) =>
        typeof l === "object" && l !== null && "type" in l ? l.type : undefined,
      );
      expect(lineTypes).toContain("text_delta");
      expect(lineTypes).toContain("usage_update");
    });
  });
});
