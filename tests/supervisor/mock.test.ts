/* oxlint-disable typescript/require-await -- mock action steps are typed () => Promise<void>; async without await is the ergonomic way to satisfy the interface in tests */
import { afterEach, describe, expect, expectTypeOf, it, vi } from "vitest";

import type {
  Driver,
  DriverSession,
  SessionEndReason,
  SessionOutcome,
  SessionPrompt,
} from "../../src/supervisor/driver.js";
import type { SessionEvent, SessionObserver } from "../../src/supervisor/events.js";
import { summarize, SUMMARY_MAX_CHARS } from "../../src/supervisor/events.js";
import { createMockDriver } from "../../src/supervisor/mock.js";
import type { MockStep } from "../../src/supervisor/mock.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePrompt(overrides: Partial<SessionPrompt> = {}): SessionPrompt {
  return {
    kickoff: "do the thing",
    cwd: "/tmp/test",
    ...overrides,
  };
}

function noopObserver(): SessionObserver {
  return () => {};
}

function collectingObserver(): { events: SessionEvent[]; observer: SessionObserver } {
  const events: SessionEvent[] = [];
  const observer: SessionObserver = (e) => {
    events.push(e);
  };
  return { events, observer };
}

// ---------------------------------------------------------------------------
// driver.ts — type-level tests for seam types
// ---------------------------------------------------------------------------

describe("SessionPrompt", () => {
  it("has required kickoff and cwd, optional systemPromptAppend and model", () => {
    expectTypeOf<SessionPrompt>().toEqualTypeOf<{
      readonly kickoff: string;
      readonly systemPromptAppend?: string;
      readonly cwd: string;
      readonly model?: string;
    }>();
  });
});

describe("SessionEndReason", () => {
  it("is a union of completed, interrupted, and error", () => {
    expectTypeOf<SessionEndReason>().toEqualTypeOf<"completed" | "interrupted" | "error">();
  });
});

describe("SessionOutcome", () => {
  it("has reason, costUsd, and optional message", () => {
    expectTypeOf<SessionOutcome>().toEqualTypeOf<{
      readonly reason: SessionEndReason;
      readonly costUsd: number;
      readonly message?: string;
    }>();
  });
});

describe("DriverSession", () => {
  it("has inject, interrupt, usage, and outcome", () => {
    expectTypeOf<DriverSession>()
      .toHaveProperty("inject")
      .toEqualTypeOf<(message: string) => void>();
    expectTypeOf<DriverSession>().toHaveProperty("interrupt").toEqualTypeOf<() => Promise<void>>();
    expectTypeOf<DriverSession>()
      .toHaveProperty("usage")
      .toEqualTypeOf<() => { readonly costUsd: number }>();
    expectTypeOf<DriverSession>()
      .toHaveProperty("outcome")
      .toEqualTypeOf<Promise<SessionOutcome>>();
  });
});

describe("Driver", () => {
  it("has start method with prompt, observer, and optional signal", () => {
    expectTypeOf<Driver>()
      .toHaveProperty("start")
      .toEqualTypeOf<
        (prompt: SessionPrompt, observer: SessionObserver, signal?: AbortSignal) => DriverSession
      >();
  });
});

// ---------------------------------------------------------------------------
// Mock driver
// ---------------------------------------------------------------------------

describe("createMockDriver", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  // script execution

  describe("when the script runs to completion", () => {
    it("emits each emit-step event to the observer in order", async () => {
      const events: SessionEvent[] = [
        { type: "agent_start", timestamp: 1, prompt: "hi", promptSummary: "hi", model: null },
        { type: "text_delta", timestamp: 2, chunk: "hello" },
      ];
      const steps: MockStep[] = [{ emit: events[0]! }, { emit: events[1]! }];
      const { events: collected, observer } = collectingObserver();
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(collected).toStrictEqual(events);
    });

    it("runs each action-step callback in order", async () => {
      const order: number[] = [];
      const steps: MockStep[] = [
        {
          action: async () => {
            order.push(1);
          },
        },
        {
          action: async () => {
            order.push(2);
          },
        },
      ];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());
      await session.outcome;

      expect(order).toStrictEqual([1, 2]);
    });

    it("emits usage_update for cost steps and updates usage()", async () => {
      const { events, observer } = collectingObserver();
      const steps: MockStep[] = [{ costUsd: 0.05 }, { costUsd: 0.1 }];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      const usageEvents = events.filter((e) => e.type === "usage_update");
      expect(usageEvents).toHaveLength(2);
      expect(usageEvents[0]).toMatchObject({ costUsd: 0.05 });
      expect(usageEvents[1]).toMatchObject({ costUsd: 0.1 });
    });

    it("resolves outcome with reason completed and last reported cost", async () => {
      const steps: MockStep[] = [{ costUsd: 0.05 }, { costUsd: 0.1 }];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());
      const outcome = await session.outcome;

      expect(outcome).toStrictEqual({ reason: "completed", costUsd: 0.1 });
    });

    it("resolves outcome with costUsd 0 when no cost steps exist", async () => {
      const steps: MockStep[] = [{ action: async () => {} }];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());
      const outcome = await session.outcome;

      expect(outcome).toStrictEqual({ reason: "completed", costUsd: 0 });
    });
  });

  // delayMs with fake timers

  describe("when steps have delayMs", () => {
    it("delays execution until the timer fires", async () => {
      vi.useFakeTimers();
      const order: string[] = [];
      const steps: MockStep[] = [
        {
          action: async () => {
            order.push("first");
          },
          delayMs: 100,
        },
        {
          action: async () => {
            order.push("second");
          },
          delayMs: 200,
        },
      ];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());

      expect(order).toStrictEqual([]);

      await vi.advanceTimersByTimeAsync(100);
      expect(order).toStrictEqual(["first"]);

      await vi.advanceTimersByTimeAsync(200);
      expect(order).toStrictEqual(["first", "second"]);

      await session.outcome;
    });
  });

  // usage()

  describe("usage()", () => {
    it("returns costUsd 0 before any cost step", () => {
      const steps: MockStep[] = [{ costUsd: 0.05 }];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());

      expect(session.usage()).toStrictEqual({ costUsd: 0 });
    });

    it("returns the latest reported cost after cost steps run", async () => {
      const steps: MockStep[] = [{ costUsd: 0.05 }, { costUsd: 0.15 }];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());
      await session.outcome;

      expect(session.usage()).toStrictEqual({ costUsd: 0.15 });
    });
  });

  // inject()

  describe("inject()", () => {
    it("records the message in the injections list", async () => {
      const driver = createMockDriver([]);

      const session = driver.start(makePrompt(), noopObserver());
      session.inject("hello");
      session.inject("world");
      await session.outcome;

      expect(session.injections).toStrictEqual(["hello", "world"]);
    });

    it("emits an inject event with summarized message", async () => {
      const { events, observer } = collectingObserver();
      const driver = createMockDriver([]);

      const session = driver.start(makePrompt(), observer);
      session.inject("hello world");
      await session.outcome;

      const injectEvents = events.filter((e) => e.type === "inject");
      expect(injectEvents).toHaveLength(1);
      expect(injectEvents[0]).toMatchObject({
        type: "inject",
        message: "hello world",
        messageSummary: summarize("hello world", SUMMARY_MAX_CHARS),
      });
    });
  });

  // interrupt()

  describe("interrupt()", () => {
    it("stops the script and resolves outcome as interrupted", async () => {
      vi.useFakeTimers();
      const order: string[] = [];
      const steps: MockStep[] = [
        {
          action: async () => {
            order.push("first");
          },
          delayMs: 100,
        },
        {
          action: async () => {
            order.push("second");
          },
          delayMs: 100,
        },
      ];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());

      await vi.advanceTimersByTimeAsync(100);
      expect(order).toStrictEqual(["first"]);

      await session.interrupt();
      await vi.advanceTimersByTimeAsync(100);

      expect(order).toStrictEqual(["first"]);

      const outcome = await session.outcome;
      expect(outcome).toStrictEqual({ reason: "interrupted", costUsd: 0 });
    });

    it("reports the last known cost in the interrupted outcome", async () => {
      const steps: MockStep[] = [
        { costUsd: 0.07 },
        {
          action: async () => {
            /* never reached */
          },
        },
      ];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());

      // Let the first step (cost) run, then interrupt before the second
      await vi.waitFor(async () => {
        expect(session.usage().costUsd).toBe(0.07);
      });
      await session.interrupt();

      const outcome = await session.outcome;
      expect(outcome).toStrictEqual({ reason: "interrupted", costUsd: 0.07 });
    });
  });

  // interrupt between steps

  describe("when interrupted between steps", () => {
    it("lets the in-flight action finish but prevents successors", async () => {
      vi.useFakeTimers();
      const log: string[] = [];
      const steps: MockStep[] = [
        {
          action: async () => {
            await new Promise<void>((r) => setTimeout(r, 500));
            log.push("slow-action-done");
          },
        },
        {
          action: async () => {
            log.push("successor");
          },
        },
      ];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());

      // Interrupt while the slow action is in flight
      await session.interrupt();

      // Let the in-flight action's timer resolve
      await vi.advanceTimersByTimeAsync(500);

      const outcome = await session.outcome;
      expect(log).toStrictEqual(["slow-action-done"]);
      expect(outcome.reason).toBe("interrupted");
    });
  });

  // error handling

  describe("when a step throws", () => {
    it("resolves outcome with reason error and the error message", async () => {
      const steps: MockStep[] = [
        {
          action: async () => {
            throw new Error("kaboom");
          },
        },
        {
          action: async () => {
            /* never reached */
          },
        },
      ];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());
      const outcome = await session.outcome;

      expect(outcome).toStrictEqual({ reason: "error", costUsd: 0, message: "kaboom" });
    });

    it("does not run subsequent steps after a throw", async () => {
      const reached: string[] = [];
      const steps: MockStep[] = [
        {
          action: async () => {
            throw new Error("fail");
          },
        },
        {
          action: async () => {
            reached.push("after-error");
          },
        },
      ];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());
      await session.outcome;

      expect(reached).toStrictEqual([]);
    });

    it("includes the last known cost in the error outcome", async () => {
      const steps: MockStep[] = [
        { costUsd: 0.03 },
        {
          action: async () => {
            throw new Error("oops");
          },
        },
      ];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver());
      const outcome = await session.outcome;

      expect(outcome).toStrictEqual({ reason: "error", costUsd: 0.03, message: "oops" });
    });
  });

  // abort signal

  describe("when the abort signal fires", () => {
    it("stops the script and resolves outcome as interrupted", async () => {
      vi.useFakeTimers();
      const controller = new AbortController();
      const log: string[] = [];
      const steps: MockStep[] = [
        {
          action: async () => {
            log.push("first");
          },
          delayMs: 100,
        },
        {
          action: async () => {
            log.push("second");
          },
          delayMs: 100,
        },
      ];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver(), controller.signal);

      await vi.advanceTimersByTimeAsync(100);
      expect(log).toStrictEqual(["first"]);

      controller.abort();
      await vi.advanceTimersByTimeAsync(100);

      const outcome = await session.outcome;
      expect(log).toStrictEqual(["first"]);
      expect(outcome).toStrictEqual({ reason: "interrupted", costUsd: 0 });
    });

    it("reports the last known cost when aborted", async () => {
      const controller = new AbortController();
      const steps: MockStep[] = [
        { costUsd: 0.12 },
        {
          action: async () => {
            /* never reached */
          },
          delayMs: 100,
        },
      ];
      const driver = createMockDriver(steps);

      const session = driver.start(makePrompt(), noopObserver(), controller.signal);

      await vi.waitFor(async () => {
        expect(session.usage().costUsd).toBe(0.12);
      });
      controller.abort();

      const outcome = await session.outcome;
      expect(outcome).toStrictEqual({ reason: "interrupted", costUsd: 0.12 });
    });
  });
});
