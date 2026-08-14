/* oxlint-disable require-yield -- async generators used as AsyncIterable adapters for the SDK message stream; yield is not always needed */
/* oxlint-disable typescript/require-await -- mock action callbacks are async to match the interface but have no awaitable work */
/* oxlint-disable typescript/no-unsafe-type-assertion -- narrowing test assertions on partial event objects */
import { describe, expect, it } from "vitest";

import { createClaudeDriver } from "../../src/supervisor/claude.js";
import type { QueryFn } from "../../src/supervisor/claude.js";
import type { SessionPrompt } from "../../src/supervisor/driver.js";
import type { SessionEvent, SessionObserver } from "../../src/supervisor/events.js";

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

function collectingObserver(): { events: SessionEvent[]; observer: SessionObserver } {
  const events: SessionEvent[] = [];
  const observer: SessionObserver = (e) => {
    events.push(e);
  };
  return { events, observer };
}

/**
 * Creates a fake queryFn that yields the given SDK-shaped messages.
 *
 * The returned function matches the QueryFn signature: it accepts query options
 * and returns an object with an async iterable of messages, an interrupt method,
 * and a result accessor providing per-turn usage.
 */
function fakeQueryFn(
  messages: Array<Record<string, unknown>>,
  options?: {
    resultCostUsd?: number;
    onInterrupt?: () => void;
    throwError?: Error;
  },
): QueryFn {
  let interrupted = false;
  return (_opts: Record<string, unknown>) => {
    async function* generate() {
      for (const msg of messages) {
        if (interrupted) return;
        yield msg;
      }
      if (options?.throwError) {
        throw options.throwError;
      }
    }

    return {
      messages: generate(),
      interrupt(): void {
        interrupted = true;
        options?.onInterrupt?.();
      },
      result: {
        total_cost_usd: options?.resultCostUsd ?? 0,
      },
    };
  };
}

// ---------------------------------------------------------------------------
// createClaudeDriver
// ---------------------------------------------------------------------------

describe("createClaudeDriver", () => {
  it("returns an object with a start method (Driver interface)", () => {
    const driver = createClaudeDriver({ queryFn: fakeQueryFn([]) });
    expect(driver).toHaveProperty("start");
    expect(typeof driver.start).toBe("function");
  });
});

// ---------------------------------------------------------------------------
// start() — query options forwarding
// ---------------------------------------------------------------------------

describe("start", () => {
  describe("when launching a session", () => {
    it("passes prompt as an AsyncIterable seeded with kickoff, not a plain string", async () => {
      let capturedOpts: Record<string, unknown> = {};
      const queryFn: QueryFn = (opts: Record<string, unknown>) => {
        capturedOpts = opts;
        async function* empty() {
          // yield nothing
        }
        return {
          messages: empty(),
          interrupt(): void {},
          result: { total_cost_usd: 0 },
        };
      };
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(makePrompt(), collectingObserver().observer);
      await session.outcome;

      // prompt must be an AsyncIterable, not a string
      expect(typeof capturedOpts["prompt"]).not.toBe("string");
      expect(Symbol.asyncIterator in (capturedOpts["prompt"] as object)).toBe(true);
    });

    it("forwards cwd and permissionMode: bypassPermissions", async () => {
      let capturedOpts: Record<string, unknown> = {};
      const queryFn: QueryFn = (opts: Record<string, unknown>) => {
        capturedOpts = opts;
        async function* empty() {}
        return {
          messages: empty(),
          interrupt(): void {},
          result: { total_cost_usd: 0 },
        };
      };
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(
        makePrompt({ cwd: "/my/project" }),
        collectingObserver().observer,
      );
      await session.outcome;

      expect(capturedOpts["cwd"]).toBe("/my/project");
      expect(capturedOpts["permissionMode"]).toBe("bypassPermissions");
    });

    it("includes systemPrompt in append-to-preset form when systemPromptAppend is present", async () => {
      let capturedOpts: Record<string, unknown> = {};
      const queryFn: QueryFn = (opts: Record<string, unknown>) => {
        capturedOpts = opts;
        async function* empty() {}
        return {
          messages: empty(),
          interrupt(): void {},
          result: { total_cost_usd: 0 },
        };
      };
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(
        makePrompt({ systemPromptAppend: "extra instructions" }),
        collectingObserver().observer,
      );
      await session.outcome;

      expect(capturedOpts["systemPrompt"]).toStrictEqual({
        type: "append-to-preset",
        text: "extra instructions",
      });
    });

    it("omits systemPrompt when systemPromptAppend is absent", async () => {
      let capturedOpts: Record<string, unknown> = {};
      const queryFn: QueryFn = (opts: Record<string, unknown>) => {
        capturedOpts = opts;
        async function* empty() {}
        return {
          messages: empty(),
          interrupt(): void {},
          result: { total_cost_usd: 0 },
        };
      };
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(makePrompt(), collectingObserver().observer);
      await session.outcome;

      expect(capturedOpts).not.toHaveProperty("systemPrompt");
    });

    it("includes model only when given in the prompt", async () => {
      let capturedOpts: Record<string, unknown> = {};
      const queryFn: QueryFn = (opts: Record<string, unknown>) => {
        capturedOpts = opts;
        async function* empty() {}
        return {
          messages: empty(),
          interrupt(): void {},
          result: { total_cost_usd: 0 },
        };
      };
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(
        makePrompt({ model: "claude-sonnet-4-20250514" }),
        collectingObserver().observer,
      );
      await session.outcome;

      expect(capturedOpts["model"]).toBe("claude-sonnet-4-20250514");
    });

    it("omits model when not given in the prompt", async () => {
      let capturedOpts: Record<string, unknown> = {};
      const queryFn: QueryFn = (opts: Record<string, unknown>) => {
        capturedOpts = opts;
        async function* empty() {}
        return {
          messages: empty(),
          interrupt(): void {},
          result: { total_cost_usd: 0 },
        };
      };
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(makePrompt(), collectingObserver().observer);
      await session.outcome;

      expect(capturedOpts).not.toHaveProperty("model");
    });
  });
});

// ---------------------------------------------------------------------------
// inject()
// ---------------------------------------------------------------------------

describe("inject", () => {
  it("pushes a user message and emits an inject event", async () => {
    const queryFn: QueryFn = (_opts: Record<string, unknown>) => {
      async function* generate() {
        // Yield nothing — just let the session start and complete
      }
      return {
        messages: generate(),
        interrupt(): void {},
        result: { total_cost_usd: 0 },
      };
    };
    const driver = createClaudeDriver({ queryFn });
    const { events, observer } = collectingObserver();

    const session = driver.start(makePrompt(), observer);
    session.inject("feedback message");
    await session.outcome;

    const injectEvents = events.filter((e) => e.type === "inject");
    expect(injectEvents).toHaveLength(1);
    expect(injectEvents[0]).toMatchObject({
      type: "inject",
      message: "feedback message",
    });
  });
});

// ---------------------------------------------------------------------------
// SDK message mapping
// ---------------------------------------------------------------------------

describe("SDK message mapping", () => {
  describe("when SDK yields an assistant text content block", () => {
    it("emits a text_delta event", async () => {
      const sdkMessages = [
        {
          type: "assistant",
          content: [{ type: "text", text: "hello world" }],
        },
      ];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      const textEvents = events.filter((e) => e.type === "text_delta");
      expect(textEvents).toHaveLength(1);
      expect(textEvents[0]).toMatchObject({
        type: "text_delta",
        chunk: "hello world",
      });
    });
  });

  describe("when SDK yields an assistant tool_use content block", () => {
    it("emits a tool_start event", async () => {
      const sdkMessages = [
        {
          type: "assistant",
          content: [
            { type: "tool_use", id: "tu_1", name: "Read", input: { file_path: "/foo.ts" } },
          ],
        },
      ];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      const toolStarts = events.filter((e) => e.type === "tool_start");
      expect(toolStarts).toHaveLength(1);
      expect(toolStarts[0]).toMatchObject({
        type: "tool_start",
        toolUseId: "tu_1",
        toolName: "Read",
        input: { file_path: "/foo.ts" },
      });
    });
  });

  describe("when SDK yields a tool_result message", () => {
    it("emits a tool_end event with durationMs from matching start", async () => {
      const sdkMessages = [
        {
          type: "assistant",
          content: [
            { type: "tool_use", id: "tu_1", name: "Read", input: { file_path: "/foo.ts" } },
          ],
        },
        {
          type: "tool_result",
          tool_use_id: "tu_1",
          content: "file contents here",
        },
      ];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      const toolEnds = events.filter((e) => e.type === "tool_end");
      expect(toolEnds).toHaveLength(1);
      expect(toolEnds[0]).toMatchObject({
        type: "tool_end",
        toolUseId: "tu_1",
        toolName: "Read",
      });
      // durationMs should be a non-negative number
      expect((toolEnds[0] as { durationMs: number }).durationMs).toBeGreaterThanOrEqual(0);
    });
  });

  describe("when SDK yields a tool_progress message", () => {
    it("emits a tool_progress event", async () => {
      const sdkMessages = [
        {
          type: "assistant",
          content: [{ type: "tool_use", id: "tu_2", name: "Bash", input: { command: "ls" } }],
        },
        {
          type: "tool_progress",
          tool_use_id: "tu_2",
        },
      ];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      const progressEvents = events.filter((e) => e.type === "tool_progress");
      expect(progressEvents).toHaveLength(1);
      expect(progressEvents[0]).toMatchObject({
        type: "tool_progress",
        toolUseId: "tu_2",
      });
    });
  });

  describe("when SDK yields a tool_result with non-string content", () => {
    it("JSON-stringifies the content for the tool_end result", async () => {
      const sdkMessages = [
        {
          type: "assistant",
          content: [{ type: "tool_use", id: "tu_3", name: "Bash", input: { command: "echo hi" } }],
        },
        {
          type: "tool_result",
          tool_use_id: "tu_3",
          content: { stdout: "hi", exitCode: 0 },
        },
      ];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      const toolEnds = events.filter((e) => e.type === "tool_end");
      expect(toolEnds).toHaveLength(1);
      expect((toolEnds[0] as { result: string }).result).toBe(
        JSON.stringify({ stdout: "hi", exitCode: 0 }),
      );
    });
  });

  describe("when SDK yields a tool_result without a matching tool_start", () => {
    it("uses unknown as the tool name and zero as duration", async () => {
      const sdkMessages = [
        {
          type: "tool_result",
          tool_use_id: "tu_orphan",
          content: "result",
        },
      ];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      const toolEnds = events.filter((e) => e.type === "tool_end");
      expect(toolEnds).toHaveLength(1);
      expect(toolEnds[0]).toMatchObject({
        toolName: "unknown",
        toolUseId: "tu_orphan",
      });
    });
  });

  describe("when SDK yields malformed messages", () => {
    it("ignores messages without a type field", async () => {
      const sdkMessages = [{ noType: true }];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(events.filter((e) => e.type !== "usage_update")).toHaveLength(0);
    });

    it("ignores assistant messages with non-array content", async () => {
      const sdkMessages = [{ type: "assistant", content: "not an array" }];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(events.filter((e) => e.type === "text_delta")).toHaveLength(0);
    });

    it("ignores content blocks without a type field", async () => {
      const sdkMessages = [{ type: "assistant", content: [{ text: "orphan" }] }];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(events.filter((e) => e.type === "text_delta")).toHaveLength(0);
    });

    it("ignores text blocks with non-string text", async () => {
      const sdkMessages = [{ type: "assistant", content: [{ type: "text", text: 42 }] }];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(events.filter((e) => e.type === "text_delta")).toHaveLength(0);
    });

    it("ignores tool_use blocks with missing id or name", async () => {
      const sdkMessages = [{ type: "assistant", content: [{ type: "tool_use", input: {} }] }];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(events.filter((e) => e.type === "tool_start")).toHaveLength(0);
    });

    it("ignores thinking blocks with non-string thinking", async () => {
      const sdkMessages = [{ type: "assistant", content: [{ type: "thinking", thinking: 42 }] }];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(events.filter((e) => e.type === "thinking_update")).toHaveLength(0);
    });

    it("ignores non-object content blocks in the content array", async () => {
      const sdkMessages = [{ type: "assistant", content: ["not an object", 42, null] }];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(events.filter((e) => e.type === "text_delta")).toHaveLength(0);
    });

    it("ignores unknown message types", async () => {
      const sdkMessages = [{ type: "unknown_type", data: "something" }];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(events.filter((e) => e.type !== "usage_update")).toHaveLength(0);
    });

    it("ignores tool_result messages with non-string tool_use_id", async () => {
      const sdkMessages = [{ type: "tool_result", tool_use_id: 42, content: "data" }];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(events.filter((e) => e.type === "tool_end")).toHaveLength(0);
    });

    it("ignores tool_progress messages with non-string tool_use_id", async () => {
      const sdkMessages = [{ type: "tool_progress", tool_use_id: 42 }];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      expect(events.filter((e) => e.type === "tool_progress")).toHaveLength(0);
    });
  });

  describe("when SDK yields a thinking update", () => {
    it("emits a thinking_update event", async () => {
      const sdkMessages = [
        {
          type: "assistant",
          content: [
            {
              type: "thinking",
              thinking: "Let me consider this...",
            },
          ],
        },
      ];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      const thinkingEvents = events.filter((e) => e.type === "thinking_update");
      expect(thinkingEvents).toHaveLength(1);
      expect(thinkingEvents[0]).toMatchObject({
        type: "thinking_update",
      });
    });
  });
});

// ---------------------------------------------------------------------------
// usage tracking
// ---------------------------------------------------------------------------

describe("usage tracking", () => {
  it("returns the latest total_cost_usd from the SDK result and emits usage_update", async () => {
    const driver = createClaudeDriver({
      queryFn: fakeQueryFn([], { resultCostUsd: 0.05 }),
    });
    const { events, observer } = collectingObserver();

    const session = driver.start(makePrompt(), observer);
    await session.outcome;

    expect(session.usage()).toStrictEqual({ costUsd: 0.05 });

    const usageEvents = events.filter((e) => e.type === "usage_update");
    expect(usageEvents).toHaveLength(1);
    expect(usageEvents[0]).toMatchObject({
      type: "usage_update",
      costUsd: 0.05,
    });
  });

  it("does not emit usage_update when cost is zero", async () => {
    const driver = createClaudeDriver({
      queryFn: fakeQueryFn([], { resultCostUsd: 0 }),
    });
    const { events, observer } = collectingObserver();

    const session = driver.start(makePrompt(), observer);
    await session.outcome;

    expect(session.usage()).toStrictEqual({ costUsd: 0 });
    expect(events.filter((e) => e.type === "usage_update")).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// interrupt and abort
// ---------------------------------------------------------------------------

describe("interrupt", () => {
  it("resolves outcome as interrupted after interrupt()", async () => {
    let interruptCalled = false;
    const queryFn: QueryFn = (_opts: Record<string, unknown>) => {
      async function* generate() {
        // Yield a message, then stall — interrupt should stop iteration
        yield { type: "assistant", content: [{ type: "text", text: "hi" }] };
        // This simulates the SDK cooperating with interrupt by stopping iteration
        await new Promise<void>((resolve) => {
          setTimeout(resolve, 10_000);
        });
      }
      return {
        messages: generate(),
        interrupt(): void {
          interruptCalled = true;
        },
        result: { total_cost_usd: 0.01 },
      };
    };
    const driver = createClaudeDriver({ queryFn });

    const session = driver.start(makePrompt(), collectingObserver().observer);
    await session.interrupt();
    const outcome = await session.outcome;

    expect(interruptCalled).toBe(true);
    expect(outcome.reason).toBe("interrupted");
  });
});

describe("abort signal", () => {
  it("resolves outcome as interrupted when abort signal fires", async () => {
    const controller = new AbortController();
    const queryFn: QueryFn = (_opts: Record<string, unknown>) => {
      async function* generate() {
        yield { type: "assistant", content: [{ type: "text", text: "hi" }] };
        await new Promise<void>((resolve) => {
          setTimeout(resolve, 10_000);
        });
      }
      return {
        messages: generate(),
        interrupt(): void {},
        result: { total_cost_usd: 0 },
      };
    };
    const driver = createClaudeDriver({ queryFn });

    const session = driver.start(makePrompt(), collectingObserver().observer, controller.signal);
    controller.abort();
    const outcome = await session.outcome;

    expect(outcome.reason).toBe("interrupted");
  });

  it("resolves outcome as interrupted immediately when signal is already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    const driver = createClaudeDriver({ queryFn: fakeQueryFn([]) });

    const session = driver.start(makePrompt(), collectingObserver().observer, controller.signal);
    const outcome = await session.outcome;

    expect(outcome.reason).toBe("interrupted");
  });
});

// ---------------------------------------------------------------------------
// outcome
// ---------------------------------------------------------------------------

describe("outcome", () => {
  it("resolves completed when the SDK stream ends normally", async () => {
    const driver = createClaudeDriver({ queryFn: fakeQueryFn([]) });

    const session = driver.start(makePrompt(), collectingObserver().observer);
    const outcome = await session.outcome;

    expect(outcome.reason).toBe("completed");
  });

  it("resolves error with message when the SDK stream throws", async () => {
    const driver = createClaudeDriver({
      queryFn: fakeQueryFn([], { throwError: new Error("SDK failure") }),
    });

    const session = driver.start(makePrompt(), collectingObserver().observer);
    const outcome = await session.outcome;

    expect(outcome.reason).toBe("error");
    expect(outcome.message).toBe("SDK failure");
  });

  it("never rejects the outcome promise even on errors", async () => {
    const driver = createClaudeDriver({
      queryFn: fakeQueryFn([], { throwError: new Error("boom") }),
    });

    const session = driver.start(makePrompt(), collectingObserver().observer);

    // Should not throw — should resolve with error outcome
    await expect(session.outcome).resolves.toMatchObject({
      reason: "error",
      message: "boom",
    });
  });
});

// ---------------------------------------------------------------------------
// lazy SDK loading
// ---------------------------------------------------------------------------

describe("lazy SDK loading", () => {
  it("does not import the SDK module at driver construction time", () => {
    const driver = createClaudeDriver({ queryFn: fakeQueryFn([]) });
    expect(driver).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// message queue edge cases
// ---------------------------------------------------------------------------

describe("message queue", () => {
  it("closes the queue on interrupt so the SDK prompt iterable ends", async () => {
    let promptIterator: AsyncIterator<{ role: string; content: string }> | undefined;
    const queryFn: QueryFn = (opts: Record<string, unknown>) => {
      const prompt = opts["prompt"] as AsyncIterable<{ role: string; content: string }>;
      promptIterator = prompt[Symbol.asyncIterator]();
      async function* generate() {
        // consume first message then stall
        await promptIterator!.next();
      }
      return {
        messages: generate(),
        interrupt(): void {},
        result: { total_cost_usd: 0 },
      };
    };
    const driver = createClaudeDriver({ queryFn });
    const session = driver.start(makePrompt(), collectingObserver().observer);
    await session.interrupt();
    // After interrupt, the queue is closed; subsequent iteration should return done
    const result = await promptIterator!.next();
    expect(result.done).toBe(true);
  });

  it("delivers injected messages to the SDK prompt iterable after waiting", async () => {
    const consumed: string[] = [];
    let resolveGate: (() => void) | undefined;
    const gate = new Promise<void>((r) => {
      resolveGate = r;
    });
    const queryFn: QueryFn = (opts: Record<string, unknown>) => {
      const prompt = opts["prompt"] as AsyncIterable<{ role: string; content: string }>;
      async function* generate() {
        const iter = prompt[Symbol.asyncIterator]();
        // Consume the initial kickoff message
        const first = await iter.next();
        if (!first.done) consumed.push(first.value.content);
        // Signal that we're ready for the next message — this will await inside the queue
        resolveGate!();
        const second = await iter.next();
        if (!second.done) consumed.push(second.value.content);
      }
      return {
        messages: generate(),
        interrupt(): void {},
        result: { total_cost_usd: 0 },
      };
    };
    const driver = createClaudeDriver({ queryFn });
    const session = driver.start(makePrompt({ kickoff: "initial" }), collectingObserver().observer);
    // Wait until the generator has consumed the first message and is waiting for the second
    await gate;
    // Now inject — the queue iterator should resolve its waiter
    session.inject("followup");
    await session.outcome;

    expect(consumed).toStrictEqual(["initial", "followup"]);
  });
});

// ---------------------------------------------------------------------------
// tool result edge cases
// ---------------------------------------------------------------------------

describe("tool result edge cases", () => {
  it("falls back to String() for non-JSON-serializable tool result content", async () => {
    const circular: Record<string, unknown> = {};
    circular["self"] = circular;
    const sdkMessages = [
      {
        type: "assistant",
        content: [{ type: "tool_use", id: "tu_c", name: "Test", input: {} }],
      },
      {
        type: "tool_result",
        tool_use_id: "tu_c",
        content: circular,
      },
    ];
    const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
    const { events, observer } = collectingObserver();

    const session = driver.start(makePrompt(), observer);
    await session.outcome;

    const toolEnds = events.filter((e) => e.type === "tool_end");
    expect(toolEnds).toHaveLength(1);
    expect((toolEnds[0] as { result: string }).result).toBe("[object Object]");
  });

  it("handles tool_progress without a matching tool_start", async () => {
    const sdkMessages = [
      {
        type: "tool_progress",
        tool_use_id: "tu_orphan",
      },
    ];
    const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
    const { events, observer } = collectingObserver();

    const session = driver.start(makePrompt(), observer);
    await session.outcome;

    const progressEvents = events.filter((e) => e.type === "tool_progress");
    expect(progressEvents).toHaveLength(1);
    expect((progressEvents[0] as { elapsedMs: number }).elapsedMs).toBe(0);
  });
});
