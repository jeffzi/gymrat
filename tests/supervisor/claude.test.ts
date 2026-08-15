/* oxlint-disable require-yield -- async generators used as AsyncIterable adapters for the SDK message stream; yield is not always needed */
/* oxlint-disable typescript/require-await -- mock action callbacks are async to match the interface but have no awaitable work */
/* oxlint-disable typescript/no-unsafe-type-assertion -- narrowing test assertions on partial event objects */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { createClaudeDriver } from "../../src/supervisor/claude.js";
import type { QueryFn } from "../../src/supervisor/claude.js";
import { capturingQueryFn, collectingObserver, makePrompt } from "../fixtures/supervisor.js";

/** Creates a fake queryFn that yields the given SDK-shaped messages. */
function fakeQueryFn(
  messages: Array<Record<string, unknown>>,
  options?: {
    throwError?: Error;
  },
): QueryFn {
  const fn = (_opts: Record<string, unknown>): AsyncIterable<Record<string, unknown>> => {
    async function* generate() {
      for (const msg of messages) {
        yield msg;
      }
      if (options?.throwError) {
        throw options.throwError;
      }
    }
    return generate();
  };
  return fn;
}

/** A queryFn that yields one text delta, then hangs well past any test's own timeout. */
function hangingQueryFn(): QueryFn {
  const fn = (_opts: Record<string, unknown>): AsyncIterable<Record<string, unknown>> => {
    async function* generate() {
      yield { type: "assistant", content: [{ type: "text", text: "hi" }] };
      await new Promise<void>((r) => {
        setTimeout(r, 10_000);
      });
    }
    return generate();
  };
  return fn;
}

// ---------------------------------------------------------------------------
// SDK as optional peer dependency
// ---------------------------------------------------------------------------

describe("SDK peer dependency", () => {
  it("declares @anthropic-ai/claude-agent-sdk as an optional peer dependency", () => {
    const pkgPath = resolve(import.meta.dirname, "../../package.json");
    const pkg = JSON.parse(readFileSync(pkgPath, "utf-8")) as Record<string, unknown>;
    const peerDeps = pkg["peerDependencies"] as Record<string, string> | undefined;
    const peerMeta = pkg["peerDependenciesMeta"] as
      | Record<string, { optional?: boolean }>
      | undefined;

    expect(peerDeps?.["@anthropic-ai/claude-agent-sdk"]).toBeDefined();
    expect(peerMeta?.["@anthropic-ai/claude-agent-sdk"]?.optional).toBe(true);
  });

  it("does not list the SDK in dependencies or devDependencies", () => {
    const pkgPath = resolve(import.meta.dirname, "../../package.json");
    const pkg = JSON.parse(readFileSync(pkgPath, "utf-8")) as Record<string, unknown>;
    const deps = pkg["dependencies"] as Record<string, string> | undefined;
    const devDeps = pkg["devDependencies"] as Record<string, string> | undefined;

    expect(deps?.["@anthropic-ai/claude-agent-sdk"]).toBeUndefined();
    expect(devDeps?.["@anthropic-ai/claude-agent-sdk"]).toBeUndefined();
  });
});

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
    it("passes prompt.kickoff as a plain string, not an AsyncIterable", async () => {
      const { queryFn, captured } = capturingQueryFn();
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(
        makePrompt({ kickoff: "hello agent" }),
        collectingObserver().observer,
      );
      await session.outcome;

      expect(typeof captured()["prompt"]).toBe("string");
      expect(captured()["prompt"]).toBe("hello agent");
    });

    it("forwards cwd and permissionMode: bypassPermissions", async () => {
      const { queryFn, captured } = capturingQueryFn();
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(
        makePrompt({ cwd: "/my/project" }),
        collectingObserver().observer,
      );
      await session.outcome;

      expect(captured()["cwd"]).toBe("/my/project");
      expect(captured()["permissionMode"]).toBe("bypassPermissions");
    });

    it("includes systemPrompt in append-to-preset form when systemPromptAppend is present", async () => {
      const { queryFn, captured } = capturingQueryFn();
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(
        makePrompt({ systemPromptAppend: "extra instructions" }),
        collectingObserver().observer,
      );
      await session.outcome;

      expect(captured()["systemPrompt"]).toStrictEqual({
        type: "append-to-preset",
        text: "extra instructions",
      });
    });

    it("omits systemPrompt when systemPromptAppend is absent", async () => {
      const { queryFn, captured } = capturingQueryFn();
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(makePrompt(), collectingObserver().observer);
      await session.outcome;

      expect(captured()).not.toHaveProperty("systemPrompt");
    });

    it("includes model only when given in the prompt", async () => {
      const { queryFn, captured } = capturingQueryFn();
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(
        makePrompt({ model: "claude-sonnet-4-20250514" }),
        collectingObserver().observer,
      );
      await session.outcome;

      expect(captured()["model"]).toBe("claude-sonnet-4-20250514");
    });

    it("omits model when not given in the prompt", async () => {
      const { queryFn, captured } = capturingQueryFn();
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(makePrompt(), collectingObserver().observer);
      await session.outcome;

      expect(captured()).not.toHaveProperty("model");
    });

    it("passes an AbortController as options.abortController", async () => {
      const { queryFn, captured } = capturingQueryFn();
      const driver = createClaudeDriver({ queryFn });

      const session = driver.start(makePrompt(), collectingObserver().observer);
      await session.outcome;

      const ac = captured()["abortController"];
      expect(ac).toBeInstanceOf(AbortController);
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
    afterEach(() => {
      vi.useRealTimers();
    });

    it("emits a tool_end event with durationMs derived from the start timestamp", async () => {
      vi.useFakeTimers({ now: 1000 });

      let yieldGate: (() => void) | undefined;
      const fn = (_opts: Record<string, unknown>): AsyncIterable<Record<string, unknown>> => {
        async function* generate() {
          yield {
            type: "assistant",
            content: [
              { type: "tool_use", id: "tu_1", name: "Read", input: { file_path: "/foo.ts" } },
            ],
          };
          await new Promise<void>((r) => {
            yieldGate = r;
          });
          yield {
            type: "tool_result",
            tool_use_id: "tu_1",
            content: "file contents here",
          };
        }
        return generate();
      };

      const driver = createClaudeDriver({ queryFn: fn });
      const { events, observer } = collectingObserver();
      const session = driver.start(makePrompt(), observer);

      await vi.advanceTimersByTimeAsync(150);
      yieldGate!();
      await session.outcome;

      const toolEnds = events.filter((e) => e.type === "tool_end");
      expect(toolEnds).toHaveLength(1);
      expect(toolEnds[0]).toMatchObject({
        type: "tool_end",
        toolUseId: "tu_1",
        toolName: "Read",
        durationMs: 150,
      });
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
        durationMs: 0,
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

    it("reports estimatedTokens equal to delta for a single thinking block", async () => {
      // "abcd" = 4 chars → Math.ceil(4/4) = 1 token
      const sdkMessages = [
        {
          type: "assistant",
          content: [{ type: "thinking", thinking: "abcd" }],
        },
      ];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      const thinkingEvents = events.filter((e) => e.type === "thinking_update");
      expect(thinkingEvents).toHaveLength(1);
      expect(thinkingEvents[0]).toMatchObject({
        estimatedTokens: 1,
        delta: 1,
      });
    });

    it("accumulates estimatedTokens across multiple thinking blocks while delta stays per-block", async () => {
      // Block 1: "abcd" = 4 chars → delta = 1, cumulative = 1
      // Block 2: "abcdefgh" = 8 chars → delta = 2, cumulative = 3
      const sdkMessages = [
        {
          type: "assistant",
          content: [{ type: "thinking", thinking: "abcd" }],
        },
        {
          type: "assistant",
          content: [{ type: "thinking", thinking: "abcdefgh" }],
        },
      ];
      const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
      const { events, observer } = collectingObserver();

      const session = driver.start(makePrompt(), observer);
      await session.outcome;

      const thinkingEvents = events.filter((e) => e.type === "thinking_update");
      expect(thinkingEvents).toHaveLength(2);
      expect(thinkingEvents[0]).toMatchObject({
        estimatedTokens: 1,
        delta: 1,
      });
      expect(thinkingEvents[1]).toMatchObject({
        estimatedTokens: 3,
        delta: 2,
      });
    });
  });
});

// ---------------------------------------------------------------------------
// cost from result-type messages
// ---------------------------------------------------------------------------

describe("cost tracking from result messages", () => {
  it("emits usage_update for each result-type message with total_cost_usd", async () => {
    const sdkMessages = [
      { type: "assistant", content: [{ type: "text", text: "working..." }] },
      { type: "result", total_cost_usd: 0.03 },
      { type: "assistant", content: [{ type: "text", text: "done" }] },
      { type: "result", total_cost_usd: 0.07 },
    ];
    const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
    const { events, observer } = collectingObserver();

    const session = driver.start(makePrompt(), observer);
    await session.outcome;

    const usageEvents = events.filter((e) => e.type === "usage_update");
    expect(usageEvents).toHaveLength(2);
    expect(usageEvents[0]).toMatchObject({ type: "usage_update", costUsd: 0.03 });
    expect(usageEvents[1]).toMatchObject({ type: "usage_update", costUsd: 0.07 });
  });

  it("reports the final cost in the outcome from the last result message", async () => {
    const sdkMessages = [
      { type: "result", total_cost_usd: 0.02 },
      { type: "result", total_cost_usd: 0.05 },
    ];
    const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
    const { observer } = collectingObserver();

    const session = driver.start(makePrompt(), observer);
    const outcome = await session.outcome;

    expect(outcome.costUsd).toBe(0.05);
  });

  it("does not emit usage_update when no result messages appear", async () => {
    const sdkMessages = [{ type: "assistant", content: [{ type: "text", text: "hello" }] }];
    const driver = createClaudeDriver({ queryFn: fakeQueryFn(sdkMessages) });
    const { events, observer } = collectingObserver();

    const session = driver.start(makePrompt(), observer);
    await session.outcome;

    expect(events.filter((e) => e.type === "usage_update")).toHaveLength(0);
  });

  it("reports costUsd 0 in outcome when no result messages provide cost", async () => {
    const driver = createClaudeDriver({ queryFn: fakeQueryFn([]) });
    const { observer } = collectingObserver();

    const session = driver.start(makePrompt(), observer);
    const outcome = await session.outcome;

    expect(outcome.costUsd).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// interrupt via AbortController
// ---------------------------------------------------------------------------

describe("interrupt", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("aborts the internal controller when interrupt() is called", async () => {
    const { queryFn, captured } = capturingQueryFn();
    const driver = createClaudeDriver({ queryFn });

    const session = driver.start(makePrompt(), collectingObserver().observer);
    await session.interrupt();

    const ac = captured()["abortController"] as AbortController;
    expect(ac.signal.aborted).toBe(true);
  });

  it("resolves outcome as interrupted after interrupt()", async () => {
    vi.useFakeTimers();
    const driver = createClaudeDriver({ queryFn: hangingQueryFn() });

    const session = driver.start(makePrompt(), collectingObserver().observer);
    await session.interrupt();
    const outcome = await session.outcome;

    expect(outcome.reason).toBe("interrupted");
  });
});

describe("abort signal", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("resolves outcome as interrupted when abort signal fires", async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const driver = createClaudeDriver({ queryFn: hangingQueryFn() });

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
    let queryFnCalled = false;
    const fn = (_opts: Record<string, unknown>): AsyncIterable<Record<string, unknown>> => {
      queryFnCalled = true;
      async function* empty() {}
      return empty();
    };

    const driver = createClaudeDriver({ queryFn: fn });

    expect(queryFnCalled).toBe(false);
    expect(driver).toHaveProperty("start");
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
