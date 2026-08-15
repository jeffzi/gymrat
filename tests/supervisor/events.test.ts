import { describe, expect, expectTypeOf, it } from "vitest";

import type { SessionEvent, SessionObserver } from "../../src/supervisor/events.js";
import {
  combineObservers,
  SUMMARY_MAX_CHARS,
  summarize,
  summarizeInput,
} from "../../src/supervisor/events.js";

// ---------------------------------------------------------------------------
// SessionEvent union — type-level tests
// ---------------------------------------------------------------------------

describe("SessionEvent", () => {
  it("covers all expected event types", () => {
    expectTypeOf<SessionEvent["type"]>().toEqualTypeOf<
      | "agent_start"
      | "thinking_update"
      | "tool_start"
      | "tool_progress"
      | "tool_end"
      | "text_delta"
      | "agent_end"
      | "agent_error"
      | "usage_update"
      | "inject"
      | "launch"
    >();
  });

  it("agent_start carries prompt, promptSummary, model, and timestamp", () => {
    expectTypeOf<Extract<SessionEvent, { type: "agent_start" }>>().toEqualTypeOf<{
      readonly type: "agent_start";
      readonly timestamp: number;
      readonly prompt: string;
      readonly promptSummary: string;
      readonly model: string | null;
    }>();
  });

  it("thinking_update carries estimatedTokens and delta", () => {
    expectTypeOf<Extract<SessionEvent, { type: "thinking_update" }>>().toEqualTypeOf<{
      readonly type: "thinking_update";
      readonly timestamp: number;
      readonly estimatedTokens: number;
      readonly delta: number;
    }>();
  });

  it("tool_start carries toolUseId, toolName, input, and inputSummary", () => {
    expectTypeOf<Extract<SessionEvent, { type: "tool_start" }>>().toEqualTypeOf<{
      readonly type: "tool_start";
      readonly timestamp: number;
      readonly toolUseId: string;
      readonly toolName: string;
      readonly input: unknown;
      readonly inputSummary: string;
    }>();
  });

  it("tool_progress carries toolUseId and elapsedMs", () => {
    expectTypeOf<Extract<SessionEvent, { type: "tool_progress" }>>().toEqualTypeOf<{
      readonly type: "tool_progress";
      readonly timestamp: number;
      readonly toolUseId: string;
      readonly elapsedMs: number;
    }>();
  });

  it("tool_end carries toolUseId, toolName, durationMs, result, and resultSummary", () => {
    expectTypeOf<Extract<SessionEvent, { type: "tool_end" }>>().toEqualTypeOf<{
      readonly type: "tool_end";
      readonly timestamp: number;
      readonly toolUseId: string;
      readonly toolName: string;
      readonly durationMs: number;
      readonly result: string;
      readonly resultSummary: string;
    }>();
  });

  it("text_delta carries chunk", () => {
    expectTypeOf<Extract<SessionEvent, { type: "text_delta" }>>().toEqualTypeOf<{
      readonly type: "text_delta";
      readonly timestamp: number;
      readonly chunk: string;
    }>();
  });

  it("agent_end carries durationMs and costUsd", () => {
    expectTypeOf<Extract<SessionEvent, { type: "agent_end" }>>().toEqualTypeOf<{
      readonly type: "agent_end";
      readonly timestamp: number;
      readonly durationMs: number;
      readonly costUsd: number;
    }>();
  });

  it("agent_error carries durationMs, costUsd (optional), subtype, and message", () => {
    expectTypeOf<Extract<SessionEvent, { type: "agent_error" }>>().toEqualTypeOf<{
      readonly type: "agent_error";
      readonly timestamp: number;
      readonly durationMs: number;
      readonly costUsd: number | undefined;
      readonly subtype: string;
      readonly message: string;
    }>();
  });

  it("usage_update carries costUsd", () => {
    expectTypeOf<Extract<SessionEvent, { type: "usage_update" }>>().toEqualTypeOf<{
      readonly type: "usage_update";
      readonly timestamp: number;
      readonly costUsd: number;
    }>();
  });

  it("inject carries message and messageSummary", () => {
    expectTypeOf<Extract<SessionEvent, { type: "inject" }>>().toEqualTypeOf<{
      readonly type: "inject";
      readonly timestamp: number;
      readonly message: string;
      readonly messageSummary: string;
    }>();
  });

  it("launch carries headSha, dirty, maxMinutes, maxUsd, model, runbookPath, and kickoffSummary", () => {
    expectTypeOf<Extract<SessionEvent, { type: "launch" }>>().toEqualTypeOf<{
      readonly type: "launch";
      readonly timestamp: number;
      readonly headSha: string;
      readonly dirty: false | { readonly fileCount: number };
      readonly maxMinutes: number;
      readonly maxUsd: number | undefined;
      readonly model: string | undefined;
      readonly runbookPath: string;
      readonly kickoffSummary: string;
    }>();
  });
});

// ---------------------------------------------------------------------------
// SessionObserver — type-level test
// ---------------------------------------------------------------------------

describe("SessionObserver", () => {
  it("is a function from SessionEvent to void", () => {
    expectTypeOf<SessionObserver>().toEqualTypeOf<(event: SessionEvent) => void>();
  });
});

// ---------------------------------------------------------------------------
// SUMMARY_MAX_CHARS
// ---------------------------------------------------------------------------

describe("SUMMARY_MAX_CHARS", () => {
  it("equals 200", () => {
    expect(SUMMARY_MAX_CHARS).toBe(200);
  });
});

// ---------------------------------------------------------------------------
// summarize
// ---------------------------------------------------------------------------

describe("summarize", () => {
  describe("when text fits within maxChars", () => {
    it("returns the text unchanged", () => {
      const text = "short text";

      const result = summarize(text, 100);

      expect(result).toBe("short text");
    });

    it("normalizes internal whitespace to single spaces", () => {
      const text = "hello   world";

      const result = summarize(text, 100);

      expect(result).toBe("hello world");
    });

    it("trims leading and trailing whitespace", () => {
      const text = "  trimmed  ";

      const result = summarize(text, 100);

      expect(result).toBe("trimmed");
    });

    it("collapses newlines into single spaces", () => {
      const text = "line1\nline2\nline3";

      const result = summarize(text, 100);

      expect(result).toBe("line1 line2 line3");
    });
  });

  describe("when text exceeds maxChars", () => {
    it("truncates and appends remaining chars and original line count", () => {
      const text = "a".repeat(300);

      const result = summarize(text, 50);

      expect(result).toMatch(/^a+… \(\d+ more chars, 1 lines\)$/);
    });

    it("reports the correct number of remaining characters", () => {
      const text = "a".repeat(300);

      const result = summarize(text, 50);

      expect(result).toBe("a".repeat(50) + "… (250 more chars, 1 lines)");
    });

    it("reports line count from the original text before collapsing", () => {
      const text = "line1\nline2\nline3\nline4";

      const result = summarize(text, 20);

      expect(result).toContain("4 lines");
    });

    it("truncates on code-point boundaries, not UTF-16 code units", () => {
      // 🎯 is U+1F3AF — a single code point represented as 2 UTF-16 code units.
      // 8 emoji = 8 code points (but 16 UTF-16 code units).
      // With maxChars=5, we keep the first 5 code points (5 emoji),
      // and report 3 remaining code points.
      const text = "🎯".repeat(8);

      const result = summarize(text, 5);

      expect(result).toBe("🎯🎯🎯🎯🎯… (3 more chars, 1 lines)");
    });

    it("counts remaining characters by code points when truncating", () => {
      // Mix ASCII and multi-byte: "ab🎯🎯cd🎯" = 7 code points.
      // maxChars=3 → keep "ab🎯", remaining = 4 code points.
      const text = "ab🎯🎯cd🎯";

      const result = summarize(text, 3);

      expect(result).toBe("ab🎯… (4 more chars, 1 lines)");
    });
  });
});

// ---------------------------------------------------------------------------
// summarizeInput
// ---------------------------------------------------------------------------

describe("summarizeInput", () => {
  it("JSON-encodes the input before summarizing", () => {
    const input = { key: "value" };

    const result = summarizeInput(input, 200);

    expect(result).toBe('{"key":"value"}');
  });

  it("falls back to String(input) for non-serializable values", () => {
    // JSON.stringify throws on this — exercises the String(input) fallback
    const circular: Record<string, unknown> = {};
    circular["self"] = circular;

    const result = summarizeInput(circular, 200);

    expect(result).toBe("[object Object]");
  });

  it("handles undefined input by summarizing the literal string 'undefined'", () => {
    const result = summarizeInput(undefined, 200);

    expect(result).toBe("undefined");
  });

  it("falls back to String(input) when JSON.stringify returns undefined for a function", () => {
    const fn = (): void => {};

    const result = summarizeInput(fn, 200);

    expect(result).toBe(String(fn));
  });

  it("falls back to String(input) when JSON.stringify returns undefined for a symbol", () => {
    const sym = Symbol("test");

    const result = summarizeInput(sym, 200);

    expect(result).toBe("Symbol(test)");
  });
});

// ---------------------------------------------------------------------------
// combineObservers
// ---------------------------------------------------------------------------

describe("combineObservers", () => {
  function makeEvent(): SessionEvent {
    return {
      type: "usage_update",
      timestamp: Date.now(),
      costUsd: 0.01,
    };
  }

  it("invokes each observer in order with the identical event object", () => {
    const received: SessionEvent[] = [];
    const obs1: SessionObserver = (e) => {
      received.push(e);
    };
    const obs2: SessionObserver = (e) => {
      received.push(e);
    };
    const combined = combineObservers(obs1, obs2);
    const event = makeEvent();

    combined(event);

    expect(received).toHaveLength(2);
    expect(received[0]).toBe(event);
    expect(received[1]).toBe(event);
  });

  it("returns a no-op observer when called with no arguments", () => {
    const combined = combineObservers();
    const event = makeEvent();

    // Act & Assert — calling it must not throw
    expect(() => {
      combined(event);
    }).not.toThrow();
  });
});
