import { messageOf } from "../errors.js";
import { createSession } from "./driver.js";
import type { Driver, SessionOutcome, SessionPrompt } from "./driver.js";
import type { SessionObserver } from "./events.js";
import { summarize, summarizeInput, SUMMARY_MAX_CHARS } from "./events.js";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Handle returned by a {@link QueryFn} call. */
export interface QueryHandle {
  readonly messages: AsyncIterable<Record<string, unknown>>;
  interrupt(): void;
  readonly result: { readonly total_cost_usd: number };
}

/** Signature of the SDK `query()` function or its injectable replacement. */
export type QueryFn = (opts: Record<string, unknown>) => QueryHandle;

/** Options for {@link createClaudeDriver}. */
export interface ClaudeDriverOptions {
  readonly queryFn?: QueryFn;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Resolves the SDK `query()` function from a dynamic import.
 *
 * The module path is a runtime variable so TypeScript does not attempt
 * static resolution — the SDK may not be installed at build time.
 */
/* v8 ignore start -- dynamic import exercised in production, not unit tests */
async function loadSdk(): Promise<QueryFn> {
  const SDK_MODULE = "@anthropic-ai/claude-agent-sdk";
  const mod: unknown = await import(SDK_MODULE);
  if (!isRecord(mod) || typeof mod["query"] !== "function") {
    throw new Error(`${SDK_MODULE} does not export a query function — is the package installed?`);
  }
  // oxlint-disable-next-line no-unsafe-type-assertion -- runtime guard above validates shape; SDK has no typed export
  return mod["query"] as QueryFn;
}
/* v8 ignore stop */

// ---------------------------------------------------------------------------
// Internal message queue — AsyncIterable that feeds user messages to the SDK
// ---------------------------------------------------------------------------

interface MessageQueue {
  readonly iterable: AsyncIterable<{ role: string; content: string }>;
  push(message: string): void;
  close(): void;
}

function createMessageQueue(initialMessage: string): MessageQueue {
  const pending: string[] = [initialMessage];
  let waiter: (() => void) | null = null;
  let closed = false;

  const iterable: AsyncIterable<{ role: string; content: string }> = {
    [Symbol.asyncIterator]() {
      return {
        async next(): Promise<IteratorResult<{ role: string; content: string }>> {
          for (;;) {
            if (pending.length > 0) {
              const msg = pending.shift()!;
              return { value: { role: "user", content: msg }, done: false };
            }
            if (closed) {
              return { value: undefined, done: true } as IteratorReturnResult<undefined>;
            }
            await new Promise<void>((r) => {
              waiter = r;
            });
          }
        },
      };
    },
  };

  return {
    iterable,
    push(message: string): void {
      pending.push(message);
      waiter?.();
      waiter = null;
    },
    close(): void {
      closed = true;
      waiter?.();
      waiter = null;
    },
  };
}

// ---------------------------------------------------------------------------
// SDK message processing
// ---------------------------------------------------------------------------

interface ProcessingContext {
  readonly observer: SessionObserver;
  readonly toolStartTimes: Map<string, number>;
  readonly toolNames: Map<string, string>;
}

function processContentBlock(block: Record<string, unknown>, ctx: ProcessingContext): void {
  const blockType = block["type"];
  if (typeof blockType !== "string") return;

  if (blockType === "text") {
    const text = block["text"];
    if (typeof text !== "string") return;
    ctx.observer({
      type: "text_delta",
      timestamp: Date.now(),
      chunk: text,
    });
  } else if (blockType === "tool_use") {
    const toolUseId = block["id"];
    const toolName = block["name"];
    if (typeof toolUseId !== "string" || typeof toolName !== "string") return;

    const input: unknown = block["input"];
    const now = Date.now();
    ctx.toolStartTimes.set(toolUseId, now);
    ctx.toolNames.set(toolUseId, toolName);
    ctx.observer({
      type: "tool_start",
      timestamp: now,
      toolUseId,
      toolName,
      input,
      inputSummary: summarizeInput(input, SUMMARY_MAX_CHARS),
    });
  } else if (blockType === "thinking") {
    const thinking = block["thinking"];
    if (typeof thinking !== "string") return;

    const estimatedTokens = Math.ceil(thinking.length / 4);
    ctx.observer({
      type: "thinking_update",
      timestamp: Date.now(),
      estimatedTokens,
      delta: estimatedTokens,
    });
  }
}

function processToolResult(msg: Record<string, unknown>, ctx: ProcessingContext): void {
  const toolUseId = msg["tool_use_id"];
  if (typeof toolUseId !== "string") return;

  const toolName = ctx.toolNames.get(toolUseId) ?? "unknown";
  const startTime = ctx.toolStartTimes.get(toolUseId);
  const durationMs = startTime !== undefined ? Date.now() - startTime : 0;

  const rawResult = msg["content"];
  let result: string;
  if (typeof rawResult === "string") {
    result = rawResult;
  } else {
    try {
      result = JSON.stringify(rawResult);
    } catch {
      result = String(rawResult);
    }
  }

  ctx.observer({
    type: "tool_end",
    timestamp: Date.now(),
    toolUseId,
    toolName,
    durationMs,
    result,
    resultSummary: summarize(result, SUMMARY_MAX_CHARS),
  });
}

function processToolProgress(msg: Record<string, unknown>, ctx: ProcessingContext): void {
  const toolUseId = msg["tool_use_id"];
  if (typeof toolUseId !== "string") return;

  const startTime = ctx.toolStartTimes.get(toolUseId);
  const elapsedMs = startTime !== undefined ? Date.now() - startTime : 0;
  ctx.observer({
    type: "tool_progress",
    timestamp: Date.now(),
    toolUseId,
    elapsedMs,
  });
}

function processSdkMessage(msg: Record<string, unknown>, ctx: ProcessingContext): void {
  const msgType = msg["type"];
  if (typeof msgType !== "string") return;

  if (msgType === "assistant") {
    const content = msg["content"];
    if (!Array.isArray(content)) return;
    for (const block of content) {
      if (isRecord(block)) {
        processContentBlock(block, ctx);
      }
    }
  } else if (msgType === "tool_result") {
    processToolResult(msg, ctx);
  } else if (msgType === "tool_progress") {
    processToolProgress(msg, ctx);
  }
}

// ---------------------------------------------------------------------------
// Driver factory
// ---------------------------------------------------------------------------

/**
 * Creates a {@link Driver} backed by the Claude Agent SDK.
 *
 * When `queryFn` is provided it replaces the real SDK, making the driver
 * testable without installing `@anthropic-ai/claude-agent-sdk`. When omitted,
 * the SDK is loaded lazily via dynamic `import()` the first time `start()` is
 * called — constructing the driver never triggers the import.
 */
export function createClaudeDriver(options: ClaudeDriverOptions = {}): Driver {
  return {
    start(prompt: SessionPrompt, observer: SessionObserver, signal?: AbortSignal) {
      const queue = createMessageQueue(prompt.kickoff);
      let costUsd = 0;
      let interruptedOutcome: SessionOutcome | undefined;
      let sdkQuery: QueryHandle | undefined;

      function doInterrupt(): void {
        interruptedOutcome = { reason: "interrupted", costUsd };
        sdkQuery?.interrupt();
        queue.close();
      }

      if (signal) {
        if (signal.aborted) {
          doInterrupt();
        } else {
          signal.addEventListener("abort", doInterrupt, { once: true });
        }
      }

      const ctx: ProcessingContext = {
        observer,
        toolStartTimes: new Map(),
        toolNames: new Map(),
      };

      // fallow-ignore-next-line complexity
      async function run(): Promise<SessionOutcome> {
        if (interruptedOutcome) return interruptedOutcome;

        /* v8 ignore next -- loadSdk fallback exercised in production, not unit tests */
        const queryFn = options.queryFn ?? (await loadSdk());

        const queryOpts: Record<string, unknown> = {
          prompt: queue.iterable,
          cwd: prompt.cwd,
          permissionMode: "bypassPermissions",
        };

        if (prompt.systemPromptAppend !== undefined) {
          queryOpts["systemPrompt"] = {
            type: "append-to-preset",
            text: prompt.systemPromptAppend,
          };
        }

        if (prompt.model !== undefined) {
          queryOpts["model"] = prompt.model;
        }

        const q = queryFn(queryOpts);
        sdkQuery = q;

        try {
          for await (const msg of q.messages) {
            // oxlint-disable-next-line no-unnecessary-condition -- interruptedOutcome is externally mutated during await
            if (interruptedOutcome) return interruptedOutcome;
            if (isRecord(msg)) {
              processSdkMessage(msg, ctx);
            }
          }

          costUsd = q.result.total_cost_usd;
          if (costUsd > 0) {
            observer({ type: "usage_update", timestamp: Date.now(), costUsd });
          }
          return { reason: "completed", costUsd };
        } catch (err: unknown) {
          // oxlint-disable-next-line no-unnecessary-condition -- interruptedOutcome is externally mutated during await
          if (interruptedOutcome) return interruptedOutcome;
          return { reason: "error", costUsd, message: messageOf(err) };
        }
      }

      return createSession({
        onMessage: (msg) => {
          queue.push(msg);
        },
        doInterrupt,
        getCostUsd: () => costUsd,
        getInterruptedOutcome: () => interruptedOutcome,
        runPromise: run(),
        observer,
      });
    },
  };
}
