import { isRecord, messageOf } from "../errors.js";
import type { Driver, DriverSession, SessionOutcome, SessionPrompt } from "./driver.js";
import type { SessionObserver } from "./events.js";
import { summarize, summarizeInput } from "./events.js";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Signature of the SDK `query()` function or its injectable replacement. */
export type QueryFn = (opts: Record<string, unknown>) => AsyncIterable<Record<string, unknown>>;

/** Options for {@link createClaudeDriver}. */
export interface ClaudeDriverOptions {
  readonly queryFn?: QueryFn;
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
// SDK message processing
// ---------------------------------------------------------------------------

interface ProcessingContext {
  readonly observer: SessionObserver;
  readonly toolStartTimes: Map<string, number>;
  readonly toolNames: Map<string, string>;
  thinkingTokens: number;
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
      inputSummary: summarizeInput(input),
    });
  } else if (blockType === "thinking") {
    const thinking = block["thinking"];
    if (typeof thinking !== "string") return;

    const delta = Math.ceil(thinking.length / 4);
    ctx.thinkingTokens += delta;
    ctx.observer({
      type: "thinking_update",
      timestamp: Date.now(),
      estimatedTokens: ctx.thinkingTokens,
      delta,
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
    resultSummary: summarize(result),
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

function processAssistantContent(msg: Record<string, unknown>, ctx: ProcessingContext): void {
  const content = msg["content"];
  if (!Array.isArray(content)) return;
  for (const block of content) {
    if (isRecord(block)) {
      processContentBlock(block, ctx);
    }
  }
}

function extractCostUpdate(
  msg: Record<string, unknown>,
  ctx: ProcessingContext,
  currentCostUsd: number,
): number {
  const totalCostUsd = msg["total_cost_usd"];
  if (typeof totalCostUsd === "number" && totalCostUsd > 0) {
    ctx.observer({ type: "usage_update", timestamp: Date.now(), costUsd: totalCostUsd });
    return totalCostUsd;
  }
  return currentCostUsd;
}

/**
 * Processes a single SDK message, updating cost when a `result`-type message
 * carries `total_cost_usd`. Returns the updated cumulative cost.
 */
function processSdkMessage(
  msg: Record<string, unknown>,
  ctx: ProcessingContext,
  currentCostUsd: number,
): number {
  const msgType = msg["type"];
  if (typeof msgType !== "string") return currentCostUsd;

  if (msgType === "assistant") {
    processAssistantContent(msg, ctx);
  } else if (msgType === "tool_result") {
    processToolResult(msg, ctx);
  } else if (msgType === "tool_progress") {
    processToolProgress(msg, ctx);
  } else if (msgType === "result") {
    return extractCostUpdate(msg, ctx, currentCostUsd);
  }

  return currentCostUsd;
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
    start(prompt: SessionPrompt, observer: SessionObserver, signal?: AbortSignal): DriverSession {
      let costUsd = 0;
      let interruptedOutcome: SessionOutcome | undefined;
      const queryAbortController = new AbortController();

      function doInterrupt(): void {
        interruptedOutcome = { reason: "interrupted", costUsd };
        queryAbortController.abort();
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
        thinkingTokens: 0,
      };

      // fallow-ignore-next-line complexity
      async function run(): Promise<SessionOutcome> {
        if (interruptedOutcome) return interruptedOutcome;

        /* v8 ignore next -- loadSdk fallback exercised in production, not unit tests */
        const queryFn = options.queryFn ?? (await loadSdk());

        const queryOpts: Record<string, unknown> = {
          prompt: prompt.kickoff,
          cwd: prompt.cwd,
          permissionMode: "bypassPermissions",
          abortController: queryAbortController,
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

        try {
          for await (const msg of q) {
            // oxlint-disable-next-line no-unnecessary-condition -- interruptedOutcome is externally mutated during await
            if (interruptedOutcome) return interruptedOutcome;
            if (isRecord(msg)) {
              costUsd = processSdkMessage(msg, ctx, costUsd);
            }
          }

          return { reason: "completed", costUsd };
        } catch (err: unknown) {
          // oxlint-disable-next-line no-unnecessary-condition -- interruptedOutcome is externally mutated during await
          if (interruptedOutcome) return interruptedOutcome;
          return { reason: "error", costUsd, message: messageOf(err) };
        }
      }

      const runPromise = run();

      return {
        interrupt(): Promise<void> {
          doInterrupt();
          return Promise.resolve();
        },

        get outcome(): Promise<SessionOutcome> {
          if (interruptedOutcome) return Promise.resolve(interruptedOutcome);
          return runPromise;
        },
      };
    },
  };
}
