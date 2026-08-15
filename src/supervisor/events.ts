// ---------------------------------------------------------------------------
// Event Types
// ---------------------------------------------------------------------------

/** Maximum character length for session event summaries. */
export const SUMMARY_MAX_CHARS = 200;

/** Emitted as the model's extended-thinking token estimate changes mid-turn. */
export interface ThinkingUpdateEvent {
  readonly type: "thinking_update";
  readonly timestamp: number;
  readonly estimatedTokens: number;
  readonly delta: number;
}

/** Emitted when the model invokes a tool. */
export interface ToolStartEvent {
  readonly type: "tool_start";
  readonly timestamp: number;
  readonly toolUseId: string;
  readonly toolName: string;
  readonly input: unknown;
  readonly inputSummary: string;
}

/** Emitted periodically while a long-running tool call is still in flight. */
export interface ToolProgressEvent {
  readonly type: "tool_progress";
  readonly timestamp: number;
  readonly toolUseId: string;
  readonly elapsedMs: number;
}

/** Emitted when a tool call completes and its result is available. */
export interface ToolEndEvent {
  readonly type: "tool_end";
  readonly timestamp: number;
  readonly toolUseId: string;
  readonly toolName: string;
  readonly durationMs: number;
  readonly result: string;
  readonly resultSummary: string;
}

/** Emitted for each chunk of assistant text as it streams in. */
export interface TextDeltaEvent {
  readonly type: "text_delta";
  readonly timestamp: number;
  readonly chunk: string;
}

/** Emitted when the driver observes updated cumulative cost. */
export interface UsageUpdateEvent {
  readonly type: "usage_update";
  readonly timestamp: number;
  readonly costUsd: number;
}

/** Written by the supervisor as the log's first line: launch provenance. */
export interface LaunchEvent {
  readonly type: "launch";
  readonly timestamp: number;
  readonly headSha: string;
  readonly dirty: false | { readonly fileCount: number };
  readonly maxMinutes: number;
  readonly maxUsd: number | undefined;
  readonly model: string | undefined;
  readonly runbookPath: string;
  readonly kickoffSummary: string;
}

/** The union of every event a session can emit to a {@link SessionObserver}. */
export type SessionEvent =
  | ThinkingUpdateEvent
  | ToolStartEvent
  | ToolProgressEvent
  | ToolEndEvent
  | TextDeltaEvent
  | UsageUpdateEvent
  | LaunchEvent;

/** Receives {@link SessionEvent}s as a session streams them. */
export type SessionObserver = (event: SessionEvent) => void;

// ---------------------------------------------------------------------------
// combineObservers
// ---------------------------------------------------------------------------

/**
 * Combines multiple observers into a single observer that invokes each one
 * in order with the same event.
 *
 * If the list is empty, returns a no-op observer that does nothing.
 * If any observer throws, the error propagates and stops further invocations.
 */
export function combineObservers(...observers: SessionObserver[]): SessionObserver {
  return (event: SessionEvent): void => {
    for (const observer of observers) {
      observer(event);
    }
  };
}

// ---------------------------------------------------------------------------
// summarize
// ---------------------------------------------------------------------------

/**
 * Produces a compact, single-line summary of text.
 *
 * If the collapsed text (whitespace normalized to single spaces) fits within
 * maxChars, returns it unchanged. Otherwise truncates and appends a suffix
 * with remaining char count and original line count.
 */
export function summarize(text: string, maxChars: number = SUMMARY_MAX_CHARS): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  const codePoints = Array.from(collapsed);

  if (codePoints.length <= maxChars) {
    return collapsed;
  }

  const remaining = codePoints.length - maxChars;
  const lineCount = text.split("\n").length;
  return `${codePoints.slice(0, maxChars).join("")}… (${remaining} more chars, ${lineCount} lines)`;
}

// ---------------------------------------------------------------------------
// summarizeInput
// ---------------------------------------------------------------------------

/**
 * Summarizes an input by JSON-stringifying it, then passing the result
 * through summarize().
 *
 * If JSON.stringify throws (e.g., circular reference) or returns undefined,
 * falls back to summarize(String(input), maxChars).
 */
export function summarizeInput(input: unknown, maxChars: number = SUMMARY_MAX_CHARS): string {
  if (input === undefined) return summarize("undefined", maxChars);
  try {
    return summarize(JSON.stringify(input), maxChars);
  } catch {
    // oxlint-disable-next-line typescript/no-base-to-string -- fallback for non-serializable values when JSON.stringify throws
    return summarize(String(input), maxChars);
  }
}
