import type { QueryFn } from "../../src/supervisor/claude.js";
import type { SessionPrompt } from "../../src/supervisor/driver.js";
import type { LaunchEvent, SessionEvent, SessionObserver } from "../../src/supervisor/events.js";

export function makePrompt(overrides: Partial<SessionPrompt> = {}): SessionPrompt {
  return {
    kickoff: "do the thing",
    cwd: "/tmp/test",
    ...overrides,
  };
}

export function collectingObserver(): { events: SessionEvent[]; observer: SessionObserver } {
  const events: SessionEvent[] = [];
  const observer: SessionObserver = (e) => {
    events.push(e);
  };
  return { events, observer };
}

export function makeLaunch(overrides: Partial<LaunchEvent> = {}): LaunchEvent {
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

export function noopObserver(): SessionObserver {
  return () => {};
}

/**
 * A queryFn that captures the options it was called with and yields nothing.
 *
 * Use this when the test needs to inspect what the driver sent to the SDK
 * (prompt shape, cwd, model, etc.) rather than what the SDK yields back.
 */
export function capturingQueryFn(): {
  queryFn: QueryFn;
  captured: () => Record<string, unknown>;
} {
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
  return { queryFn, captured: () => capturedOpts };
}
