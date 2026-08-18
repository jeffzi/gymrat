import type { QueryFn } from "../../src/supervisor/claude.js";
import type { SessionPrompt } from "../../src/supervisor/driver.js";
import type { LaunchEvent, SessionEvent, SessionObserver } from "../../src/supervisor/events.js";

/** Build a SessionPrompt with a default kickoff and cwd, overridable per field. */
export function makePrompt(overrides: Partial<SessionPrompt> = {}): SessionPrompt {
  return {
    kickoff: "do the thing",
    cwd: "/tmp/test",
    ...overrides,
  };
}

/** An observer plus the array of events it has recorded, in order. */
export function collectingObserver(): { events: SessionEvent[]; observer: SessionObserver } {
  const events: SessionEvent[] = [];
  const observer: SessionObserver = (e) => {
    events.push(e);
  };
  return { events, observer };
}

/** Build a LaunchEvent with sensible defaults, overridable per field. */
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

/** An observer that discards every event, for tests that don't inspect them. */
export function noopObserver(): SessionObserver {
  return () => {};
}

/** A queryFn that captures the options it was called with and yields nothing. */
export function capturingQueryFn(): {
  queryFn: QueryFn;
  captured: () => Record<string, unknown>;
} {
  let capturedOpts: Record<string, unknown> = {};
  const fn = (opts: Record<string, unknown>): AsyncIterable<Record<string, unknown>> => {
    capturedOpts = opts;
    async function* empty() {}
    return empty();
  };
  return { queryFn: fn, captured: () => capturedOpts };
}
