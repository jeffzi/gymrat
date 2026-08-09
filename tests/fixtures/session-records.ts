import type { ResolvedConfig } from "../../src/config.js";
import type { IterationRecord, KeepRecord, SessionRecord } from "../../src/session/records.js";

/** The instant every fixture record in this file was written at. */
export const AT = "2026-08-08T14:15:30.000Z";

/** A commit SHA fixture records point at; not a real commit. */
const COMMIT = "b".repeat(40);

/** A settled run configuration, geomean-led unless a test names its own primary. */
export function resolvedConfig(overrides: Partial<ResolvedConfig> = {}): ResolvedConfig {
  return {
    bench: "npm run bench",
    adapter: "metric-lines",
    samples: 10,
    timeoutSeconds: 1800,
    unstableNoisePct: 200,
    primary: "geomean",
    hooks: "gymrat.hooks",
    ...overrides,
  };
}

/**
 * The session header a started session writes, with every field overridable.
 *
 * `sessionId` drives the default `branch`, so a caller overriding just the id
 * still gets a matching branch; a caller after a divergent branch overrides
 * both explicitly.
 */
export function sessionRecord(overrides: Partial<SessionRecord> = {}): SessionRecord {
  const sessionId = overrides.sessionId ?? "20260808-141530-a3f2";
  return {
    type: "session",
    schemaVersion: 1,
    sessionId,
    createdAt: AT,
    baseline: { ref: "main", sha: "a".repeat(40) },
    branch: `gymrat/${sessionId}`,
    worktrees: {
      experiment: "/repo/.gymrat/worktrees/experiment",
      baseline: "/repo/.gymrat/worktrees/baseline",
    },
    config: {
      bench: "npm run bench",
      adapter: "metric-lines",
      samples: 10,
      timeoutSeconds: 1800,
      primary: "geomean",
      hooks: "gymrat.hooks",
    },
    ...overrides,
  };
}

/** A measured iteration numbered 1, improved unless overridden. */
export function iterationRecord(overrides: Partial<IterationRecord> = {}): IterationRecord {
  return {
    type: "iteration",
    seq: 1,
    at: AT,
    samples: { experiment: [{ total_ms: 14100 }], baseline: [{ total_ms: 15200 }] },
    metrics: {
      total_ms: {
        deltaPct: -7.2,
        verdict: "improved",
        method: "signed-rank",
        p: 0.002,
        noisePct: 1.4,
        gating: true,
        confirmed: false,
      },
    },
    primary: { kind: "geomean", deltaPct: -7.2 },
    outcome: "improved",
    targetReached: false,
    ...overrides,
  };
}

/** A keep that committed the iteration numbered `seq`, with every field overridable. */
export function committedKeep(
  seq: number,
  overrides: Partial<Omit<KeepRecord, "type" | "status">> = {},
): KeepRecord {
  return {
    type: "keep",
    seq,
    at: AT,
    status: "committed",
    commit: COMMIT,
    message: "cache the regex",
    checks: { configured: true, passed: true },
    ...overrides,
  };
}
