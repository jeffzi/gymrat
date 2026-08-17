import { expect } from "vitest";

import type { BenchlessConfig, ResolvedConfig } from "../../src/config.js";
import { sessionJsonlPath } from "../../src/session/paths.js";
import type {
  DiscardRecord,
  FinalizeRecord,
  HookRecord,
  IterationRecord,
  KeepRecord,
  SessionLogRecord,
  SessionRecord,
} from "../../src/session/records.js";
import { appendRecord } from "../../src/session/store.js";
import { SESSION_ID } from "./constants.js";

/**
 * Like `Partial<T>` but allows explicit `undefined` values — the semantics
 * `Partial` had before `exactOptionalPropertyTypes`. Callers pass `undefined`
 * to erase a default the helper sets; `withDefaults` removes those keys
 * so the resulting object satisfies the strict type.
 */
export type LoosePartial<T> = { [K in keyof T]?: T[K] | undefined };

/**
 * Build `defaults` with selected keys overridden.
 *
 * Entries in `overrides` whose value is `undefined` are dropped — the default
 * stays — so callers can pass `{ prop: undefined }` to erase a fixture default
 * without violating `exactOptionalPropertyTypes`.
 */
export function withDefaults<T extends object>(defaults: T, overrides: LoosePartial<T>): T {
  const result = { ...defaults };
  for (const [key, value] of Object.entries(overrides)) {
    if (value !== undefined) {
      (result as Record<string, unknown>)[key] = value;
    } else {
      delete (result as Record<string, unknown>)[key];
    }
  }
  return result;
}

/** The instant every fixture record in this file was written at. */
export const AT = "2026-08-08T14:15:30.000Z";

/** A commit SHA fixture records point at; not a real commit. */
export const COMMIT = "b".repeat(40);

/** The squash commit SHA finalize fixtures point at; distinct from {@link COMMIT}. */
const SQUASH_COMMIT = "c".repeat(40);

/** A settled configuration without a bench command, geomean-led unless overridden. */
export function benchlessConfig(overrides: LoosePartial<BenchlessConfig> = {}): BenchlessConfig {
  return withDefaults<BenchlessConfig>(
    {
      adapter: "metric-lines",
      samples: 10,
      timeoutSeconds: 1800,
      unstableNoisePct: 200,
      primary: "geomean",
    },
    overrides,
  );
}

/** A settled run configuration, geomean-led unless a test names its own primary. */
export function resolvedConfig(overrides: LoosePartial<ResolvedConfig> = {}): ResolvedConfig {
  return withDefaults<ResolvedConfig>({ ...benchlessConfig(), bench: "npm run bench" }, overrides);
}

/**
 * The session header a started session writes, with every field overridable.
 *
 * `sessionId` drives the default `branch`, so a caller overriding just the id
 * still gets a matching branch; a caller after a divergent branch overrides
 * both explicitly.
 */
export function sessionRecord(overrides: LoosePartial<SessionRecord> = {}): SessionRecord {
  const sessionId = overrides.sessionId ?? SESSION_ID;
  return withDefaults<SessionRecord>(
    {
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
      },
    },
    overrides,
  );
}

/** A measured iteration numbered 1, improved unless overridden. */
export function iterationRecord(overrides: LoosePartial<IterationRecord> = {}): IterationRecord {
  return withDefaults<IterationRecord>(
    {
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
    },
    overrides,
  );
}

/** A `HookRecord`, but with `durationMs` a matcher instead of a number — never a real one to assert against. */
type ExpectedHookRecord = Omit<HookRecord, "durationMs"> & { durationMs: unknown };

/**
 * The `HookRecord` `runHook` produces, for asserting with `toStrictEqual`.
 *
 * `durationMs` is nondeterministic, so it is always `expect.any(Number)` rather
 * than a value a caller could supply. `timedOut` defaults to `false`; every
 * other field is the caller's to name.
 */
export function expectedHookRecord(
  overrides: Omit<HookRecord, "type" | "durationMs" | "timedOut"> &
    Partial<Pick<HookRecord, "timedOut">>,
): ExpectedHookRecord {
  return {
    type: "hook",
    timedOut: false,
    stderrBytes: 0,
    ...overrides,
    // oxlint-disable-next-line typescript/no-unsafe-assignment -- vitest asymmetric matcher
    durationMs: expect.any(Number),
  };
}

/** A discard of the iteration numbered `seq`. */
export function discardRecord(seq: number): DiscardRecord {
  return { type: "discard", seq, at: AT };
}

/** The record that closes a session, with every field overridable. */
export function finalizeRecord(overrides: LoosePartial<FinalizeRecord> = {}): FinalizeRecord {
  return withDefaults<FinalizeRecord>(
    {
      type: "finalize",
      at: AT,
      branch: `gymrat/${SESSION_ID}-final`,
      commit: SQUASH_COMMIT,
      message: "squash 1 kept iteration",
    },
    overrides,
  );
}

/** A keep that committed the iteration numbered `seq`, with every field overridable. */
export function committedKeep(
  seq: number,
  overrides: LoosePartial<Omit<KeepRecord, "type" | "status">> = {},
): KeepRecord {
  return withDefaults<KeepRecord>(
    {
      type: "keep",
      seq,
      at: AT,
      status: "committed",
      commit: COMMIT,
      message: "cache the regex",
      checks: { configured: true, passed: true },
    },
    overrides,
  );
}

/** A keep the checks gate refused, leaving the iteration numbered `seq` uncommitted. */
export function blockedKeep(
  seq: number,
  overrides: LoosePartial<Omit<KeepRecord, "type" | "status">> = {},
): KeepRecord {
  return withDefaults<KeepRecord>(
    {
      type: "keep",
      seq,
      at: AT,
      status: "blocked",
      reason: "checks-failed",
      checks: { configured: true, passed: false },
    },
    overrides,
  );
}

/**
 * Write a session log opening on `header` and holding `history` after it.
 *
 * The header is appended first, then each history record in order.
 */
export function writeSessionLog(
  root: string,
  header: SessionRecord,
  history: SessionLogRecord[] = [],
): void {
  appendRecord(sessionJsonlPath(root), header);
  for (const record of history) {
    appendRecord(sessionJsonlPath(root), record);
  }
}
