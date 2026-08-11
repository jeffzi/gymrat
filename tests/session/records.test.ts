import { describe, expect, expectTypeOf, it } from "vitest";

import { GymratError } from "../../src/errors.js";
import type {
  BaselineRecord,
  DiscardRecord,
  FinalizeRecord,
  HookRecord,
  IterationRecord,
  KeepRecord,
  SessionLogRecord,
  SessionRecord,
} from "../../src/session/records.js";
import { parseRecord } from "../../src/session/records.js";
import { captureGymratError } from "../fixtures/errors.js";

const AT = "2026-08-08T14:15:30.000Z";
const SHA = "a".repeat(40);
const COMMIT = "b".repeat(40);

const sessionRecord: SessionRecord = {
  type: "session",
  schemaVersion: 1,
  sessionId: "20260808-141530-a3f2",
  createdAt: AT,
  baseline: { ref: "main", sha: SHA },
  branch: "gymrat/20260808-141530-a3f2",
  worktrees: { experiment: "/repo/.gymrat/experiment", baseline: "/repo/.gymrat/baseline" },
  config: {
    bench: "npm run bench",
    adapter: "metric-lines",
    samples: 10,
    timeoutSeconds: 1800,
    primary: "geomean",
  },
};

const baselineRecord: BaselineRecord = {
  type: "baseline",
  at: AT,
  label: "main",
  samples: [{ total_ms: 15200 }, { total_ms: 15184 }],
};

const metricVerdict: NonNullable<IterationRecord["metrics"][string]> = {
  deltaPct: -7.2,
  verdict: "improved",
  method: "signed-rank",
  p: 0.002,
  noisePct: 1.4,
  gating: true,
  confirmed: false,
};

const iterationRecord: IterationRecord = {
  type: "iteration",
  seq: 1,
  at: AT,
  samples: {
    experiment: [{ total_ms: 14100 }, { total_ms: 14088 }],
    baseline: [{ total_ms: 15200 }, { total_ms: 15190 }],
  },
  metrics: { total_ms: metricVerdict },
  primary: { kind: "geomean", deltaPct: -7.2 },
  outcome: "improved",
  targetReached: false,
};

const committedKeepRecord: KeepRecord = {
  type: "keep",
  seq: 1,
  at: AT,
  status: "committed",
  commit: COMMIT,
  message: "cache the regex",
  checks: { configured: true, passed: true },
};

const blockedKeepRecord: KeepRecord = {
  type: "keep",
  seq: 2,
  at: AT,
  status: "blocked",
  reason: "checks-failed",
  checks: { configured: true, passed: false },
};

const discardRecord: DiscardRecord = { type: "discard", seq: 3, at: AT };

const hookRecord: HookRecord = {
  type: "hook",
  stage: "before",
  seq: 4,
  exitCode: 0,
  durationMs: 120,
  stdoutBytes: 80,
  timedOut: false,
};

const finalizeRecord: FinalizeRecord = {
  type: "finalize",
  at: AT,
  branch: "gymrat/20260808-141530-a3f2-final",
  commit: COMMIT,
  message: "squash 3 kept iterations",
};

/** Copy of `record` without `key`, widened to `unknown` so `parseRecord` accepts it. */
function omitting<T extends object>(record: T, key: keyof T & string): unknown {
  const clone: Partial<T> = { ...record };
  delete clone[key];
  return clone;
}

/** Copy of `record` with `patch` merged over it, widened to `unknown`. */
function patching(record: object, patch: Record<string, unknown>): unknown {
  return { ...record, ...patch };
}

/** Matches an error message that names `field` as the failing location. */
function mentioning(field: string): RegExp {
  return new RegExp(`\\b${field.replaceAll(".", "\\.")}\\b`);
}

/** Registers a parametrized case asserting `parseRecord` rejects `value`, naming `field` in the error. */
function itRejectsNamingField(
  cases: { description: string; value: unknown; field: string }[],
): void {
  it.each(cases)("rejects $description", ({ value, field }) => {
    const act = (): SessionLogRecord => parseRecord(value);

    expect(act).toThrow(GymratError);
    expect(act).toThrow(mentioning(field));
  });
}

describe("parseRecord", () => {
  describe("when the record satisfies the schema for its type", () => {
    it.each([
      { description: "a session", record: sessionRecord },
      {
        description: "a session carrying the optional prepare and filter commands",
        record: patching(sessionRecord, {
          config: {
            ...sessionRecord.config,
            prepare: "npm run build",
            filter: "npm run bench -- --filter {names}",
          },
        }),
      },
      {
        description: "a session carrying the optional hook commands",
        record: patching(sessionRecord, {
          config: {
            ...sessionRecord.config,
            hooks: { before: "npm run warm-cache", after: "npm run cool-down" },
          },
        }),
      },
      { description: "a baseline", record: baselineRecord },
      { description: "an iteration", record: iterationRecord },
      {
        description: "an iteration whose metric verdict omits the optional statistics",
        record: patching(iterationRecord, {
          metrics: {
            total_ms: {
              deltaPct: -7.2,
              verdict: "improved",
              method: "band",
              gating: true,
              confirmed: false,
            },
          },
        }),
      },
      {
        description: "an iteration whose deltas a zero baseline median left undefined",
        record: patching(iterationRecord, {
          metrics: { total_ms: { ...metricVerdict, deltaPct: null, verdict: "no-signal" } },
          primary: { kind: "geomean", deltaPct: null },
          outcome: "no-signal",
        }),
      },
      {
        description: "an iteration that reran to confirm",
        record: patching(iterationRecord, {
          confirm: {
            ran: true,
            filtered: ["total_ms"],
            samples: { experiment: [{ total_ms: 14120 }], baseline: [{ total_ms: 15170 }] },
          },
        }),
      },
      { description: "a committed keep", record: committedKeepRecord },
      {
        description: "a committed keep without a message",
        record: omitting(committedKeepRecord, "message"),
      },
      { description: "a blocked keep", record: blockedKeepRecord },
      {
        description: "a keep blocked before anything was measured",
        record: patching(blockedKeepRecord, {
          seq: 0,
          reason: "nothing-measured",
          checks: { configured: false },
        }),
      },
      { description: "a discard", record: discardRecord },
      { description: "a hook", record: hookRecord },
      { description: "a finalize", record: finalizeRecord },
    ])("returns $description record unchanged", ({ record }) => {
      const result = parseRecord(record);

      expect(result).toStrictEqual(record);
    });
  });

  describe("when a required field is missing", () => {
    itRejectsNamingField([
      {
        description: "a session without its session id",
        value: omitting(sessionRecord, "sessionId"),
        field: "sessionId",
      },
      {
        description: "a session whose config omits the bench command",
        value: patching(sessionRecord, { config: omitting(sessionRecord.config, "bench") }),
        field: "config.bench",
      },
      {
        description: "a baseline without samples",
        value: omitting(baselineRecord, "samples"),
        field: "samples",
      },
      {
        description: "an iteration without metrics",
        value: omitting(iterationRecord, "metrics"),
        field: "metrics",
      },
      {
        description: "an iteration whose metric verdict drops its delta instead of nulling it",
        value: patching(iterationRecord, {
          metrics: { total_ms: omitting(metricVerdict, "deltaPct") },
        }),
        field: "metrics.total_ms.deltaPct",
      },
      {
        description: "an iteration whose primary drops its delta instead of nulling it",
        value: patching(iterationRecord, {
          primary: omitting(iterationRecord.primary, "deltaPct"),
        }),
        field: "primary.deltaPct",
      },
      {
        description: "a keep without a status",
        value: omitting(committedKeepRecord, "status"),
        field: "status",
      },
      {
        description: "a discard without a timestamp",
        value: omitting(discardRecord, "at"),
        field: "at",
      },
      {
        description: "a hook without an exit code",
        value: omitting(hookRecord, "exitCode"),
        field: "exitCode",
      },
      {
        description: "a finalize without the commit it squashed onto",
        value: omitting(finalizeRecord, "commit"),
        field: "commit",
      },
    ]);
  });

  describe("when a field violates its schema", () => {
    itRejectsNamingField([
      {
        description: "a session pinned to another schema version",
        value: patching(sessionRecord, { schemaVersion: 2 }),
        field: "schemaVersion",
      },
      {
        description: "a session whose baseline ref is not a string",
        value: patching(sessionRecord, { baseline: { ref: 42, sha: SHA } }),
        field: "baseline.ref",
      },
      {
        description: "a session whose sample count is fractional",
        value: patching(sessionRecord, {
          config: { ...sessionRecord.config, samples: 10.5 },
        }),
        field: "config.samples",
      },
      {
        description: "a session whose config snapshot holds the superseded string hooks value",
        value: patching(sessionRecord, {
          config: { ...sessionRecord.config, hooks: "gymrat.hooks" },
        }),
        field: "config.hooks",
      },
      {
        description: "a baseline whose samples hold non-numeric values",
        value: patching(baselineRecord, { samples: [{ total_ms: "15200" }] }),
        field: "samples.0.total_ms",
      },
      {
        description: "an iteration numbered below one",
        value: patching(iterationRecord, { seq: 0 }),
        field: "seq",
      },
      {
        description: "an iteration with an outcome outside the tri-state",
        value: patching(iterationRecord, { outcome: "unknown" }),
        field: "outcome",
      },
      {
        description: "an iteration whose metric verdict omits its gating flag",
        value: patching(iterationRecord, {
          metrics: { total_ms: omitting(metricVerdict, "gating") },
        }),
        field: "metrics.total_ms.gating",
      },
      {
        description: "an iteration whose target flag is not a boolean",
        value: patching(iterationRecord, { targetReached: "false" }),
        field: "targetReached",
      },
      {
        description: "a keep with an unrecognized status",
        value: patching(committedKeepRecord, { status: "pending" }),
        field: "status",
      },
      {
        description: "a keep blocked for an unrecognized reason",
        value: patching(blockedKeepRecord, { reason: "bored" }),
        field: "reason",
      },
      {
        description: "a keep numbered below zero",
        value: patching(committedKeepRecord, { seq: -1 }),
        field: "seq",
      },
      {
        description: "a discard numbered fractionally",
        value: patching(discardRecord, { seq: 1.5 }),
        field: "seq",
      },
      {
        description: "a hook run at an unrecognized stage",
        value: patching(hookRecord, { stage: "during" }),
        field: "stage",
      },
      {
        description: "a hook whose timeout flag is not a boolean",
        value: patching(hookRecord, { timedOut: 0 }),
        field: "timedOut",
      },
      {
        description: "a finalize whose branch is not a string",
        value: patching(finalizeRecord, { branch: 42 }),
        field: "branch",
      },
    ]);
  });

  describe("when the record carries a key its schema does not declare", () => {
    itRejectsNamingField([
      {
        description: "at the top level",
        value: patching(discardRecord, { note: "why not" }),
        field: "note",
      },
      {
        description: "inside a nested object",
        value: patching(sessionRecord, {
          config: { ...sessionRecord.config, retries: 3 },
        }),
        field: "config.retries",
      },
      {
        description: "inside a metric verdict",
        value: patching(iterationRecord, {
          metrics: { total_ms: { ...metricVerdict, band: 1.4 } },
        }),
        field: "metrics.total_ms.band",
      },
    ]);
  });

  describe("when the value cannot be matched to a record type", () => {
    it.each([
      { description: "null", value: null },
      { description: "a number", value: 42 },
      { description: "a string", value: "session" },
      { description: "an array", value: [sessionRecord] },
      {
        description: "an object with no type discriminator",
        value: omitting(discardRecord, "type"),
      },
      { description: "an object whose type is not a string", value: { type: 42 } },
      { description: "an object with an unknown type", value: { type: "banana" } },
    ])("rejects $description", ({ value }) => {
      const act = (): SessionLogRecord => parseRecord(value);

      expect(act).toThrow(GymratError);
    });

    it("names the unrecognized type in the message", () => {
      const act = (): SessionLogRecord => parseRecord({ type: "banana", seq: 1 });

      expect(act).toThrow(mentioning("banana"));
    });

    it("names finalize among the types it does recognize", () => {
      const error = captureGymratError(() => parseRecord({ type: "banana", seq: 1 }));

      expect(error.hint).toContain("finalize");
    });
  });

  describe("when the caller narrows the returned union", () => {
    it("pins the return type to the full record union", () => {
      const record = parseRecord(hookRecord);

      expectTypeOf(record).toEqualTypeOf<SessionLogRecord>();
    });
  });
});
