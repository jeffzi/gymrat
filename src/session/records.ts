import { Type } from "@sinclair/typebox";
import type { Static, TSchema } from "@sinclair/typebox";

import { GymratError } from "../errors.js";
import {
  compile,
  describeKey,
  expected,
  nameKeyedRecordOptions,
  parse,
  type SchemaIssue,
  strictObjectOptions,
} from "../schema.js";

/** Version of the session JSONL format these schemas describe. */
const SCHEMA_VERSION = 1;

const stringSchema = Type.String(expected("a string"));
const numberSchema = Type.Number(expected("a number"));
const booleanSchema = Type.Boolean(expected("a boolean"));
const positiveIntegerSchema = Type.Integer({ ...expected("a positive integer"), minimum: 1 });

/**
 * A percentage a figure moved by, or `null` where the ratio has no value.
 *
 * A baseline median of zero leaves the ratio nothing to normalize against, so the
 * engine computes `NaN` — a defined answer to a degenerate input, never an error.
 * `JSON.stringify` writes `NaN` as `null`, so `null` is the form such a delta
 * reaches the log in, and reading it back must not brick the session.
 *
 * Required all the same: a writer that drops the key altogether has lost the
 * figure rather than measured an undefined one, and is still a broken writer.
 */
const deltaPctSchema = Type.Union([numberSchema, Type.Null()], expected("a number or null"));

/**
 * Rounds of raw samples, in the shape an adapter emits: one flat name → value map per round.
 *
 * Raw values are stored rather than summary statistics so a later statistics change is a
 * rendering change, not a schema break.
 */
const sampleRoundsSchema = Type.Array(
  Type.Record(Type.String(), numberSchema, nameKeyedRecordOptions),
  expected("an array of objects mapping metric names to numbers"),
);

/** The hook commands a session was started with; a stage left out ran nothing. */
const nonEmptyStringSchema = Type.String({ ...expected("a non-empty string"), minLength: 1 });
const sessionHooksSchema = Type.Object(
  { before: Type.Optional(nonEmptyStringSchema), after: Type.Optional(nonEmptyStringSchema) },
  strictObjectOptions,
);

const sessionConfigSchema = Type.Object(
  {
    bench: stringSchema,
    prepare: Type.Optional(stringSchema),
    adapter: stringSchema,
    samples: positiveIntegerSchema,
    timeoutSeconds: positiveIntegerSchema,
    primary: stringSchema,
    filter: Type.Optional(stringSchema),
    hooks: Type.Optional(sessionHooksSchema),
  },
  strictObjectOptions,
);

const sessionRecordSchema = Type.Object(
  {
    type: Type.Literal("session"),
    schemaVersion: Type.Literal(SCHEMA_VERSION, expected(String(SCHEMA_VERSION))),
    sessionId: stringSchema,
    createdAt: stringSchema,
    baseline: Type.Object({ ref: stringSchema, sha: stringSchema }, strictObjectOptions),
    branch: stringSchema,
    worktrees: Type.Object(
      { experiment: stringSchema, baseline: stringSchema },
      strictObjectOptions,
    ),
    config: sessionConfigSchema,
  },
  strictObjectOptions,
);

const baselineRecordSchema = Type.Object(
  {
    type: Type.Literal("baseline"),
    at: stringSchema,
    label: stringSchema,
    samples: sampleRoundsSchema,
  },
  strictObjectOptions,
);

const metricVerdictSchema = Type.Object(
  {
    deltaPct: deltaPctSchema,
    // The value set `MetricVerdict["verdict"]` admits — see `Verdict` and
    // `ApproximateVerdictValue` in verdict/verdict.ts.
    verdict: Type.Union(
      [
        Type.Literal("improved"),
        Type.Literal("regressed"),
        Type.Literal("no-signal"),
        Type.Literal("unstable"),
      ],
      expected(`"improved", "regressed", "no-signal" or "unstable"`),
    ),
    // `MetricVerdict["method"]` in verdict/verdict.ts.
    method: Type.Union(
      [Type.Literal("signed-rank"), Type.Literal("band"), Type.Literal("exact")],
      expected(`"signed-rank", "band" or "exact"`),
    ),
    p: Type.Optional(numberSchema),
    noisePct: Type.Optional(numberSchema),
    gating: booleanSchema,
    confirmed: booleanSchema,
  },
  strictObjectOptions,
);

const pairedSamplesSchema = Type.Object(
  { experiment: sampleRoundsSchema, baseline: sampleRoundsSchema },
  strictObjectOptions,
);

const iterationRecordSchema = Type.Object(
  {
    type: Type.Literal("iteration"),
    seq: positiveIntegerSchema,
    at: stringSchema,
    samples: pairedSamplesSchema,
    metrics: Type.Record(Type.String(), metricVerdictSchema, nameKeyedRecordOptions),
    confirm: Type.Optional(
      Type.Object(
        {
          ran: booleanSchema,
          filtered: Type.Array(stringSchema, expected("an array of strings")),
          // Present only when the rerun skipped something it was asked about,
          // so a log written before the field existed still parses.
          absent: Type.Optional(Type.Array(stringSchema, expected("an array of strings"))),
          samples: pairedSamplesSchema,
        },
        strictObjectOptions,
      ),
    ),
    primary: Type.Object(
      // A geomean primary is nameless — it aggregates every gating metric — so
      // the name only exists on the variant that names one.
      {
        // `LoopPrimary["kind"]` in report/loop.ts.
        kind: Type.Union(
          [Type.Literal("geomean"), Type.Literal("metric")],
          expected(`"geomean" or "metric"`),
        ),
        name: Type.Optional(stringSchema),
        deltaPct: deltaPctSchema,
      },
      strictObjectOptions,
    ),
    outcome: Type.Union(
      [Type.Literal("improved"), Type.Literal("regressed"), Type.Literal("no-signal")],
      expected(`"improved", "regressed" or "no-signal"`),
    ),
    targetReached: booleanSchema,
  },
  strictObjectOptions,
);

/**
 * Counter shared by the records that settle an iteration.
 *
 * A keep blocked before anything was measured, and the hooks that run around the first
 * iteration, carry `0` — no iteration has been minted yet.
 */
const settledSeqSchema = Type.Integer({ ...expected("a non-negative integer"), minimum: 0 });

const byteCountSchema = Type.Integer({ ...expected("a non-negative integer"), minimum: 0 });

const keepRecordSchema = Type.Object(
  {
    type: Type.Literal("keep"),
    seq: settledSeqSchema,
    at: stringSchema,
    status: Type.Union(
      [Type.Literal("committed"), Type.Literal("blocked")],
      expected(`"committed" or "blocked"`),
    ),
    commit: Type.Optional(stringSchema),
    message: Type.Optional(stringSchema),
    reason: Type.Optional(
      Type.Union(
        [
          Type.Literal("checks-failed"),
          Type.Literal("gating-regression"),
          Type.Literal("nothing-measured"),
        ],
        expected(`"checks-failed", "gating-regression" or "nothing-measured"`),
      ),
    ),
    checks: Type.Object(
      {
        configured: booleanSchema,
        passed: Type.Optional(booleanSchema),
        // Carried by a keep the checks blocked, whose report relays the command's
        // output cut to the relay limit: the counts are what the command wrote,
        // so a reader of the log can tell the relay was cut.
        stdoutBytes: Type.Optional(byteCountSchema),
        stderrBytes: Type.Optional(byteCountSchema),
      },
      strictObjectOptions,
    ),
  },
  strictObjectOptions,
);

const discardRecordSchema = Type.Object(
  { type: Type.Literal("discard"), seq: settledSeqSchema, at: stringSchema },
  strictObjectOptions,
);

const hookRecordSchema = Type.Object(
  {
    type: Type.Literal("hook"),
    stage: Type.Union(
      [Type.Literal("before"), Type.Literal("after")],
      expected(`"before" or "after"`),
    ),
    seq: settledSeqSchema,
    exitCode: Type.Integer(expected("an integer")),
    durationMs: numberSchema,
    stdoutBytes: Type.Integer(expected("an integer")),
    timedOut: booleanSchema,
  },
  strictObjectOptions,
);

const finalizeRecordSchema = Type.Object(
  {
    type: Type.Literal("finalize"),
    at: stringSchema,
    branch: stringSchema,
    commit: stringSchema,
    message: stringSchema,
  },
  strictObjectOptions,
);

/** Opens a session log: the session's identity, worktrees, and a config snapshot for provenance. */
export type SessionRecord = Static<typeof sessionRecordSchema>;
/** A labelled set of baseline samples, written by `measure --record`. */
export type BaselineRecord = Static<typeof baselineRecordSchema>;
/** One measured edit: its raw samples, per-metric verdicts, and the outcome the agent acts on. */
export type IterationRecord = Static<typeof iterationRecordSchema>;
/** The settlement of an iteration: committed, or blocked with the reason it was refused. */
export type KeepRecord = Static<typeof keepRecordSchema>;
/** The reverted settlement of an iteration. */
export type DiscardRecord = Static<typeof discardRecordSchema>;
/** One hook invocation around an iteration. */
export type HookRecord = Static<typeof hookRecordSchema>;
/** Closes a session: the branch and squash commit its kept work was collapsed onto. */
export type FinalizeRecord = Static<typeof finalizeRecordSchema>;

/** Any line of a session log, discriminated on `type`. */
export type SessionLogRecord =
  | SessionRecord
  | BaselineRecord
  | IterationRecord
  | KeepRecord
  | DiscardRecord
  | HookRecord
  | FinalizeRecord;

/**
 * Word a schema failure as a session-record error.
 *
 * Unknown keys must be caught before the last branch: no sub-schema exists for a key the
 * schema never declared, so the phrase they carry is the containing object's `an object`.
 */
function recordMessage(issue: SchemaIssue): string {
  if (issue.kind === "unknown-key") {
    return `Unknown session record key: ${describeKey(issue.path)}`;
  }
  return `Invalid session record value for ${issue.path}: expected ${issue.expected}, got ${JSON.stringify(issue.value)}`;
}

/** Compile `schema` once, and word every failure it later finds as a record error. */
function parserFor<T extends TSchema>(schema: T): (value: unknown) => Static<T> {
  const validator = compile(schema);
  return (value) => parse(validator, value, recordMessage);
}

/**
 * One parser per `type` a session log line can carry.
 *
 * Keyed rather than switched so a record type added to {@link SessionLogRecord}
 * without a schema is a missing-key compile error, not a runtime fall-through to
 * the unknown-type throw.
 */
const recordParsers: Record<SessionLogRecord["type"], (value: unknown) => SessionLogRecord> = {
  session: parserFor(sessionRecordSchema),
  baseline: parserFor(baselineRecordSchema),
  iteration: parserFor(iterationRecordSchema),
  keep: parserFor(keepRecordSchema),
  discard: parserFor(discardRecordSchema),
  hook: parserFor(hookRecordSchema),
  finalize: parserFor(finalizeRecordSchema),
};

/** Whether `type` is one the log declares, and therefore a key {@link recordParsers} holds. */
function isRecordType(type: string): type is SessionLogRecord["type"] {
  return Object.hasOwn(recordParsers, type);
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Validate one parsed session-log line against the schema for its `type`.
 *
 * Returns the caller's own object, narrowed to the matching record type. Throws a
 * {@link GymratError} when the value carries no recognized `type`, or when it violates
 * that type's schema.
 */
export function parseRecord(value: unknown): SessionLogRecord {
  if (!isJsonObject(value)) {
    throw new GymratError(
      `Invalid session record: expected a JSON object, got ${JSON.stringify(value)}`,
    );
  }
  const type: unknown = value["type"];
  if (typeof type === "string" && isRecordType(type)) {
    return recordParsers[type](value);
  }
  throw new GymratError(
    `Unknown session record type: ${JSON.stringify(type)}`,
    // Read off the registry so a record type added to `recordParsers` cannot
    // leave the hint naming a set the parser no longer accepts.
    `Expected one of: ${Object.keys(recordParsers).join(", ")}.`,
  );
}
