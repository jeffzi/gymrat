import type { SchemaOptions, Static, TSchema } from "@sinclair/typebox";
import { TypeCompiler } from "@sinclair/typebox/compiler";
import { ValueErrorType, type ValueError } from "@sinclair/typebox/errors";

import { GymratError } from "./errors.js";

/**
 * Schema options carrying the phrase a caller wants to see in its own error message.
 *
 * Compose it into a schema literal by spreading it alongside other options, or pass it
 * as the whole options object:
 *
 * ```ts
 * Type.Integer({ ...expected("a sample count"), minimum: 1 })
 * Type.String(expected("a metric name"))
 * ```
 *
 * The phrase travels on the schema it is written next to, so a failure deep inside a
 * nested object reports the phrase belonging to the field that failed. The exception is
 * an undeclared key: no sub-schema exists for it, so the phrase reported is the
 * containing object's own. Classify on {@link SchemaIssue.kind} before reading the
 * phrase.
 *
 * The return type is narrowed rather than the full `SchemaOptions`, whose
 * `[prop: string]: any` index signature would otherwise reach every caller.
 */
export function expected(phrase: string): Pick<SchemaOptions, "description"> {
  return { description: phrase };
}

/** Shared options for object schemas: rejects non-objects and disallows unknown keys. */
export const strictObjectOptions = { ...expected("an object"), additionalProperties: false };

/**
 * Shared options for the record schemas whose keys are names an adapter supplies —
 * metric names and kind names.
 *
 * `Type.Record(Type.String(), …)` compiles to `patternProperties` with `^(.*)$`, and
 * neither `.` nor an unanchored `$` spans a line terminator — so a key containing one
 * matches no pattern at all. Without `additionalProperties: false` such a key would be
 * an unconstrained extra property, admitting its entry unchecked; with it, the key is
 * rejected outright.
 */
export const nameKeyedRecordOptions = { ...expected("an object"), additionalProperties: false };

/** A single validation failure, described in terms a caller can word a message from. */
export interface SchemaIssue {
  /**
   * Dotted path to the failing location; empty when the whole value is wrong.
   *
   * Optimized for reading, not for round-tripping: a key that itself contains a `.`
   * is indistinguishable from a nesting step. Callers needing an unambiguous path
   * read {@link SchemaIssue.error}'s JSON Pointer instead.
   */
  path: string;
  /**
   * Whether the value carries a key the schema does not declare, or a bad value.
   *
   * A *required* key that is absent is a bad value, not an unknown key: it reports
   * `"invalid-value"` with a `value` of `undefined` at the missing key's path.
   */
  kind: "unknown-key" | "invalid-value";
  /** The phrase from {@link expected}, or the underlying validator's message. */
  expected: string;
  /** The value at the failing location. */
  value: unknown;
  /**
   * The raw underlying error, kept as an escape hatch. Callers needing more than the
   * fields above — nested union-variant errors, for instance — read it from here rather
   * than growing this interface.
   */
  error: ValueError;
}

/** A schema compiled once, ready to validate many values. */
export interface Validator<T extends TSchema> {
  check: (value: unknown) => value is Static<T>;
  firstIssue: (value: unknown) => SchemaIssue | undefined;
}

/**
 * Convert a JSON Pointer (RFC 6901) to a dotted path, undoing the pointer escaping.
 *
 * Escapes are undone per segment and in the order the RFC mandates: `~1` before `~0`,
 * so a key holding the literal text `~1` (escaped as `~01`) survives the round trip
 * instead of collapsing into a slash.
 */
function toDottedPath(pointer: string): string {
  return pointer
    .slice(1)
    .split("/")
    .map((segment) => segment.replaceAll("~1", "/").replaceAll("~0", "~"))
    .join(".");
}

function toIssue(error: ValueError): SchemaIssue {
  return {
    path: toDottedPath(error.path),
    kind:
      error.type === ValueErrorType.ObjectAdditionalProperties ? "unknown-key" : "invalid-value",
    expected: error.schema.description ?? error.message,
    value: error.value,
    error,
  };
}

/**
 * Compile a schema into a reusable validator.
 *
 * Compilation happens here, once, rather than on each validation: a schema built from a
 * construct the compiler cannot handle fails at startup instead of on the first value it
 * is asked to check.
 */
export function compile<T extends TSchema>(schema: T): Validator<T> {
  const compiled = TypeCompiler.Compile(schema);
  return {
    check: (value: unknown): value is Static<T> => compiled.Check(value),
    firstIssue: (value: unknown): SchemaIssue | undefined => {
      const error = compiled.Errors(value).First();
      return error === undefined ? undefined : toIssue(error);
    },
  };
}

/**
 * Validate a value, or throw an error worded by the caller.
 *
 * `toMessage` receives the first issue and owns every word the user reads — this module
 * supplies structure, never phrasing. The issue is chained as the thrown error's `cause`
 * so a catch site further up can still reach the path and the raw `ValueError`.
 *
 * Returns the caller's own reference, not a copy: a caller that has narrowed a parsed
 * object can hand it straight on without rebuilding it field by field.
 */
export function parse<T extends TSchema>(
  validator: Validator<T>,
  value: unknown,
  toMessage: (issue: SchemaIssue) => string,
): Static<T> {
  if (validator.check(value)) {
    return value;
  }
  // `Check` runs the compiled function while `Errors` walks the schema interpretively,
  // so the two are separate traversals; TypeBox keeps them in agreement, and a failing
  // check therefore always yields an error. Asserting it beats a fallback branch: this
  // module owns no wording, so it has nothing to say if the impossible happens.
  const issue = validator.firstIssue(value)!;
  throw new GymratError(toMessage(issue), undefined, { cause: issue });
}
