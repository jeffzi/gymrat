import { Type } from "@sinclair/typebox";
import type { TSchema } from "@sinclair/typebox";
import { describe, it, expect, expectTypeOf } from "vitest";

import type { SchemaIssue } from "../src/schema.js";
import { compile, describeKey, expected, parse } from "../src/schema.js";

function firstIssueOf(schema: TSchema, value: unknown): SchemaIssue {
  const issue = compile(schema).firstIssue(value);
  if (issue === undefined) {
    throw new Error(`no issue reported for ${JSON.stringify(value)}`);
  }
  return issue;
}

function describeIssue(issue: SchemaIssue): string {
  return `${issue.path} must be ${issue.expected}`;
}

const benchSchema = Type.Object({ bench: Type.String() });

describe("expected", () => {
  describe("when composed into a schema", () => {
    it.each([
      {
        description: "spread alongside other options",
        schema: Type.Integer({ ...expected("a positive integer"), minimum: 1 }),
        value: 0,
        phrase: "a positive integer",
      },
      {
        description: "passed directly as the options object",
        schema: Type.String(expected("a label")),
        value: 42,
        phrase: "a label",
      },
    ])("carries the phrase into the issue when $description", ({ schema, value, phrase }) => {
      const issue = firstIssueOf(schema, value);

      expect(issue.expected).toBe(phrase);
    });
  });
});

describe("describeKey", () => {
  describe("when the path names a key", () => {
    it.each([
      { description: "a top-level key", path: "bench" },
      { description: "a nested key", path: "hooks.before" },
      { description: "a key containing a slash", path: "decode/time" },
    ])("returns $description unchanged", ({ path }) => {
      const result = describeKey(path);

      expect(result).toBe(path);
    });
  });

  describe("when the path is the empty key", () => {
    it("quotes it so the reader can see a key is there and that it is empty", () => {
      const result = describeKey("");

      expect(result).toBe('""');
    });
  });

  describe("when the path ends in an empty key nested under a parent", () => {
    it("quotes the empty key so it does not read as a stray trailing dot", () => {
      const schema = Type.Object({ metrics: Type.Object({ "": Type.String() }) });
      const issue = firstIssueOf(schema, { metrics: { "": 1 } });

      const result = describeKey(issue.path);

      expect(result).toBe('metrics.""');
    });
  });
});

describe("compile", () => {
  describe("when the schema uses a construct the compiler cannot handle", () => {
    it("throws at compile time instead of deferring to validation time", () => {
      expect(() => compile(Type.Unsafe({}))).toThrow();
    });
  });
});

describe("Validator.check", () => {
  describe("when given a value to check", () => {
    it.each([
      { description: "satisfies the schema", value: { bench: "my-bench" }, verdict: true },
      { description: "violates the schema", value: { bench: 42 }, verdict: false },
    ])("returns $verdict when the value $description", ({ value, verdict }) => {
      const validator = compile(benchSchema);

      const result = validator.check(value);

      expect(result).toBe(verdict);
    });
  });

  describe("when used as a type guard", () => {
    it("narrows the value to the schema's static type", () => {
      const validator = compile(benchSchema);

      expectTypeOf(validator.check).guards.toEqualTypeOf<{ bench: string }>();
    });
  });
});

describe("Validator.firstIssue", () => {
  describe("when the value satisfies the schema", () => {
    it("returns undefined", () => {
      const validator = compile(benchSchema);

      const issue = validator.firstIssue({ bench: "my-bench" });

      expect(issue).toBeUndefined();
    });
  });

  describe("when several parts of the value fail", () => {
    it("describes only the first reported failure", () => {
      const schema = Type.Object({ bench: Type.String(), adapter: Type.String() });

      const issue = firstIssueOf(schema, { bench: 1, adapter: 2 });

      expect(issue.path).toBe("bench");
    });
  });

  describe("when reporting the failing location", () => {
    it.each([
      {
        description: "a nested property",
        schema: Type.Object({ metrics: Type.Object({ latency: Type.String() }) }),
        value: { metrics: { latency: 1 } },
        path: "metrics.latency",
      },
      {
        description: "a key containing a slash",
        schema: Type.Object({ "decode/time": Type.String() }),
        value: { "decode/time": 1 },
        path: "decode/time",
      },
      {
        description: "a key containing a tilde followed by a one",
        schema: Type.Object({ "metric~1": Type.String() }),
        value: { "metric~1": 1 },
        path: "metric~1",
      },
      {
        description: "the whole value",
        schema: Type.String(),
        value: 1,
        path: "",
      },
    ])("reports $description as the dotted path $path", ({ schema, value, path }) => {
      const issue = firstIssueOf(schema, value);

      expect(issue.path).toBe(path);
    });
  });

  describe("when classifying the failure", () => {
    it.each([
      {
        description: "a key the schema does not declare",
        schema: Type.Object({ bench: Type.String() }, { additionalProperties: false }),
        value: { bench: "my-bench", typo: 1 },
        kind: "unknown-key",
      },
      {
        description: "a missing required property",
        schema: benchSchema,
        value: {},
        kind: "invalid-value",
      },
    ])("classifies $description as $kind", ({ schema, value, kind }) => {
      const issue = firstIssueOf(schema, value);

      expect(issue.kind).toBe(kind);
    });
  });

  describe("when the failing sub-schema carries no phrase", () => {
    it("falls back to the underlying validator's own message", () => {
      const issue = firstIssueOf(benchSchema, { bench: 1 });

      expect(issue.expected).toBe(issue.error.message);
    });
  });

  describe("when the failure is nested inside the value", () => {
    it("exposes the value at the failing location rather than the whole input", () => {
      const schema = Type.Object({ metrics: Type.Object({ latency: Type.String() }) });

      const issue = firstIssueOf(schema, { metrics: { latency: 42 } });

      expect(issue.value).toBe(42);
    });
  });

  describe("when the caller needs more than the flattened fields", () => {
    it("retains the raw error with its unconverted JSON-Pointer path", () => {
      const schema = Type.Object({ "decode/time": Type.String() });

      const issue = firstIssueOf(schema, { "decode/time": 1 });

      expect(issue.error.path).toBe("/decode~1time");
    });
  });
});

describe("parse", () => {
  describe("when the value satisfies the schema", () => {
    it("returns the caller's own object typed as the schema's static type", () => {
      const validator = compile(benchSchema);
      const value = { bench: "my-bench" };

      const result = parse(validator, value, describeIssue);

      expect(result).toBe(value);
      expectTypeOf(result).toEqualTypeOf<{ bench: string }>();
    });
  });

  describe("when the value violates the schema", () => {
    it("throws an error worded by the caller from the first issue", () => {
      const validator = compile(
        Type.Object({ samples: Type.Integer({ ...expected("a positive integer"), minimum: 1 }) }),
      );

      expect(() => parse(validator, { samples: 0 }, describeIssue)).toThrow(
        "samples must be a positive integer",
      );
    });

    it("chains the issue as the error's cause so a catch site can still reach it", () => {
      const validator = compile(benchSchema);

      let cause: unknown;
      try {
        parse(validator, { bench: 1 }, describeIssue);
      } catch (error) {
        cause = error instanceof Error ? error.cause : undefined;
      }

      expect(cause).toMatchObject({ path: "bench", kind: "invalid-value" });
    });
  });
});
