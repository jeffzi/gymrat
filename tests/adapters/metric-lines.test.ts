import { afterEach, describe, expect, it, vi } from "vitest";

import metricLinesAdapter from "../../src/adapters/metric-lines.js";
import { AdapterError } from "../../src/adapters/types.js";
import { captureStderr } from "../fixtures/console.js";
import { metricRecord } from "../fixtures/metrics.js";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("metric-lines adapter", () => {
  describe("parse()", () => {
    describe("basic parsing", () => {
      it.each([
        ["integer", "METRIC foo=42", metricRecord({ foo: 42 })],
        ["decimal", "METRIC bar=3.14", metricRecord({ bar: 3.14 })],
        ["leading whitespace", "  METRIC foo=42", metricRecord({ foo: 42 })],
        ["trailing whitespace", "METRIC foo=42  ", metricRecord({ foo: 42 })],
        ["leading and trailing whitespace", "  METRIC foo=42  ", metricRecord({ foo: 42 })],
      ])("parses METRIC line with %s", (_, stdout, expected) => {
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(expected);
      });
    });

    describe("last-= split", () => {
      it.each([
        ["two equals", "METRIC k=v=3.14", metricRecord({ "k=v": 3.14 })],
        ["multiple equals", "METRIC a=b=c=d=5", metricRecord({ "a=b=c=d": 5 })],
      ])("splits at LAST = with %s", (_, stdout, expected) => {
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(expected);
      });
    });

    describe("number grammar", () => {
      it.each([
        ["negative integer", "METRIC val=-12", metricRecord({ val: -12 })],
        ["negative decimal", "METRIC val=-3.14", metricRecord({ val: -3.14 })],
        ["positive sign", "METRIC val=+5", metricRecord({ val: 5 })],
        ["positive decimal with sign", "METRIC val=+5.0", metricRecord({ val: 5.0 })],
        ["small decimal", "METRIC val=0.001", metricRecord({ val: 0.001 })],
        ["scientific notation (negative exponent)", "METRIC val=1e-9", metricRecord({ val: 1e-9 })],
        ["scientific notation (positive exponent)", "METRIC val=1e9", metricRecord({ val: 1e9 })],
        ["scientific notation (uppercase E)", "METRIC val=1E-9", metricRecord({ val: 1e-9 })],
      ])("parses %s", (_, stdout, expected) => {
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(expected);
      });
    });

    describe("ignore non-matching lines", () => {
      it("ignores lines without METRIC prefix", () => {
        const stdout = "some other output\nMETRIC valid=1\nother log line";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ valid: 1 }));
      });

      it("ignores lines that don't start with METRIC (case-sensitive)", () => {
        const stdout = "metric foo=42\nMetric bar=3.14\nMETRIC valid=1";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ valid: 1 }));
      });

      it("requires a space after METRIC — METRICfoo=42 is not a metric line", () => {
        const stdout = "METRICfoo=42\nMETRIC valid=1";
        captureStderr(() => {
          const result = metricLinesAdapter.parse(stdout);
          expect(result).toStrictEqual(metricRecord({ valid: 1 }));
        });
      });

      it("extracts METRIC lines from mixed output", () => {
        const stdout = "Starting benchmark...\nMETRIC foo=42\nRunning test\nMETRIC bar=3.14\nDone";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ foo: 42, bar: 3.14 }));
      });
    });

    describe("near-miss METRIC prefix warning", () => {
      it.each([
        { description: "no space after METRIC", offending: "METRICfoo=42" },
        { description: "a longer word starting with METRIC", offending: "METRICS foo=1" },
        { description: "an underscore instead of a space", offending: "METRIC_foo=1" },
      ])("warns about a line with $description", ({ offending }) => {
        const stderr = captureStderr(() => {
          metricLinesAdapter.parse(`${offending}\nMETRIC valid=1`);
        });

        expect(stderr).toContain(`Failed to parse METRIC line: ${offending}`);
      });

      it.each(["some other output", "Starting benchmark...", "metric foo=42"])(
        "stays silent for the unrelated line %s",
        (unrelated) => {
          const stderr = captureStderr(() => {
            metricLinesAdapter.parse(`${unrelated}\nMETRIC valid=1`);
          });

          expect(stderr).toBe("");
        },
      );
    });

    describe("malformed METRIC warning", () => {
      it.each([
        { description: "without value", offending: "METRIC foo" },
        { description: "with only =", offending: "METRIC =5" },
        { description: "with non-numeric value", offending: "METRIC foo=bar" },
        { description: "with empty value", offending: "METRIC foo=" },
        { description: "NaN value", offending: "METRIC foo=NaN" },
        { description: "Infinity value", offending: "METRIC foo=Infinity" },
        { description: "negative Infinity value", offending: "METRIC foo=-Infinity" },
      ])("warning names the offending line for METRIC $description", ({ offending }) => {
        const stderr = captureStderr(() => {
          metricLinesAdapter.parse(`${offending}\nMETRIC valid=1`);
        });

        expect(stderr).toContain(`Failed to parse METRIC line: ${offending}`);
      });

      it.each([
        ["empty value", "METRIC x=1\nMETRIC x=\nMETRIC x=3"],
        ["whitespace-only value", "METRIC x=1\nMETRIC x=   \nMETRIC x=3"],
      ])("excludes a %s from the metric's samples instead of reading it as zero", (_, stdout) => {
        captureStderr(() => {
          const result = metricLinesAdapter.parse(stdout);
          expect(result).toStrictEqual(metricRecord({ x: 2 }));
        });
      });

      it("continues parsing after malformed line", () => {
        const stdout = "METRIC foo=bar\nMETRIC valid=42";
        captureStderr(() => {
          const result = metricLinesAdapter.parse(stdout);
          expect(result).toStrictEqual(metricRecord({ valid: 42 }));
        });
      });
    });

    describe("injected warning sink", () => {
      it("hands the warning to the sink and leaves stderr untouched", () => {
        const warnings: string[] = [];

        const stderr = captureStderr(() => {
          metricLinesAdapter.parse("METRIC foo=bar\nMETRIC valid=1", (message) => {
            warnings.push(message);
          });
        });

        expect.soft(warnings).toStrictEqual(["Failed to parse METRIC line: METRIC foo=bar"]);
        expect(stderr).toBe("");
      });
    });

    describe("repeated metric name → median", () => {
      it("returns median for odd count of values", () => {
        const stdout = "METRIC x=1\nMETRIC x=3\nMETRIC x=2";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ x: 2 }));
      });

      it("returns average of two middle values for even count", () => {
        const stdout = "METRIC x=1\nMETRIC x=2\nMETRIC x=3\nMETRIC x=4";
        const result = metricLinesAdapter.parse(stdout);
        // sorted: [1, 2, 3, 4], median = (2 + 3) / 2 = 2.5
        expect(result).toStrictEqual(metricRecord({ x: 2.5 }));
      });

      it("returns single value for one occurrence", () => {
        const stdout = "METRIC x=42";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ x: 42 }));
      });

      it("computes median separately for each metric name", () => {
        const stdout = "METRIC x=1\nMETRIC x=3\nMETRIC y=10\nMETRIC y=20\nMETRIC y=30";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ x: 2, y: 20 }));
      });

      it("handles repeated metrics with decimal values", () => {
        const stdout = "METRIC x=1.5\nMETRIC x=2.5\nMETRIC x=3.5";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ x: 2.5 }));
      });
    });

    describe("metric names that collide with Object.prototype", () => {
      it.each(["toString", "constructor", "valueOf", "hasOwnProperty"])(
        "leaves %s absent when the bench never emitted it",
        (inherited) => {
          const result = metricLinesAdapter.parse("METRIC foo=42");

          expect(result[inherited]).toBeUndefined();
        },
      );

      it("parses a metric named __proto__ as an ordinary metric", () => {
        const result = metricLinesAdapter.parse("METRIC __proto__=1\nMETRIC __proto__=3");

        expect(Object.entries(result)).toStrictEqual([["__proto__", 2]]);
      });
    });

    describe("zero metrics → AdapterError", () => {
      it.each([
        ["no valid METRIC lines found", "some output\nwith no metrics"],
        ["empty string", ""],
        ["only malformed METRIC lines present", "METRIC foo\nMETRIC bar=baz"],
      ])("throws AdapterError when %s", (_description, stdout) => {
        captureStderr(() => {
          const parse = () => metricLinesAdapter.parse(stdout);
          expect(parse).toThrow(AdapterError);
          expect(parse).toThrow(/^No valid METRIC lines found$/);
        });
      });
    });
  });

  describe("defaults()", () => {
    it.each(["foo", "latency", "memory", "custom_metric"])(
      "returns direction: lower for metric %s",
      (metricName) => {
        const result = metricLinesAdapter.defaults(metricName);
        expect(result).toStrictEqual({ direction: "lower" });
      },
    );
  });
});
