import metricLinesAdapter from "../../src/adapters/metric-lines.js";

describe("metric-lines adapter", () => {
  describe("parse()", () => {
    describe("basic parsing", () => {
      it.each([
        ["integer", "METRIC foo=42", { foo: 42 }],
        ["decimal", "METRIC bar=3.14", { bar: 3.14 }],
        ["leading whitespace", "  METRIC foo=42", { foo: 42 }],
        ["trailing whitespace", "METRIC foo=42  ", { foo: 42 }],
        ["leading and trailing whitespace", "  METRIC foo=42  ", { foo: 42 }],
      ])("parses METRIC line with %s", (_, stdout, expected) => {
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(expected);
      });
    });

    describe("last-= split", () => {
      it.each([
        ["two equals", "METRIC k=v=3.14", { "k=v": 3.14 }],
        ["multiple equals", "METRIC a=b=c=d=5", { "a=b=c=d": 5 }],
      ])("splits at LAST = with %s", (_, stdout, expected) => {
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(expected);
      });
    });

    describe("number grammar", () => {
      it.each([
        ["negative integer", "METRIC val=-12", { val: -12 }],
        ["negative decimal", "METRIC val=-3.14", { val: -3.14 }],
        ["positive sign", "METRIC val=+5", { val: 5 }],
        ["positive decimal with sign", "METRIC val=+5.0", { val: 5.0 }],
        ["small decimal", "METRIC val=0.001", { val: 0.001 }],
        ["scientific notation (negative exponent)", "METRIC val=1e-9", { val: 1e-9 }],
        ["scientific notation (positive exponent)", "METRIC val=1e9", { val: 1e9 }],
        ["scientific notation (uppercase E)", "METRIC val=1E-9", { val: 1e-9 }],
      ])("parses %s", (_, stdout, expected) => {
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual(expected);
      });
    });

    describe("ignore non-matching lines", () => {
      it("ignores lines without METRIC prefix", () => {
        const stdout = "some other output\nMETRIC valid=1\nother log line";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual({ valid: 1 });
      });

      it("ignores lines that don't start with METRIC (case-sensitive)", () => {
        const stdout = "metric foo=42\nMetric bar=3.14\nMETRIC valid=1";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual({ valid: 1 });
      });

      it("extracts METRIC lines from mixed output", () => {
        const stdout = "Starting benchmark...\nMETRIC foo=42\nRunning test\nMETRIC bar=3.14\nDone";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual({ foo: 42, bar: 3.14 });
      });
    });

    describe("malformed METRIC warning", () => {
      it.each([
        ["without value", "METRIC foo\nMETRIC valid=1", /METRIC/],
        ["with only =", "METRIC =5\nMETRIC valid=1", /METRIC/],
        ["with non-numeric value", "METRIC foo=bar\nMETRIC valid=1", /METRIC/],
        ["NaN value", "METRIC foo=NaN\nMETRIC valid=1", /METRIC/],
        ["Infinity value", "METRIC foo=Infinity\nMETRIC valid=1", /METRIC/],
        ["negative Infinity value", "METRIC foo=-Infinity\nMETRIC valid=1", /METRIC/],
      ])("emits warning for METRIC %s", (_, stdout, pattern) => {
        const stderr = captureStderr(() => {
          metricLinesAdapter.parse(stdout);
        });
        expect(stderr).toMatch(pattern);
      });

      it.each([
        ["without value", "METRIC foo\nMETRIC valid=1", /foo/],
        ["with non-numeric value", "METRIC foo=bar\nMETRIC valid=1", /foo=bar/],
      ])("warning for METRIC %s includes the problematic input", (_, stdout, pattern) => {
        const stderr = captureStderr(() => {
          metricLinesAdapter.parse(stdout);
        });
        expect(stderr).toMatch(pattern);
      });

      it("continues parsing after malformed line", () => {
        const stdout = "METRIC foo=bar\nMETRIC valid=42";
        captureStderr(() => {
          const result = metricLinesAdapter.parse(stdout);
          expect(result).toStrictEqual({ valid: 42 });
        });
      });
    });

    describe("repeated metric name → median", () => {
      it("returns median for odd count of values", () => {
        const stdout = "METRIC x=1\nMETRIC x=3\nMETRIC x=2";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual({ x: 2 });
      });

      it("returns average of two middle values for even count", () => {
        const stdout = "METRIC x=1\nMETRIC x=2\nMETRIC x=3\nMETRIC x=4";
        const result = metricLinesAdapter.parse(stdout);
        // sorted: [1, 2, 3, 4], median = (2 + 3) / 2 = 2.5
        expect(result).toStrictEqual({ x: 2.5 });
      });

      it("returns single value for one occurrence", () => {
        const stdout = "METRIC x=42";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual({ x: 42 });
      });

      it("computes median separately for each metric name", () => {
        const stdout = "METRIC x=1\nMETRIC x=3\nMETRIC y=10\nMETRIC y=20\nMETRIC y=30";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual({ x: 2, y: 20 });
      });

      it("handles repeated metrics with decimal values", () => {
        const stdout = "METRIC x=1.5\nMETRIC x=2.5\nMETRIC x=3.5";
        const result = metricLinesAdapter.parse(stdout);
        expect(result).toStrictEqual({ x: 2.5 });
      });
    });

    describe("zero metrics → AdapterError", () => {
      it("throws AdapterError when no valid METRIC lines found", () => {
        const stdout = "some output\nwith no metrics";
        expect(() => metricLinesAdapter.parse(stdout)).toThrow("AdapterError");
      });

      it("throws AdapterError for empty string", () => {
        expect(() => metricLinesAdapter.parse("")).toThrow("AdapterError");
      });

      it("throws AdapterError when only malformed METRIC lines present", () => {
        const stdout = "METRIC foo\nMETRIC bar=baz";
        expect(() => metricLinesAdapter.parse(stdout)).toThrow("AdapterError");
      });
    });
  });

  describe("defaults()", () => {
    it.each(["foo", "latency", "memory", "custom_metric"])(
      "returns direction: lower for metric %s",
      (metricName) => {
        const result = metricLinesAdapter.defaults(metricName);
        expect(result).toStrictEqual({ direction: "lower" });
        expect(result).not.toHaveProperty("unit");
      },
    );
  });
});

// Helper to capture stderr
function captureStderr(fn: () => void): string {
  const originalWarn = console.warn;
  let capturedOutput = "";

  console.warn = (...args: unknown[]) => {
    capturedOutput += args.join(" ");
  };

  try {
    fn();
  } finally {
    console.warn = originalWarn;
  }

  return capturedOutput;
}
