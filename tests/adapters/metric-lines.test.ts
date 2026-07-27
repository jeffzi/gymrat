import metricLinesAdapter from "../../src/adapters/metric-lines.js";
import { AdapterError } from "../../src/adapters/types.js";

afterEach(() => {
  vi.restoreAllMocks();
});

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
        { description: "without value", offending: "METRIC foo" },
        { description: "with only =", offending: "METRIC =5" },
        { description: "with non-numeric value", offending: "METRIC foo=bar" },
        { description: "NaN value", offending: "METRIC foo=NaN" },
        { description: "Infinity value", offending: "METRIC foo=Infinity" },
        { description: "negative Infinity value", offending: "METRIC foo=-Infinity" },
      ])("warning names the offending line for METRIC $description", ({ offending }) => {
        const stderr = captureStderr(() => {
          metricLinesAdapter.parse(`${offending}\nMETRIC valid=1`);
        });

        expect(stderr).toContain(`Failed to parse METRIC line: ${offending}`);
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
      it.each([
        ["no valid METRIC lines found", "some output\nwith no metrics"],
        ["empty string", ""],
        ["only malformed METRIC lines present", "METRIC foo\nMETRIC bar=baz"],
      ])("throws AdapterError when %s", (_description, stdout) => {
        const parse = () => metricLinesAdapter.parse(stdout);
        expect(parse).toThrow(AdapterError);
        expect(parse).toThrow(/^No valid METRIC lines found$/);
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

/**
 * Run `fn` with console.warn spied out and return everything it warned.
 *
 * The spy is restored by the suite-level `afterEach`, so an early return or
 * throw inside `fn` cannot leak the patched console into the next test.
 */
function captureStderr(fn: () => void): string {
  const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

  fn();

  return warnSpy.mock.calls.map((args) => args.join(" ")).join("");
}
