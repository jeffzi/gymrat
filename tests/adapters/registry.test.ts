import { describe, expect, it } from "vitest";

import { AdapterError, getAdapter } from "../../src/adapters/index.js";
import { metricRecord } from "../fixtures/metrics.js";

describe("getAdapter registry", () => {
  const testAdapters = [
    {
      name: "metric-lines",
      parseInput: "METRIC foo=42",
      parseExpected: metricRecord({ foo: 42 }),
      defaultsInput: "test_metric",
      defaultsExpected: { direction: "lower" },
    },
    {
      name: "mitata",
      parseInput: JSON.stringify({
        benchmarks: [
          {
            alias: "test",
            runs: [
              {
                name: "test",
                args: {},
                stats: { p50: 42 },
              },
            ],
          },
        ],
      }),
      parseExpected: metricRecord({ "test/time": 42 }),
      defaultsInput: "test/time",
      defaultsExpected: {
        direction: "lower",
        unit: "ns",
        kind: "time",
        shortName: "test",
      },
    },
  ];

  it.each(testAdapters)(
    "getAdapter('$name') returns an adapter with the expected shape",
    (spec) => {
      const adapter = getAdapter(spec.name);

      expect(adapter.name).toBe(spec.name);
      expect(adapter.parse(spec.parseInput)).toStrictEqual(spec.parseExpected);
      expect(adapter.defaults(spec.defaultsInput)).toStrictEqual(spec.defaultsExpected);
    },
  );

  describe("getAdapter('unknown')", () => {
    it("throws Error (not AdapterError) with valid names listed", () => {
      expect(() => getAdapter("unknown")).toThrow(/metric-lines/);
      expect(() => getAdapter("unknown")).toThrow(/mitata/);
      expect(() => getAdapter("unknown")).not.toThrow(AdapterError);
    });
  });
});
