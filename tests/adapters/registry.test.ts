import { describe, expect, it } from "vitest";

import { getAdapter } from "../../src/adapters/index.js";
import { GymratError } from "../../src/errors.js";
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

  it.each(testAdapters)("getAdapter('$name') returns an adapter with matching name", (spec) => {
    const adapter = getAdapter(spec.name);

    expect(adapter.name).toBe(spec.name);
  });

  it.each(testAdapters)("getAdapter('$name').parse() parses the expected input", (spec) => {
    const adapter = getAdapter(spec.name);

    expect(adapter.parse(spec.parseInput)).toStrictEqual(spec.parseExpected);
  });

  it.each(testAdapters)("getAdapter('$name').defaults() returns the expected defaults", (spec) => {
    const adapter = getAdapter(spec.name);

    expect(adapter.defaults(spec.defaultsInput)).toStrictEqual(spec.defaultsExpected);
  });

  describe("getAdapter('unknown')", () => {
    it("throws GymratError with valid adapter names listed in the hint", () => {
      expect(() => getAdapter("unknown")).toThrow(GymratError);
      expect(() => getAdapter("unknown")).toThrow(/Unknown adapter/);
    });

    it("lists valid adapter names in the hint", () => {
      let hint: string | undefined;
      try {
        getAdapter("unknown");
      } catch (e) {
        if (e instanceof GymratError) hint = e.hint;
      }
      expect(hint).toMatch(/metric-lines/);
      expect(hint).toMatch(/mitata/);
    });
  });
});
