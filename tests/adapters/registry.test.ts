import { AdapterError, getAdapter } from "../../src/adapters/index.js";

describe("getAdapter registry", () => {
  const testAdapters = [
    {
      name: "metric-lines",
      parseInput: "METRIC foo=42",
      parseExpected: { foo: 42 },
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
      parseExpected: { "test/time": 42 },
      defaultsInput: "test/time",
      defaultsExpected: { direction: "lower", unit: "ns" },
    },
  ];

  it.each(testAdapters)(
    "getAdapter('$name') returns an adapter with the expected shape",
    (spec) => {
      const adapter = getAdapter(spec.name);

      expect(adapter.name).toBe(spec.name);
      expect(typeof adapter.parse).toBe("function");
      expect(typeof adapter.defaults).toBe("function");
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
