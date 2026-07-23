describe("getAdapter registry", () => {
  const testAdapters = [
    {
      name: "metric-lines",
      validName: "metric-lines",
      parseInput: "METRIC foo=42",
      parseExpected: { foo: 42 },
      defaultsInput: "test_metric",
      defaultsExpected: { direction: "lower" },
    },
    {
      name: "mitata",
      validName: "mitata",
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
    "getAdapter('$validName') returns adapter with name $name",
    async (spec) => {
      const { getAdapter } = await import("../../src/adapters/index.js");
      const adapter = getAdapter(spec.validName);
      expect(adapter.name).toBe(spec.name);
    },
  );

  it.each(testAdapters)("getAdapter('$validName') adapter.parse() works", async (spec) => {
    const { getAdapter } = await import("../../src/adapters/index.js");
    const adapter = getAdapter(spec.validName);
    const result = adapter.parse(spec.parseInput);
    expect(result).toStrictEqual(spec.parseExpected);
  });

  it.each(testAdapters)("getAdapter('$validName') adapter.defaults() works", async (spec) => {
    const { getAdapter } = await import("../../src/adapters/index.js");
    const adapter = getAdapter(spec.validName);
    const result = adapter.defaults(spec.defaultsInput);
    expect(result).toStrictEqual(spec.defaultsExpected);
  });

  describe("getAdapter('unknown')", () => {
    it("throws Error (not AdapterError) with valid names listed", async () => {
      const { getAdapter, AdapterError } = await import("../../src/adapters/index.js");
      expect(() => getAdapter("unknown")).toThrow(/metric-lines/);
      expect(() => getAdapter("unknown")).toThrow(/mitata/);
      expect(() => getAdapter("unknown")).not.toThrow(AdapterError);
    });
  });
});
