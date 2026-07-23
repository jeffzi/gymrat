import mitataAdapter from "../../src/adapters/mitata.js";

describe("mitata adapter", () => {
  describe("parse()", () => {
    describe("basic JSON parsing", () => {
      const jsonFixture = {
        benchmarks: [
          {
            alias: "encode",
            runs: [
              {
                name: "encode",
                args: {},
                stats: { p50: 42 },
              },
            ],
          },
        ],
      };

      it.each([
        ["no preamble or trailer", JSON.stringify(jsonFixture)],
        ["preamble", `some preamble\nmore output\n${JSON.stringify(jsonFixture)}`],
        ["trailer", `${JSON.stringify(jsonFixture)}\ntrailing output\nmore output`],
        ["preamble and trailer", `preamble\n${JSON.stringify(jsonFixture)}\ntrailer`],
      ])("parses JSON with %s", (_, stdout) => {
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "encode/time": 42 });
      });
    });

    describe("metric naming for parameterized benchmarks", () => {
      it("replaces $key with key=value in alias", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "decode/$text",
              runs: [
                {
                  name: "decode/digits",
                  args: { text: "digits" },
                  stats: { p50: 42 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "decode/text=digits/time": 42 });
      });

      it("handles multiple parameters", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "op/$a/$b",
              runs: [
                {
                  name: "op/x/y",
                  args: { a: "x", b: "y" },
                  stats: { p50: 50 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "op/a=x/b=y/time": 50 });
      });

      it("replaces multiple occurrences of same parameter", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test/$x/sep/$x",
              runs: [
                {
                  name: "test/1/sep/1",
                  args: { x: "1" },
                  stats: { p50: 99 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "test/x=1/sep/x=1/time": 99 });
      });
    });

    describe("p50 value extraction", () => {
      it("uses stats.p50 as the metric value", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test",
              runs: [
                {
                  name: "test",
                  args: {},
                  stats: { p50: 123.456 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "test/time": 123.456 });
      });

      it("preserves decimal precision in p50", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test",
              runs: [
                {
                  name: "test",
                  args: {},
                  stats: { p50: 0.0791015625 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "test/time": 0.0791015625 });
      });
    });

    describe("heap metric emission", () => {
      it("emits heap metric when stats.heap.avg is present", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test",
              runs: [
                {
                  name: "test",
                  args: {},
                  stats: {
                    p50: 42,
                    heap: { avg: 1024 },
                  },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({
          "test/time": 42,
          "test/heap": 1024,
        });
      });

      it("emits heap metric for parameterized benchmark", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "decode/$text",
              runs: [
                {
                  name: "decode/digits",
                  args: { text: "digits" },
                  stats: {
                    p50: 10,
                    heap: { avg: 256 },
                  },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({
          "decode/text=digits/time": 10,
          "decode/text=digits/heap": 256,
        });
      });

      it("skips heap metric when avg is missing", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test",
              runs: [
                {
                  name: "test",
                  args: {},
                  stats: {
                    p50: 42,
                    heap: { total: 1024 },
                  },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "test/time": 42 });
      });

      it("skips heap metric when heap is missing entirely", () => {
        const stdout = JSON.stringify({
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
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "test/time": 42 });
      });
    });

    describe("multiple runs per benchmark", () => {
      it("emits metrics for each run in a benchmark", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "decode/$text",
              runs: [
                {
                  name: "decode/digits",
                  args: { text: "digits" },
                  stats: { p50: 10 },
                },
                {
                  name: "decode/words",
                  args: { text: "words" },
                  stats: { p50: 20 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({
          "decode/text=digits/time": 10,
          "decode/text=words/time": 20,
        });
      });

      it("emits heap metrics for each run", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "decode/$text",
              runs: [
                {
                  name: "decode/digits",
                  args: { text: "digits" },
                  stats: { p50: 10, heap: { avg: 256 } },
                },
                {
                  name: "decode/words",
                  args: { text: "words" },
                  stats: { p50: 20, heap: { avg: 512 } },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({
          "decode/text=digits/time": 10,
          "decode/text=digits/heap": 256,
          "decode/text=words/time": 20,
          "decode/text=words/heap": 512,
        });
      });
    });

    describe("multiple benchmarks", () => {
      it("emits metrics for all benchmarks", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "encode",
              runs: [
                {
                  name: "encode",
                  args: {},
                  stats: { p50: 42 },
                },
              ],
            },
            {
              alias: "decode",
              runs: [
                {
                  name: "decode",
                  args: {},
                  stats: { p50: 100 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({
          "encode/time": 42,
          "decode/time": 100,
        });
      });
    });

    describe("error handling", () => {
      it("throws AdapterError when no JSON object found", () => {
        const stdout = "not valid json at all";
        expect(() => mitataAdapter.parse(stdout)).toThrow("AdapterError");
      });

      it("throws AdapterError when JSON between braces is malformed", () => {
        const stdout = "preamble { invalid json } trailer";
        expect(() => mitataAdapter.parse(stdout)).toThrow("AdapterError");
      });

      it("throws AdapterError when benchmarks array is missing", () => {
        const stdout = JSON.stringify({ something: "else" });
        expect(() => mitataAdapter.parse(stdout)).toThrow("AdapterError");
      });

      it("throws AdapterError when benchmarks array is empty", () => {
        const stdout = JSON.stringify({ benchmarks: [] });
        expect(() => mitataAdapter.parse(stdout)).toThrow("AdapterError");
      });

      it("throws AdapterError when no runs have valid stats", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test",
              runs: [
                {
                  name: "test",
                  args: {},
                  stats: {},
                },
              ],
            },
          ],
        });
        expect(() => mitataAdapter.parse(stdout)).toThrow("AdapterError");
      });

      it("skips non-object benchmarks entries", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            null,
            42,
            "string",
            {
              alias: "valid",
              runs: [{ name: "valid", args: {}, stats: { p50: 1 } }],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "valid/time": 1 });
      });

      it("skips benchmarks with non-string alias or missing runs", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            { alias: 42, runs: [] },
            { runs: [{ args: {}, stats: { p50: 1 } }] },
            {
              alias: "valid",
              runs: [{ name: "valid", args: {}, stats: { p50: 1 } }],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "valid/time": 1 });
      });

      it("skips runs with non-object args or stats", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test",
              runs: [
                { args: "not-object", stats: { p50: 1 } },
                { args: {}, stats: "not-object" },
                { args: {}, stats: { p50: 1 } },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "test/time": 1 });
      });

      it("skips non-object runs", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test",
              runs: [null, 42, { args: {}, stats: { p50: 1 } }],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "test/time": 1 });
      });
    });

    describe("error field handling", () => {
      it("skips runs with error field", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test/$x",
              runs: [
                {
                  name: "test/a",
                  args: { x: "a" },
                  error: "something went wrong",
                  stats: { p50: 10 },
                },
                {
                  name: "test/b",
                  args: { x: "b" },
                  stats: { p50: 20 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "test/x=b/time": 20 });
      });

      it("throws AdapterError if all runs have errors", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test",
              runs: [
                {
                  name: "test",
                  args: {},
                  error: "something failed",
                  stats: { p50: 10 },
                },
              ],
            },
          ],
        });
        expect(() => mitataAdapter.parse(stdout)).toThrow("AdapterError");
      });

      it("skips null error field", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test",
              runs: [
                {
                  name: "test",
                  args: {},
                  error: null,
                  stats: { p50: 10 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual({ "test/time": 10 });
      });
    });

    describe("real fixture", () => {
      it("parses the fixture file correctly", async () => {
        // Using dynamic import to load JSON from fixture
        const fixtureModule = await import("../../tests/fixtures/mitata.json", {
          with: { type: "json" },
        });
        const fixtureJson = fixtureModule.default;
        const result = mitataAdapter.parse(JSON.stringify(fixtureJson));

        // From the fixture: two benchmarks
        // 1. decode/$text with two runs: digits and words
        // 2. encode with one run
        expect(result).toHaveProperty("decode/text=digits/time");
        expect(result).toHaveProperty("decode/text=digits/heap");
        expect(result).toHaveProperty("decode/text=words/time");
        expect(result).toHaveProperty("decode/text=words/heap");
        expect(result).toHaveProperty("encode/time");
        expect(result).toHaveProperty("encode/heap");

        // Values from fixture
        expect(result["decode/text=digits/time"]).toStrictEqual(4.0791015625);
        expect(result["decode/text=words/time"]).toStrictEqual(7.8125);
        expect(result["encode/time"]).toStrictEqual(42.66357421875);
      });
    });
  });

  describe("defaults()", () => {
    it.each([
      ["test/time", { direction: "lower", unit: "ns" }],
      ["encode/time", { direction: "lower", unit: "ns" }],
      ["decode/x=1/time", { direction: "lower", unit: "ns" }],
      ["complex/a=1/b=2/time", { direction: "lower", unit: "ns" }],
    ])("returns direction: lower with ns unit for /time metric %s", (metricName, expected) => {
      const result = mitataAdapter.defaults(metricName);
      expect(result).toStrictEqual(expected);
    });

    it.each([
      ["test/heap", { direction: "lower", unit: "bytes" }],
      ["encode/heap", { direction: "lower", unit: "bytes" }],
      ["decode/x=1/heap", { direction: "lower", unit: "bytes" }],
      ["complex/a=1/b=2/heap", { direction: "lower", unit: "bytes" }],
    ])("returns direction: lower with bytes unit for /heap metric %s", (metricName, expected) => {
      const result = mitataAdapter.defaults(metricName);
      expect(result).toStrictEqual(expected);
    });

    it.each(["custom_metric", "test", "test/throughput", "test/ops"])(
      "does not include unit for metric %s",
      (metricName) => {
        const result = mitataAdapter.defaults(metricName);
        expect(result).toStrictEqual({ direction: "lower" });
        expect(result).not.toHaveProperty("unit");
      },
    );
  });
});
