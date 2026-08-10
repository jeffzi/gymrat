import { afterEach, describe, expect, it, vi } from "vitest";

import mitataAdapter from "../../src/adapters/mitata.js";
import { AdapterError } from "../../src/adapters/types.js";
import { captureStderr } from "../fixtures/console.js";
import { metricRecord } from "../fixtures/metrics.js";

afterEach(() => {
  vi.restoreAllMocks();
});

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
        expect(result).toStrictEqual(metricRecord({ "encode/time": 42 }));
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
        expect(result).toStrictEqual(metricRecord({ "decode/text=digits/time": 42 }));
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
        expect(result).toStrictEqual(metricRecord({ "op/a=x/b=y/time": 50 }));
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
        expect(result).toStrictEqual(metricRecord({ "test/x=1/sep/x=1/time": 99 }));
      });
    });

    describe("alias substitution with hostile argument values", () => {
      it.each([
        ["whole-match reference", "a$&b"],
        ["prefix reference", "a$`b"],
        ["suffix reference", "a$'b"],
        ["escaped dollar", "a$$b"],
      ])("keeps a %s in an argument value literal", (_, value) => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "decode/$text",
              runs: [
                {
                  name: "decode/hostile",
                  args: { text: value },
                  stats: { p50: 42 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ [`decode/text=${value}/time`]: 42 }));
      });

      it("does not substitute a placeholder introduced by an earlier argument value", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "op/$a/$b",
              runs: [
                {
                  name: "op",
                  args: { a: "$b", b: "y" },
                  stats: { p50: 7 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ "op/a=$b/b=y/time": 7 }));
      });
    });

    describe("metric name collisions", () => {
      const aliasMissingPlaceholder = JSON.stringify({
        benchmarks: [
          {
            alias: "decode",
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

      it("warns naming the metric two runs of one benchmark collide on", () => {
        const stderr = captureStderr(() => {
          mitataAdapter.parse(aliasMissingPlaceholder);
        });

        expect(stderr).toContain("Duplicate metric name: decode/time");
      });

      it("keeps the last value written for a colliding metric", () => {
        captureStderr(() => {
          const result = mitataAdapter.parse(aliasMissingPlaceholder);
          expect(result).toStrictEqual(metricRecord({ "decode/time": 20 }));
        });
      });

      it("routes the collision warning to the warn sink it is given, not stderr", () => {
        const warnings: string[] = [];

        const stderr = captureStderr(() => {
          mitataAdapter.parse(aliasMissingPlaceholder, (message) => warnings.push(message));
        });

        expect({ warnings, stderr }).toStrictEqual({
          warnings: [expect.stringContaining("Duplicate metric name: decode/time")],
          stderr: "",
        });
      });

      it("warns when two benchmarks share an alias", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "encode",
              runs: [{ name: "encode", args: {}, stats: { p50: 1 } }],
            },
            {
              alias: "encode",
              runs: [{ name: "encode", args: {}, stats: { p50: 2 } }],
            },
          ],
        });

        const stderr = captureStderr(() => {
          mitataAdapter.parse(stdout);
        });

        expect(stderr).toContain("Duplicate metric name: encode/time");
      });
    });

    describe("p50 value extraction", () => {
      it.each([
        { description: "uses stats.p50 as the metric value", p50: 123.456 },
        { description: "preserves decimal precision", p50: 0.0791015625 },
      ])("$description", ({ p50 }) => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test",
              runs: [
                {
                  name: "test",
                  args: {},
                  stats: { p50 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ "test/time": p50 }));
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
        expect(result).toStrictEqual(
          metricRecord({
            "test/time": 42,
            "test/heap": 1024,
          }),
        );
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
        expect(result).toStrictEqual(
          metricRecord({
            "decode/text=digits/time": 10,
            "decode/text=digits/heap": 256,
          }),
        );
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
        expect(result).toStrictEqual(metricRecord({ "test/time": 42 }));
      });
    });

    describe("non-finite statistics", () => {
      it.each([
        ["positive infinity", "1e999"],
        ["negative infinity", "-1e999"],
      ])("skips a run whose p50 is %s", (_, literal) => {
        const stdout = `{"benchmarks":[{"alias":"test/$x","runs":[
          {"name":"test/a","args":{"x":"a"},"stats":{"p50":${literal}}},
          {"name":"test/b","args":{"x":"b"},"stats":{"p50":20}}
        ]}]}`;

        const result = mitataAdapter.parse(stdout);

        expect(result).toStrictEqual(metricRecord({ "test/x=b/time": 20 }));
      });

      it("throws AdapterError when every run has a non-finite p50", () => {
        const stdout = `{"benchmarks":[{"alias":"test","runs":[
          {"name":"test","args":{},"stats":{"p50":1e999}}
        ]}]}`;

        const parse = () => mitataAdapter.parse(stdout);

        expect(parse).toThrow(AdapterError);
        expect(parse).toThrow(/^No valid benchmark runs found$/);
      });

      it("keeps the time metric but records no heap metric when heap.avg is non-finite", () => {
        const stdout = `{"benchmarks":[{"alias":"test","runs":[
          {"name":"test","args":{},"stats":{"p50":42,"heap":{"avg":1e999}}}
        ]}]}`;

        const result = mitataAdapter.parse(stdout);

        expect(result).toStrictEqual(metricRecord({ "test/time": 42 }));
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
        expect(result).toStrictEqual(
          metricRecord({
            "decode/text=digits/time": 10,
            "decode/text=words/time": 20,
          }),
        );
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
        expect(result).toStrictEqual(
          metricRecord({
            "decode/text=digits/time": 10,
            "decode/text=digits/heap": 256,
            "decode/text=words/time": 20,
            "decode/text=words/heap": 512,
          }),
        );
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
        expect(result).toStrictEqual(
          metricRecord({
            "encode/time": 42,
            "decode/time": 100,
          }),
        );
      });
    });

    describe("error handling", () => {
      it("throws AdapterError when no JSON object found", () => {
        const stdout = "not valid json at all";
        const parse = () => mitataAdapter.parse(stdout);
        expect(parse).toThrow(AdapterError);
        expect(parse).toThrow(/^No JSON object found in stdout$/);
      });

      it("throws AdapterError when JSON between braces is malformed", () => {
        const stdout = "preamble { invalid json } trailer";
        const parse = () => mitataAdapter.parse(stdout);
        expect(parse).toThrow(AdapterError);
        expect(parse).toThrow(/^Failed to parse JSON: /);
      });

      it("throws AdapterError when benchmarks array is missing", () => {
        const stdout = JSON.stringify({ something: "else" });
        const parse = () => mitataAdapter.parse(stdout);
        expect(parse).toThrow(AdapterError);
        expect(parse).toThrow(/^JSON missing benchmarks array$/);
      });

      it("throws AdapterError when benchmarks array is empty", () => {
        const stdout = JSON.stringify({ benchmarks: [] });
        const parse = () => mitataAdapter.parse(stdout);
        expect(parse).toThrow(AdapterError);
        expect(parse).toThrow(/^benchmarks array is empty$/);
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
        const parse = () => mitataAdapter.parse(stdout);
        expect(parse).toThrow(AdapterError);
        expect(parse).toThrow(/^No valid benchmark runs found$/);
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
        expect(result).toStrictEqual(metricRecord({ "valid/time": 1 }));
      });

      it("skips benchmarks with non-string alias or missing runs", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            { alias: 42, runs: [] },
            { alias: "orphan" },
            { runs: [{ args: {}, stats: { p50: 1 } }] },
            {
              alias: "valid",
              runs: [{ name: "valid", args: {}, stats: { p50: 1 } }],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ "valid/time": 1 }));
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
        expect(result).toStrictEqual(metricRecord({ "test/time": 1 }));
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
        expect(result).toStrictEqual(metricRecord({ "test/time": 1 }));
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
        expect(result).toStrictEqual(metricRecord({ "test/x=b/time": 20 }));
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
        const parse = () => mitataAdapter.parse(stdout);
        expect(parse).toThrow(AdapterError);
        expect(parse).toThrow(/^No valid benchmark runs found$/);
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
        expect(result).toStrictEqual(metricRecord({ "test/time": 10 }));
      });
    });

    describe("real fixture", () => {
      it("parses the fixture file correctly", async () => {
        const fixtureModule = await import("../../tests/fixtures/mitata.json", {
          with: { type: "json" },
        });
        const fixtureJson = fixtureModule.default;
        const result = mitataAdapter.parse(JSON.stringify(fixtureJson));

        expect(result).toStrictEqual(
          metricRecord({
            "decode/text=digits/time": 4.0791015625,
            "decode/text=digits/heap": 0.13420623129857714,
            "decode/text=words/time": 7.8125,
            "decode/text=words/heap": 0.14746411878141288,
            "encode/time": 42.66357421875,
            "encode/heap": 80.1967411655276,
          }),
        );
      });
    });
  });

  describe("defaults()", () => {
    it.each([
      { metricName: "test/time", shortName: "test" },
      { metricName: "encode/time", shortName: "encode" },
      { metricName: "decode/x=1/time", shortName: "decode/x=1" },
      { metricName: "complex/a=1/b=2/time", shortName: "complex/a=1/b=2" },
    ])(
      "describes $metricName as a lower-is-better time metric shown as $shortName",
      ({ metricName, shortName }) => {
        const result = mitataAdapter.defaults(metricName);
        expect(result).toStrictEqual({
          direction: "lower",
          unit: "ns",
          kind: "time",
          shortName,
        });
      },
    );

    it.each([
      { metricName: "test/heap", shortName: "test" },
      { metricName: "encode/heap", shortName: "encode" },
      { metricName: "decode/x=1/heap", shortName: "decode/x=1" },
      { metricName: "complex/a=1/b=2/heap", shortName: "complex/a=1/b=2" },
    ])(
      "describes $metricName as a lower-is-better memory metric shown as $shortName",
      ({ metricName, shortName }) => {
        const result = mitataAdapter.defaults(metricName);
        expect(result).toStrictEqual({
          direction: "lower",
          unit: "bytes",
          kind: "memory",
          shortName,
        });
      },
    );

    it.each(["custom_metric", "test", "test/throughput", "test/ops"])(
      "omits unit, kind, and shortName for unrecognized metric %s",
      (metricName) => {
        const result = mitataAdapter.defaults(metricName);
        expect(result).toStrictEqual({ direction: "lower" });
      },
    );
  });
});
