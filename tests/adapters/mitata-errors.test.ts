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

    describe("non-primitive run-argument serialization", () => {
      it("serializes an object argument value via JSON rather than [object Object]", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "bench/$opts",
              runs: [
                {
                  name: "bench/cfg",
                  args: { opts: { size: 100 } },
                  stats: { p50: 5 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ 'bench/opts={"size":100}/time': 5 }));
      });

      it("keeps distinct metric names for distinct object argument values", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "bench/$opts",
              runs: [
                {
                  name: "bench/a",
                  args: { opts: { size: 100 } },
                  stats: { p50: 5 },
                },
                {
                  name: "bench/b",
                  args: { opts: { size: 200 } },
                  stats: { p50: 10 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(
          metricRecord({
            'bench/opts={"size":100}/time': 5,
            'bench/opts={"size":200}/time': 10,
          }),
        );
      });

      it("serializes object keys in sorted order for deterministic names", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "bench/$opts",
              runs: [
                {
                  name: "bench/cfg",
                  args: { opts: { z: 1, a: 2 } },
                  stats: { p50: 5 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ 'bench/opts={"a":2,"z":1}/time': 5 }));
      });

      it("serializes an array argument value via JSON", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "bench/$items",
              runs: [
                {
                  name: "bench/list",
                  args: { items: [1, 2, 3] },
                  stats: { p50: 7 },
                },
              ],
            },
          ],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ "bench/items=[1,2,3]/time": 7 }));
      });
    });

    describe("non-finite p50 or malformed shape warnings", () => {
      it("warns when stats.p50 is missing (malformed shape)", () => {
        const stdout = JSON.stringify({
          benchmarks: [
            {
              alias: "test/$x",
              runs: [
                { name: "test/a", args: { x: "a" }, stats: {} },
                { name: "test/b", args: { x: "b" }, stats: { p50: 20 } },
              ],
            },
          ],
        });
        let result: Record<string, number> | undefined;
        const stderr = captureStderr(() => {
          result = mitataAdapter.parse(stdout);
        });

        expect.soft(stderr).toContain("test/$x");
        expect(result).toStrictEqual(metricRecord({ "test/x=b/time": 20 }));
      });

      it("routes the skip warning to the warn sink, not stderr", () => {
        const stdout = `{"benchmarks":[{"alias":"test","runs":[
          {"name":"test","args":{},"stats":{"p50":1e999}},
          {"name":"test2","args":{},"stats":{"p50":20}}
        ]}]}`;

        const warnings: string[] = [];
        const stderr = captureStderr(() => {
          mitataAdapter.parse(stdout, (message) => warnings.push(message));
        });

        expect.soft(warnings.length).toBeGreaterThan(0);
        expect.soft(warnings[0]).toMatch(/non-finite/i);
        expect(stderr).toBe("");
      });
    });

    describe("brace-aware extractJson", () => {
      it("parses when banner text before the JSON contains braces", () => {
        const json = JSON.stringify({
          benchmarks: [
            {
              alias: "encode",
              runs: [{ name: "encode", args: {}, stats: { p50: 42 } }],
            },
          ],
        });
        const stdout = `cpu: {model}\nruntime: bun {version}\n\n${json}`;
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ "encode/time": 42 }));
      });

      it("parses when banner text after the JSON contains braces", () => {
        const json = JSON.stringify({
          benchmarks: [
            {
              alias: "encode",
              runs: [{ name: "encode", args: {}, stats: { p50: 42 } }],
            },
          ],
        });
        const stdout = `${json}\nfooter: {info}`;
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ "encode/time": 42 }));
      });

      it("parses when braces appear both before and after the JSON", () => {
        const json = JSON.stringify({
          benchmarks: [
            {
              alias: "encode",
              runs: [{ name: "encode", args: {}, stats: { p50: 42 } }],
            },
          ],
        });
        const stdout = `cpu: {model}\n${json}\nfooter: {info}`;
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ "encode/time": 42 }));
      });

      it("still throws AdapterError when no complete JSON object exists", () => {
        const stdout = "cpu: {model}\nno json here\nfooter: {info}";
        const parse = () => mitataAdapter.parse(stdout);
        expect(parse).toThrow(AdapterError);
        expect(parse).toThrow(/^Failed to parse JSON:/);
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
});
