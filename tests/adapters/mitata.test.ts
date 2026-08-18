import { afterEach, describe, expect, it, vi } from "vitest";

import mitataAdapter, { findJsonCandidates } from "../../src/adapters/mitata.js";
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
      it.each([
        {
          description: "replaces $key with key=value in alias",
          alias: "decode/$text",
          name: "decode/digits",
          args: { text: "digits" },
          p50: 42,
          metricName: "decode/text=digits/time",
        },
        {
          description: "handles multiple parameters",
          alias: "op/$a/$b",
          name: "op/x/y",
          args: { a: "x", b: "y" },
          p50: 50,
          metricName: "op/a=x/b=y/time",
        },
        {
          description: "replaces multiple occurrences of same parameter",
          alias: "test/$x/sep/$x",
          name: "test/1/sep/1",
          args: { x: "1" },
          p50: 99,
          metricName: "test/x=1/sep/x=1/time",
        },
        {
          description: "preserves a stray dollar that matches no argument key",
          alias: "test/$x/$unknown",
          name: "test/1/$unknown",
          args: { x: "1" },
          p50: 77,
          metricName: "test/x=1/$unknown/time",
        },
      ])("$description", ({ alias, name, args, p50, metricName }) => {
        const stdout = JSON.stringify({
          benchmarks: [{ alias, runs: [{ name, args, stats: { p50 } }] }],
        });
        const result = mitataAdapter.parse(stdout);
        expect(result).toStrictEqual(metricRecord({ [metricName]: p50 }));
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

    describe("metric names carrying a line terminator", () => {
      /**
       * Characters JavaScript regexes treat as line terminators.
       *
       * A metric name holding one of these could never be matched back by an
       * anchored key pattern, so the run is dropped instead of recorded.
       */
      const LINE_TERMINATORS = [
        { description: "a line feed", char: "\n" },
        { description: "a carriage return", char: "\r" },
        { description: "a line separator U+2028", char: "\u{2028}" },
        { description: "a paragraph separator U+2029", char: "\u{2029}" },
      ];

      it.each(LINE_TERMINATORS)(
        "warns and skips a benchmark whose alias contains $description",
        ({ char }) => {
          const offendingAlias = `enc${char}ode`;
          const stdout = JSON.stringify({
            benchmarks: [
              {
                alias: offendingAlias,
                runs: [{ name: "encode", args: {}, stats: { p50: 42 } }],
              },
              {
                alias: "valid",
                runs: [{ name: "valid", args: {}, stats: { p50: 1 } }],
              },
            ],
          });
          let result: Record<string, number> | undefined;

          const stderr = captureStderr(() => {
            result = mitataAdapter.parse(stdout);
          });

          expect
            .soft(stderr)
            .toContain(`Skipping run with a line terminator in its metric name: ${offendingAlias}`);
          expect(result).toStrictEqual(metricRecord({ "valid/time": 1 }));
        },
      );

      it.each(LINE_TERMINATORS)(
        "warns and skips a run whose argument value contains $description",
        ({ char }) => {
          const stdout = JSON.stringify({
            benchmarks: [
              {
                alias: "decode/$text",
                runs: [
                  { name: "decode/offending", args: { text: `di${char}gits` }, stats: { p50: 10 } },
                  { name: "decode/words", args: { text: "words" }, stats: { p50: 20 } },
                ],
              },
            ],
          });
          let result: Record<string, number> | undefined;

          const stderr = captureStderr(() => {
            result = mitataAdapter.parse(stdout);
          });

          expect
            .soft(stderr)
            .toContain("Skipping run with a line terminator in its metric name: decode/$text");
          expect(result).toStrictEqual(metricRecord({ "decode/text=words/time": 20 }));
        },
      );
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

      it("warns about the collision and keeps the last value written", () => {
        let result: Record<string, number> | undefined;
        const stderr = captureStderr(() => {
          result = mitataAdapter.parse(aliasMissingPlaceholder);
        });

        expect.soft(stderr).toContain("Duplicate metric name: decode/time");
        expect(result).toStrictEqual(metricRecord({ "decode/time": 20 }));
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
      ])("skips and warns about a run whose p50 is %s", (_, literal) => {
        const stdout = `{"benchmarks":[{"alias":"test/$x","runs":[
          {"name":"test/a","args":{"x":"a"},"stats":{"p50":${literal}}},
          {"name":"test/b","args":{"x":"b"},"stats":{"p50":20}}
        ]}]}`;

        let result: Record<string, number> | undefined;
        const stderr = captureStderr(() => {
          result = mitataAdapter.parse(stdout);
        });

        expect.soft(stderr).toContain("test/$x");
        expect.soft(stderr).toMatch(/non-finite/i);
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

    describe("empty shortName fallback", () => {
      it.each([
        { metricName: "/time", kind: "time", unit: "ns" as const },
        { metricName: "/heap", kind: "memory", unit: "bytes" as const },
      ])(
        "falls back to full metric name for $metricName instead of empty shortName",
        ({ metricName, kind, unit }) => {
          const result = mitataAdapter.defaults(metricName);
          expect(result).toStrictEqual({
            direction: "lower",
            unit,
            kind,
            shortName: metricName,
          });
        },
      );
    });
  });
});

describe("findJsonCandidates", () => {
  it("extracts a single JSON object from a plain string", () => {
    const input = '{"key": "value"}';
    const result = findJsonCandidates(input);
    expect(result).toStrictEqual(['{"key": "value"}']);
  });

  it("extracts multiple top-level JSON candidates", () => {
    const input = '{"a": 1} some text {"b": 2}';
    const result = findJsonCandidates(input);
    expect(result).toStrictEqual(['{"a": 1}', '{"b": 2}']);
  });

  it("ignores braces inside double-quoted strings", () => {
    const input = '{"key": "value with {braces} inside"}';
    const result = findJsonCandidates(input);
    expect(result).toStrictEqual(['{"key": "value with {braces} inside"}']);
  });

  it("handles escaped quotes within strings", () => {
    const input = '{"key": "value with \\"escaped\\" quotes and {braces}"}';
    const result = findJsonCandidates(input);
    expect(result).toStrictEqual(['{"key": "value with \\"escaped\\" quotes and {braces}"}']);
  });

  it("handles nested braces correctly", () => {
    const input = '{"outer": {"inner": {"deep": 1}}}';
    const result = findJsonCandidates(input);
    expect(result).toStrictEqual(['{"outer": {"inner": {"deep": 1}}}']);
  });

  it("returns an empty array when input contains no braces", () => {
    const result = findJsonCandidates("no braces here at all");
    expect(result).toStrictEqual([]);
  });

  it("skips unbalanced opening braces", () => {
    const input = "prefix { incomplete";
    const result = findJsonCandidates(input);
    expect(result).toStrictEqual([]);
  });

  it("extracts a candidate surrounded by non-JSON text with braces", () => {
    const input = 'cpu: {model}\n{"benchmarks": []}\nfooter: {info}';
    const result = findJsonCandidates(input);
    expect(result).toStrictEqual(["{model}", '{"benchmarks": []}', "{info}"]);
  });
});
