import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { ResolvedConfig } from "../src/config.js";
import { loadConfigFile, resolveConfig } from "../src/config.js";
import { GymratError } from "../src/errors.js";
import { metricRecord } from "./fixtures/metrics.js";
import { freshRoot } from "./fixtures/scratch-repo.js";

/** Byte-order mark that editors on Windows prepend to UTF-8 files: `EF BB BF`. */
const UTF8_BOM = "\u{FEFF}";

/**
 * Loop keys `resolveConfig` always fills in, even for compare/measure runs that
 * never read them.
 */
const LOOP_DEFAULTS = { primary: "geomean" } as const;

/**
 * Full loop configuration, shared between the `loadConfigFile` parsing test and
 * the `resolveConfig` merging test for the loop keys.
 */
const LOOP_CONFIG = {
  checks: "npm test",
  filter: "npm run bench -- {names}",
  primary: "decode/time",
  stop: { targetValue: 1.5, maxIterations: 20 },
  hooks: { before: "npm run warm-cache", after: "npm run cool-down" },
};

/**
 * Characters JavaScript regexes treat as line terminators.
 *
 * A `^…$` key pattern stops at every one of them, so any of these embedded in a
 * config key can smuggle the rest of the key past validation.
 */
const LINE_BREAKS = [
  { description: "a line feed", char: "\n" },
  { description: "a carriage return", char: "\r" },
  { description: "a line separator", char: "\u{2028}" },
  { description: "a paragraph separator", char: "\u{2029}" },
];

function writeRawConfigFile(content: string): { dir: string; configPath: string } {
  const dir = freshRoot("gymrat-");
  const configPath = path.join(dir, "gymrat.json");
  fs.writeFileSync(configPath, content);
  return { dir, configPath };
}

function createConfigFile(content: Record<string, unknown>): { dir: string; configPath: string } {
  return writeRawConfigFile(JSON.stringify(content));
}

/** Registers an `afterEach` that restores the process cwd captured at call time. */
function restoreCwdAfterEach(): void {
  const originalCwd = process.cwd();
  afterEach(() => {
    process.chdir(originalCwd);
  });
}

describe("loadConfigFile", () => {
  describe("when the config file does not exist", () => {
    it("returns an empty config object", () => {
      const nonexistentPath = path.join(os.tmpdir(), `nonexistent-${Date.now()}.json`);
      const result = loadConfigFile(nonexistentPath);

      expect(result).toStrictEqual({});
    });
  });

  describe("when the config file does not exist and the caller requires it", () => {
    it("throws a GymratError naming the missing path", () => {
      const nonexistentPath = path.join(os.tmpdir(), `nonexistent-${Date.now()}.json`);
      const act = (): void => {
        loadConfigFile(nonexistentPath, { required: true });
      };

      expect(act).toThrow(GymratError);
      expect(act).toThrow(nonexistentPath);
    });
  });

  describe("when the config path is not a readable file", () => {
    it.each([{ options: undefined }, { options: { required: true } }])(
      "throws a GymratError naming the path (options: $options)",
      ({ options }) => {
        const dir = freshRoot("gymrat-");

        expect(() => loadConfigFile(dir, options)).toThrow(GymratError);
        expect(() => loadConfigFile(dir, options)).toThrow(dir);
      },
    );
  });

  describe("when the config file contains valid JSON with known keys", () => {
    it("returns the parsed config with bench key", () => {
      const { configPath } = createConfigFile({ bench: "custom-bench" });

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual({ bench: "custom-bench" });
    });

    it("returns the parsed config with all known keys", () => {
      const config = {
        bench: "bench-name",
        prepare: "prepare-cmd",
        adapter: "adapter-name",
        samples: 10,
        timeoutSeconds: 30,
        // Fractional: unlike samples, the noise threshold need not be an integer.
        unstableNoisePct: 150.5,
        metrics: {
          metric1: { direction: "lower" as const, gating: true, exact: false },
          metric2: { direction: "higher" as const },
        },
      };
      const { configPath } = createConfigFile(config);

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual(config);
    });

    it("returns the parsed config with partial metrics metadata", () => {
      const config = {
        metrics: {
          responseTime: { direction: "lower" as const },
          throughput: { gating: true },
        },
      };
      const { configPath } = createConfigFile(config);

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual(config);
    });
  });

  describe("when the config file contains invalid JSON", () => {
    it("throws a GymratError that includes the file path", () => {
      const { configPath } = writeRawConfigFile("{ invalid json }");
      const act = (): void => {
        loadConfigFile(configPath);
      };

      expect(act).toThrow(GymratError);
      expect(act).toThrow(configPath);
    });
  });

  describe("when the config file is prefixed with a UTF-8 BOM", () => {
    it("parses the config as if the BOM were absent", () => {
      const config = { bench: "bom-bench", samples: 5 };
      const { configPath } = writeRawConfigFile(`${UTF8_BOM}${JSON.stringify(config)}`);

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual(config);
    });
  });

  describe("when the config file JSON root is not an object", () => {
    it.each([
      { description: "an array", json: "[]" },
      { description: "a string", json: '"bench"' },
      { description: "a number", json: "3" },
      { description: "a boolean", json: "true" },
      { description: "null", json: "null" },
    ])("throws naming the file and the expected JSON object for $description", ({ json }) => {
      const { configPath } = writeRawConfigFile(json);
      const act = (): void => {
        loadConfigFile(configPath);
      };

      expect(act).toThrow(configPath);
      expect(act).toThrow(/JSON object/);
    });
  });

  describe("when the config file contains unknown top-level keys", () => {
    it("throws an error that names the unknown key", () => {
      const { configPath } = createConfigFile({ unknownKey: "value" });

      expect(() => loadConfigFile(configPath)).toThrow(/unknownKey/);
    });

    it("throws when there is an unknown key mixed with known keys", () => {
      const { configPath } = createConfigFile({ bench: "name", badKey: "value" });

      expect(() => loadConfigFile(configPath)).toThrow(/badKey/);
    });

    it("names an empty-string key as a quoted empty key rather than a non-object root", () => {
      const { configPath } = createConfigFile({ "": 1 });
      const act = (): void => {
        loadConfigFile(configPath);
      };

      expect.soft(act).toThrow('Unknown config key: ""');
      expect(act).not.toThrow(/JSON object/);
    });
  });

  describe("when the config file contains an empty object", () => {
    it("returns an empty config object", () => {
      const { configPath } = createConfigFile({});

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual({});
    });
  });

  describe("when a string-typed key holds a non-string value", () => {
    it.each([
      { key: "bench", description: "a number", value: 42 },
      { key: "bench", description: "an array", value: ["a"] },
      { key: "prepare", description: "a boolean", value: true },
      { key: "prepare", description: "an object", value: { cmd: "x" } },
      { key: "adapter", description: "null", value: null },
      { key: "checks", description: "a number", value: 42 },
      { key: "filter", description: "an array", value: ["a"] },
      { key: "primary", description: "null", value: null },
    ])("throws naming $key when it is $description", ({ key, value }) => {
      const { configPath } = createConfigFile({ [key]: value });

      expect(() => loadConfigFile(configPath)).toThrow(new RegExp(`${key}.*string`));
    });
  });

  describe("when a non-empty-string key holds an empty string", () => {
    it.each([
      { key: "checks" },
      { key: "bench" },
      { key: "prepare" },
      { key: "adapter" },
      { key: "runbook" },
      { key: "primary" },
    ])("throws naming $key and the non-empty requirement", ({ key }) => {
      const { configPath } = createConfigFile({ [key]: "" });
      const act = (): void => {
        loadConfigFile(configPath);
      };

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(new RegExp(`${key}.*non-empty`));
    });
  });

  describe("when a positive-integer key holds an invalid value", () => {
    it.each([
      { key: "samples", description: "a string", value: "ten" },
      { key: "samples", description: "a non-integer", value: 1.5 },
      { key: "samples", description: "zero", value: 0 },
      { key: "timeoutSeconds", description: "a negative number", value: -1 },
      { key: "timeoutSeconds", description: "a boolean", value: true },
      { key: "timeoutSeconds", description: "null", value: null },
    ])("throws naming $key when it is $description", ({ key, value }) => {
      const { configPath } = createConfigFile({ [key]: value });

      expect(() => loadConfigFile(configPath)).toThrow(new RegExp(`${key}.*positive integer`));
    });
  });

  describe("when timeoutSeconds exceeds the millisecond timer cap", () => {
    it("throws naming timeoutSeconds and the cap the flag path reports", () => {
      const { configPath } = createConfigFile({ timeoutSeconds: 2_147_484 });
      const act = (): void => {
        loadConfigFile(configPath);
      };

      expect.soft(act).toThrow(GymratError);
      expect.soft(act).toThrow(/timeoutSeconds/);
      expect(act).toThrow(/no greater than 2147483/);
    });
  });

  describe("when timeoutSeconds sits exactly on the millisecond timer cap", () => {
    it("accepts the value", () => {
      const { configPath } = createConfigFile({ timeoutSeconds: 2_147_483 });

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual({ timeoutSeconds: 2_147_483 });
    });
  });

  describe("when unstableNoisePct holds an invalid value", () => {
    it.each([
      { description: "a string", value: "loud" },
      { description: "zero", value: 0 },
      { description: "a negative number", value: -5 },
      { description: "a boolean", value: true },
      { description: "null", value: null },
      { description: "below the noise floor", value: 0.25 },
    ])(
      "throws naming unstableNoisePct and the 0.5 noise floor when it is $description",
      ({ value }) => {
        const { configPath } = createConfigFile({ unstableNoisePct: value });
        const act = (): void => {
          loadConfigFile(configPath);
        };

        expect.soft(act).toThrow(/unstableNoisePct.*0\.5/);
        expect(act).toThrow(/noise floor/);
      },
    );
  });

  describe("when unstableNoisePct sits exactly on the noise floor", () => {
    it("accepts the value", () => {
      const { configPath } = createConfigFile({ unstableNoisePct: 0.5 });

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual({ unstableNoisePct: 0.5 });
    });
  });

  describe("when metrics is not an object", () => {
    it.each([
      { description: "an array", value: [] },
      { description: "a string", value: "latency" },
      { description: "a number", value: 3 },
      { description: "null", value: null },
    ])("throws naming metrics when it is $description", ({ value }) => {
      const { configPath } = createConfigFile({ metrics: value });

      expect(() => loadConfigFile(configPath)).toThrow(/metrics.*object/);
    });
  });

  describe("when a metrics entry is not an object", () => {
    it("throws naming the offending metric", () => {
      const { configPath } = createConfigFile({ metrics: { latency: "lower" } });

      expect(() => loadConfigFile(configPath)).toThrow(/metrics\.latency.*object/);
    });
  });

  describe("when a metrics entry under an empty-string key has an invalid value", () => {
    it("quotes the empty key so the path does not end with a trailing dot", () => {
      const { configPath } = createConfigFile({ metrics: { "": 5 } });

      expect(() => loadConfigFile(configPath)).toThrow(/metrics\."".*object/);
    });
  });

  describe("when a metrics entry has an invalid direction", () => {
    it.each([
      { description: "an unknown string", value: "sideways" },
      { description: "wrongly capitalized", value: "Lower" },
      { description: "a boolean", value: true },
      { description: "null", value: null },
    ])("throws naming metrics.latency.direction when it is $description", ({ value }) => {
      const { configPath } = createConfigFile({ metrics: { latency: { direction: value } } });

      expect(() => loadConfigFile(configPath)).toThrow(
        /metrics\.latency\.direction.*"lower".*"higher"/,
      );
    });
  });

  describe("when a metrics entry has a non-boolean flag", () => {
    it.each([
      { field: "gating", description: "a string", value: "yes" },
      { field: "gating", description: "null", value: null },
      { field: "exact", description: "a number", value: 1 },
    ])("throws naming metrics.latency.$field when it is $description", ({ field, value }) => {
      const { configPath } = createConfigFile({ metrics: { latency: { [field]: value } } });

      expect(() => loadConfigFile(configPath)).toThrow(
        new RegExp(`metrics\\.latency\\.${field}.*boolean`),
      );
    });
  });

  describe("when a metrics entry contains an unknown key", () => {
    it("throws an error that names the offending key", () => {
      const { configPath } = createConfigFile({
        metrics: { latency: { direction: "lower", threshold: "higher" } },
      });

      expect(() => loadConfigFile(configPath)).toThrow(/metrics\.latency\.threshold/);
    });
  });

  describe("when a metrics key embeds a line-break character", () => {
    it.each(LINE_BREAKS)("throws naming metrics when the key embeds $description", ({ char }) => {
      const smuggled = `latency${char}direction: 999, gating: 0`;
      const { configPath } = createConfigFile({
        metrics: { [smuggled]: { direction: "lower" } },
      });
      const act = (): void => {
        loadConfigFile(configPath);
      };

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(/metrics/);
    });
  });

  describe("when a kinds key embeds a line-break character", () => {
    it("throws naming kinds", () => {
      const smuggled = "memory\ngating: 999";
      const { configPath } = createConfigFile({ kinds: { [smuggled]: { gating: false } } });
      const act = (): void => {
        loadConfigFile(configPath);
      };

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(/kinds/);
    });
  });

  describe("when the config file has a kinds section", () => {
    it("returns the parsed per-kind overrides", () => {
      const config = { kinds: { memory: { gating: false }, time: {} } };
      const { configPath } = createConfigFile(config);

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual(config);
    });
  });

  describe("when kinds is not an object", () => {
    it.each([
      { description: "an array", value: [] },
      { description: "a string", value: "memory" },
      { description: "a number", value: 3 },
      { description: "null", value: null },
    ])("throws naming kinds when it is $description", ({ value }) => {
      const { configPath } = createConfigFile({ kinds: value });

      expect(() => loadConfigFile(configPath)).toThrow(/kinds.*object/);
    });
  });

  describe("when a kinds entry is not an object", () => {
    it("throws naming the offending kind", () => {
      const { configPath } = createConfigFile({ kinds: { memory: false } });

      expect(() => loadConfigFile(configPath)).toThrow(/kinds\.memory.*object/);
    });
  });

  describe("when a kinds entry has a non-boolean gating flag", () => {
    it.each([
      { description: "a string", value: "yes" },
      { description: "a number", value: 1 },
      { description: "null", value: null },
    ])("throws naming kinds.memory.gating when it is $description", ({ value }) => {
      const { configPath } = createConfigFile({ kinds: { memory: { gating: value } } });

      expect(() => loadConfigFile(configPath)).toThrow(/kinds\.memory\.gating.*boolean/);
    });
  });

  describe("when a kinds entry contains an unknown key", () => {
    it("throws an error that names the offending key by its dotted path", () => {
      const { configPath } = createConfigFile({
        kinds: { memory: { gating: false, threshold: 5 } },
      });

      expect(() => loadConfigFile(configPath)).toThrow(
        /Unknown config key: kinds\.memory\.threshold/,
      );
    });
  });

  describe("when the config file has a runbook key", () => {
    it("returns the parsed runbook value", () => {
      const { configPath } = createConfigFile({ runbook: "RUNBOOK.md" });

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual({ runbook: "RUNBOOK.md" });
    });
  });

  describe("when the config file has loop keys", () => {
    it("returns the parsed loop configuration", () => {
      const { configPath } = createConfigFile(LOOP_CONFIG);

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual(LOOP_CONFIG);
    });
  });

  describe("when hooks holds a command object", () => {
    it.each([
      { description: "only a before command", hooks: { before: "npm run warm-cache" } },
      { description: "only an after command", hooks: { after: "npm run cool-down" } },
    ])("returns the parsed hooks when it declares $description", ({ hooks }) => {
      const { configPath } = createConfigFile({ hooks });

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual({ hooks });
    });
  });

  describe("when hooks is not a command object", () => {
    it.each([
      { description: "the superseded directory-name string", value: "gymrat.hooks" },
      { description: "an array", value: [] },
      { description: "a boolean", value: true },
      { description: "null", value: null },
    ])("throws naming hooks when it is $description", ({ value }) => {
      const { configPath } = createConfigFile({ hooks: value });

      expect(() => loadConfigFile(configPath)).toThrow(/hooks.*object/);
    });
  });

  describe("when a hooks command is not a non-empty string", () => {
    it.each([
      { stage: "before", description: "an empty string", value: "" },
      { stage: "after", description: "an empty string", value: "" },
      { stage: "before", description: "a number", value: 42 },
      { stage: "after", description: "null", value: null },
    ])("throws naming hooks.$stage when it is $description", ({ stage, value }) => {
      const { configPath } = createConfigFile({ hooks: { [stage]: value } });
      const act = (): void => {
        loadConfigFile(configPath);
      };

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(new RegExp(`hooks\\.${stage}.*non-empty string`));
    });
  });

  describe("when the hooks section contains an unknown key", () => {
    it("throws an error that names the offending key by its dotted path", () => {
      const { configPath } = createConfigFile({
        hooks: { before: "npm run warm-cache", during: "npm run mid" },
      });

      expect(() => loadConfigFile(configPath)).toThrow(/Unknown config key: hooks\.during/);
    });
  });

  describe("when a stop field holds an invalid value", () => {
    it.each([
      {
        field: "targetValue",
        description: "a string",
        value: "fast",
        pattern: /stop\.targetValue.*number/,
      },
      {
        field: "maxIterations",
        description: "zero",
        value: 0,
        pattern: /stop\.maxIterations.*positive integer/,
      },
      {
        field: "maxIterations",
        description: "a non-integer",
        value: 1.5,
        pattern: /stop\.maxIterations.*positive integer/,
      },
    ])("throws naming stop.$field when it is $description", ({ field, value, pattern }) => {
      const { configPath } = createConfigFile({ stop: { [field]: value } });

      expect(() => loadConfigFile(configPath)).toThrow(pattern);
    });
  });

  describe("when the stop section contains an unknown key", () => {
    it("throws an error that names the offending key by its dotted path", () => {
      const { configPath } = createConfigFile({ stop: { targetValue: 1, patience: 3 } });

      expect(() => loadConfigFile(configPath)).toThrow(/Unknown config key: stop\.patience/);
    });
  });
});

describe("resolveConfig", () => {
  let tmpdir: string;
  restoreCwdAfterEach();

  describe("when flags and config are empty", () => {
    it("returns defaults for adapter, samples, timeoutSeconds, unstableNoisePct, primary, and no hooks", () => {
      // resolveConfig falls back to ./gymrat.json, so run from a dir that has none
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);

      const result = resolveConfig({ bench: "my-bench" });

      expect(result).toStrictEqual({
        bench: "my-bench",
        adapter: "metric-lines",
        samples: 10,
        timeoutSeconds: 1800,
        unstableNoisePct: 200,
        ...LOOP_DEFAULTS,
      });
    });
  });

  describe("when config file has values and flags are empty", () => {
    it("uses config file values over defaults", () => {
      const config = {
        bench: "config-bench",
        adapter: "custom-adapter",
        samples: 20,
        timeoutSeconds: 3600,
        unstableNoisePct: 150.5,
      };
      tmpdir = createConfigFile(config).dir;
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result).toStrictEqual({ ...config, ...LOOP_DEFAULTS });
    });
  });

  describe("when flags and config both provide values", () => {
    it("uses flag values over config file values", () => {
      tmpdir = createConfigFile({
        bench: "config-bench",
        adapter: "config-adapter",
        samples: 20,
      }).dir;
      process.chdir(tmpdir);

      const result = resolveConfig({
        bench: "flag-bench",
        adapter: "flag-adapter",
      });

      expect(result).toStrictEqual({
        bench: "flag-bench",
        adapter: "flag-adapter",
        samples: 20,
        timeoutSeconds: 1800,
        unstableNoisePct: 200,
        ...LOOP_DEFAULTS,
      });
    });
  });

  describe("when flags provide values and no config", () => {
    it("uses flag values over defaults", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);

      const result = resolveConfig({
        bench: "flag-bench",
        adapter: "flag-adapter",
        samples: 25,
        timeout: 900,
      });

      expect(result).toStrictEqual({
        bench: "flag-bench",
        adapter: "flag-adapter",
        samples: 25,
        timeoutSeconds: 900,
        unstableNoisePct: 200,
        ...LOOP_DEFAULTS,
      });
    });
  });

  describe("when bench is missing from both flags and config", () => {
    it("throws a GymratError that mentions --bench and the config file", () => {
      tmpdir = createConfigFile({}).dir;
      process.chdir(tmpdir);
      const act = (): ResolvedConfig => resolveConfig({});

      expect.soft(act).toThrow(GymratError);
      expect.soft(act).toThrow(/--bench/);
      expect(act).toThrow(/config file/);
    });
  });

  describe("when prepare is provided", () => {
    it("includes prepare in resolved config", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);

      const result = resolveConfig({
        bench: "my-bench",
        prepare: "prepare-cmd",
      });

      expect(result.prepare).toBe("prepare-cmd");
    });
  });

  describe("when a non-empty-string flag holds an empty string", () => {
    it.each([
      { key: "bench", flags: { bench: "" } },
      { key: "prepare", flags: { bench: "my-bench", prepare: "" } },
      { key: "adapter", flags: { bench: "my-bench", adapter: "" } },
      { key: "config", flags: { bench: "my-bench", config: "" } },
    ])("throws naming --$key and the non-empty requirement", ({ key, flags }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      const act = (): ResolvedConfig => resolveConfig(flags);

      // The flag spelling, not the config key: the flag is what the user typed.
      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(new RegExp(`--${key}.*non-empty`));
    });
  });

  describe("when config file path is specified", () => {
    it("loads config from the specified path", () => {
      tmpdir = freshRoot("gymrat-");
      const customConfigPath = path.join(tmpdir, "custom-config.json");
      fs.writeFileSync(customConfigPath, JSON.stringify({ bench: "custom-bench" }));

      const result = resolveConfig({ config: customConfigPath });

      expect(result.bench).toBe("custom-bench");
    });
  });

  describe("when the specified config file path does not exist", () => {
    it("throws an error naming the missing path instead of falling back to defaults", () => {
      tmpdir = freshRoot("gymrat-");
      const missingConfigPath = path.join(tmpdir, "typo.json");

      expect(() => resolveConfig({ bench: "my-bench", config: missingConfigPath })).toThrow(
        missingConfigPath,
      );
    });
  });

  describe("when the config file has a metrics section", () => {
    it("propagates the per-metric overrides to the resolved config", () => {
      const metrics = {
        "decode/time": { direction: "higher" as const, gating: false, exact: true },
      };
      tmpdir = createConfigFile({ bench: "config-bench", metrics }).dir;
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result).toStrictEqual({
        bench: "config-bench",
        adapter: "metric-lines",
        samples: 10,
        timeoutSeconds: 1800,
        unstableNoisePct: 200,
        ...LOOP_DEFAULTS,
        metrics: metricRecord(metrics),
      });
    });
  });

  describe("when the config file has a kinds section", () => {
    it("propagates the per-kind overrides to the resolved config", () => {
      const kinds = { memory: { gating: false } };
      tmpdir = createConfigFile({ bench: "config-bench", kinds }).dir;
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result).toStrictEqual({
        bench: "config-bench",
        adapter: "metric-lines",
        samples: 10,
        timeoutSeconds: 1800,
        unstableNoisePct: 200,
        ...LOOP_DEFAULTS,
        kinds: metricRecord(kinds),
      });
    });
  });

  describe("when the config file has no metrics section", () => {
    it("omits metrics from the resolved config", () => {
      tmpdir = createConfigFile({ bench: "config-bench" }).dir;
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result).not.toHaveProperty("metrics");
    });
  });

  describe("when timeout flag is provided", () => {
    it("uses timeout from flags over config file value", () => {
      tmpdir = createConfigFile({ timeoutSeconds: 3600 }).dir;
      process.chdir(tmpdir);

      const result = resolveConfig({
        bench: "my-bench",
        timeout: 1200,
      });

      expect(result.timeoutSeconds).toBe(1200);
    });
  });

  describe("when the config file has loop keys", () => {
    it("propagates them to the resolved config over the loop defaults", () => {
      tmpdir = createConfigFile({ bench: "config-bench", ...LOOP_CONFIG }).dir;
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result).toStrictEqual({
        bench: "config-bench",
        adapter: "metric-lines",
        samples: 10,
        timeoutSeconds: 1800,
        unstableNoisePct: 200,
        ...LOOP_CONFIG,
      });
    });
  });

  describe("when filter omits the {names} placeholder", () => {
    it("throws a GymratError naming filter and the required placeholder", () => {
      tmpdir = createConfigFile({ bench: "config-bench", filter: "npm run bench" }).dir;
      process.chdir(tmpdir);
      const act = (): ResolvedConfig => resolveConfig({});

      expect.soft(act).toThrow(GymratError);
      expect.soft(act).toThrow(/filter/);
      expect(act).toThrow(/\{names\}/);
    });
  });

  describe("when stop.targetValue is combined with a geomean primary", () => {
    it.each([
      { description: "explicitly", overrides: { primary: "geomean" } },
      { description: "by default", overrides: {} },
    ])("throws a GymratError when geomean is chosen $description", ({ overrides }) => {
      tmpdir = createConfigFile({
        bench: "config-bench",
        stop: { targetValue: 1.5 },
        ...overrides,
      }).dir;
      process.chdir(tmpdir);
      const act = (): ResolvedConfig => resolveConfig({});

      expect.soft(act).toThrow(GymratError);
      expect.soft(act).toThrow(/targetValue/);
      expect(act).toThrow(/geomean/);
    });
  });

  describe("when stop sets only maxIterations and primary defaults to geomean", () => {
    it("resolves without demanding a named primary metric", () => {
      tmpdir = createConfigFile({ bench: "config-bench", stop: { maxIterations: 5 } }).dir;
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result.stop).toStrictEqual({ maxIterations: 5 });
    });
  });
});
