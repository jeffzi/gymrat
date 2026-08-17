import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { Adapter } from "../src/adapters/types.js";
import type { BenchlessConfig, CliFlags, ResolvedConfig } from "../src/config.js";
import {
  inspectConfig,
  loadConfigFile,
  resolveBenchlessConfig,
  resolveConfig,
  resolveMetricMeta,
} from "../src/config.js";
import { GymratError } from "../src/errors.js";
import { createMockAdapter as createMockAdapterBase } from "./fixtures/adapter.js";
import { metricMeta } from "./fixtures/comparison-result.js";
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

function createMockAdapter(
  defaults: Adapter["defaults"] = () => ({ direction: "lower" as const }),
): Adapter {
  return createMockAdapterBase({ name: "test-adapter", defaults });
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

describe("resolveConfig, GYMRAT_* environment variables", () => {
  let tmpdir: string;
  restoreCwdAfterEach();
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe("when a GYMRAT_* env var is set and its corresponding flag is absent", () => {
    it.each([
      {
        envVar: "GYMRAT_BENCH",
        envValue: "env-bench",
        flags: {},
        field: "bench",
        expected: "env-bench",
      },
      {
        envVar: "GYMRAT_PREPARE",
        envValue: "env-prepare",
        flags: { bench: "b" },
        field: "prepare",
        expected: "env-prepare",
      },
      {
        envVar: "GYMRAT_ADAPTER",
        envValue: "env-adapter",
        flags: { bench: "b" },
        field: "adapter",
        expected: "env-adapter",
      },
      {
        envVar: "GYMRAT_SAMPLES",
        envValue: "42",
        flags: { bench: "b" },
        field: "samples",
        expected: 42,
      },
      {
        envVar: "GYMRAT_TIMEOUT",
        envValue: "900",
        flags: { bench: "b" },
        field: "timeoutSeconds",
        expected: 900,
      },
    ])("uses $envVar as $field", ({ envVar, envValue, flags, field, expected }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv(envVar, envValue);

      const result = resolveConfig(flags);

      expect(result).toHaveProperty(field, expected);
    });
  });

  describe("when a GYMRAT_* env var is set and the config file provides the same field", () => {
    it.each([
      {
        envVar: "GYMRAT_BENCH",
        envValue: "env-bench",
        config: { bench: "config-bench" },
        field: "bench",
        expected: "env-bench",
      },
      {
        envVar: "GYMRAT_PREPARE",
        envValue: "env-prepare",
        config: { bench: "b", prepare: "config-prepare" },
        field: "prepare",
        expected: "env-prepare",
      },
      {
        envVar: "GYMRAT_ADAPTER",
        envValue: "env-adapter",
        config: { bench: "b", adapter: "config-adapter" },
        field: "adapter",
        expected: "env-adapter",
      },
      {
        envVar: "GYMRAT_SAMPLES",
        envValue: "42",
        config: { bench: "b", samples: 20 },
        field: "samples",
        expected: 42,
      },
      {
        envVar: "GYMRAT_TIMEOUT",
        envValue: "900",
        config: { bench: "b", timeoutSeconds: 3600 },
        field: "timeoutSeconds",
        expected: 900,
      },
    ])(
      "uses $envVar over the config file for $field",
      ({ envVar, envValue, config, field, expected }) => {
        tmpdir = createConfigFile(config).dir;
        process.chdir(tmpdir);
        vi.stubEnv(envVar, envValue);

        const result = resolveConfig({});

        expect(result).toHaveProperty(field, expected);
      },
    );
  });

  describe("when a string GYMRAT_* env var holds an empty string", () => {
    it.each([
      { envVar: "GYMRAT_BENCH", flags: {} },
      { envVar: "GYMRAT_PREPARE", flags: { bench: "b" } },
      { envVar: "GYMRAT_ADAPTER", flags: { bench: "b" } },
    ])("throws a GymratError naming $envVar", ({ envVar, flags }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv(envVar, "");
      const act = (): ResolvedConfig => resolveConfig(flags);

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(new RegExp(`${envVar}.*non-empty`));
    });
  });

  describe("when GYMRAT_SAMPLES holds an invalid value", () => {
    it.each([
      { description: "a non-numeric string", value: "abc" },
      { description: "a non-integer", value: "1.5" },
      { description: "zero", value: "0" },
      { description: "a negative number", value: "-1" },
      { description: "an empty string", value: "" },
    ])("throws a GymratError naming GYMRAT_SAMPLES when it is $description", ({ value }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_SAMPLES", value);
      const act = (): ResolvedConfig => resolveConfig({ bench: "my-bench" });

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(/GYMRAT_SAMPLES.*positive integer/);
    });
  });

  describe("when GYMRAT_TIMEOUT holds an invalid value", () => {
    it.each([
      { description: "a non-numeric string", value: "abc" },
      { description: "a non-integer", value: "1.5" },
      { description: "zero", value: "0" },
      { description: "a negative number", value: "-1" },
      { description: "an empty string", value: "" },
    ])("throws a GymratError naming GYMRAT_TIMEOUT when it is $description", ({ value }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_TIMEOUT", value);
      const act = (): ResolvedConfig => resolveConfig({ bench: "my-bench" });

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(/GYMRAT_TIMEOUT.*positive integer/);
    });
  });

  describe("when GYMRAT_TIMEOUT exceeds the millisecond timer cap", () => {
    it("throws a GymratError naming GYMRAT_TIMEOUT and the cap", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_TIMEOUT", "2147484");
      const act = (): ResolvedConfig => resolveConfig({ bench: "my-bench" });

      expect.soft(act).toThrow(GymratError);
      expect.soft(act).toThrow(/GYMRAT_TIMEOUT/);
      expect(act).toThrow(/no greater than 2147483/);
    });
  });

  describe("when GYMRAT_CONFIG names an existing config file and --config is absent", () => {
    it("loads config from the env-specified path", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      const envConfigPath = path.join(tmpdir, "env-config.json");
      fs.writeFileSync(envConfigPath, JSON.stringify({ bench: "env-config-bench" }));
      vi.stubEnv("GYMRAT_CONFIG", envConfigPath);

      const result = resolveConfig({});

      expect(result.bench).toBe("env-config-bench");
    });
  });

  describe("when GYMRAT_CONFIG names a file that does not exist", () => {
    it("throws an error naming the missing path", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      const missingPath = path.join(tmpdir, "typo.json");
      vi.stubEnv("GYMRAT_CONFIG", missingPath);

      expect(() => resolveConfig({ bench: "my-bench" })).toThrow(missingPath);
    });
  });

  describe("when GYMRAT_CONFIG is set", () => {
    it("bypasses the implicit gymrat.json in the working directory", () => {
      tmpdir = createConfigFile({ bench: "implicit-bench", adapter: "implicit-adapter" }).dir;
      process.chdir(tmpdir);
      const envConfigPath = path.join(tmpdir, "alt-config.json");
      fs.writeFileSync(envConfigPath, JSON.stringify({ bench: "alt-bench" }));
      vi.stubEnv("GYMRAT_CONFIG", envConfigPath);

      const result = resolveConfig({});

      expect(result.bench).toBe("alt-bench");
      // adapter falls back to default, not the implicit gymrat.json value
      expect(result.adapter).toBe("metric-lines");
    });
  });

  describe("when GYMRAT_CONFIG holds an empty string", () => {
    it("throws a GymratError naming GYMRAT_CONFIG", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_CONFIG", "");
      const act = (): ResolvedConfig => resolveConfig({ bench: "my-bench" });

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(/GYMRAT_CONFIG.*non-empty/);
    });
  });
});

describe("resolveBenchlessConfig", () => {
  let tmpdir: string;
  restoreCwdAfterEach();

  describe("when the config flag holds an empty string", () => {
    it("throws naming --config and the non-empty requirement", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      const act = (): BenchlessConfig => resolveBenchlessConfig({ config: "" });

      expect.soft(act).toThrow(GymratError);
      expect(act).toThrow(/--config.*non-empty/);
    });
  });
});

/**
 * Both resolvers settle the implicit `gymrat.json` through the same lookup, so a
 * base directory has to reach it through either one.
 */
const CONFIG_RESOLVERS = [
  {
    name: "resolveConfig",
    resolve: (flags: CliFlags, baseDir?: string): BenchlessConfig => resolveConfig(flags, baseDir),
  },
  {
    name: "resolveBenchlessConfig",
    resolve: (flags: CliFlags, baseDir?: string): BenchlessConfig =>
      resolveBenchlessConfig(flags, baseDir),
  },
];

describe.each(CONFIG_RESOLVERS)("$name, given a base directory", ({ resolve }) => {
  restoreCwdAfterEach();

  /**
   * A directory holding a `gymrat.json` of `baseConfig`, with a nested directory
   * holding one of `nestedConfig` — the two the lookup has to choose between.
   */
  function createNestedConfigDirs(
    baseConfig: Record<string, unknown>,
    nestedConfig: Record<string, unknown>,
  ): { baseDir: string; nestedDir: string } {
    const baseDir = freshRoot("gymrat-");
    const nestedDir = path.join(baseDir, "packages", "core");
    fs.mkdirSync(nestedDir, { recursive: true });
    fs.writeFileSync(path.join(baseDir, "gymrat.json"), JSON.stringify(baseConfig));
    fs.writeFileSync(path.join(nestedDir, "gymrat.json"), JSON.stringify(nestedConfig));
    return { baseDir, nestedDir };
  }

  describe("when the base directory and the working directory each hold a gymrat.json", () => {
    it("reads the base directory's", () => {
      // Arrange
      const { baseDir, nestedDir } = createNestedConfigDirs(
        { bench: "a-bench", checks: "base-checks" },
        { bench: "a-bench", checks: "cwd-checks" },
      );
      process.chdir(nestedDir);

      // Act
      const result = resolve({}, baseDir);

      // Assert
      expect(result.checks).toBe("base-checks");
    });
  });

  describe("when --config names a path relative to the working directory", () => {
    it("reads the named file, leaving the base directory's gymrat.json unread", () => {
      // Arrange
      const { baseDir, nestedDir } = createNestedConfigDirs(
        { bench: "a-bench", checks: "base-checks" },
        { bench: "a-bench", checks: "cwd-checks" },
      );
      fs.writeFileSync(
        path.join(nestedDir, "custom.json"),
        JSON.stringify({ bench: "a-bench", checks: "named-checks" }),
      );
      process.chdir(nestedDir);

      // Act
      const result = resolve({ config: "custom.json" }, baseDir);

      // Assert
      expect(result.checks).toBe("named-checks");
    });
  });
});

describe.each(CONFIG_RESOLVERS)("$name, runbook resolution", ({ resolve }) => {
  let tmpdir: string;
  restoreCwdAfterEach();

  describe("when the config file has no runbook key", () => {
    it("omits runbook from the resolved config", () => {
      tmpdir = createConfigFile({ bench: "a-bench" }).dir;
      process.chdir(tmpdir);

      const result = resolve({});

      expect(result).not.toHaveProperty("runbook");
    });
  });

  describe("when the config file names a runbook that exists as a file", () => {
    it("resolves the runbook path to an absolute path against the config directory", () => {
      tmpdir = createConfigFile({ bench: "a-bench", runbook: "RUNBOOK.md" }).dir;
      fs.writeFileSync(path.join(tmpdir, "RUNBOOK.md"), "# Steps\n");
      process.chdir(tmpdir);

      const result = resolve({});

      expect(result.runbook).toBe(path.join(tmpdir, "RUNBOOK.md"));
    });
  });

  describe("when the runbook path does not resolve to an existing file", () => {
    it("throws a GymratError naming the field and the path", () => {
      tmpdir = createConfigFile({ bench: "a-bench", runbook: "missing.md" }).dir;
      process.chdir(tmpdir);
      const act = (): BenchlessConfig => resolve({});

      expect.soft(act).toThrow(GymratError);
      expect.soft(act).toThrow(/runbook/);
      expect(act).toThrow(/missing\.md/);
    });
  });

  describe("when the runbook path points to a directory instead of a file", () => {
    it("throws a GymratError naming the field and the path", () => {
      tmpdir = createConfigFile({ bench: "a-bench", runbook: "docs" }).dir;
      fs.mkdirSync(path.join(tmpdir, "docs"));
      process.chdir(tmpdir);
      const act = (): BenchlessConfig => resolve({});

      expect.soft(act).toThrow(GymratError);
      expect.soft(act).toThrow(/runbook/);
      expect(act).toThrow(/docs/);
    });
  });
});

describe.each(CONFIG_RESOLVERS)("$name, implicit lookup in a git repository", ({ resolve }) => {
  restoreCwdAfterEach();

  describe("when no baseDir is passed and the cwd is inside a git repository", () => {
    it("finds gymrat.json at the repository root, not the cwd", () => {
      const repoRoot = freshRoot("gymrat-repo-");
      execFileSync("git", ["init"], { cwd: repoRoot, stdio: "ignore" });
      fs.writeFileSync(
        path.join(repoRoot, "gymrat.json"),
        JSON.stringify({ bench: "repo-bench", checks: "repo-checks" }),
      );
      const nestedDir = path.join(repoRoot, "packages", "core");
      fs.mkdirSync(nestedDir, { recursive: true });
      process.chdir(nestedDir);

      const result = resolve({ bench: "flag-bench" });

      expect(result.checks).toBe("repo-checks");
    });
  });
});

describe("resolveMetricMeta", () => {
  describe("when configMetrics is undefined and adapter returns direction only", () => {
    it("defaults gating true, exact false, kind other, and shortName to the metric name", () => {
      const mockAdapter = createMockAdapter();

      const result = resolveMetricMeta(["response-time"], undefined, mockAdapter);

      expect(result).toStrictEqual(metricRecord({ "response-time": metricMeta("response-time") }));
    });
  });

  describe("when adapter returns direction with unit", () => {
    it("includes the unit in resolved metadata", () => {
      const mockAdapter = createMockAdapter(() => ({
        direction: "lower" as const,
        unit: "ns",
      }));

      const result = resolveMetricMeta(["response-time"], undefined, mockAdapter);

      expect(result).toStrictEqual(
        metricRecord({ "response-time": metricMeta("response-time", { unit: "ns" }) }),
      );
    });
  });

  describe("when adapter reports a kind and a short name", () => {
    it("carries both onto the resolved metadata", () => {
      const mockAdapter = createMockAdapter(() => ({
        direction: "lower" as const,
        kind: "memory",
        shortName: "heap",
      }));

      const result = resolveMetricMeta(["bench-a/heap"], undefined, mockAdapter);

      expect(result).toStrictEqual(
        metricRecord({ "bench-a/heap": metricMeta("heap", { kind: "memory" }) }),
      );
    });
  });

  describe("when config provides direction override", () => {
    it("overrides adapter direction", () => {
      const mockAdapter = createMockAdapter();
      const configMetrics = {
        throughput: { direction: "higher" as const },
      };

      const result = resolveMetricMeta(["throughput"], configMetrics, mockAdapter);

      expect(result).toStrictEqual(
        metricRecord({ throughput: metricMeta("throughput", { direction: "higher" }) }),
      );
    });
  });

  describe("when config provides gating override", () => {
    it("overrides the default gating true", () => {
      const mockAdapter = createMockAdapter();
      const configMetrics = {
        "response-time": { gating: false },
      };

      const result = resolveMetricMeta(["response-time"], configMetrics, mockAdapter);

      expect(result).toStrictEqual(
        metricRecord({ "response-time": metricMeta("response-time", { gating: false }) }),
      );
    });
  });

  describe("when config provides exact override", () => {
    it("overrides the default exact false", () => {
      const mockAdapter = createMockAdapter();
      const configMetrics = {
        "response-time": { exact: true },
      };

      const result = resolveMetricMeta(["response-time"], configMetrics, mockAdapter);

      expect(result).toStrictEqual(
        metricRecord({ "response-time": metricMeta("response-time", { exact: true }) }),
      );
    });
  });

  describe("when config provides only gating override (partial)", () => {
    it("uses adapter direction, overrides gating, defaults exact to false", () => {
      const mockAdapter = createMockAdapter(() => ({
        direction: "lower" as const,
        unit: "bytes",
      }));
      const configMetrics = {
        "memory-usage": { gating: false },
      };

      const result = resolveMetricMeta(["memory-usage"], configMetrics, mockAdapter);

      expect(result).toStrictEqual(
        metricRecord({
          "memory-usage": metricMeta("memory-usage", { unit: "bytes", gating: false }),
        }),
      );
    });
  });

  describe("when multiple metric names are provided", () => {
    it("resolves metadata for each metric", () => {
      const mockAdapter = createMockAdapter((name: string) => {
        if (name === "response-time") return { direction: "lower" as const, unit: "ns" };
        if (name === "throughput") return { direction: "higher" as const };
        return { direction: "lower" as const };
      });
      const configMetrics = {
        "response-time": { gating: false },
        throughput: { exact: true },
      };

      const result = resolveMetricMeta(["response-time", "throughput"], configMetrics, mockAdapter);

      expect(result).toStrictEqual(
        metricRecord({
          "response-time": metricMeta("response-time", { unit: "ns", gating: false }),
          throughput: metricMeta("throughput", { direction: "higher", exact: true }),
        }),
      );
    });
  });

  describe("when config contains entries for nonexistent metrics", () => {
    it("ignores config for metrics not in metricNames", () => {
      const mockAdapter = createMockAdapter();
      const configMetrics = {
        "response-time": { gating: false },
        unused: { gating: true, exact: true },
      };

      const result = resolveMetricMeta(["response-time"], configMetrics, mockAdapter);

      expect(result).toStrictEqual(
        metricRecord({ "response-time": metricMeta("response-time", { gating: false }) }),
      );
    });
  });

  describe("when a kind sets gating", () => {
    it("applies that gating only to the metrics reporting that kind", () => {
      const mockAdapter = createMockAdapter((name: string) =>
        name.endsWith("/heap")
          ? { direction: "lower" as const, kind: "memory", shortName: "heap" }
          : { direction: "lower" as const, kind: "time", shortName: "time" },
      );
      const configKinds = { memory: { gating: false } };

      const result = resolveMetricMeta(
        ["bench-a/heap", "bench-a/time"],
        undefined,
        mockAdapter,
        configKinds,
      );

      expect(result).toStrictEqual(
        metricRecord({
          "bench-a/heap": metricMeta("heap", { gating: false, kind: "memory" }),
          "bench-a/time": metricMeta("time", { kind: "time" }),
        }),
      );
    });
  });

  describe("when a kind entry names a kind no metric reports", () => {
    it("ignores the entry", () => {
      const mockAdapter = createMockAdapter(() => ({
        direction: "lower" as const,
        kind: "memory",
        shortName: "heap",
      }));
      const configKinds = { io: { gating: false } };

      const result = resolveMetricMeta(["bench-a/heap"], undefined, mockAdapter, configKinds);

      expect(result).toStrictEqual(
        metricRecord({ "bench-a/heap": metricMeta("heap", { kind: "memory" }) }),
      );
    });
  });

  describe("when a metric entry and its kind disagree about gating", () => {
    it("lets the exact metric name win over the kind", () => {
      const mockAdapter = createMockAdapter((name: string) => ({
        direction: "lower" as const,
        kind: "memory",
        shortName: name.split("/").at(-1) ?? name,
      }));
      const configMetrics = { "bench-a/heap": { gating: true } };
      const configKinds = { memory: { gating: false } };

      const result = resolveMetricMeta(
        ["bench-a/heap", "bench-a/rss"],
        configMetrics,
        mockAdapter,
        configKinds,
      );

      expect(result).toStrictEqual(
        metricRecord({
          "bench-a/heap": metricMeta("heap", { kind: "memory" }),
          "bench-a/rss": metricMeta("rss", { kind: "memory", gating: false }),
        }),
      );
    });
  });
});

describe("inspectConfig", () => {
  let tmpdir: string;
  restoreCwdAfterEach();

  describe("when no config file exists and flags are empty", () => {
    it("returns all defaults with no problems, no bench, and no config path", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);

      const result = inspectConfig({});

      expect(result).toStrictEqual({
        configPath: undefined,
        configExists: false,
        problems: [],
        config: {
          adapter: "metric-lines",
          samples: 10,
          timeoutSeconds: 1800,
          unstableNoisePct: 200,
          primary: "geomean",
        },
      });
    });
  });

  describe("when flags provide bench and no config file exists", () => {
    it("includes bench alongside the default settled config", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);

      const result = inspectConfig({ bench: "flag-bench" });

      expect.soft(result.problems).toStrictEqual([]);
      expect.soft(result.config).toStrictEqual({
        adapter: "metric-lines",
        samples: 10,
        timeoutSeconds: 1800,
        unstableNoisePct: 200,
        primary: "geomean",
      });
      expect(result.bench).toBe("flag-bench");
    });
  });

  describe("when a valid config file provides values and bench", () => {
    it("returns the settled config with bench and the config path", () => {
      const fileConfig = {
        bench: "config-bench",
        adapter: "custom-adapter",
        samples: 20,
        timeoutSeconds: 3600,
        unstableNoisePct: 150.5,
      };
      tmpdir = createConfigFile(fileConfig).dir;
      process.chdir(tmpdir);

      const result = inspectConfig({});

      expect.soft(result.configPath).toBe(path.join(tmpdir, "gymrat.json"));
      expect.soft(result.configExists).toBe(true);
      expect.soft(result.problems).toStrictEqual([]);
      expect.soft(result.config).toStrictEqual({
        adapter: "custom-adapter",
        samples: 20,
        timeoutSeconds: 3600,
        unstableNoisePct: 150.5,
        primary: "geomean",
      });
      expect(result.bench).toBe("config-bench");
    });
  });

  describe("when flags override config file values", () => {
    it("uses flag values in the settled config", () => {
      tmpdir = createConfigFile({
        bench: "config-bench",
        adapter: "config-adapter",
        samples: 20,
      }).dir;
      process.chdir(tmpdir);

      const result = inspectConfig({ bench: "flag-bench", adapter: "flag-adapter" });

      expect.soft(result.problems).toStrictEqual([]);
      expect.soft(result.config?.adapter).toBe("flag-adapter");
      expect.soft(result.config?.samples).toBe(20);
      expect(result.bench).toBe("flag-bench");
    });
  });

  describe("when the config file has loop keys", () => {
    it("includes them in the settled config", () => {
      tmpdir = createConfigFile({ bench: "config-bench", ...LOOP_CONFIG }).dir;
      process.chdir(tmpdir);

      const result = inspectConfig({});

      expect.soft(result.problems).toStrictEqual([]);
      expect.soft(result.config).toStrictEqual({
        adapter: "metric-lines",
        samples: 10,
        timeoutSeconds: 1800,
        unstableNoisePct: 200,
        ...LOOP_CONFIG,
      });
      expect(result.bench).toBe("config-bench");
    });
  });

  describe("when the config file names a runbook that exists", () => {
    it("resolves the runbook path in the settled config", () => {
      tmpdir = createConfigFile({ bench: "config-bench", runbook: "RUNBOOK.md" }).dir;
      fs.writeFileSync(path.join(tmpdir, "RUNBOOK.md"), "# Steps\n");
      process.chdir(tmpdir);

      const result = inspectConfig({});

      expect.soft(result.problems).toStrictEqual([]);
      expect(result.config?.runbook).toBe(path.join(tmpdir, "RUNBOOK.md"));
    });
  });

  describe("when --config names a path that does not exist", () => {
    it("reports a problem naming the missing path and omits the settled config", () => {
      tmpdir = freshRoot("gymrat-");
      const missingPath = path.join(tmpdir, "typo.json");

      const result = inspectConfig({ bench: "my-bench", config: missingPath });

      expect.soft(result.configPath).toBe(missingPath);
      expect.soft(result.configExists).toBe(false);
      expect
        .soft(result.problems)
        .toEqual(expect.arrayContaining([expect.stringContaining(missingPath)]));
      expect(result).not.toHaveProperty("config");
    });
  });

  describe("when the config file contains invalid JSON", () => {
    it("reports a problem that includes the file path", () => {
      const { dir, configPath } = writeRawConfigFile("{ invalid json }");
      process.chdir(dir);

      const result = inspectConfig({});

      expect.soft(result.configPath).toBe(configPath);
      expect.soft(result.configExists).toBe(true);
      expect
        .soft(result.problems)
        .toEqual(expect.arrayContaining([expect.stringContaining(configPath)]));
      expect(result).not.toHaveProperty("config");
    });
  });

  describe("when the config file JSON root is not an object", () => {
    it("reports a problem mentioning JSON object", () => {
      const { dir } = writeRawConfigFile("[]");
      process.chdir(dir);

      const result = inspectConfig({});

      expect
        .soft(result.problems)
        .toEqual(expect.arrayContaining([expect.stringMatching(/JSON object/)]));
      expect(result).not.toHaveProperty("config");
    });
  });

  describe("when the config file has multiple schema issues", () => {
    it("collects all issues instead of stopping at the first", () => {
      tmpdir = createConfigFile({
        bench: 42,
        samples: "bad",
        adapter: 123,
      }).dir;
      process.chdir(tmpdir);

      const result = inspectConfig({});

      expect(result.problems.length).toBeGreaterThanOrEqual(3);
      expect(result.problems).toEqual(
        expect.arrayContaining([
          expect.stringMatching(/bench/),
          expect.stringMatching(/samples/),
          expect.stringMatching(/adapter/),
        ]),
      );
      expect(result).not.toHaveProperty("config");
    });
  });

  describe("when filter omits the {names} placeholder", () => {
    it("reports a problem naming filter and the required placeholder", () => {
      tmpdir = createConfigFile({
        bench: "config-bench",
        filter: "npm run bench",
      }).dir;
      process.chdir(tmpdir);

      const result = inspectConfig({});

      expect(result.problems).toEqual(
        expect.arrayContaining([expect.stringMatching(/filter.*\{names\}/)]),
      );
    });
  });

  describe("when stop.targetValue is combined with a geomean primary", () => {
    it("reports a problem naming targetValue and geomean", () => {
      tmpdir = createConfigFile({
        bench: "config-bench",
        stop: { targetValue: 1.5 },
      }).dir;
      process.chdir(tmpdir);

      const result = inspectConfig({});

      expect(result.problems).toEqual(
        expect.arrayContaining([
          expect.stringMatching(/targetValue.*geomean|geomean.*targetValue/),
        ]),
      );
    });
  });

  describe("when the runbook path does not resolve to an existing file", () => {
    it("reports a problem naming the field and the path", () => {
      tmpdir = createConfigFile({
        bench: "config-bench",
        runbook: "missing.md",
      }).dir;
      process.chdir(tmpdir);

      const result = inspectConfig({});

      expect
        .soft(result.problems)
        .toEqual(expect.arrayContaining([expect.stringMatching(/runbook/)]));
      expect(result.problems.join("\n")).toMatch(/missing\.md/);
    });
  });

  describe("when a flag holds an empty string", () => {
    it.each([
      { key: "bench", flags: { bench: "" } },
      { key: "prepare", flags: { bench: "my-bench", prepare: "" } },
      { key: "adapter", flags: { bench: "my-bench", adapter: "" } },
      { key: "config", flags: { bench: "my-bench", config: "" } },
    ])("reports a problem naming --$key", ({ key, flags }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);

      const result = inspectConfig(flags);

      expect(result.problems).toEqual(
        expect.arrayContaining([expect.stringMatching(new RegExp(`--${key}.*non-empty`))]),
      );
    });
  });

  describe("when multiple flags hold empty strings", () => {
    it("collects all empty-string problems", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);

      const result = inspectConfig({ bench: "", adapter: "" });

      expect(result.problems.length).toBeGreaterThanOrEqual(2);
      expect(result.problems).toEqual(
        expect.arrayContaining([
          expect.stringMatching(/--bench.*non-empty/),
          expect.stringMatching(/--adapter.*non-empty/),
        ]),
      );
    });
  });

  describe("when a base directory is provided", () => {
    it("reads the base directory's gymrat.json instead of the cwd's", () => {
      const baseDir = freshRoot("gymrat-");
      fs.writeFileSync(
        path.join(baseDir, "gymrat.json"),
        JSON.stringify({ bench: "base-bench", checks: "base-checks" }),
      );
      tmpdir = freshRoot("gymrat-");
      fs.writeFileSync(
        path.join(tmpdir, "gymrat.json"),
        JSON.stringify({ bench: "cwd-bench", checks: "cwd-checks" }),
      );
      process.chdir(tmpdir);

      const result = inspectConfig({}, baseDir);

      expect.soft(result.bench).toBe("base-bench");
      expect(result.configPath).toBe(path.join(baseDir, "gymrat.json"));
    });
  });
});

describe("inspectConfig, GYMRAT_* environment variables", () => {
  let tmpdir: string;
  restoreCwdAfterEach();
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe("when a string GYMRAT_* env var holds an empty string", () => {
    it.each([
      { envVar: "GYMRAT_BENCH", flags: {} },
      { envVar: "GYMRAT_PREPARE", flags: { bench: "b" } },
      { envVar: "GYMRAT_ADAPTER", flags: { bench: "b" } },
      { envVar: "GYMRAT_CONFIG", flags: { bench: "b" } },
    ])("reports a problem naming $envVar", ({ envVar, flags }) => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv(envVar, "");

      const result = inspectConfig(flags);

      expect(result.problems).toEqual(
        expect.arrayContaining([expect.stringMatching(new RegExp(`${envVar}.*non-empty`))]),
      );
    });
  });

  describe("when GYMRAT_CONFIG names a file that does not exist", () => {
    it("reports a problem naming the missing path", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      const missingPath = path.join(tmpdir, "typo.json");
      vi.stubEnv("GYMRAT_CONFIG", missingPath);

      const result = inspectConfig({ bench: "my-bench" });

      expect(result.problems).toEqual(
        expect.arrayContaining([expect.stringContaining(missingPath)]),
      );
    });
  });

  describe("when GYMRAT_SAMPLES holds an invalid value", () => {
    it("reports a problem naming GYMRAT_SAMPLES", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_SAMPLES", "abc");

      const result = inspectConfig({ bench: "my-bench" });

      expect(result.problems).toEqual(
        expect.arrayContaining([expect.stringMatching(/GYMRAT_SAMPLES.*positive integer/)]),
      );
    });
  });

  describe("when GYMRAT_TIMEOUT holds an invalid value", () => {
    it("reports a problem naming GYMRAT_TIMEOUT", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_TIMEOUT", "abc");

      const result = inspectConfig({ bench: "my-bench" });

      expect(result.problems).toEqual(
        expect.arrayContaining([expect.stringMatching(/GYMRAT_TIMEOUT.*positive integer/)]),
      );
    });
  });

  describe("when GYMRAT_TIMEOUT exceeds the millisecond timer cap", () => {
    it("reports a problem naming GYMRAT_TIMEOUT and the cap", () => {
      tmpdir = freshRoot("gymrat-");
      process.chdir(tmpdir);
      vi.stubEnv("GYMRAT_TIMEOUT", "2147484");

      const result = inspectConfig({ bench: "my-bench" });

      expect
        .soft(result.problems)
        .toEqual(expect.arrayContaining([expect.stringMatching(/GYMRAT_TIMEOUT/)]));
      expect(result.problems.join("\n")).toMatch(/no greater than 2147483/);
    });
  });
});
