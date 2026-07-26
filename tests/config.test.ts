import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { describe, it, expect, afterEach } from "vitest";

import type { Adapter } from "../src/adapters/types.js";
import { loadConfigFile, resolveConfig, resolveMetricMeta } from "../src/config.js";

function createConfigFile(content: Record<string, unknown>, filename = "gymrat.json"): string {
  const tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-"));
  fs.writeFileSync(path.join(tmpdir, filename), JSON.stringify(content));
  return tmpdir;
}

function createMockAdapter(
  defaults: Adapter["defaults"] = () => ({ direction: "lower" as const }),
): Adapter {
  return {
    name: "test-adapter",
    parse: () => ({}),
    defaults,
  };
}

describe("loadConfigFile", () => {
  let tmpdir: string;

  afterEach(() => {
    if (tmpdir && fs.existsSync(tmpdir)) {
      fs.rmSync(tmpdir, { recursive: true, force: true });
    }
  });

  describe("when the config file does not exist", () => {
    it("returns an empty config object", () => {
      const nonexistentPath = path.join(os.tmpdir(), `nonexistent-${Date.now()}.json`);
      const result = loadConfigFile(nonexistentPath);

      expect(result).toStrictEqual({});
    });
  });

  describe("when the config file does not exist and the caller requires it", () => {
    it("throws an error naming the missing path", () => {
      const nonexistentPath = path.join(os.tmpdir(), `nonexistent-${Date.now()}.json`);

      expect(() => loadConfigFile(nonexistentPath, { required: true })).toThrow(nonexistentPath);
    });
  });

  describe("when the config path is not a readable file", () => {
    it("propagates the filesystem error instead of returning an empty config", () => {
      tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-"));

      expect(() => loadConfigFile(tmpdir)).toThrow(/EISDIR/);
    });

    it("propagates the filesystem error even when the caller requires the file", () => {
      tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-"));

      expect(() => loadConfigFile(tmpdir, { required: true })).toThrow(/EISDIR/);
    });
  });

  describe("when the config file contains valid JSON with known keys", () => {
    it("returns the parsed config with bench key", () => {
      tmpdir = createConfigFile({ bench: "custom-bench" });
      const configPath = path.join(tmpdir, "gymrat.json");

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
        metrics: {
          metric1: { direction: "lower" as const, gating: true, exact: false },
          metric2: { direction: "higher" as const },
        },
      };
      tmpdir = createConfigFile(config);
      const configPath = path.join(tmpdir, "gymrat.json");

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
      tmpdir = createConfigFile(config);
      const configPath = path.join(tmpdir, "gymrat.json");

      const result = loadConfigFile(configPath);

      expect(result).toStrictEqual(config);
    });
  });

  describe("when the config file contains invalid JSON", () => {
    it("throws an error that includes the file path", () => {
      tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-"));
      const configPath = path.join(tmpdir, "gymrat.json");
      fs.writeFileSync(configPath, "{ invalid json }");

      expect(() => loadConfigFile(configPath)).toThrow(configPath);
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
      tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-"));
      const configPath = path.join(tmpdir, "gymrat.json");
      fs.writeFileSync(configPath, json);
      const act = (): void => {
        loadConfigFile(configPath);
      };

      expect(act).toThrow(configPath);
      expect(act).toThrow(/JSON object/);
    });
  });

  describe("when the config file contains unknown top-level keys", () => {
    it("throws an error that names the unknown key", () => {
      tmpdir = createConfigFile({ unknownKey: "value" });
      const configPath = path.join(tmpdir, "gymrat.json");

      expect(() => loadConfigFile(configPath)).toThrow(/unknownKey/);
    });

    it("throws when there is an unknown key mixed with known keys", () => {
      tmpdir = createConfigFile({ bench: "name", badKey: "value" });
      const configPath = path.join(tmpdir, "gymrat.json");

      expect(() => loadConfigFile(configPath)).toThrow(/badKey/);
    });
  });

  describe("when the config file contains an empty object", () => {
    it("returns an empty config object", () => {
      tmpdir = createConfigFile({});
      const configPath = path.join(tmpdir, "gymrat.json");

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
    ])("throws naming $key when it is $description", ({ key, value }) => {
      tmpdir = createConfigFile({ [key]: value });
      const configPath = path.join(tmpdir, "gymrat.json");

      expect(() => loadConfigFile(configPath)).toThrow(new RegExp(`${key}.*string`));
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
      tmpdir = createConfigFile({ [key]: value });
      const configPath = path.join(tmpdir, "gymrat.json");

      expect(() => loadConfigFile(configPath)).toThrow(new RegExp(`${key}.*positive integer`));
    });
  });

  describe("when metrics is not an object", () => {
    it.each([
      { description: "an array", value: [] },
      { description: "a string", value: "latency" },
      { description: "a number", value: 3 },
      { description: "null", value: null },
    ])("throws naming metrics when it is $description", ({ value }) => {
      tmpdir = createConfigFile({ metrics: value });
      const configPath = path.join(tmpdir, "gymrat.json");

      expect(() => loadConfigFile(configPath)).toThrow(/metrics.*object/);
    });
  });

  describe("when a metrics entry is not an object", () => {
    it("throws naming the offending metric", () => {
      tmpdir = createConfigFile({ metrics: { latency: "lower" } });
      const configPath = path.join(tmpdir, "gymrat.json");

      expect(() => loadConfigFile(configPath)).toThrow(/metrics\.latency.*object/);
    });
  });

  describe("when a metrics entry has an invalid direction", () => {
    it.each([
      { description: "an unknown string", value: "sideways" },
      { description: "wrongly capitalized", value: "Lower" },
      { description: "a boolean", value: true },
      { description: "null", value: null },
    ])("throws naming metrics.latency.direction when it is $description", ({ value }) => {
      tmpdir = createConfigFile({ metrics: { latency: { direction: value } } });
      const configPath = path.join(tmpdir, "gymrat.json");

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
      tmpdir = createConfigFile({ metrics: { latency: { [field]: value } } });
      const configPath = path.join(tmpdir, "gymrat.json");

      expect(() => loadConfigFile(configPath)).toThrow(
        new RegExp(`metrics\\.latency\\.${field}.*boolean`),
      );
    });
  });

  describe("when a metrics entry contains an unknown key", () => {
    it("throws an error that names the offending key", () => {
      tmpdir = createConfigFile({
        metrics: { latency: { direction: "lower", threshold: "higher" } },
      });
      const configPath = path.join(tmpdir, "gymrat.json");

      expect(() => loadConfigFile(configPath)).toThrow(/metrics\.latency\.threshold/);
    });
  });
});

describe("resolveConfig", () => {
  let tmpdir: string;
  const originalCwd = process.cwd();

  afterEach(() => {
    if (tmpdir && fs.existsSync(tmpdir)) {
      fs.rmSync(tmpdir, { recursive: true, force: true });
    }
    process.chdir(originalCwd);
  });

  describe("when flags and config are empty", () => {
    it("returns defaults for adapter, samples, timeoutSeconds", () => {
      // resolveConfig falls back to ./gymrat.json, so run from a dir that has none
      tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-"));
      process.chdir(tmpdir);

      const result = resolveConfig({ bench: "my-bench" });

      expect(result).toStrictEqual({
        bench: "my-bench",
        adapter: "metric-lines",
        samples: 10,
        timeoutSeconds: 1800,
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
      };
      tmpdir = createConfigFile(config);
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result).toStrictEqual(config);
    });
  });

  describe("when flags and config both provide values", () => {
    it("uses flag values over config file values", () => {
      tmpdir = createConfigFile({
        bench: "config-bench",
        adapter: "config-adapter",
        samples: 20,
      });
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
      });
    });
  });

  describe("when flags provide values and no config", () => {
    it("uses flag values over defaults", () => {
      tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-"));
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
      });
    });
  });

  describe("when bench is missing from both flags and config", () => {
    it.each([
      { pattern: /--bench/, description: "mentions --bench" },
      { pattern: /config file/, description: "mentions config file" },
    ])("throws an error that $description", ({ pattern }) => {
      tmpdir = createConfigFile({});
      process.chdir(tmpdir);

      expect(() => resolveConfig({})).toThrow(pattern);
    });
  });

  describe("when prepare is provided", () => {
    it("includes prepare in resolved config", () => {
      tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-"));
      process.chdir(tmpdir);

      const result = resolveConfig({
        bench: "my-bench",
        prepare: "prepare-cmd",
      });

      expect(result.prepare).toBe("prepare-cmd");
    });
  });

  describe("when config file path is specified", () => {
    it("loads config from the specified path", () => {
      tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-"));
      const customConfigPath = path.join(tmpdir, "custom-config.json");
      fs.writeFileSync(customConfigPath, JSON.stringify({ bench: "custom-bench" }));

      const result = resolveConfig({ config: customConfigPath });

      expect(result.bench).toBe("custom-bench");
    });
  });

  describe("when the specified config file path does not exist", () => {
    it("throws an error naming the missing path instead of falling back to defaults", () => {
      tmpdir = fs.mkdtempSync(path.join(os.tmpdir(), "gymrat-"));
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
      tmpdir = createConfigFile({ bench: "config-bench", metrics });
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result).toStrictEqual({
        bench: "config-bench",
        adapter: "metric-lines",
        samples: 10,
        timeoutSeconds: 1800,
        metrics,
      });
    });
  });

  describe("when the config file has no metrics section", () => {
    it("omits metrics from the resolved config", () => {
      tmpdir = createConfigFile({ bench: "config-bench" });
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result).not.toHaveProperty("metrics");
    });
  });

  describe("when timeout flag is provided", () => {
    it("uses timeout from flags over config file value", () => {
      tmpdir = createConfigFile({ timeoutSeconds: 3600 });
      process.chdir(tmpdir);

      const result = resolveConfig({
        bench: "my-bench",
        timeout: 1200,
      });

      expect(result.timeoutSeconds).toBe(1200);
    });
  });
});

describe("resolveMetricMeta", () => {
  describe("when configMetrics is undefined and adapter returns direction only", () => {
    it("resolves to adapter direction with gating true and exact false, no unit", () => {
      const mockAdapter = createMockAdapter();

      const result = resolveMetricMeta(["response-time"], undefined, mockAdapter);

      expect(result).toStrictEqual({
        "response-time": {
          direction: "lower",
          gating: true,
          exact: false,
        },
      });
    });
  });

  describe("when adapter returns direction with unit", () => {
    it("includes the unit in resolved metadata", () => {
      const mockAdapter = createMockAdapter(() => ({
        direction: "lower" as const,
        unit: "ns",
      }));

      const result = resolveMetricMeta(["response-time"], undefined, mockAdapter);

      expect(result).toStrictEqual({
        "response-time": {
          direction: "lower",
          unit: "ns",
          gating: true,
          exact: false,
        },
      });
    });
  });

  describe("when config provides direction override", () => {
    it("overrides adapter direction", () => {
      const mockAdapter = createMockAdapter();
      const configMetrics = {
        throughput: { direction: "higher" as const },
      };

      const result = resolveMetricMeta(["throughput"], configMetrics, mockAdapter);

      expect(result).toStrictEqual({
        throughput: {
          direction: "higher",
          gating: true,
          exact: false,
        },
      });
    });
  });

  describe("when config provides gating override", () => {
    it("overrides the default gating true", () => {
      const mockAdapter = createMockAdapter();
      const configMetrics = {
        "response-time": { gating: false },
      };

      const result = resolveMetricMeta(["response-time"], configMetrics, mockAdapter);

      expect(result).toStrictEqual({
        "response-time": {
          direction: "lower",
          gating: false,
          exact: false,
        },
      });
    });
  });

  describe("when config provides exact override", () => {
    it("overrides the default exact false", () => {
      const mockAdapter = createMockAdapter();
      const configMetrics = {
        "response-time": { exact: true },
      };

      const result = resolveMetricMeta(["response-time"], configMetrics, mockAdapter);

      expect(result).toStrictEqual({
        "response-time": {
          direction: "lower",
          gating: true,
          exact: true,
        },
      });
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

      expect(result).toStrictEqual({
        "memory-usage": {
          direction: "lower",
          unit: "bytes",
          gating: false,
          exact: false,
        },
      });
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

      expect(result).toStrictEqual({
        "response-time": {
          direction: "lower",
          unit: "ns",
          gating: false,
          exact: false,
        },
        throughput: {
          direction: "higher",
          gating: true,
          exact: true,
        },
      });
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

      expect(result).toStrictEqual({
        "response-time": {
          direction: "lower",
          gating: false,
          exact: false,
        },
      });
    });
  });
});
