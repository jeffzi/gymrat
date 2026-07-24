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
      const result = resolveConfig({ bench: "my-bench" });

      expect(result).toStrictEqual({
        bench: "my-bench",
        adapter: "metric-lines",
        samples: 10,
        timeoutSeconds: 1800,
      });
    });

    it.each([
      { pattern: /--bench/, description: "mentions --bench" },
      { pattern: /config file/, description: "mentions config file" },
    ])("throws error when bench is missing that $description", ({ pattern }) => {
      expect(() => resolveConfig({})).toThrow(pattern);
    });

    it("resolves prepare as undefined when not provided", () => {
      const result = resolveConfig({ bench: "my-bench" });

      expect(result.prepare).toBeUndefined();
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

  describe("when bench is provided", () => {
    it("includes bench in resolved config from flags", () => {
      const result = resolveConfig({ bench: "my-bench" });

      expect(result.bench).toBe("my-bench");
    });

    it("includes bench in resolved config from config file", () => {
      tmpdir = createConfigFile({ bench: "config-bench" });
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result.bench).toBe("config-bench");
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

  describe("when config file path is not specified", () => {
    it("loads from ./gymrat.json in current working directory", () => {
      tmpdir = createConfigFile({ bench: "default-bench" });
      process.chdir(tmpdir);

      const result = resolveConfig({});

      expect(result.bench).toBe("default-bench");
    });
  });

  describe("when timeout flag is provided", () => {
    it("maps timeout flag to timeoutSeconds in resolved config", () => {
      const result = resolveConfig({
        bench: "my-bench",
        timeout: 900,
      });

      expect(result.timeoutSeconds).toBe(900);
    });

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
