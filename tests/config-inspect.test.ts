import fs from "node:fs";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { Adapter } from "../src/adapters/types.js";
import { inspectConfig } from "../src/config-inspect.js";
import { resolveMetricMeta } from "../src/config.js";
import { createMockAdapter as createMockAdapterBase } from "./fixtures/adapter.js";
import { metricMeta } from "./fixtures/comparison-result.js";
import { metricRecord } from "./fixtures/metrics.js";
import { freshRoot } from "./fixtures/scratch-repo.js";

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

      const result = inspectConfig({
        bench: "flag-bench",
        adapter: "flag-adapter",
        samples: 5,
        timeout: 30,
      });

      expect.soft(result.problems).toStrictEqual([]);
      expect.soft(result.config?.adapter).toBe("flag-adapter");
      expect.soft(result.config?.samples).toBe(5);
      expect.soft(result.config?.timeoutSeconds).toBe(30);
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
      expect(result.problems).toStrictEqual(
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

      expect(result.problems).toStrictEqual(
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

      expect(result.problems).toStrictEqual(
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

      expect(result.problems).toStrictEqual(
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
      expect(result.problems).toStrictEqual(
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

      expect(result.problems).toStrictEqual(
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

      expect(result.problems).toStrictEqual(
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

      expect(result.problems).toStrictEqual(
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

      expect(result.problems).toStrictEqual(
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
