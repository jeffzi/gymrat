import fs from "node:fs";
import path from "node:path";

import type { Adapter } from "./adapters/types.js";

export interface ConfigFile {
  bench?: string;
  prepare?: string;
  adapter?: string;
  samples?: number;
  timeoutSeconds?: number;
  metrics?: Record<
    string,
    {
      direction?: "lower" | "higher";
      gating?: boolean;
      exact?: boolean;
    }
  >;
}

export interface CliFlags {
  bench?: string;
  prepare?: string;
  adapter?: string;
  samples?: number;
  timeout?: number;
  config?: string;
}

export interface ResolvedConfig {
  bench: string;
  prepare?: string;
  adapter: string;
  samples: number;
  timeoutSeconds: number;
}

export type ResolvedMetricMeta = {
  direction: "lower" | "higher";
  gating: boolean;
  exact: boolean;
  unit?: "ns" | "bytes";
};

function isRecord(val: unknown): val is Record<string, unknown> {
  return val !== null && typeof val === "object" && !Array.isArray(val);
}

const KNOWN_KEYS = new Set<string>([
  "bench",
  "prepare",
  "adapter",
  "samples",
  "timeoutSeconds",
  "metrics",
]);

const DEFAULTS = {
  adapter: "metric-lines",
  samples: 10,
  timeoutSeconds: 1800,
} as const;

export function loadConfigFile(configPath: string): ConfigFile {
  let content: string;

  try {
    content = fs.readFileSync(configPath, "utf-8");
  } catch (err) {
    if (err instanceof Error && "code" in err && err.code === "ENOENT") {
      return {};
    }
    throw err;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch (err) {
    throw new Error(
      `Failed to parse config file at ${configPath}: ${err instanceof Error ? err.message : "unknown error"}`,
      { cause: err },
    );
  }

  if (!isRecord(parsed)) {
    return {};
  }

  for (const key of Object.keys(parsed)) {
    if (!KNOWN_KEYS.has(key)) {
      throw new Error(`Unknown config key: ${key}`);
    }
  }
  return parsed;
}

export function resolveConfig(flags: CliFlags): ResolvedConfig {
  const configPath = flags.config ?? path.join(process.cwd(), "gymrat.json");
  const configFile = loadConfigFile(configPath);

  const bench = flags.bench ?? configFile.bench;
  const prepare = flags.prepare ?? configFile.prepare;
  const adapter = flags.adapter ?? configFile.adapter ?? DEFAULTS.adapter;
  const samples = flags.samples ?? configFile.samples ?? DEFAULTS.samples;
  const timeoutSeconds = flags.timeout ?? configFile.timeoutSeconds ?? DEFAULTS.timeoutSeconds;

  if (!bench) {
    throw new Error("bench is required. Provide it via --bench flag or in config file.");
  }

  return {
    bench,
    adapter,
    samples,
    timeoutSeconds,
    ...(prepare !== undefined && { prepare }),
  };
}

/**
 * Resolve per-metric metadata by merging adapter defaults with config overrides.
 * Config entries for metrics not in metricNames are silently ignored.
 */
export function resolveMetricMeta(
  metricNames: readonly string[],
  configMetrics: ConfigFile["metrics"],
  adapter: Adapter,
): Record<string, ResolvedMetricMeta> {
  const result: Record<string, ResolvedMetricMeta> = {};

  for (const metricName of metricNames) {
    const adapterDefaults = adapter.defaults(metricName);
    const configEntry = configMetrics?.[metricName];

    const resolved: ResolvedMetricMeta = {
      direction: configEntry?.direction ?? adapterDefaults.direction,
      gating: configEntry?.gating ?? true,
      exact: configEntry?.exact ?? false,
    };

    if (adapterDefaults.unit !== undefined) {
      resolved.unit = adapterDefaults.unit;
    }

    result[metricName] = resolved;
  }

  return result;
}
