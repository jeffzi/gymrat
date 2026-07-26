import fs from "node:fs";
import path from "node:path";

import type { Adapter } from "./adapters/types.js";

/** The `gymrat.json` schema — every field optional, since flags can supply any of them. */
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

/** Command-line overrides, named after the flags rather than the config keys. */
export interface CliFlags {
  bench?: string;
  prepare?: string;
  adapter?: string;
  samples?: number;
  timeout?: number;
  config?: string;
}

/** A settled run configuration: every value a run needs is present, defaults already applied. */
export interface ResolvedConfig {
  bench: string;
  prepare?: string;
  adapter: string;
  samples: number;
  timeoutSeconds: number;
  metrics?: ConfigMetrics;
}

/** One metric's settled metadata, after adapter defaults and config overrides are merged. */
export type ResolvedMetricMeta = {
  direction: "lower" | "higher";
  gating: boolean;
  exact: boolean;
  unit?: "ns" | "bytes";
};

function isRecord(val: unknown): val is Record<string, unknown> {
  return val !== null && typeof val === "object" && !Array.isArray(val);
}

type ConfigMetrics = NonNullable<ConfigFile["metrics"]>;
type ConfigMetricEntry = ConfigMetrics[string];

function invalidValue(key: string, expected: string, value: unknown): Error {
  return new Error(
    `Invalid config value for ${key}: expected ${expected}, got ${JSON.stringify(value)}`,
  );
}

function requireString(key: string, value: unknown): string {
  if (typeof value !== "string") {
    throw invalidValue(key, "a string", value);
  }
  return value;
}

function requirePositiveInteger(key: string, value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw invalidValue(key, "a positive integer", value);
  }
  return value;
}

function requireBoolean(key: string, value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw invalidValue(key, "a boolean", value);
  }
  return value;
}

function requireDirection(key: string, value: unknown): "lower" | "higher" {
  if (value === "lower" || value === "higher") {
    return value;
  }
  throw invalidValue(key, `"lower" or "higher"`, value);
}

function requireMetrics(value: unknown): ConfigMetrics {
  if (!isRecord(value)) {
    throw invalidValue("metrics", "an object", value);
  }

  const metrics: ConfigMetrics = {};
  for (const [name, rawEntry] of Object.entries(value)) {
    const entryKey = `metrics.${name}`;
    if (!isRecord(rawEntry)) {
      throw invalidValue(entryKey, "an object", rawEntry);
    }

    const entry: ConfigMetricEntry = {};
    for (const [field, fieldValue] of Object.entries(rawEntry)) {
      switch (field) {
        case "direction":
          entry.direction = requireDirection(`${entryKey}.direction`, fieldValue);
          break;
        case "gating":
        case "exact":
          entry[field] = requireBoolean(`${entryKey}.${field}`, fieldValue);
          break;
        default:
          throw new Error(`Unknown config key: ${entryKey}.${field}`);
      }
    }
    metrics[name] = entry;
  }
  return metrics;
}

const DEFAULTS = {
  adapter: "metric-lines",
  samples: 10,
  timeoutSeconds: 1800,
} as const;

/**
 * Read and validate a config file.
 *
 * A missing file yields an empty config so the implicit `gymrat.json` lookup can
 * fall through to defaults. Pass `required: true` when the user named the path
 * explicitly — a typo must fail loudly rather than silently run with defaults.
 */
export function loadConfigFile(
  configPath: string,
  options: { required?: boolean } = {},
): ConfigFile {
  let content: string;

  try {
    content = fs.readFileSync(configPath, "utf-8");
  } catch (err) {
    if (err instanceof Error && "code" in err && err.code === "ENOENT") {
      if (options.required === true) {
        throw new Error(`Config file not found at ${configPath}`, { cause: err });
      }
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
    throw new Error(
      `Invalid config file at ${configPath}: expected a JSON object, got ${JSON.stringify(parsed)}`,
    );
  }

  const config: ConfigFile = {};
  for (const [key, value] of Object.entries(parsed)) {
    switch (key) {
      case "bench":
      case "prepare":
      case "adapter":
        config[key] = requireString(key, value);
        break;
      case "samples":
      case "timeoutSeconds":
        config[key] = requirePositiveInteger(key, value);
        break;
      case "metrics":
        config.metrics = requireMetrics(value);
        break;
      default:
        throw new Error(`Unknown config key: ${key}`);
    }
  }
  return config;
}

/**
 * Settle a run configuration from flags, config file, and built-in defaults.
 *
 * Precedence runs flags → config file → `DEFAULTS`, so a flag always wins over
 * the same key in the file. With no `--config`, `./gymrat.json` is looked up
 * implicitly and its absence is not an error; an explicit path is loaded with
 * `required: true` so a typo throws instead of silently running on defaults.
 *
 * `bench` has no default and must come from one of the two sources — a run
 * without it throws.
 */
export function resolveConfig(flags: CliFlags): ResolvedConfig {
  const explicitPath = flags.config;
  const configPath = explicitPath ?? path.join(process.cwd(), "gymrat.json");
  const configFile = loadConfigFile(configPath, { required: explicitPath !== undefined });

  const bench = flags.bench ?? configFile.bench;
  const prepare = flags.prepare ?? configFile.prepare;
  const adapter = flags.adapter ?? configFile.adapter ?? DEFAULTS.adapter;
  const samples = flags.samples ?? configFile.samples ?? DEFAULTS.samples;
  const timeoutSeconds = flags.timeout ?? configFile.timeoutSeconds ?? DEFAULTS.timeoutSeconds;
  const metrics = configFile.metrics;

  if (!bench) {
    throw new Error("bench is required. Provide it via --bench flag or in config file.");
  }

  return {
    bench,
    adapter,
    samples,
    timeoutSeconds,
    ...(prepare !== undefined && { prepare }),
    ...(metrics !== undefined && { metrics }),
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
