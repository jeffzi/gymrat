import fs from "node:fs";
import path from "node:path";

import { Type } from "@sinclair/typebox";
import type { Static } from "@sinclair/typebox";

import type { Adapter, MetricDefaults } from "./adapters/types.js";
import { GymratError } from "./errors.js";
import { metricRecord } from "./metric-record.js";
import { compile, expected, parse, type SchemaIssue } from "./schema.js";
import { MAX_TIMEOUT_SECONDS } from "./timer-limits.js";
import { DEFAULT_UNSTABLE_NOISE_PCT } from "./verdict/verdict.js";

/** Shared options for object schemas: rejects non-objects and disallows unknown keys. */
const strictObjectOptions = { ...expected("an object"), additionalProperties: false };

const metricEntrySchema = Type.Object(
  {
    direction: Type.Optional(
      Type.Union([Type.Literal("lower"), Type.Literal("higher")], expected(`"lower" or "higher"`)),
    ),
    gating: Type.Optional(Type.Boolean(expected("a boolean"))),
    exact: Type.Optional(Type.Boolean(expected("a boolean"))),
  },
  strictObjectOptions,
);

/**
 * Metric names are unconstrained, so any string key is accepted.
 *
 * This compiles to `patternProperties` with `^(.*)$`, and `.` does not match a newline —
 * a metric name containing one slips through without its entry being checked. Harmless in
 * practice, since such a name cannot come out of a bench command's metric lines.
 */
const metricsSchema = Type.Record(Type.String(), metricEntrySchema, expected("an object"));

const kindEntrySchema = Type.Object(
  { gating: Type.Optional(Type.Boolean(expected("a boolean"))) },
  strictObjectOptions,
);

/** Kind names come from the adapter, so — like metric names — any string key is accepted. */
const kindsSchema = Type.Record(Type.String(), kindEntrySchema, expected("an object"));

/** The `gymrat.json` schema — every field optional, since flags can supply any of them. */
const configFileSchema = Type.Object(
  {
    bench: Type.Optional(Type.String(expected("a string"))),
    prepare: Type.Optional(Type.String(expected("a string"))),
    adapter: Type.Optional(Type.String(expected("a string"))),
    samples: Type.Optional(Type.Integer({ ...expected("a positive integer"), minimum: 1 })),
    timeoutSeconds: Type.Optional(
      Type.Integer({
        ...expected("a positive integer"),
        minimum: 1,
        maximum: MAX_TIMEOUT_SECONDS,
      }),
    ),
    // A noise threshold is a percentage, not a count, so fractional values are allowed.
    unstableNoisePct: Type.Optional(
      Type.Number({ ...expected("a positive number"), exclusiveMinimum: 0 }),
    ),
    metrics: Type.Optional(metricsSchema),
    kinds: Type.Optional(kindsSchema),
  },
  strictObjectOptions,
);

const configFileValidator = compile(configFileSchema);

/** The shape of `gymrat.json` after schema validation — every field optional, since CLI flags can supply any of them. */
export type ConfigFile = Static<typeof configFileSchema>;
/** A string-keyed record of per-metric overrides (direction, gating, exact), derived from the config file's `metrics` section. */
export type ConfigMetrics = Static<typeof metricsSchema>;
/** A kind-keyed record of overrides that apply to every metric of that kind, derived from the config file's `kinds` section. */
export type ConfigKinds = Static<typeof kindsSchema>;

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
  unstableNoisePct: number;
  metrics?: ConfigMetrics;
  kinds?: ConfigKinds;
}

/** One metric's settled metadata, after adapter defaults and config overrides are merged. */
export type ResolvedMetricMeta = {
  direction: MetricDefaults["direction"];
  gating: boolean;
  exact: boolean;
  kind: string;
  shortName: string;
  unit?: MetricDefaults["unit"];
};

/**
 * Word a schema failure as a config error.
 *
 * A root-level failure means the whole file is the wrong shape, so it names the file
 * rather than a key; everything else names the dotted path the reader wrote in JSON.
 * Unknown keys must be caught before the last branch: no sub-schema exists for a key the
 * schema never declared, so the phrase they carry is the containing object's `an object`.
 *
 * Root is detected on the raw JSON Pointer rather than {@link SchemaIssue.path}, whose
 * dotted form renders both the root and a top-level `""` key as the empty string.
 */
function configMessage(configPath: string, issue: SchemaIssue): string {
  if (issue.error.path === "") {
    return `Invalid config file at ${configPath}: expected a JSON object, got ${JSON.stringify(issue.value)}`;
  }
  if (issue.kind === "unknown-key") {
    return `Unknown config key: ${issue.path}`;
  }
  return `Invalid config value for ${issue.path}: expected ${issue.expected}, got ${JSON.stringify(issue.value)}`;
}

const DEFAULTS = {
  adapter: "metric-lines",
  samples: 10,
  timeoutSeconds: 1800,
  unstableNoisePct: DEFAULT_UNSTABLE_NOISE_PCT,
} as const;

function isFileNotFoundError(err: unknown): boolean {
  return err instanceof Error && "code" in err && err.code === "ENOENT";
}

function readConfigContent(configPath: string, required: boolean): string | undefined {
  try {
    return fs.readFileSync(configPath, "utf-8");
  } catch (err) {
    if (isFileNotFoundError(err)) {
      if (required) {
        throw new GymratError(`Config file not found at ${configPath}`, undefined, { cause: err });
      }
      return undefined;
    }
    throw err;
  }
}

function parseJsonContent(content: string, configPath: string): unknown {
  // A UTF-8 BOM survives decoding as U+FEFF, and JSON.parse rejects it as an
  // unexpected token — strip it so BOM-prefixed files read the same as plain ones.
  const json = content.startsWith("\u{FEFF}") ? content.slice(1) : content;
  try {
    return JSON.parse(json);
  } catch (err) {
    throw new GymratError(
      `Failed to parse config file at ${configPath}: ${err instanceof Error ? err.message : "unknown error"}`,
      undefined,
      { cause: err },
    );
  }
}

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
  const content = readConfigContent(configPath, options.required === true);
  if (content === undefined) {
    return {};
  }
  const parsed = parseJsonContent(content, configPath);
  return parse(configFileValidator, parsed, (issue) => configMessage(configPath, issue));
}

/**
 * Settle a run configuration from flags, config file, and built-in defaults.
 *
 * A flag always wins over the same key in the file. With no `--config`,
 * `./gymrat.json` is looked up implicitly and its absence is not an error; an
 * explicit path is loaded with `required: true` so a typo throws instead of
 * silently running on defaults.
 *
 * `bench` has no default and must come from one of the two sources — a run
 * without it throws.
 */
export function resolveConfig(flags: CliFlags): ResolvedConfig {
  const configPath = flags.config ?? path.join(process.cwd(), "gymrat.json");
  const configFile = loadConfigFile(configPath, { required: flags.config !== undefined });

  const bench = flags.bench ?? configFile.bench;
  const prepare = flags.prepare ?? configFile.prepare;
  const adapter = flags.adapter ?? configFile.adapter ?? DEFAULTS.adapter;
  const samples = flags.samples ?? configFile.samples ?? DEFAULTS.samples;
  const timeoutSeconds = flags.timeout ?? configFile.timeoutSeconds ?? DEFAULTS.timeoutSeconds;
  const unstableNoisePct = configFile.unstableNoisePct ?? DEFAULTS.unstableNoisePct;
  const metrics = configFile.metrics ? metricRecord(Object.entries(configFile.metrics)) : undefined;
  const kinds = configFile.kinds ? metricRecord(Object.entries(configFile.kinds)) : undefined;

  if (!bench) {
    throw new GymratError("bench is required. Provide it via --bench flag or in config file.");
  }

  return {
    bench,
    adapter,
    samples,
    timeoutSeconds,
    unstableNoisePct,
    ...(prepare !== undefined && { prepare }),
    ...(metrics !== undefined && { metrics }),
    ...(kinds !== undefined && { kinds }),
  };
}

/** The kind an adapter's metric falls under when it reports none. */
const DEFAULT_METRIC_KIND = "other";

function resolveOneMetric(
  metricName: string,
  configMetrics: ConfigFile["metrics"],
  adapter: Adapter,
  configKinds: ConfigFile["kinds"],
): ResolvedMetricMeta {
  const adapterDefaults = adapter.defaults(metricName);
  const configEntry = configMetrics?.[metricName];
  const kind = adapterDefaults.kind ?? DEFAULT_METRIC_KIND;

  return {
    direction: configEntry?.direction ?? adapterDefaults.direction,
    gating: configEntry?.gating ?? configKinds?.[kind]?.gating ?? true,
    exact: configEntry?.exact ?? false,
    kind,
    shortName: adapterDefaults.shortName ?? metricName,
    ...(adapterDefaults.unit !== undefined && { unit: adapterDefaults.unit }),
  };
}

/**
 * Resolve per-metric metadata by merging adapter defaults with config overrides.
 *
 * Gating is settled by the narrowest source that names the metric: an exact
 * `metrics` entry, then the `kinds` entry for the kind the adapter reports, then
 * gating on. Config entries — metric or kind — that nothing in the run matches
 * are silently ignored.
 */
export function resolveMetricMeta(
  metricNames: readonly string[],
  configMetrics: ConfigFile["metrics"],
  adapter: Adapter,
  configKinds?: ConfigFile["kinds"],
): Record<string, ResolvedMetricMeta> {
  return metricRecord(
    metricNames.map((name): [string, ResolvedMetricMeta] => [
      name,
      resolveOneMetric(name, configMetrics, adapter, configKinds),
    ]),
  );
}
