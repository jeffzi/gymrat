import fs from "node:fs";
import path from "node:path";

import { Type } from "@sinclair/typebox";
import type { Static } from "@sinclair/typebox";

import type { Adapter, MetricDefaults } from "./adapters/types.js";
import { GymratError, hasErrorCode } from "./errors.js";
import { metricRecord } from "./metric-record.js";
import { compile, expected, parse, type SchemaIssue } from "./schema.js";
import { MAX_TIMEOUT_SECONDS } from "./timer-limits.js";
import { DEFAULT_UNSTABLE_NOISE_PCT, NOISE_FLOOR_PCT } from "./verdict/verdict.js";

/** Shared options for object schemas: rejects non-objects and disallows unknown keys. */
const strictObjectOptions = { ...expected("an object"), additionalProperties: false };

/** Shared schema for optional string fields — reused across every plain string config key. */
const optionalStringSchema = Type.Optional(Type.String(expected("a string")));

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
 * Shared options for the record schemas whose keys are names supplied by an adapter.
 *
 * `Type.Record(Type.String(), …)` compiles to `patternProperties` with `^(.*)$`, and
 * neither `.` nor an unanchored `$` spans a line terminator — so a key containing one
 * matches no pattern at all. Without `additionalProperties: false` such a key would be
 * an unconstrained extra property, admitting its entry unchecked; with it, the key is
 * rejected outright.
 */
const nameKeyedRecordOptions = { ...expected("an object"), additionalProperties: false };

/** Metric names are unconstrained, so any single-line string key is accepted. */
const metricsSchema = Type.Record(Type.String(), metricEntrySchema, nameKeyedRecordOptions);

const kindEntrySchema = Type.Object(
  { gating: Type.Optional(Type.Boolean(expected("a boolean"))) },
  strictObjectOptions,
);

/** Kind names come from the adapter, so — like metric names — any single-line string key is accepted. */
const kindsSchema = Type.Record(Type.String(), kindEntrySchema, nameKeyedRecordOptions);

/** When a loop stops: a value the primary metric must reach, an iteration budget, or both. */
const stopSchema = Type.Object(
  {
    // A target is a metric value, and no metric is constrained to whole numbers.
    targetValue: Type.Optional(Type.Number(expected("a number"))),
    maxIterations: Type.Optional(Type.Integer({ ...expected("a positive integer"), minimum: 1 })),
  },
  strictObjectOptions,
);

/** The `gymrat.json` schema — every field optional, since flags can supply any of them. */
const configFileSchema = Type.Object(
  {
    bench: optionalStringSchema,
    prepare: optionalStringSchema,
    adapter: optionalStringSchema,
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
      Type.Number({
        ...expected(`a number at or above the ${NOISE_FLOOR_PCT}% noise floor`),
        minimum: NOISE_FLOOR_PCT,
      }),
    ),
    metrics: Type.Optional(metricsSchema),
    kinds: Type.Optional(kindsSchema),
    checks: optionalStringSchema,
    filter: optionalStringSchema,
    primary: optionalStringSchema,
    stop: Type.Optional(stopSchema),
    hooks: optionalStringSchema,
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
/** The loop's stop conditions, derived from the config file's `stop` section. */
export type ConfigStop = Static<typeof stopSchema>;

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
  checks?: string;
  filter?: string;
  primary: string;
  stop?: ConfigStop;
  hooks: string;
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
 * Word an "invalid value" failure the same way whether TypeBox found it while validating
 * the file or {@link resolveConfig} found it while cross-checking two settled fields.
 */
function invalidValueMessage(field: string, expectedPhrase: string, value: unknown): string {
  return `Invalid config value for ${field}: expected ${expectedPhrase}, got ${JSON.stringify(value)}`;
}

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
  return invalidValueMessage(issue.path, issue.expected, issue.value);
}

/** The primary that aggregates every gating metric rather than naming one. */
const GEOMEAN_PRIMARY = "geomean";

/** The token a `filter` command must carry, where the loop substitutes the benchmark names. */
const FILTER_PLACEHOLDER = "{names}";

const DEFAULTS = {
  adapter: "metric-lines",
  samples: 10,
  timeoutSeconds: 1800,
  unstableNoisePct: DEFAULT_UNSTABLE_NOISE_PCT,
  primary: GEOMEAN_PRIMARY,
  hooks: "gymrat.hooks",
} as const;

function isFileNotFoundError(err: unknown): boolean {
  return hasErrorCode(err, "ENOENT");
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
 * Cross-field rules that the schema alone cannot express.
 *
 * `filter` must carry its placeholder, and `stop.targetValue` only makes sense
 * when `primary` names a metric (the geomean is a ratio, not a metric value).
 */
function validateLoopKeys(config: { filter?: string; primary: string; stop?: ConfigStop }): void {
  if (config.filter !== undefined && !config.filter.includes(FILTER_PLACEHOLDER)) {
    throw new GymratError(
      invalidValueMessage(
        "filter",
        `a string containing the ${FILTER_PLACEHOLDER} placeholder`,
        config.filter,
      ),
    );
  }
  if (config.stop?.targetValue !== undefined && config.primary === GEOMEAN_PRIMARY) {
    throw new GymratError(
      `Invalid config value for stop.targetValue: it needs primary to name a metric, not ${JSON.stringify(GEOMEAN_PRIMARY)}`,
    );
  }
}

/**
 * Merge flags → config file → built-in defaults into one partial record.
 *
 * `bench` stays optional — the caller validates its presence and spreads the
 * required value into the return.
 */
function mergeConfig(flags: CliFlags, configFile: ConfigFile): Omit<ResolvedConfig, "bench"> {
  return {
    adapter: flags.adapter ?? configFile.adapter ?? DEFAULTS.adapter,
    samples: flags.samples ?? configFile.samples ?? DEFAULTS.samples,
    timeoutSeconds: flags.timeout ?? configFile.timeoutSeconds ?? DEFAULTS.timeoutSeconds,
    unstableNoisePct: configFile.unstableNoisePct ?? DEFAULTS.unstableNoisePct,
    primary: configFile.primary ?? DEFAULTS.primary,
    hooks: configFile.hooks ?? DEFAULTS.hooks,
    ...((flags.prepare ?? configFile.prepare)
      ? { prepare: flags.prepare ?? configFile.prepare }
      : undefined),
    ...(configFile.metrics
      ? { metrics: metricRecord(Object.entries(configFile.metrics)) }
      : undefined),
    ...(configFile.kinds ? { kinds: metricRecord(Object.entries(configFile.kinds)) } : undefined),
    ...(configFile.checks ? { checks: configFile.checks } : undefined),
    ...(configFile.filter ? { filter: configFile.filter } : undefined),
    ...(configFile.stop ? { stop: configFile.stop } : undefined),
  };
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
  const merged = mergeConfig(flags, configFile);

  const bench = flags.bench ?? configFile.bench;
  if (!bench) {
    throw new GymratError("bench is required. Provide it via --bench flag or in config file.");
  }
  validateLoopKeys(merged);

  return { ...merged, bench };
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
