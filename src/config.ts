import fs from "node:fs";
import path from "node:path";

import { Type } from "@sinclair/typebox";
import type { Static } from "@sinclair/typebox";

import type { Adapter, MetricDefaults } from "./adapters/types.js";
import { GymratError, hasErrorCode } from "./errors.js";
import { metricRecord } from "./metric-record.js";
import {
  compile,
  describeKey,
  expected,
  nameKeyedRecordOptions,
  parse,
  type SchemaIssue,
  strictObjectOptions,
} from "./schema.js";
import { MAX_TIMEOUT_SECONDS } from "./timer-limits.js";
import { DEFAULT_UNSTABLE_NOISE_PCT, NOISE_FLOOR_PCT } from "./verdict/verdict.js";

/** Shared schema for optional string fields — reused across every plain string config key. */
const optionalStringSchema = Type.Optional(Type.String(expected("a string")));

/**
 * Shared schema for optional strings that name work to do — a command to run, or the
 * benchmark a run targets.
 *
 * Empty is rejected rather than read as absent: an empty command runs as a no-op
 * shell that exits 0, so a blank entry would report as having run and succeeded
 * without ever executing anything, and an empty `bench` selects nothing while
 * looking like a benchmark was chosen.
 */
const optionalNonEmptyStringSchema = Type.Optional(
  Type.String({ ...expected("a non-empty string"), minLength: 1 }),
);

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

/** Commands the loop runs around each iteration; a stage left out runs nothing. */
const hooksSchema = Type.Object(
  { before: optionalNonEmptyStringSchema, after: optionalNonEmptyStringSchema },
  strictObjectOptions,
);

/** The `gymrat.json` schema — every field optional, since flags can supply any of them. */
const configFileSchema = Type.Object(
  {
    bench: optionalNonEmptyStringSchema,
    prepare: optionalNonEmptyStringSchema,
    adapter: optionalNonEmptyStringSchema,
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
    checks: optionalNonEmptyStringSchema,
    filter: optionalStringSchema,
    primary: optionalStringSchema,
    stop: Type.Optional(stopSchema),
    hooks: Type.Optional(hooksSchema),
  },
  strictObjectOptions,
);

const configFileValidator = compile(configFileSchema);

/** The shape of `gymrat.json` after schema validation — every field optional, since CLI flags can supply any of them. */
type ConfigFile = Static<typeof configFileSchema>;
/** A string-keyed record of per-metric overrides (direction, gating, exact), derived from the config file's `metrics` section. */
export type ConfigMetrics = Static<typeof metricsSchema>;
/** A kind-keyed record of overrides that apply to every metric of that kind, derived from the config file's `kinds` section. */
export type ConfigKinds = Static<typeof kindsSchema>;
/** The loop's stop conditions, derived from the config file's `stop` section. */
export type ConfigStop = Static<typeof stopSchema>;
/** The per-stage hook commands, derived from the config file's `hooks` section. */
type ConfigHooks = Static<typeof hooksSchema>;

/** Command-line overrides, named after the flags rather than the config keys. */
export interface CliFlags {
  bench?: string;
  prepare?: string;
  adapter?: string;
  samples?: number;
  timeout?: number;
  config?: string;
}

/**
 * A settled configuration for a command that never benches, defaults already applied.
 *
 * A bench command is what a benchmark run needs, not what gymrat needs: `status`
 * and `keep` read settings like `checks` and `stop` and run no benchmark, so a
 * repository they work in is fully configured without one.
 */
export interface BenchlessConfig {
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
  hooks?: ConfigHooks;
}

/** A settled run configuration: every value a run needs is present, including the bench command it runs. */
export interface ResolvedConfig extends BenchlessConfig {
  bench: string;
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
    return `Unknown config key: ${describeKey(issue.path)}`;
  }
  return invalidValueMessage(issue.path, issue.expected, issue.value);
}

/** The primary that aggregates every gating metric rather than naming one. */
export const GEOMEAN_PRIMARY = "geomean";

/** The token a `filter` command must carry, where the loop substitutes the benchmark names. */
export const FILTER_PLACEHOLDER = "{names}";

const DEFAULTS = {
  adapter: "metric-lines",
  samples: 10,
  timeoutSeconds: 1800,
  unstableNoisePct: DEFAULT_UNSTABLE_NOISE_PCT,
  primary: GEOMEAN_PRIMARY,
} as const;

function readConfigContent(configPath: string, required: boolean): string | undefined {
  try {
    return fs.readFileSync(configPath, "utf-8");
  } catch (err) {
    if (hasErrorCode(err, "ENOENT")) {
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
 * Reject a flag given an empty value.
 *
 * Flags bypass the file schema, so `--bench ""` is the one way an empty string still
 * reaches a resolved field; it is a value the user got wrong, not a value left out.
 * The message names the flag rather than the config key, because the flag is what
 * the user typed.
 */
function assertFlagNotEmpty(field: string, value: string | undefined): void {
  if (value === "") {
    throw new GymratError(invalidValueMessage(`--${field}`, "a non-empty string", value));
  }
}

/**
 * Merge flags → config file → built-in defaults into every setting but `bench`.
 *
 * `bench` is settled separately: the commands that bench spread it into the
 * result, and the commands that do not never ask for it.
 */
function mergeConfig(flags: CliFlags, configFile: ConfigFile): BenchlessConfig {
  const prepare = flags.prepare ?? configFile.prepare;
  return {
    adapter: flags.adapter ?? configFile.adapter ?? DEFAULTS.adapter,
    samples: flags.samples ?? configFile.samples ?? DEFAULTS.samples,
    timeoutSeconds: flags.timeout ?? configFile.timeoutSeconds ?? DEFAULTS.timeoutSeconds,
    unstableNoisePct: configFile.unstableNoisePct ?? DEFAULTS.unstableNoisePct,
    primary: configFile.primary ?? DEFAULTS.primary,
    ...(prepare !== undefined ? { prepare } : undefined),
    ...(configFile.metrics !== undefined
      ? { metrics: metricRecord(Object.entries(configFile.metrics)) }
      : undefined),
    ...(configFile.kinds !== undefined
      ? { kinds: metricRecord(Object.entries(configFile.kinds)) }
      : undefined),
    ...(configFile.checks !== undefined ? { checks: configFile.checks } : undefined),
    ...(configFile.filter !== undefined ? { filter: configFile.filter } : undefined),
    ...(configFile.stop !== undefined ? { stop: configFile.stop } : undefined),
    ...(configFile.hooks !== undefined ? { hooks: configFile.hooks } : undefined),
  };
}

/**
 * Settle every setting but `bench` from flags, config file, and built-in defaults.
 *
 * A flag always wins over the same key in the file. With no `--config`,
 * `./gymrat.json` is looked up implicitly and its absence is not an error; an
 * explicit path is loaded with `required: true` so a typo throws instead of
 * silently running on defaults.
 *
 * Use this for a command that runs no benchmark: it settles the same values
 * {@link resolveConfig} does, without asking for a bench command none of them
 * would ever run.
 */
export function resolveBenchlessConfig(flags: CliFlags): BenchlessConfig {
  return settleConfig(flags).config;
}

/**
 * Settle a run configuration from flags, config file, and built-in defaults.
 *
 * Everything {@link resolveBenchlessConfig} settles, plus the bench command the
 * run executes. `bench` has no default and must come from a flag or the config
 * file — a run without it throws.
 */
export function resolveConfig(flags: CliFlags): ResolvedConfig {
  const { config, bench } = settleConfig(flags);
  if (bench === undefined) {
    throw new GymratError("bench is required. Provide it via --bench flag or in config file.");
  }

  return { ...config, bench };
}

/** Settle the shared configuration, and report what `bench` the sources named — if any. */
function settleConfig(flags: CliFlags): { config: BenchlessConfig; bench: string | undefined } {
  assertFlagNotEmpty("bench", flags.bench);
  assertFlagNotEmpty("prepare", flags.prepare);
  assertFlagNotEmpty("adapter", flags.adapter);
  const configPath = flags.config ?? path.join(process.cwd(), "gymrat.json");
  const configFile = loadConfigFile(configPath, { required: flags.config !== undefined });
  const config = mergeConfig(flags, configFile);
  validateLoopKeys(config);

  return { config, bench: flags.bench ?? configFile.bench };
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
