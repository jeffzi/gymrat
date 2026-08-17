import fs from "node:fs";
import path from "node:path";

import { Type } from "@sinclair/typebox";
import type { Static } from "@sinclair/typebox";

import type { Adapter, MetricDefaults } from "./adapters/types.js";
import { assertNever, GymratError, hasErrorCode, messageOf } from "./errors.js";
import { runGit } from "./git.js";
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
    // The phrase names the cap because a single description covers both bounds, and a
    // value over the cap is the one a reader cannot guess the limit of.
    timeoutSeconds: Type.Optional(
      Type.Integer({
        ...expected(`a positive integer no greater than ${MAX_TIMEOUT_SECONDS}`),
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
    runbook: optionalNonEmptyStringSchema,
    filter: optionalStringSchema,
    primary: optionalNonEmptyStringSchema,
    stop: Type.Optional(stopSchema),
    hooks: Type.Optional(hooksSchema),
  },
  strictObjectOptions,
);

/** Validator compiled from the config-file schema; used to check a scaffolded or loaded config file. */
export const configFileValidator = compile(configFileSchema);

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
  runbook?: string;
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
  return invalidValueMessage(describeKey(issue.path), issue.expected, issue.value);
}

/** The config file basename the CLI writes, loads, and probes for. */
export const CONFIG_FILENAME = "gymrat.json";

/** The primary that aggregates every gating metric rather than naming one. */
export const GEOMEAN_PRIMARY = "geomean";

/** The token a `filter` command must carry, where the loop substitutes the benchmark names. */
export const FILTER_PLACEHOLDER = "{names}";

/** Built-in fallbacks for the {@link BenchlessConfig} fields no flag, env var, or config file set. */
export const CONFIG_DEFAULTS = {
  adapter: "metric-lines",
  samples: 10,
  timeoutSeconds: 1800,
  unstableNoisePct: DEFAULT_UNSTABLE_NOISE_PCT,
  primary: GEOMEAN_PRIMARY,
} as const;

type ConfigReadResult =
  | { kind: "parsed"; parsed: unknown }
  | { kind: "absent" }
  | { kind: "error"; message: string; cause: unknown };

/**
 * Read and JSON-parse a config file without throwing.
 *
 * Both the throwing path ({@link loadConfigFile}) and the collecting path
 * ({@link loadConfigFileCollecting}) funnel through here so that every
 * file-I/O and JSON-parse rule has a single implementation.
 */
function readConfigSource(configPath: string): ConfigReadResult {
  let content: string;
  try {
    content = fs.readFileSync(configPath, "utf-8");
  } catch (err) {
    if (hasErrorCode(err, "ENOENT")) {
      return { kind: "absent" };
    }
    return {
      kind: "error",
      message: `Cannot read config file at ${configPath}: ${messageOf(err)}`,
      cause: err,
    };
  }

  // A UTF-8 BOM survives decoding as U+FEFF, and JSON.parse rejects it as an
  // unexpected token — strip it so BOM-prefixed files read the same as plain ones.
  const json = content.startsWith("\u{FEFF}") ? content.slice(1) : content;
  try {
    return { kind: "parsed", parsed: JSON.parse(json) as unknown };
  } catch (err) {
    return {
      kind: "error",
      message: `Failed to parse config file at ${configPath}: ${messageOf(err)}`,
      cause: err,
    };
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
  const result = readConfigSource(configPath);
  switch (result.kind) {
    case "absent":
      if (options.required) {
        throw new GymratError(`Config file not found at ${configPath}`);
      }
      return {};
    case "error":
      throw new GymratError(result.message, undefined, { cause: result.cause });
    case "parsed":
      return parse(configFileValidator, result.parsed, (issue) => configMessage(configPath, issue));
    default:
      return assertNever(result);
  }
}

/**
 * Collecting counterpart of {@link loadConfigFile}: reads, parses, and validates
 * the config file, returning the parsed result alongside any problems rather than
 * throwing on the first.
 *
 * `configFile` is defined when the file was successfully read, parsed, and validated
 * (or absent-but-optional). When any step fails, `configFile` is `undefined` and the
 * problems array explains why.
 */
function loadConfigFileCollecting(
  configFilePath: string,
  required: boolean,
): { configFile?: ConfigFile; exists: boolean; problems: string[] } {
  const result = readConfigSource(configFilePath);
  switch (result.kind) {
    case "absent":
      if (required) {
        return { exists: false, problems: [`Config file not found at ${configFilePath}`] };
      }
      return { configFile: {}, exists: false, problems: [] };
    case "error":
      return { exists: true, problems: [result.message] };
    case "parsed":
      if (configFileValidator.check(result.parsed)) {
        return { configFile: result.parsed, exists: true, problems: [] };
      }
      return {
        exists: true,
        problems: configFileValidator
          .issues(result.parsed)
          .map((issue) => configMessage(configFilePath, issue)),
      };
    default:
      return assertNever(result);
  }
}

/**
 * Cross-field rules that the schema alone cannot express, returning all violations.
 *
 * `filter` must carry its placeholder, and `stop.targetValue` only makes sense
 * when `primary` names a metric (the geomean is a ratio, not a metric value).
 */
export function loopKeyProblems(config: {
  filter?: string;
  primary: string;
  stop?: ConfigStop;
}): string[] {
  const problems: string[] = [];
  if (config.filter !== undefined && !config.filter.includes(FILTER_PLACEHOLDER)) {
    problems.push(
      invalidValueMessage(
        "filter",
        `a string containing the ${FILTER_PLACEHOLDER} placeholder`,
        config.filter,
      ),
    );
  }
  if (config.stop?.targetValue !== undefined && config.primary === GEOMEAN_PRIMARY) {
    problems.push(
      `Invalid config value for stop.targetValue: it needs primary to name a metric, not ${JSON.stringify(GEOMEAN_PRIMARY)}`,
    );
  }
  return problems;
}

/** Throw the first cross-field violation, or do nothing when all rules hold. */
function validateLoopKeys(config: { filter?: string; primary: string; stop?: ConfigStop }): void {
  const problems = loopKeyProblems(config);
  if (problems.length > 0) throw new GymratError(problems[0]!);
}

/**
 * Return a problem string when a flag value is the empty string, or undefined when valid.
 *
 * Flags bypass the file schema, so `--bench ""` is the one way an empty string still
 * reaches a resolved field; it is a value the user got wrong, not a value left out.
 * The message names the flag rather than the config key, because the flag is what
 * the user typed.
 *
 * `--config ""` needs the same guard for a different reason: it is not nullish, so it
 * wins the fallback to `./gymrat.json` and is then loaded as a required path — leaving
 * the user with a missing-file error that names no file.
 */
function flagProblem(field: string, value: string | undefined): string | undefined {
  if (value === "") {
    return invalidValueMessage(`--${field}`, "a non-empty string", value);
  }
  return undefined;
}

/** Throw when a flag value is the empty string. */
function assertFlagNotEmpty(field: string, value: string | undefined): void {
  const problem = flagProblem(field, value);
  if (problem !== undefined) throw new GymratError(problem);
}

function assignDefined<T extends object, K extends keyof T>(
  target: T,
  key: K,
  value: T[K] | undefined,
): void {
  if (value !== undefined) target[key] = value;
}

/**
 * Check a `GYMRAT_*` string env var, returning the value or a problem string.
 *
 * Returns an empty object when the variable is unset so the next source in the
 * precedence chain (config file, then built-in default) can supply the value.
 */
function envStringResult(envVar: string): { value?: string; problem?: string } {
  const raw = process.env[envVar];
  if (raw === undefined) return {};
  if (raw === "") {
    return { problem: `Invalid value for ${envVar}: expected a non-empty string, got ""` };
  }
  return { value: raw };
}

/** Throwing wrapper around {@link envStringResult}. */
function readEnvString(envVar: string): string | undefined {
  const { value, problem } = envStringResult(envVar);
  if (problem !== undefined) throw new GymratError(problem);
  return value;
}

/**
 * Check a `GYMRAT_*` numeric env var, returning the parsed value or a problem string.
 *
 * When `max` is supplied the cap is included in the error phrase so the user
 * sees the allowed range in a single message.
 */
function envPositiveIntResult(envVar: string, max?: number): { value?: number; problem?: string } {
  const raw = process.env[envVar];
  if (raw === undefined) return {};
  const n = Number(raw);
  const phrase =
    max !== undefined ? `a positive integer no greater than ${max}` : "a positive integer";
  if (!Number.isInteger(n) || n < 1 || (max !== undefined && n > max)) {
    return {
      problem: `Invalid value for ${envVar}: expected ${phrase}, got ${JSON.stringify(raw)}`,
    };
  }
  return { value: n };
}

/** One `GYMRAT_*` string field's association between its `CliFlags` key, env var name, and reader. */
const STRING_ENV_FIELDS: readonly {
  field: "bench" | "prepare" | "adapter";
  envVar: string;
  reader: (envVar: string) => { value?: string; problem?: string };
}[] = [
  { field: "bench", envVar: "GYMRAT_BENCH", reader: envStringResult },
  { field: "prepare", envVar: "GYMRAT_PREPARE", reader: envStringResult },
  { field: "adapter", envVar: "GYMRAT_ADAPTER", reader: envStringResult },
];

/** One `GYMRAT_*` numeric field's association between its `CliFlags` key, env var name, and reader. */
const NUMBER_ENV_FIELDS: readonly {
  field: "samples" | "timeout";
  envVar: string;
  reader: (envVar: string) => { value?: number; problem?: string };
}[] = [
  { field: "samples", envVar: "GYMRAT_SAMPLES", reader: envPositiveIntResult },
  {
    field: "timeout",
    envVar: "GYMRAT_TIMEOUT",
    reader: (n) => envPositiveIntResult(n, MAX_TIMEOUT_SECONDS),
  },
];

/**
 * Read `GYMRAT_*` environment variables for fields whose CLI flag is absent.
 *
 * Only env vars whose corresponding flag is `undefined` are consulted, so a
 * flag always wins without the env var's validation ever firing.  `GYMRAT_CONFIG`
 * is handled separately in {@link settleConfig} because it affects which file is
 * loaded, not a field in the resolved config.
 */
function readEnvFlags(flags: CliFlags): CliFlags {
  const result: CliFlags = {};
  for (const { field, envVar, reader } of STRING_ENV_FIELDS) {
    if (flags[field] === undefined) {
      const { value, problem } = reader(envVar);
      if (problem !== undefined) throw new GymratError(problem);
      assignDefined(result, field, value);
    }
  }
  for (const { field, envVar, reader } of NUMBER_ENV_FIELDS) {
    if (flags[field] === undefined) {
      const { value, problem } = reader(envVar);
      if (problem !== undefined) throw new GymratError(problem);
      assignDefined(result, field, value);
    }
  }
  return result;
}

/**
 * Collecting counterpart of {@link readEnvFlags}: returns all valid env-flag
 * values alongside any validation problems, rather than bailing on the first.
 */
function collectOneEnv<T>(
  envVar: string,
  reader: (name: string) => { value?: T; problem?: string },
  problems: string[],
): T | undefined {
  const r = reader(envVar);
  if (r.problem !== undefined) problems.push(r.problem);
  return r.value;
}

function collectEnvFlags(flags: CliFlags): { flags: CliFlags; problems: string[] } {
  const problems: string[] = [];
  const result: CliFlags = {};
  for (const { field, envVar, reader } of STRING_ENV_FIELDS) {
    if (flags[field] === undefined) {
      assignDefined(result, field, collectOneEnv(envVar, reader, problems));
    }
  }
  for (const { field, envVar, reader } of NUMBER_ENV_FIELDS) {
    if (flags[field] === undefined) {
      assignDefined(result, field, collectOneEnv(envVar, reader, problems));
    }
  }
  return { flags: result, problems };
}

/**
 * Return a problem string when a `runbook` does not resolve to an existing file.
 *
 * Resolved relative to `baseDir` rather than the process's cwd, matching how
 * the implicit `./gymrat.json` lookup itself is anchored — a runbook path is
 * authored relative to the repo the config lives in.
 */
function runbookProblem(runbook: string, baseDir: string | undefined): string | undefined {
  const resolvedPath = path.resolve(baseDir ?? process.cwd(), runbook);
  let stat: fs.Stats | undefined;
  try {
    stat = fs.statSync(resolvedPath);
  } catch (err) {
    if (!hasErrorCode(err, "ENOENT")) {
      return `Cannot read runbook path ${runbook}: ${messageOf(err)}`;
    }
  }
  if (stat === undefined || !stat.isFile()) {
    return invalidValueMessage("runbook", "a path to an existing file", runbook);
  }
  return undefined;
}

/** Throwing wrapper around {@link runbookProblem}. */
function assertRunbookExists(runbook: string, baseDir: string | undefined): void {
  const problem = runbookProblem(runbook, baseDir);
  if (problem !== undefined) throw new GymratError(problem);
}

/**
 * Merge flags → config file → built-in defaults into every setting but `bench`.
 *
 * `bench` is settled separately: the commands that bench spread it into the
 * result, and the commands that do not never ask for it.
 */
// fallow-ignore-next-line complexity
function mergeConfig(flags: CliFlags, configFile: ConfigFile): BenchlessConfig {
  const prepare = flags.prepare ?? configFile.prepare;
  return {
    adapter: flags.adapter ?? configFile.adapter ?? CONFIG_DEFAULTS.adapter,
    samples: flags.samples ?? configFile.samples ?? CONFIG_DEFAULTS.samples,
    timeoutSeconds: flags.timeout ?? configFile.timeoutSeconds ?? CONFIG_DEFAULTS.timeoutSeconds,
    unstableNoisePct: configFile.unstableNoisePct ?? CONFIG_DEFAULTS.unstableNoisePct,
    primary: configFile.primary ?? CONFIG_DEFAULTS.primary,
    ...(prepare !== undefined ? { prepare } : undefined),
    ...(configFile.metrics !== undefined
      ? { metrics: metricRecord(Object.entries(configFile.metrics)) }
      : undefined),
    ...(configFile.kinds !== undefined
      ? { kinds: metricRecord(Object.entries(configFile.kinds)) }
      : undefined),
    ...(configFile.checks !== undefined ? { checks: configFile.checks } : undefined),
    ...(configFile.runbook !== undefined ? { runbook: configFile.runbook } : undefined),
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
export function resolveBenchlessConfig(flags: CliFlags, baseDir?: string): BenchlessConfig {
  return settleConfig(flags, baseDir).config;
}

/**
 * Settle a run configuration from flags, config file, and built-in defaults.
 *
 * Everything {@link resolveBenchlessConfig} settles, plus the bench command the
 * run executes. `bench` has no default and must come from a flag or the config
 * file — a run without it throws.
 */
export function resolveConfig(flags: CliFlags, baseDir?: string): ResolvedConfig {
  const { config, bench } = settleConfig(flags, baseDir);
  if (bench === undefined) {
    throw new GymratError("bench is required. Provide it via --bench flag or in config file.");
  }

  return { ...config, bench };
}

/**
 * Find the anchor directory for the implicit `gymrat.json` lookup.
 *
 * Inside a git repository the config lives at the repo root — the file is
 * authored there and moving cwd into a subdirectory should not lose it.
 * Outside a repository (or when git is unavailable) the lookup falls back to
 * the process's cwd, which is where the user invoked the command.
 */
function findImplicitBase(): string {
  try {
    return runGit(["rev-parse", "--show-toplevel"], process.cwd()).trim();
  } catch {
    return process.cwd();
  }
}

/** Settle the shared configuration, and report what `bench` the sources named — if any. */
// fallow-ignore-next-line complexity
function settleConfig(
  flags: CliFlags,
  baseDir?: string,
): { config: BenchlessConfig; bench: string | undefined } {
  assertFlagNotEmpty("bench", flags.bench);
  assertFlagNotEmpty("prepare", flags.prepare);
  assertFlagNotEmpty("adapter", flags.adapter);
  assertFlagNotEmpty("config", flags.config);

  // Env vars fill in for absent flags: flag > env > file > default.
  const effective: CliFlags = { ...flags, ...readEnvFlags(flags) };

  // --config > GYMRAT_CONFIG > implicit gymrat.json
  const envConfigPath = flags.config === undefined ? readEnvString("GYMRAT_CONFIG") : undefined;
  const explicitConfig = flags.config ?? envConfigPath;
  const configPath = explicitConfig ?? path.join(baseDir ?? findImplicitBase(), CONFIG_FILENAME);
  const configFile = loadConfigFile(configPath, { required: explicitConfig !== undefined });

  const config = mergeConfig(effective, configFile);
  validateLoopKeys(config);

  if (config.runbook !== undefined) {
    const configDir = path.dirname(configPath);
    assertRunbookExists(config.runbook, configDir);
    config.runbook = path.resolve(configDir, config.runbook);
  }

  return { config, bench: effective.bench ?? configFile.bench };
}

/** The outcome of a non-throwing config inspection. */
export interface ConfigInspection {
  /** The resolved path to the config file, or undefined when no file was found. */
  configPath: string | undefined;
  /** Whether the resolved config path points to an existing file. */
  configExists: boolean;
  /** Human-worded problems found during inspection; empty when the config is clean. */
  problems: string[];
  /** The settled configuration, present only when no problems were found. */
  config?: BenchlessConfig;
  /** The resolved bench command, present only when clean and a bench value exists. */
  bench?: string;
}

/**
 * Resolve which config file to load, load and validate it, and report any problems.
 *
 * When the config source itself is broken (empty `--config` flag, empty
 * `GYMRAT_CONFIG`), file loading is skipped and an empty config file is
 * returned so the merge can still produce defaults.
 */
function resolveConfigSource(
  flags: CliFlags,
  baseDir: string | undefined,
): {
  configPath: string | undefined;
  configExists: boolean;
  configFile: ConfigFile | undefined;
  problems: string[];
} {
  const problems: string[] = [];

  let envConfigPath: string | undefined;
  let envConfigFailed = false;
  if (flags.config === undefined) {
    const r = envStringResult("GYMRAT_CONFIG");
    if (r.problem !== undefined) {
      problems.push(r.problem);
      envConfigFailed = true;
    }
    envConfigPath = r.value;
  }

  const explicitConfig = (flags.config !== "" ? flags.config : undefined) ?? envConfigPath;
  const skipLoading = flags.config === "" || envConfigFailed;

  if (skipLoading) {
    return { configPath: undefined, configExists: false, configFile: {}, problems };
  }

  const resolvedPath = explicitConfig ?? path.join(baseDir ?? findImplicitBase(), CONFIG_FILENAME);
  const required = explicitConfig !== undefined;
  const fileResult = loadConfigFileCollecting(resolvedPath, required);
  problems.push(...fileResult.problems);

  return {
    configPath: required || fileResult.exists ? resolvedPath : undefined,
    configExists: fileResult.exists,
    configFile: fileResult.configFile,
    problems,
  };
}

/**
 * Inspect the config without throwing, collecting every problem the throwing
 * path would have surfaced as a single {@link GymratError}.
 *
 * The function mirrors the validation that {@link settleConfig} performs —
 * flag, env-var, schema, cross-field, and runbook checks — but accumulates
 * all problems rather than bailing on the first.
 */
function collectFlagProblems(flags: CliFlags): string[] {
  const problems: string[] = [];
  for (const field of ["bench", "prepare", "adapter", "config"] as const) {
    const p = flagProblem(field, flags[field]);
    if (p !== undefined) problems.push(p);
  }
  return problems;
}

function buildEffectiveFlags(flags: CliFlags, envFlags: CliFlags): CliFlags {
  const effective: CliFlags = { ...envFlags };
  for (const field of ["bench", "prepare", "adapter"] as const) {
    if (flags[field] !== undefined && flags[field] !== "") effective[field] = flags[field];
  }
  if (flags.samples !== undefined) effective.samples = flags.samples;
  if (flags.timeout !== undefined) effective.timeout = flags.timeout;
  return effective;
}

function resolveRunbook(
  config: BenchlessConfig,
  configPath: string | undefined,
  problems: string[],
): void {
  if (config.runbook === undefined || configPath === undefined) return;
  const configDir = path.dirname(configPath);
  const rp = runbookProblem(config.runbook, configDir);
  if (rp !== undefined) {
    problems.push(rp);
  } else {
    config.runbook = path.resolve(configDir, config.runbook);
  }
}

export function inspectConfig(flags: CliFlags, baseDir?: string): ConfigInspection {
  const problems = collectFlagProblems(flags);

  const envResult = collectEnvFlags(flags);
  problems.push(...envResult.problems);

  const effective = buildEffectiveFlags(flags, envResult.flags);

  const source = resolveConfigSource(flags, baseDir);
  problems.push(...source.problems);
  const { configPath, configExists, configFile } = source;

  if (configFile === undefined) {
    return { configPath, configExists, problems };
  }

  const config = mergeConfig(effective, configFile);
  problems.push(...loopKeyProblems(config));
  resolveRunbook(config, configPath, problems);

  if (problems.length > 0) {
    return { configPath, configExists, problems };
  }

  const bench = effective.bench ?? configFile.bench;
  return {
    configPath,
    configExists,
    problems: [],
    config,
    ...(bench !== undefined ? { bench } : undefined),
  };
}

/** The kind an adapter's metric falls under when it reports none. */
export const DEFAULT_METRIC_KIND = "other";

// fallow-ignore-next-line complexity
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
