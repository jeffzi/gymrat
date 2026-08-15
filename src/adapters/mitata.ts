import { messageOf } from "../errors.js";
import { metricRecord } from "../metric-record.js";
import type { Adapter, MetricDefaults, WarnSink } from "./types.js";
import { AdapterError, warnToStderr } from "./types.js";

function isRecord(val: unknown): val is Record<string, unknown> {
  return typeof val === "object" && val !== null;
}

/**
 * Characters a metric name may not carry.
 *
 * All four are line terminators to JavaScript's regular-expression engine, so an
 * anchored name check can never match a name holding one — gymrat must never
 * write a session record it cannot read back. Unlike `metric-lines`, whose input
 * is split on `\n` and `\r` before any name is read, mitata's JSON can carry
 * every one of them inside an alias or an argument value.
 */
const FORBIDDEN_NAME_CHARS = /[\n\r\u{2028}\u{2029}]/u;

/**
 * Find the outermost `{…}` JSON object using a brace/string-aware scan.
 *
 * Mitata's banner lines can contain bare braces (`cpu: {model}`) that trip up a
 * naive first-`{`-to-last-`}` slice. The scan tracks brace depth while skipping
 * quoted strings, so only a balanced top-level `{…}` pair is returned.
 */
// fallow-ignore-next-line complexity
function extractJson(stdout: string): Record<string, unknown> {
  let depth = 0;
  let start = -1;
  let lastParseError: unknown;

  for (let i = 0; i < stdout.length; i++) {
    const ch = stdout[i];
    if (ch === '"') {
      i++;
      while (i < stdout.length && stdout[i] !== '"') {
        if (stdout[i] === "\\") i++;
        i++;
      }
      continue;
    }
    if (ch === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "}") {
      depth--;
      if (depth === 0 && start !== -1) {
        const slice = stdout.slice(start, i + 1);
        try {
          const parsed: unknown = JSON.parse(slice);
          if (isRecord(parsed)) return parsed;
        } catch (err) {
          lastParseError = err;
        }
        start = -1;
      }
      if (depth < 0) depth = 0;
    }
  }

  if (lastParseError !== undefined) {
    throw new AdapterError(`Failed to parse JSON: ${messageOf(lastParseError)}`);
  }
  throw new AdapterError("No JSON object found in stdout");
}

function parseBenchmarks(json: Record<string, unknown>): unknown[] {
  const benchmarks = json.benchmarks;
  if (!Array.isArray(benchmarks)) {
    throw new AdapterError("JSON missing benchmarks array");
  }
  if (benchmarks.length === 0) {
    throw new AdapterError("benchmarks array is empty");
  }

  return benchmarks;
}

/**
 * Store `value` under `name`, warning when it displaces an earlier reading.
 *
 * A collision means two runs resolved to one metric name — an alias missing the
 * `$placeholder` for the argument that varies, or two benchmarks sharing an
 * alias — so the report would silently show only the last run's numbers.
 */
function recordMetric(
  metrics: Record<string, number>,
  name: string,
  value: number,
  warn: WarnSink,
): void {
  if (Object.hasOwn(metrics, name)) {
    warn(
      `Duplicate metric name: ${name} (keeping the last value; give the benchmark aliases distinct $placeholders to separate the runs)`,
    );
  }
  metrics[name] = value;
}

/** Read `stats.p50`, warning and returning `null` when it is missing or non-finite. */
function resolveP50(stats: Record<string, unknown>, alias: string, warn: WarnSink): number | null {
  const p50 = stats.p50;
  if (typeof p50 !== "number") {
    warn(`Skipping run with malformed stats shape: ${alias} (stats.p50 is not a number)`);
    return null;
  }
  if (!Number.isFinite(p50)) {
    warn(`Skipping run with non-finite p50: ${alias} (${p50})`);
    return null;
  }
  return p50;
}

/** Build the metric name prefix, warning and returning `null` when it carries a line terminator. */
function resolveMetricPrefix(
  alias: string,
  args: Record<string, unknown>,
  warn: WarnSink,
): string | null {
  const prefix = buildMetricNamePrefix(alias, args);
  if (FORBIDDEN_NAME_CHARS.test(prefix)) {
    warn(
      `Skipping run with a line terminator in its metric name: ${alias} (the alias or one of its argument values carries one)`,
    );
    return null;
  }
  return prefix;
}

/** Record `<prefix>/heap` from `stats.heap.avg` when mitata measured it. */
function recordHeapMetric(
  stats: Record<string, unknown>,
  prefix: string,
  metrics: Record<string, number>,
  warn: WarnSink,
): void {
  const heap = stats.heap;
  if (isRecord(heap) && typeof heap.avg === "number" && Number.isFinite(heap.avg)) {
    recordMetric(metrics, `${prefix}/heap`, heap.avg, warn);
  }
}

function extractRunMetrics(
  run: unknown,
  alias: string,
  metrics: Record<string, number>,
  warn: WarnSink,
): void {
  if (!isRecord(run)) return;
  if ("error" in run && run.error !== null) return;

  const args = run.args;
  const stats = run.stats;
  if (!isRecord(args) || !isRecord(stats)) return;

  const p50 = resolveP50(stats, alias, warn);
  if (p50 === null) return;

  const prefix = resolveMetricPrefix(alias, args, warn);
  if (prefix === null) return;

  recordMetric(metrics, `${prefix}/time`, p50, warn);
  recordHeapMetric(stats, prefix, metrics, warn);
}

/**
 * Serialize a run-argument value for inclusion in a metric name.
 *
 * Primitives keep their `String()` form; objects and arrays are serialized via
 * `JSON.stringify` with sorted keys so that two structurally equal objects
 * always produce the same metric name.
 */
function serializeArgValue(value: unknown): string {
  if (typeof value !== "object" || value === null) return String(value);
  return JSON.stringify(value, (_key: string, v: unknown) => {
    if (typeof v !== "object" || v === null || Array.isArray(v)) return v;
    const sorted: Record<string, unknown> = {};
    for (const k of Object.keys(v).toSorted()) {
      // oxlint-disable-next-line no-unsafe-type-assertion -- v is a non-null, non-array object
      sorted[k] = (v as Record<string, unknown>)[k];
    }
    return sorted;
  });
}

/**
 * Substitute each `$key` in `alias` with `key=value`, scanning the alias once.
 *
 * Both the scan and the literal splice are deliberate. `String.replace` reads
 * `$&`, `` $` ``, `$'` and `$<n>` in its replacement argument as patterns, and
 * an argument value is user data that may contain them. Repeated passes would
 * be just as wrong the other way: an earlier value containing `$b` would be
 * eaten by the pass for key `b`. Substituted text is written straight to the
 * output, so it is never scanned again.
 *
 * Keys are matched longest-first, so an alias of `$ab` picks the argument `ab`
 * over the argument `a`.
 */
function buildMetricNamePrefix(alias: string, args: Record<string, unknown>): string {
  const keys = Object.keys(args).toSorted((a, b) => b.length - a.length);

  let result = "";
  let cursor = 0;
  let dollar = alias.indexOf("$");
  while (dollar !== -1) {
    const key = keys.find((candidate) => alias.startsWith(candidate, dollar + 1));
    if (key === undefined) {
      result += alias.slice(cursor, dollar + 1);
      cursor = dollar + 1;
    } else {
      result += alias.slice(cursor, dollar) + `${key}=${serializeArgValue(args[key])}`;
      cursor = dollar + 1 + key.length;
    }
    dollar = alias.indexOf("$", cursor);
  }

  return result + alias.slice(cursor);
}

function extractBenchmarkMetrics(
  benchmark: unknown,
  metrics: Record<string, number>,
  warn: WarnSink,
): void {
  if (!isRecord(benchmark)) return;

  const alias = benchmark.alias;
  const runs = benchmark.runs;
  if (typeof alias !== "string" || !Array.isArray(runs)) return;

  for (const run of runs) {
    extractRunMetrics(run, alias, metrics, warn);
  }
}

const METRIC_SUFFIXES = [
  { suffix: "/time", unit: "ns", kind: "time" },
  { suffix: "/heap", unit: "bytes", kind: "memory" },
] as const satisfies readonly { suffix: string; unit: MetricDefaults["unit"]; kind: string }[];

/**
 * Adapter for bench scripts that print the JSON `mitata --json` writes.
 *
 * Each benchmark becomes `<alias>/time` and, when mitata measured it,
 * `<alias>/heap`.
 */
const mitataAdapter: Adapter = {
  name: "mitata",

  /**
   * Reads mitata's JSON output — a `benchmarks` array whose entries carry an
   * `alias` and a list of `runs`.
   *
   * The JSON is located by slicing between the first `{` and the last `}` so that
   * banner text mitata prints around it does not have to be stripped by the user.
   *
   * Each run yields `<alias>/time` from `stats.p50` and, when mitata measured it,
   * `<alias>/heap` from `stats.heap.avg`. For parameterized benchmarks the `$name`
   * placeholders in the alias are substituted with `name=value`, so one mitata
   * benchmark becomes one metric per argument combination rather than collapsing
   * them all onto the same name.
   *
   * Runs that errored, reported a non-finite `p50`, or resolved to a metric name
   * carrying a line terminator are skipped rather than failing the parse — a
   * single bad argument combination should not discard the rest of the run — but a
   * parse that finds no usable run at all raises {@link AdapterError}. Both the
   * skipped-name notice and a collision between two runs landing on one metric
   * name go to `warn`, which defaults to stderr; on a collision the last run still
   * wins.
   */
  parse(stdout: string, warn: WarnSink = warnToStderr): Record<string, number> {
    const json = extractJson(stdout);
    const benchmarks = parseBenchmarks(json);
    const metrics = metricRecord<number>();
    for (const benchmark of benchmarks) {
      extractBenchmarkMetrics(benchmark, metrics, warn);
    }

    if (Object.keys(metrics).length === 0) {
      throw new AdapterError("No valid benchmark runs found");
    }

    return metrics;
  },

  defaults(metricName: string): MetricDefaults {
    for (const { suffix, unit, kind } of METRIC_SUFFIXES) {
      if (metricName.endsWith(suffix)) {
        return {
          direction: "lower",
          unit,
          kind,
          shortName: metricName.slice(0, -suffix.length),
        };
      }
    }

    return { direction: "lower" };
  },
};

export default mitataAdapter;
