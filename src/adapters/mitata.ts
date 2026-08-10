import { messageOf } from "../errors.js";
import { metricRecord } from "../metric-record.js";
import type { Adapter, MetricDefaults, WarnSink } from "./types.js";
import { AdapterError, warnToStderr } from "./types.js";

function isRecord(val: unknown): val is Record<string, unknown> {
  return typeof val === "object" && val !== null;
}

function extractJson(stdout: string): Record<string, unknown> {
  const startIdx = stdout.indexOf("{");
  const endIdx = stdout.lastIndexOf("}");

  if (startIdx < 0 || startIdx >= endIdx) {
    throw new AdapterError("No JSON object found in stdout");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout.slice(startIdx, endIdx + 1));
  } catch (err) {
    throw new AdapterError(`Failed to parse JSON: ${messageOf(err)}`);
  }

  /* v8 ignore next 3 -- first-`{`-to-last-`}` slice cannot parse to a non-object without throwing */
  if (!isRecord(parsed)) {
    throw new AdapterError("JSON must be an object");
  }

  return parsed;
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

function parseMitataStats(
  run: unknown,
): { args: Record<string, unknown>; p50: number; heapAvg?: number } | undefined {
  if (!isRecord(run)) return undefined;
  if ("error" in run && run.error !== null) return undefined;

  const args = run.args;
  const stats = run.stats;
  if (!isRecord(args) || !isRecord(stats)) return undefined;

  // `1e999` in JSON parses to Infinity, so a reading that overflowed is a number
  // here. It would poison any median or geomean it entered, so the run counts as
  // unusable in exactly the way a malformed one does.
  const p50 = stats.p50;
  if (typeof p50 !== "number" || !Number.isFinite(p50)) return undefined;

  const heap = stats.heap;
  const heapAvg =
    isRecord(heap) && typeof heap.avg === "number" && Number.isFinite(heap.avg)
      ? heap.avg
      : undefined;
  return { args, p50, ...(heapAvg === undefined ? {} : { heapAvg }) };
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

function extractRunMetrics(
  run: unknown,
  alias: string,
  metrics: Record<string, number>,
  warn: WarnSink,
): void {
  const parsed = parseMitataStats(run);
  if (parsed === undefined) return;

  const prefix = buildMetricNamePrefix(alias, parsed.args);
  recordMetric(metrics, `${prefix}/time`, parsed.p50, warn);

  if (parsed.heapAvg !== undefined) {
    recordMetric(metrics, `${prefix}/heap`, parsed.heapAvg, warn);
  }
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
      result += alias.slice(cursor, dollar) + `${key}=${String(args[key])}`;
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
   * Runs that errored or reported a non-finite `p50` are skipped rather than
   * failing the parse — a single bad argument combination should not discard the
   * rest of the run — but a parse that finds no usable run at all raises
   * {@link AdapterError}. Two runs landing on one metric name go to `warn`, which
   * defaults to stderr; the last one still wins.
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
