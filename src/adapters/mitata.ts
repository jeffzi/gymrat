import type { Adapter, MetricDefaults } from "./types.js";
import { AdapterError } from "./types.js";

function isRecord(val: unknown): val is Record<string, unknown> {
  return typeof val === "object" && val !== null;
}

function extractJson(stdout: string): Record<string, unknown> {
  const startIdx = stdout.indexOf("{");
  const endIdx = stdout.lastIndexOf("}");

  if (startIdx === -1 || endIdx === -1 || startIdx >= endIdx) {
    throw new AdapterError("No JSON object found in stdout");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout.slice(startIdx, endIdx + 1)) as unknown;
  } catch (err) {
    throw new AdapterError(
      `Failed to parse JSON: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

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

function extractRunMetrics(run: unknown, alias: string, metrics: Record<string, number>): boolean {
  if (!isRecord(run)) return false;
  if ("error" in run && run.error !== null) return false;

  const args = run.args;
  const stats = run.stats;
  if (!isRecord(args) || !isRecord(stats)) return false;

  const p50 = stats.p50;
  if (typeof p50 !== "number") return false;

  const prefix = buildMetricNamePrefix(alias, args);
  metrics[`${prefix}/time`] = p50;

  const heap = stats.heap;
  if (isRecord(heap) && typeof heap.avg === "number") {
    metrics[`${prefix}/heap`] = heap.avg;
  }

  return true;
}

function buildMetricNamePrefix(alias: string, args: Record<string, unknown>): string {
  let result = alias;
  for (const [key, value] of Object.entries(args)) {
    result = result.replaceAll(`$${key}`, `${key}=${String(value)}`);
  }
  return result;
}

/**
 * Reads mitata's JSON output — the shape `mitata --json` writes, with a
 * `benchmarks` array whose entries carry an `alias` and a list of `runs`.
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
 * Runs that errored are skipped rather than failing the parse — a single bad
 * argument combination should not discard the rest of the run — but a parse that
 * finds no usable run at all raises {@link AdapterError}.
 */
const mitataAdapter: Adapter = {
  name: "mitata",

  parse(stdout: string): Record<string, number> {
    const json = extractJson(stdout);
    const benchmarks = parseBenchmarks(json);
    const metrics: Record<string, number> = {};
    let metricsFound = 0;

    for (const benchmark of benchmarks) {
      if (!isRecord(benchmark)) continue;

      const alias = benchmark.alias;
      const runs = benchmark.runs;
      if (typeof alias !== "string" || !Array.isArray(runs)) continue;

      for (const run of runs) {
        if (extractRunMetrics(run, alias, metrics)) {
          metricsFound++;
        }
      }
    }

    if (metricsFound === 0) {
      throw new AdapterError("No valid benchmark runs found");
    }

    return metrics;
  },

  defaults(metricName: string): MetricDefaults {
    if (metricName.endsWith("/time")) {
      return { direction: "lower", unit: "ns" };
    }

    if (metricName.endsWith("/heap")) {
      return { direction: "lower", unit: "bytes" };
    }

    return { direction: "lower" };
  },
};

export default mitataAdapter;
