import type { Adapter, MetricDefaults } from "./types.js";
import { AdapterError } from "./types.js";

function isRecord(val: unknown): val is Record<string, unknown> {
  return typeof val === "object" && val !== null;
}

function extractJson(stdout: string): Record<string, unknown> {
  const startIdx = stdout.indexOf("{");
  const endIdx = stdout.lastIndexOf("}");

  if (startIdx === -1 || endIdx === -1 || startIdx >= endIdx) {
    throw new AdapterError("AdapterError: No JSON object found in stdout");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(stdout.slice(startIdx, endIdx + 1)) as unknown;
  } catch (err) {
    throw new AdapterError(
      `AdapterError: Failed to parse JSON: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  if (!isRecord(parsed)) {
    throw new AdapterError("AdapterError: JSON must be an object");
  }

  return parsed;
}

function parseBenchmarks(json: Record<string, unknown>): unknown[] {
  const benchmarks = json.benchmarks;
  if (!Array.isArray(benchmarks)) {
    throw new AdapterError("AdapterError: JSON missing benchmarks array");
  }
  if (benchmarks.length === 0) {
    throw new AdapterError("AdapterError: benchmarks array is empty");
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
      throw new AdapterError("AdapterError: No valid benchmark runs found");
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
