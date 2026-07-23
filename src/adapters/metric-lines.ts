import type { Adapter, MetricDefaults } from "./types.js";
import { AdapterError } from "./types.js";

// Parses METRIC name=value lines; for repeated metrics returns median.
// Splits at LAST = since metric names may contain =. Rejects non-finite values.
const metricLinesAdapter: Adapter = {
  name: "metric-lines",

  parse(stdout: string): Record<string, number> {
    const lines = stdout.split("\n");
    const metrics = new Map<string, number[]>();

    for (const line of lines) {
      const trimmed = line.trim();

      if (!trimmed.startsWith("METRIC")) {
        continue;
      }

      const afterMetric = trimmed.slice(6).trim();
      const lastEqIndex = afterMetric.lastIndexOf("=");

      if (lastEqIndex === -1 || lastEqIndex === 0) {
        console.warn(`Failed to parse METRIC line: ${trimmed}`);
        continue;
      }

      const name = afterMetric.slice(0, lastEqIndex);
      const valueStr = afterMetric.slice(lastEqIndex + 1);
      const value = Number(valueStr);

      if (!Number.isFinite(value)) {
        console.warn(`Failed to parse METRIC line: ${trimmed}`);
        continue;
      }

      if (!metrics.has(name)) {
        metrics.set(name, []);
      }
      metrics.get(name)!.push(value);
    }

    if (metrics.size === 0) {
      throw new AdapterError("AdapterError: No valid METRIC lines found");
    }

    const result: Record<string, number> = {};
    for (const [name, values] of metrics.entries()) {
      result[name] = computeMedian(values);
    }

    return result;
  },

  defaults(_metricName: string): MetricDefaults {
    return { direction: "lower" };
  },
};

function computeMedian(values: number[]): number {
  const sorted = values.toSorted((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

export default metricLinesAdapter;
