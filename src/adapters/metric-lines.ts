import { computeMedian } from "../math.js";
import type { Adapter, MetricDefaults } from "./types.js";
import { AdapterError } from "./types.js";

const METRIC_PREFIX = "METRIC";

function parseMetricLine(line: string): { name: string; value: number } | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith(METRIC_PREFIX)) return null;

  const afterMetric = trimmed.slice(METRIC_PREFIX.length).trim();
  const lastEqIndex = afterMetric.lastIndexOf("=");

  if (lastEqIndex <= 0) {
    console.warn(`Failed to parse METRIC line: ${trimmed}`);
    return null;
  }

  const value = Number(afterMetric.slice(lastEqIndex + 1));
  if (!Number.isFinite(value)) {
    console.warn(`Failed to parse METRIC line: ${trimmed}`);
    return null;
  }

  return { name: afterMetric.slice(0, lastEqIndex), value };
}

/**
 * Adapter for bench scripts that print `METRIC name=value` lines to stdout.
 *
 * When a metric name appears more than once, the median of all its values is
 * returned.
 */
const metricLinesAdapter: Adapter = {
  name: "metric-lines",

  parse(stdout: string): Record<string, number> {
    const parsed = stdout
      .split("\n")
      .map(parseMetricLine)
      .filter((r): r is { name: string; value: number } => r !== null);

    if (parsed.length === 0) {
      throw new AdapterError("No valid METRIC lines found");
    }

    const metrics = new Map<string, number[]>();
    for (const { name, value } of parsed) {
      if (!metrics.has(name)) metrics.set(name, []);
      metrics.get(name)!.push(value);
    }

    return Object.fromEntries(
      [...metrics.entries()].map(([name, values]) => [name, computeMedian(values)]),
    );
  },

  defaults(_metricName: string): MetricDefaults {
    return { direction: "lower" };
  },
};

export default metricLinesAdapter;
