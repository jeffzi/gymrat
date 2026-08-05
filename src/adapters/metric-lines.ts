import { computeMedian } from "../math.js";
import { metricRecord } from "../metric-record.js";
import type { Adapter, MetricDefaults } from "./types.js";
import { AdapterError } from "./types.js";

const METRIC_PREFIX = "METRIC";

function parseMetricLine(line: string): { name: string; value: number } | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith(`${METRIC_PREFIX} `)) return null;

  const reject = (): null => {
    console.warn(`Failed to parse METRIC line: ${trimmed}`);
    return null;
  };

  const afterMetric = trimmed.slice(METRIC_PREFIX.length).trim();
  const lastEqIndex = afterMetric.lastIndexOf("=");

  if (lastEqIndex <= 0) {
    return reject();
  }

  // `Number("")` and `Number("   ")` are 0, so an unset shell variable would
  // otherwise be recorded as a genuine zero reading and pull the median down.
  const rawValue = afterMetric.slice(lastEqIndex + 1);
  const value = Number(rawValue);
  if (rawValue.trim() === "" || !Number.isFinite(value)) {
    return reject();
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
      .filter((r) => r !== null);

    if (parsed.length === 0) {
      throw new AdapterError("No valid METRIC lines found");
    }

    const grouped = Map.groupBy(parsed, ({ name }) => name);
    const medians = metricRecord<number>();
    for (const [name, values] of grouped) {
      medians[name] = computeMedian(values.map((v) => v.value));
    }
    return medians;
  },

  defaults(_metricName: string): MetricDefaults {
    return { direction: "lower" };
  },
};

export default metricLinesAdapter;
