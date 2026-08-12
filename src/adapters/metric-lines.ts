import { computeMedian } from "../math.js";
import { metricRecord } from "../metric-record.js";
import type { Adapter, MetricDefaults, WarnSink } from "./types.js";
import { AdapterError, warnToStderr } from "./types.js";

const METRIC_PREFIX = "METRIC";

// A bench that redraws a progress line ends it with a bare `\r`, so a split on
// `\n` alone folds the metric printed after it into the progress text.
const LINE_TERMINATOR = /\r\n|[\n\r]/;

/**
 * Characters a metric name may not carry.
 *
 * U+2028 (line separator) and U+2029 (paragraph separator) are line
 * terminators to JavaScript's regular-expression engine, so a name holding one
 * cannot pass the session record's name check — gymrat must never write a
 * record it cannot read back.
 */
const FORBIDDEN_NAME_CHARS = new RegExp(`[${String.fromCodePoint(0x2028, 0x2029)}]`, "u");

function parseMetricLine(line: string, warn: WarnSink): { name: string; value: number } | null {
  const trimmed = line.trim();

  const reject = (): null => {
    warn(`Failed to parse METRIC line: ${trimmed}`);
    return null;
  };

  if (!trimmed.startsWith(`${METRIC_PREFIX} `)) {
    // A line starting with METRIC but missing the separating space
    // (`METRICfoo=1`, `METRIC_foo=1`, `METRICS foo=1`) is a typo in the bench
    // script, not ordinary output, so it is reported rather than dropped.
    return trimmed.startsWith(METRIC_PREFIX) ? reject() : null;
  }

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

  const name = afterMetric.slice(0, lastEqIndex);
  if (FORBIDDEN_NAME_CHARS.test(name)) {
    return reject();
  }

  return { name, value };
}

/**
 * Adapter for bench scripts that print `METRIC name=value` lines to stdout.
 *
 * When a metric name appears more than once, the median of all its values is
 * returned.
 */
const metricLinesAdapter: Adapter = {
  name: "metric-lines",

  parse(stdout: string, warn: WarnSink = warnToStderr): Record<string, number> {
    const parsed = stdout
      .split(LINE_TERMINATOR)
      .map((line) => parseMetricLine(line, warn))
      .filter((r) => r !== null);

    if (parsed.length === 0) {
      throw new AdapterError("No valid METRIC lines found");
    }

    return metricRecord(
      Array.from(
        Map.groupBy(parsed, ({ name }) => name),
        ([name, values]) => [name, computeMedian(values.map((v) => v.value))],
      ),
    );
  },

  defaults(_metricName: string): MetricDefaults {
    return { direction: "lower" };
  },
};

export default metricLinesAdapter;
