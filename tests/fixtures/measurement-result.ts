import type { MeasurementResult, MetricMeasurement } from "../../src/report/types.js";
import { metricMeta } from "./comparison-result.js";

/**
 * One metric of a single-target run: what it measured, and how steady it was.
 *
 * `spread` is a percentage of the median, the same figure a comparison prints
 * beside a side's value. Passing `undefined` is how a caller pins the
 * single-sample case, where there is no run-to-run jitter to report.
 */
export function measuredMetric(
  options: {
    median?: number | undefined;
    spread?: number | undefined;
    shortName?: string;
    kind?: string;
    unit?: "ns" | "bytes";
    gating?: boolean;
  } = {},
): MetricMeasurement {
  const { shortName = "time", kind = "other", unit, gating = true } = options;
  const median = "median" in options ? options.median : 100;
  const spread = "spread" in options ? options.spread : 1;
  return {
    ...(median !== undefined && { median }),
    ...(spread !== undefined && { spread }),
    meta: metricMeta(shortName, { kind, unit, gating }),
  };
}

/**
 * A measurement of a clean single-target run with no metrics.
 *
 * Shared by the renderer tests so the text and JSON reports are driven with the
 * same shape `measure()` returns.
 */
export function createMeasurementResult(
  overrides: Partial<MeasurementResult> = {},
): MeasurementResult {
  return {
    label: "main",
    samples: 10,
    adapter: "mitata",
    metrics: {},
    rounds: [],
    worktreesRemoved: 0,
    worktreesLeftBehind: [],
    worktreePruneError: undefined,
    ...overrides,
  };
}

/**
 * A measurement spanning a gating `time` kind and an informational `memory` kind.
 *
 * `time` holds a two-metric `entity` group beside an ungrouped `warmup`, so its
 * rendered section carries both a group block and a bare row; `memory` holds one
 * ungrouped metric and gates nothing, so it pins the informational tag.
 */
export function twoKindMeasurement(overrides: Partial<MeasurementResult> = {}): MeasurementResult {
  return createMeasurementResult({
    metrics: {
      "entity.alive_check/time": measuredMetric({
        kind: "time",
        shortName: "entity.alive_check",
        unit: "ns",
      }),
      "entity.spawn/time": measuredMetric({
        kind: "time",
        shortName: "entity.spawn",
        median: 104,
        unit: "ns",
      }),
      "warmup/time": measuredMetric({ kind: "time", shortName: "warmup", unit: "ns" }),
      "encode/heap": measuredMetric({
        kind: "memory",
        shortName: "encode",
        median: 93,
        unit: "bytes",
        gating: false,
      }),
    },
    configKinds: { memory: { gating: false } },
    ...overrides,
  });
}
