import { metricRecord as createMetricRecord } from "../../src/metric-record.js";

/**
 * Ergonomic wrapper for test expectations: accepts an object literal instead of
 * the iterable the production `metricRecord` takes.
 *
 * Delegates to the production function so prototype-stripping logic lives in one
 * place, and `toStrictEqual` sees the same prototype chain the pipeline produces.
 */
export function metricRecord<T>(entries: Record<string, T> = {}): Record<string, T> {
  return createMetricRecord(Object.entries(entries));
}
