/**
 * A metric-keyed record built the way the pipeline builds them: with no prototype.
 *
 * Metric names come straight from benchmark output, so a bench is free to emit
 * one called `toString` or `__proto__`. Every record the pipeline keys by metric
 * name is therefore created without an `Object.prototype` chain, and
 * `toStrictEqual` compares prototypes — an expectation for such a record has to
 * be built the same way.
 */
export function metricRecord<T>(entries: Record<string, T>): Record<string, T> {
  const record: Record<string, T> = { ...entries };
  Object.setPrototypeOf(record, null);
  return record;
}
