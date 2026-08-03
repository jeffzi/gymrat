/**
 * Build a record keyed by metric name, with no prototype.
 *
 * Metric names come straight from benchmark output, so a bench is free to emit
 * one called `toString`, `constructor` or `__proto__`. On an ordinary object
 * those names are not free: reading one a target never reported hands back an
 * inherited function instead of `undefined`, so a one-sided metric looks present
 * on both sides, and assigning `__proto__` re-parents the object rather than
 * storing the value. Keeping every metric-keyed record prototype-less is what
 * makes such a name an ordinary key throughout the pipeline.
 *
 * `JSON.stringify` treats the result exactly like a plain object.
 */
export function metricRecord<T>(entries: Iterable<readonly [string, T]> = []): Record<string, T> {
  const record: Record<string, T> = Object.fromEntries(entries);
  Object.setPrototypeOf(record, null);
  return record;
}
